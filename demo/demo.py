import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel

from typing import Dict

import time
import json
import os
import os.path as osp
import sys

from pathlib import Path

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '..', 'common'))
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '..', 'main'))

# from HandNeRF code
from model import get_model as get_handnerf_model
from utils.ray_sampling import sample_ray
from utils.geometry3d_utils import get_grid_points_from_bounds, Rx, Ry, Rz
from config import cfg, mano_layer
from utils.vis import save_obj

# from HandOccNet code
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '..'))
from HandOccNet.main.model import get_model as get_handoccnet_model
from HandOccNet.common.utils.preprocessing import load_img, process_bbox, generate_patch_image
from HandOccNet.common.utils.camera import PerspectiveCamera
# for vis
from HandOccNet.common.utils.vis import vis_keypoints_with_skeleton
from HandOccNet.common.utils.mano import MANO 
mano = MANO()


# data parsing
def make_cam(cam_data):
    # setup camera
    camera = PerspectiveCamera()
    camera.focal_length_x = torch.full([1], cam_data['fx'])
    camera.focal_length_y = torch.full([1], cam_data['fy'])
    camera.center = torch.tensor([cam_data['cx'], cam_data['cy']]).unsqueeze(0)

    camera.rotation.requires_grad = False
    camera.translation.requires_grad = False
    camera.name = 'wrist_cam'

    return camera

# for handnerf input
def load_input_view_data(K, bbox, input_img_shape, aug_R, data: Dict[str, np.ndarray] = None):
    if data is None:
        data = {}
   
    # adjust K
    new_K = K.copy()
    new_K[0, 0] = K[0, 0] * input_img_shape[1] / bbox[2]
    new_K[1, 1] = K[1, 1] * input_img_shape[0] / bbox[3]
    new_K[0, 2] = (K[0, 2] - bbox[0]) * input_img_shape[1] / bbox[2]
    new_K[1, 2] = (K[1, 2] - bbox[1]) * input_img_shape[0] / bbox[3]
    data['input_view_K'] = new_K

    """ aug R, t to prevent 3D conv overfitting """
    input_cam_R = np.eye(3, dtype=np.float32)
    input_cam_t = np.zeros((3,1), dtype=np.float32)

    # augmenting rotation matrix that rotates coordinate axes of the input camera

    # get camera orientation and location
    ref_cam_ori = input_cam_R.T @ aug_R
    ref_cam_center = - input_cam_R.T @ input_cam_t

    # new transformation matrices
    ref_cam_R = ref_cam_ori.T
    ref_cam_t = - ref_cam_ori.T @ ref_cam_center

    # get camera orientation and location
    new_R = input_cam_R @ ref_cam_R.T  # actually just aug_R
    new_t = input_cam_t - new_R @ ref_cam_t

    data['input_view_R'] = new_R
    data['input_view_t'] = new_t

    # save inverse matrices to transform 3d points back to the exact input camera coordinate system
    data['inv_aug_R'] = input_cam_R @ ref_cam_R.T
    data['inv_aug_t'] = input_cam_t - data['inv_aug_R'] @ ref_cam_t
    
    return ref_cam_R, ref_cam_t, data

# for handnerf input
def load_world_3d_data(mode, hand_mesh, input_mesh_voxel_size=[0.005, 0.005, 0.005], bound_padding=0.25, data: Dict[str, np.ndarray] = None):
    """
    hand_mesh: (778, 3) in world
    # pred_hand_mesh: (778, 3) in world, estimated by off-the-shelf. 

    // Returns //
    hand_center: (3,1) in world
    world_bounds: (2,3) in world
    world_hand_mesh_voxel_coord: (778, 3), coordinates of world mesh vertices in the descritized volume
    world_mesh_voxel_out_sh: (3,) np.int32, shape of the world mesh volume
    """
    if data is None:
        data = {}

    # (778, 3)
    mesh = np.concatenate([hand_mesh])

    # get center
    hand_center = mesh.mean(axis=0)[:, None]  # (3,1)

    # obtain the world bounds for point sampling
    min_xyz = np.min(mesh, axis=0)
    max_xyz = np.max(mesh, axis=0)
    min_xyz -= bound_padding
    max_xyz += bound_padding
    world_bounds = np.stack([min_xyz, max_xyz], axis=0)

    # construct the voxel coordinate in the world coordinate
    dhw = hand_mesh[:, [2, 1, 0]]
    assert dhw.shape[0] == 778, "CAUTION! Don't use object mesh for dhw!"
    min_dhw = min_xyz[[2, 1, 0]]
    max_dhw = max_xyz[[2, 1, 0]]
    voxel_size = np.array(input_mesh_voxel_size, dtype=np.float32)
    world_hand_mesh_voxel_coord = np.round(
        (dhw - min_dhw) / voxel_size).astype(np.int32)

    # construct the output shape; that includes BOTH hand and object
    world_mesh_voxel_out_sh = np.ceil(
        (max_dhw - min_dhw) / voxel_size).astype(np.int32)
    x = 32
    # discretize world_mesh_voxel_out_sh to x (32) by rounding up; world_mesh_voxel_out_sh will be one of 32, 64, 96, ...
    world_mesh_voxel_out_sh = (world_mesh_voxel_out_sh | (x - 1)) + 1

    assert (world_hand_mesh_voxel_coord[:, 0] >= 0).all() and (world_hand_mesh_voxel_coord[:, 0] < world_mesh_voxel_out_sh[0]).all() and (world_hand_mesh_voxel_coord[:, 1] >= 0).all() and (
        world_hand_mesh_voxel_coord[:, 1] < world_mesh_voxel_out_sh[1]).all() and (world_hand_mesh_voxel_coord[:, 2] >= 0).all() and (world_hand_mesh_voxel_coord[:, 2] < world_mesh_voxel_out_sh[2]).all()

    data['world_bounds'] = world_bounds
    data['world_hand_mesh_voxel_coord'] = world_hand_mesh_voxel_coord
    data['world_mesh_voxel_out_sh'] = world_mesh_voxel_out_sh

    if mode == 'recon3d':
        # voxel resolution for marching cube
        world_grid_points = get_grid_points_from_bounds(world_bounds, cfg.mc_voxel_size, mode)
        world_inside = np.ones_like(world_grid_points[..., 0])
        print("Reconstructing... Input world points shape: ", world_grid_points.shape)
        data['inside'] = world_inside
        data['pts'] = world_grid_points.astype(np.float32)
    
    return hand_center

# for handnerf input
def load_2d_render_view(img, world_bounds, R, t, K, camera_center, world_center, render_img_shape=(256,256), render_whole=True, data: Dict[str, np.ndarray] = None):
    if data is None:
        data = {}
    new_K = K.copy()
    new_K[0, 0] = K[0, 0] * render_img_shape[1] / bbox[2]
    new_K[1, 1] = K[1, 1] * render_img_shape[0] / bbox[3]
    new_K[0, 2] = (K[0, 2] - bbox[0]) * render_img_shape[1] / bbox[2]
    new_K[1, 2] = (K[1, 2] - bbox[1]) * render_img_shape[0] / bbox[3]

    mask = np.ones_like(img[:, :, 0])

    rgb, fg_mask, ray_o, ray_d, near, far, coord_, mask_at_box = sample_ray(
        img, mask, new_K, R, t, world_bounds, 1024, 'test', render_whole, camera_center[:, 0], world_center[:, 0], from_world_bounds=True)

    # org_mask is for evaluating object and hand separately
    org_mask = fg_mask.copy()

    # Make sure fg_mask contains [0,1]
    fg_mask[fg_mask != 0] = 1.

    _data = {
        'rgb': rgb, 'fg_mask': fg_mask, 'org_mask': org_mask, 'ray_o': ray_o, 'ray_d': ray_d, 'near': near, 'far': far, 'mask_at_box': mask_at_box
    }
    data.update(_data)
    return data



def load_handnerf_input(mode, K, cam_hand_mesh, cam_hand_joints, mano_side='right', input_img_shape=(256,256)):
    # input_cam_R = np.eye(3)
    # input_cam_t = np.zeros(3)
    rotate_degrees = [np.pi/10, np.pi/10, np.pi/10]
    aug_R = Rz(rotate_degrees[2]) @ Ry(rotate_degrees[1]) @ Rx(rotate_degrees[0])

    input_view_parameters = {}
    aug_cam_R, aug_cam_t, _  = load_input_view_data(K, bbox, input_img_shape, aug_R, data=input_view_parameters)
    # 'input_view_R': input_view_R, 'input_view_t': input_view_t, 'input_view_K': input_view_K, inv_aug_R, inv_aug_t

    world_hand_mesh = (aug_cam_R @ cam_hand_mesh.T).T + aug_cam_t.T  # just 3D cam mesh in auged space
    world_hand_joints = (aug_cam_R @ cam_hand_joints.T).T + aug_cam_t.T
    world_3d_data = {'world_hand_mesh': world_hand_mesh, 'world_hand_joints': world_hand_joints, 'mano_side': mano_side}
    world_center = load_world_3d_data(mode, hand_mesh=world_hand_mesh, data=world_3d_data)
    # + 'world_bounds': world_bounds, 'world_hand_mesh_voxel_coord': world_hand_mesh_voxel_coord, 'world_mesh_voxel_out_sh': world_mesh_voxel_out_sh, 

    camera_center = - input_view_parameters['input_view_R'].T @ input_view_parameters['input_view_t'] 
    rendering_rays = {}
    if mode != 'recon3d':
        load_2d_render_view(img, world_3d_data['world_bounds'], input_view_parameters['input_view_R'], input_view_parameters['input_view_t'], K, camera_center, world_center, render_img_shape=input_img_shape, render_whole=True, data=rendering_rays)
    # 'rgb': rgb, 'fg_mask': fg_mask, 'org_mask': org_mask, 'ray_o': ray_o, 'ray_d': ray_d, 'near': near, 'far': far, 'mask_at_box': mask_at_box

    return input_view_parameters, world_3d_data, rendering_rays 



# run handnerf
def run_handnerf(mode, save_dir, model, img, cam_data, input_img_shape, hand_info):
    
    K = np.array([[cam_data['fx'], 0., cam_data['cx']], 
              [ 0., cam_data['fy'], cam_data['cy']], 
              [0., 0., 1.]], dtype=np.float32)
    cam_hand_mesh, cam_hand_joints, mano_side = hand_info
    input_view_parameters, world_3d_data, rendering_rays = \
        load_handnerf_input(mode, K, cam_hand_mesh, cam_hand_joints, mano_side, input_img_shape)

    for input_item in [input_view_parameters, world_3d_data, rendering_rays]:
        for k, v in input_item.items():
            if type(v) == np.ndarray:
                input_item[k] = torch.from_numpy(v).cuda()[None, ...]
            if 'input' in k:
                input_item[k] = input_item[k][None, ...]

    start = time.time()
    with torch.no_grad():
        # DP issue in model()...
        output = model.module.forward(img, input_view_parameters, world_3d_data, rendering_rays)
    end = time.time()

    print(f"HandNeRF GPU run: {end-start:.4f}" )

    if mode == 'recon3d':

        # transform 3d points to the exact input camera coorindate
        inv_aug_R = input_view_parameters['inv_aug_R'].cpu() # (1, 3, 3) 
        inv_aug_t = input_view_parameters['inv_aug_t'].cpu() # (1, 3, 1)
        row = torch.Tensor([0, 0, 0, 1]).reshape(1,1,4)
        tmp = torch.cat([inv_aug_R, inv_aug_t], dim=-1)
        Rt_4by4 = torch.cat([tmp, row], dim=1)
        for mesh_name, mesh in output.items(): # Trimesh mesh object
            if 'object' not in mesh_name:
                continue

            # exception handling
            if mesh.vertices.size == 0:
                print(f"No valid mesh for {mesh_name}")
                continue
            
            mesh.apply_transform(Rt_4by4[0].numpy())
            file_path = osp.join(save_dir, f'handnerf_{mesh_name}.ply')

            mesh.export(file_path)
            print(f"Saved mesh to {file_path}")

# run handoccnet
def run_handoccnet(model, img, camera):
    # forward pass to the model
    inputs = {'img': img} # cfg.input_img_shape[1], cfg.input_img_shape[0], 3
    targets = {}
    meta_info = {}
    with torch.no_grad():
        out = model(inputs, targets, meta_info, 'test')
    img = (img[0].cpu().numpy().transpose(1, 2, 0)*255).astype(np.uint8) # 
    verts_cam = out['mesh_coord_cam'][0].cpu().numpy()
    
    # get hand mesh's scale and translation by fitting joint cam to joint img
    joint_img, joints_cam = out['joints_coord_img'], out['joints_coord_cam']
    
    # denormalize joint_img from 0 ~ 1 to actual 0 ~ original height and width
    H, W = img.shape[:2]
    joint_img[:, :, 0] *= W
    joint_img[:, :, 1] *= H
    torch_bb2img_trans = torch.tensor(bb2img_trans).to(joint_img)
    homo_joint_img = torch.cat([joint_img, torch.ones_like(joint_img[:, :, :1])], dim=2)
    org_res_joint_img = homo_joint_img @ torch_bb2img_trans.transpose(0, 1)

    # depth initialization
    # depth = np.asarray(Image.open(depth_path))

    hand_scale, hand_translation = model.module.get_mesh_scale_trans(org_res_joint_img, joints_cam, camera, None)
    
    hand_scale, hand_translation = hand_scale.detach().cpu().numpy(), hand_translation.detach().cpu().numpy()
    org_res_joint_img = org_res_joint_img[0].cpu().numpy()
    joints_cam = joints_cam[0].detach().cpu().numpy()
    joints_cam = hand_scale * joints_cam + hand_translation
    verts_cam = hand_scale * verts_cam + hand_translation

    return verts_cam, joints_cam, org_res_joint_img

if __name__ == '__main__':
    mode = 'recon3d'
    rgb_path = 'input.jpg' #
    save_dir = './output'
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    
    # Input setting
    cam_data = {
        "fx": 425.4889026887061,
        "fy": 430.7943259159451,
        "cx": 430.4445264563152,
        "cy": 279.6149582222223,
    }
    mano_hand_side = 'right'
    input_img_shape = (256, 256)
    bbox_path = 'input.json' 
    with open(bbox_path, 'r') as f:
        ann = json.load(f)
    xs, ys = [], []
    for label in ann['shapes']:
        xyxy = np.array(label['points'], dtype=np.float32).reshape(-1)
        xs.extend([xyxy[0], xyxy[2]])
        ys.extend([xyxy[1], xyxy[3]])
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    bbox = [x1, y1, x2 - x1, y2 - y1]
    
    # HandNeRF Setting
    cfg.mc_voxel_size = [0.005, 0.005, 0.005]
    cfg.do_object_segmentation = True

    # load handoccnet
    model_path = 'handoccnet_demo_model.pth.tar'
    assert osp.exists(model_path), 'Cannot find model at ' + model_path
    print('Load checkpoint from {}'.format(model_path))
    handoccnet_model = get_handoccnet_model('test')
    handoccnet_model = DataParallel(handoccnet_model).cuda()
    
    ckpt = torch.load(model_path)
    handoccnet_model.load_state_dict(ckpt['network'], strict=True)
    handoccnet_model.eval()

    # load handnerf
    model_path = 'handnerf_demo_model.pth.tar' 
    assert osp.exists(model_path), 'Cannot find model at ' + model_path
    print('Load checkpoint from {}'.format(model_path))
    
    handnerf_model = get_handnerf_model(mode)
    handnerf_model = DataParallel(handnerf_model).cuda()
    ckpt = torch.load(model_path)
    handnerf_model.load_state_dict(ckpt['network'], strict=True)
    handnerf_model.eval()

    # load cam
    camera = make_cam(cam_data)
    camera.cuda()

    # load input image
    original_img = load_img(rgb_path)
    
    start_time = time.time() 

    original_img_height, original_img_width = original_img.shape[:2]
    bbox = process_bbox(bbox, original_img_width, original_img_height)
    img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, input_img_shape) 
    
    # process for handoccnet
    transform = transforms.ToTensor()
    input_img = transform(img.astype(np.float32))/255
    input_img = input_img.cuda()[None,:,:,:]
    # Run HandOccNet
    verts_cam, joints_cam, org_res_joint_img = run_handoccnet(handoccnet_model, input_img, camera)

    handoccnet_time = time.time()
    save_obj(verts_cam, f=mano_layer[mano_hand_side].th_faces.numpy(), file_name=osp.join(save_dir, 'handoccnet_input_hand.obj'))


    """ visualize """
    np_joint_img = org_res_joint_img
    np_joint_img = np.concatenate([np_joint_img, np.ones_like(np_joint_img[:, :1])], axis=1)
    vis_img = original_img.astype(np.uint8)[:, :, ::-1]
    pred_joint_img_overlay = vis_keypoints_with_skeleton(vis_img, np_joint_img.T, mano.skeleton)
   
    pred_joint_img_overlay = cv2.rectangle(pred_joint_img_overlay.copy(), (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (255, 0, 0), 2)

    # cv2.imshow('2d prediction', pred_joint_img_overlay)
    save_path = osp.join(save_dir, f'handoccnet_2d_prediction.png')
    cv2.imwrite(save_path, pred_joint_img_overlay)
    
    projected_joints = camera(torch.from_numpy(joints_cam[None]).cuda())
    np_joint_img = projected_joints[0].cpu().numpy()
    np_joint_img = np.concatenate([np_joint_img, np.ones_like(np_joint_img[:, :1])], axis=1)

    vis_img = original_img.astype(np.uint8)[:, :, ::-1]
    pred_joint_img_overlay = vis_keypoints_with_skeleton(vis_img, np_joint_img.T, mano.skeleton)
    # cv2.imshow('projection', pred_joint_img_overlay)
    # cv2.waitKey(0)
    save_path = osp.join(save_dir, f'handoccnet_3d_projection.png')
    cv2.imwrite(save_path, pred_joint_img_overlay)
    
    # process for handnerf
    input_img = torch.from_numpy(img.astype(np.float32))/255
    input_img = input_img.cuda()[None, None, :, :, :]

    hand_info = (verts_cam, joints_cam, mano_hand_side)
    print("It takes some time to do segmentation...")
    run_handnerf(mode, save_dir, handnerf_model, input_img, cam_data, input_img_shape, hand_info)

    end_time = time.time()

    print(f"Process time: {end_time - start_time:.4f}, HandOccNet: {handoccnet_time - start_time:.4f}, HandNeRF: {end_time - handoccnet_time:.4f}")