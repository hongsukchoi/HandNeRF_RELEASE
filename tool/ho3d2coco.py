import os
import os.path as osp
import argparse
import sys
import glob
import pickle
import json
import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation as R
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, '../common')
from utils.vis import vis_2d_keypoints, vis_mesh

# From HO-3D v3 repository: https://github.com/shreyashampali/ho3d/blob/master/vis_pcl_all_cameras.py
img_height, img_width = 480, 640
depth_threshold = 800

# multiCamSeqs = [
#     'ABF1', # bleach
#     # 'BB1', # banana
#     'GPMF1', # spam can
#     # 'GSF1',  # scissors
#     # 'MDF1', # drill
#     'SB1', # bleach
#     'ShSu1',  # sugar box
#     'SiBF1',  # banana
# Eval
#     'SMu4',  # red mug
#     'MPM1',  # spam
#     'AP1'  # blue bottle?
# ]

multiCamSeqs = {
    'train':
    [
        'ABF1',  # bleach; object id: 12
        'GPMF1',  # spam can; object id: 9
        'GSF1',  # scissors; object id: 10
    ],
    # 'evaluation': no segmentation available for rendering evaluation and training 
    # [
    #     'SB1',  # bleach; different subject
    #     'MPM1',  # spam; different subject
    #     'GSF1',  # scissors
    # ]
}

split_frames = {
    'ABF1': {'train': [range(200, 800), range(1000, 1250)], 'test': [range(800, 1000), range(1250, 2000)]},
    'GPMF1': {'train': [range(0, 500), range(700, 900)], 'test': [range(500, 700), range(900, 2000)]},
    'GSF1': {'train': [range(0, 500), range(700, 1300)], 'test': [range(500, 700), range(1300, 2000)]},
}

sys.path.insert(0, './manopth')
from manopth.manolayer import ManoLayer

# mano_layer = ManoLayer(ncomps=45, center_idx=0, flat_hand_mean=True,  side="right", mano_root='manopth/mano/models', use_pca=False)
mano_layer = ManoLayer(ncomps=45, center_idx=None, flat_hand_mean=True,  side="right", mano_root='manopth/mano/models', use_pca=False)


jointsMapManoToSimple = [0,
                         13, 14, 15, 16,
                         1, 2, 3, 17,
                         4, 5, 6, 18,
                         10, 11, 12, 19,
                         7, 8, 9, 20]
skeleton = ((0, 1), (0, 5), (0, 9), (0, 13), (0, 17), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (
            7, 8), (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20))
        
# OpenGL to OpenCV (convention)
coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)


def get_bbox(joint_img, joint_valid):

    x_img, y_img = joint_img[:, 0], joint_img[:, 1]
    x_img = x_img[joint_valid == 1]
    y_img = y_img[joint_valid == 1]
    xmin = min(x_img)
    ymin = min(y_img)
    xmax = max(x_img)
    ymax = max(y_img)

    x_center = (xmin+xmax)/2.
    width = xmax-xmin
    xmin = x_center - 0.5*width*1.2
    xmax = x_center + 0.5*width*1.2

    y_center = (ymin+ymax)/2.
    height = ymax-ymin
    ymin = y_center - 0.5*height*1.2
    ymax = y_center + 0.5*height*1.2

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox

def read_xyz(file_name='points.xyz'):
    vertices = []
    with open(file_name, 'r') as f:
        for line in f:
            xyz_parts = line.split(' ')
            xyz = [float(xyz_parts[0]), float(xyz_parts[1]), float(xyz_parts[2][:-1])]
            vertices.append(xyz)
    return vertices

def load_pickle_data(f_name):
    """ Loads the pickle data """
    if not os.path.exists(f_name):
        raise Exception('Unable to find annotations picle file at %s. Aborting.'%(f_name))
    with open(f_name, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)

    return pickle_data


def read_annotation(base_dir, seq_name, file_id, split):
    meta_filename = os.path.join(
        base_dir, split, seq_name, 'meta', file_id + '.pkl')

    pkl_data = load_pickle_data(meta_filename)

    return pkl_data

def get_intrinsics(filename):
    with open(filename, 'r') as f:
        line = f.readline()
    line = line.strip()
    items = line.split(',')
    for item in items:
        if 'fx' in item:
            fx = float(item.split(':')[1].strip())
        elif 'fy' in item:
            fy = float(item.split(':')[1].strip())
        elif 'ppx' in item:
            ppx = float(item.split(':')[1].strip())
        elif 'ppy' in item:
            ppy = float(item.split(':')[1].strip())

    camMat = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
    return camMat

def rotation_angle(angle, rot_mat):
    per_rdg, _ = cv2.Rodrigues(angle)
    resrot, _ = cv2.Rodrigues(np.dot(rot_mat, per_rdg))
    return resrot[:, 0].astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dexycb_dir', type=str, default='')
    parser.add_argument('--ho3d_dir', type=str, default='')

    args = parser.parse_args()

    return args
    
if __name__ == '__main__':
    args = parse_args()
    dexycb_dir = args.dexycb_dir #  #'/data/hongsuk/DexYCB'
    baseDir = args.ho3d_dir # '/data/hongsuk/HO3D_v3/HO3D_v3' 

    # Load object meshes
    object_mesh_xyz_path_list = glob.glob(
        osp.join(dexycb_dir, 'models') + '/*/points.xyz')
    object_mesh_dict = {}  # '002_master_chef_can': [[]]
    for path in object_mesh_xyz_path_list:
        mesh_name = path.split('/')[-2]
        mesh = read_xyz(path)
        object_mesh_dict[mesh_name] = np.array(mesh, dtype=np.float32)

    split = 'train'
    my_split = 'test'
    seqName_list = sorted(os.listdir(f'{baseDir}/{split}/'))

    # world 3d anntotation data;
    world3d_data = {}  # db_idx: val

    images = []
    annotations = []
    img_id = 0
    annot_id = 0

    check_obj_id = set()
    for seqName_path in seqName_list:  
        seqName = osp.basename(seqName_path) # convert from absolute path
        # has multi-view calibration?
        baseCamSeq = seqName[:-1] # doesn't exist actually
        cam_name: int = int(seqName[-1]) # 0,1,2,3,4
        if not (baseCamSeq in multiCamSeqs[split]):  # ABF10 -> ABF1
            continue

    
        calibDir = osp.join(baseDir, 'calibration', baseCamSeq, 'calibration')
        cams_order = np.loadtxt(osp.join(calibDir, 'cam_orders.txt')).astype('uint8').tolist()  # ex. [1,3,2,4,0]
        cam_idx = cams_order.index(cam_name)
        # print(cam_name)
        _org_cam2world = np.loadtxt(osp.join(calibDir, 'trans_{}.txt'.format(cam_idx)))
        _org_K = get_intrinsics(osp.join(calibDir, 'cam_{}_intrinsics.txt'.format(cam_idx))) #.tolist()


        img_path_list = sorted(glob.glob(osp.join(baseDir, split,  seqName, 'rgb', '*.jpg')))
        seg_path_list = sorted(glob.glob(osp.join(baseDir, split,  seqName, 'seg', '*.png')))
        assert len(img_path_list) == len(seg_path_list), "Make sure img and seg are paired"
        for img_path, seg_path in  tqdm(zip(img_path_list, seg_path_list)):
            # make sure not to overwrite
            cam2world = _org_cam2world.copy()
            _K = _org_K.copy()

            img_dict = {}
            img_dict['id'] = img_id
            img_dict['file_name'] = osp.basename(img_path)
            img_dict['seg_file_name'] = osp.basename(seg_path)  # blud: hand, green: object
            
            assert img_dict['file_name'].split('.')[0] == img_dict['seg_file_name'].split('.')[0] , "Make sure img and seg are paired"
            scene_identifier = f"{baseCamSeq}_{img_dict['file_name'].split('.')[0]}"

            img_dict['sequence_name'] = seqName  # 'ABF10''
            frame_idx = int(osp.basename(img_path)[:-4])  # 000001.jpg -> 1
            img_dict['frame_idx'] = frame_idx
            img_dict['width'] = img_width
            img_dict['height'] = img_height

            # split video by frame
            if frame_idx not in split_frames[baseCamSeq][my_split][0] and frame_idx not in split_frames[baseCamSeq][my_split][1]:
                continue
            
            # # TEMP
            # if frame_idx % 5 != 0:
            #     continue

            images.append(img_dict)

            annot_dict = {}
            annot_dict['id'] = annot_id
            annot_dict['image_id'] = img_id
            dataset_annot = read_annotation(baseDir, seqName, f'{frame_idx:04d}',  split)

            annot_dict['object_name'] = dataset_annot['objName']
            annot_dict['object_id'] = dataset_annot['objLabel']
            check_obj_id.add((seqName_path, dataset_annot['objLabel']))

            # object 2 camera
            objRot = dataset_annot['objRot']
            if objRot is None:
                continue
            objRot = objRot.astype(np.float32)  # [:,0]  # (3,) or (3,1)
            if len(objRot.shape) > 1:
                objRot = objRot[:,0]
            obj2cam_R = R.from_rotvec(objRot).as_matrix()  # (3,3)
            obj2cam_t = dataset_annot['objTrans'].astype(np.float32)  # (3,)
            obj2cam = np.eye(4, dtype=np.float32)
            # change coordinate
            obj2cam[:3, :3], obj2cam[:3, 3] = coord_change_mat @ obj2cam_R, coord_change_mat @  obj2cam_t
            annot_dict['obj2cam'] = obj2cam.tolist()
            
            # camera parameters
            K = dataset_annot['camMat']
            assert (K == _K).all(), "Are you using the correct camera intrinsics?"
            annot_dict['K'] = K.tolist()
            world2cam = np.linalg.inv(cam2world)
            annot_dict['world2cam'] = world2cam.tolist()
            
            """ --------- Get MANO mesh --------- """
            # ------------------------------------
            mano_pose = dataset_annot['handPose']  # (48)
            mano_trans = dataset_annot['handTrans'] # (3)  # in mm scale
            mano_shape = dataset_annot['handBeta'] # (10)
            
            # OpenGL to OpenCV coordinate; https://redstarhong.tistory.com/269
            mano_pose[:3] = rotation_angle(mano_pose[:3], coord_change_mat)
            
            # get mesh vertices and joints
            mano_pose, mano_shape = torch.from_numpy(mano_pose)[None, :], torch.from_numpy(mano_shape)[None, :]  # to torch Tensor
            verts, joints, manojoints2cam = mano_layer(th_pose_coeffs=mano_pose, th_betas=mano_shape)
            # verts, joints are in the camera coordinate, mm scale, and joints's are like: index 0, index 1, index 2, index 3, index 4 (finger tip)
            verts, joints, manojoints2cam = verts[0].numpy() / 1000, joints[0].numpy() / 1000, manojoints2cam[0].numpy()
            # verts: (778,3) joints: (21,3) manojoints2cam: (16,4,4)

            # OpenGL to OpenCV coordinate; https://redstarhong.tistory.com/269
            mano_trans = - joints[0] + coord_change_mat @ joints[0] + coord_change_mat @ mano_trans
            mano_trans = mano_trans.reshape(1, 3)
            mano_verts, mano_joints = verts + mano_trans, joints + mano_trans

            annot_dict['mano_pose_cam'] = mano_pose[0].numpy().tolist()
            annot_dict['mano_trans_cam'] = mano_trans.tolist()
            annot_dict['mano_shape'] = mano_shape[0].numpy().tolist()
            # ------------------------------------

            """ Transform verts, joints, mano2cam, obj meshes to the world coordinate"""
            Rot, t = cam2world[:3, :3], cam2world[:3, 3:]
            # ------------------------------------
            world_mano_verts = (Rot @ mano_verts.T).T + t.reshape(-1,3)
            world_mano_joints = (Rot @ mano_joints.T).T + t.reshape(-1,3)
            
            manojoints2cam[:, :3, 3] = manojoints2cam[:, :3, 3]  + mano_trans  # add the translation 
            cam2manojoints = np.linalg.inv(manojoints2cam)
            world2manojoints = cam2manojoints @ world2cam[None, :, :]
            
            object_mesh = object_mesh_dict[dataset_annot['objName']]
            cam_object_mesh = (obj2cam[:3, :3] @ object_mesh.T).T + obj2cam[:3, 3:].reshape(-1,3)
            world_object_mesh = (Rot @ cam_object_mesh.T).T + t.reshape(-1, 3)

            # assign world data for once. do not repeat for every camera
            if scene_identifier not in world3d_data.keys():  
                world3d_data[scene_identifier] = defaultdict(list)

                world3d_data[scene_identifier]['vert'] = world_mano_verts.tolist()
                world3d_data[scene_identifier]['joint'] = world_mano_joints.tolist()
                world3d_data[scene_identifier]['world2manojoints'] = world2manojoints.tolist()
                world3d_data[scene_identifier]['object_mesh'] = world_object_mesh.tolist()
            # ------------------------------------

            # finger tip locations of mano joints and joints_cam are different; seems like the authors added fingers tips after the mesh trasnformation to the camera coordinate
            # joints_cam = dataset_annot['handJoints3D'][jointsMapManoToSimple] @ coord_change_mat.T 
        
            # visualization
            joints_world = world_mano_joints# np.concatenate([world_mano_verts, world_object_mesh])
            joints_cam = (world2cam[:3, :3] @ joints_world.T).T + (world2cam[:3, 3:]).T
            joints_img = joints_cam @ K.T
            joints_img = joints_img / joints_img[:, 2:]
            # tmpimg = cv2.imread(img_path)
            # newimg = vis_2d_keypoints(tmpimg, joints_img)
            # newimg = vis_mesh(tmpimg, joints_img)
            # cv2.imshow("skeleton", newimg)
            # cv2.waitKey(0)
            # import pdb; pdb.set_trace()
            bbox = get_bbox(joints_img, np.ones_like(joints_img[:, 0]))
            annot_dict['bbox'] = bbox.tolist()

            annotations.append(annot_dict)
            annot_id += 1
            img_id += 1
        
    print(check_obj_id)
    print(img_id, annot_id)
    output = {'images': images, 'annotations': annotations}
    Path(osp.join(baseDir, 'annotations')).mkdir(exist_ok=True, parents=True)

    output_path = osp.join(baseDir, 'annotations', f'HO3Dv3_partial_{my_split}_multiseq_coco.json')
    with open(output_path, 'w') as f:
        json.dump(output, f)
    print('Saved at ' + output_path)

    dump_path = osp.join(baseDir, 'annotations', f'HO3Dv3_partial_{my_split}_multiseq_world_data.json')
    with open(dump_path, 'w') as f:
        json.dump(world3d_data, f)
    print('Saved at ' + dump_path)
