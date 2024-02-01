import os.path as osp
import glob
import json
import yaml
import copy
import random
import numpy as np
import torch
import cv2

from typing import List
from collections import defaultdict
from tqdm import tqdm
from pycocotools.coco import COCO

from config import cfg
from utils.preprocessing import load_img, process_bbox, get_affine_trans_mat, get_intersection, rigid_align
from utils.geometry3d_utils import calc_3d_iou, get_grid_points_from_bounds, prepare_inside_world_pts, prepare_inside_world_object_pts, project, Rx, Ry, Rz
from utils.ray_sampling import sample_ray, sample_object_ray, get_bound_2d_mask
from utils.vis import vis_mesh, generate_rendering_path_from_multiviews, save_obj
from constant import OBJ_SEG_IDX, HAND_SEG_IDX, ROOT_JOINT_IDX


class HO3D(torch.utils.data.Dataset):
    def __init__(self, mode, target_objects, target_sequence_frame=''):
        super(HO3D, self).__init__()

        self.mode = mode  # experiment mode: 'train', 'test'
        self.subset = mode  #'train', 'test'
        self.data_dir = osp.join(cfg.data_dir, 'HO3D', 'data')
        self.image_height, self.image_width = 480, 640
        self.target_objects: list = target_objects  # [-1]  # 12: '021_bleach_cleanser',9: '010_potted_meat_can', 10: '037_scissors',
        self.target_subjects: list = [] # HO3D does not provide subject info
        self.target_sequence_frame: str = target_sequence_frame

        self.include_no_intersection_data = False
        self.render_whole = False
        self.sample_rays_from_world_bounds = True
        if self.mode == 'test'  and cfg.test_mode == 'render_rotate' and self.target_sequence_frame == '':
            raise ValueError("[Dataloader] We recommend you to select one sequence_frame when testing 'render_rotate', for accurate depth denormalization and better visualization.")
            # raise ValueError("[Dataloader] When testing 'render_rotate', you need to select one sequence_frame!")

        self.pred_hand_meshes_path = osp.join(self.data_dir, 'annotations', f'HO3Dv3_HandNeRF_{cfg.test_config}_testset_HandOccNet_pred.npy')
        if cfg.use_pred_hand:
            self.pred_hand_meshes = np.load(self.pred_hand_meshes_path, allow_pickle=True)
            # {
            # ''train/ABF10/rgb/0828.jpg': [mesh_coord_cam, joints_coord_cam, mano_joints2cam, mano_pose_aa],
            # ...
            # }
            self.do_rigid_align = False
        
        self.subsample_ratio = 2 # 10 if self.mode != 'train' else 3 
        self.iou_3d_thr = 0.03 # 3%  # contact threshold using 3d iou of hand and object meshes
        self.bound_padding = 0.03  # 0.001 == 1mm. should be enough to include object around the hand
        self.input_camera_postfix = '10'

        self.datalist = self._load_data()
    
    def _load_helper(self, datalist, data, do_world2inputcam=True):
        # parser function that changes each idx data's coordinate frame
        data['inv_aug_R'] = np.eye(3, dtype=np.float32)
        data['inv_aug_t'] = np.zeros((3, 1), dtype=np.float32)
        if do_world2inputcam:

            ref_cam_idx = data['input_view_indices'][0]
            ref_cam_R = copy.deepcopy(data['R_list'][ref_cam_idx])
            ref_cam_t = copy.deepcopy(data['t_list'][ref_cam_idx])

            # aug to prevent 3D conv overfitting
            rotate_degrees = [np.pi/10, np.pi/10, np.pi/10]
            # augmenting rotation matrix that rotates coordinate axes of the input camera
            aug_R = Rz(rotate_degrees[2]) @ Ry(rotate_degrees[1]) @ Rx(rotate_degrees[0])
            
            # get camera orientation and location
            # ref_cam_ori = aug_R @ ref_cam_R.T
            ref_cam_ori = ref_cam_R.T @ aug_R # multiply from right! to make the ref_cam eventually be aug_R
            ref_cam_center = - ref_cam_R.T @ ref_cam_t

            # new transformation matrices
            ref_cam_R_org = ref_cam_R.copy()
            ref_cam_t_org = ref_cam_t.copy()
            ref_cam_R = ref_cam_ori.T
            ref_cam_t = - ref_cam_ori.T @ ref_cam_center

            # save inverse matrices to transform 3d points back to the exact input camera coordinate system
            data['inv_aug_R'] = ref_cam_R_org @ ref_cam_R.T
            data['inv_aug_t'] = ref_cam_t_org - data['inv_aug_R'] @ ref_cam_t

            # map world coordinate to inpt cam coordinate
            self._transform_dictdata_world2inputcam(data, ref_cam_R, ref_cam_t)

        datalist.append(data)

    def _transform_dictdata_world2inputcam(self, data, R, t):
        # parse every data in the world coordinate to the (first) input camera coordinate
        # do the inplace replacement
        """
        data: 
            camera_center_list: [(3,1), ...]
            object_mesh: (num_points, 3)
            hand_mesh: (778,3)
            hand_joints: (21,3)

            R_list: [(3,3), ...]
            t_list: [(3,1), ...]
            world2manojoints: (16,4,4)

        R: (3,3)
        t: (3,1)
        """
        for i, cam_center in enumerate(data['camera_center_list']):
            new_cam_center = R @ cam_center + t
            data['camera_center_list'][i] = new_cam_center

        if cfg.use_pred_hand:
            for i, pred_mesh in enumerate(data['pred_world_hand_mesh_list']):
                if pred_mesh is not None:
                    new_pred_mesh = (R @ pred_mesh.T + t).T
                    data['pred_world_hand_mesh_list'][i] = new_pred_mesh

            for i, pred_joint in enumerate(data['pred_world_hand_joints_list']):
                if pred_joint is not None:
                    new_pred_joint = (R @ pred_joint.T + t).T
                    data['pred_world_hand_joints_list'][i] = new_pred_joint

            # (num_cams, 16, 4, 4)
            for i, pred_world2manojoint in enumerate(data['pred_world2manojoints']):
                if pred_world2manojoint is not None:
                    R_, t_ = pred_world2manojoint[:, :3,
                                                  :3], pred_world2manojoint[:, :3, 3:]
                    new_R_ = R_ @ R.T[None, :, :]
                    new_t_ = t_ - new_R_ @ t[None, :, :]
                    data['pred_world2manojoints'][i][:, :3, :3] = new_R_
                    data['pred_world2manojoints'][i][:, :3, 3:] = new_t_


        point_keys = ['object_mesh', 'hand_mesh', 'hand_joints']
        for key in point_keys:
            original = data[key]  # (N, 3)
            new = (R @ original.T + t).T
            data[key] = new

        for i, world2manojoint in enumerate(data['world2manojoints']):
            R_, t_ = world2manojoint[:3, :3], world2manojoint[:3, 3:]
            new_R_ = R_ @ R.T
            new_t_ = t_ - new_R_ @ t
            data['world2manojoints'][i][:3, :3] = new_R_
            data['world2manojoints'][i][:3, 3:] = new_t_

        if self.mode == 'test' and cfg.test_mode == 'render_rotate':
            new_R_ = data['rotating_R'] @ R.T  # (3,3)
            new_t_ = data['rotating_t'] - new_R_ @ t   # (3,1)
            data['rotating_R'] = new_R_
            data['rotating_t'] = new_t_

            new_cam_center = R @ data['rotating_camera_center'] + t
            data['rotating_camera_center'] = new_cam_center

        for i, (R_, t_) in enumerate(zip(data['R_list'], data['t_list'])):
            new_R_ = R_ @ R.T
            new_t_ = t_ - new_R_ @ t
            data['R_list'][i] = new_R_
            data['t_list'][i] = new_t_
    
    def _load_data(self):
        db = COCO(osp.join(self.data_dir, 'annotations', f'HO3Dv3_partial_{self.subset}_multiseq_coco.json'))
        with open(osp.join(self.data_dir, 'annotations', f'HO3Dv3_partial_{self.subset}_multiseq_world_data.json')) as f:
            world_data_dict = json.load(f)

        object_set = set()
        data_dict = defaultdict(lambda: defaultdict(list))
        aid_list = db.anns.keys()
        for aid in tqdm(aid_list):
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_shape = (img['height'], img['width'])

            # path, directories
            sequence_name = img['sequence_name'] # 'ABF10'
            base_sequence_name = sequence_name[:-1]  # 'ABF1'
            frame_name = img['file_name']  # 0000.jpg
            sequence_frame_name = f'{base_sequence_name}_{frame_name.split(".")[0]}'  # ABF1_0000

            # object 
            object_name = ann['object_name']
            object_id = ann['object_id']
            # img, seg path // I use only train set of original HO3D
            img_path = osp.join(self.data_dir, 'train', sequence_name, 'rgb', frame_name)
            seg_path = osp.join(self.data_dir, 'train', sequence_name, 'seg', img['seg_file_name'])

            
            """ Check skip """
            if self.target_sequence_frame != '' and sequence_frame_name != self.target_sequence_frame:
                continue
            if self.target_objects[0] != -1 and object_id not in self.target_objects:
                continue
            if img['frame_idx'] % self.subsample_ratio != 0:
                continue


            """ Load camera parameters """
            K = np.array(ann['K'], np.float32)  # (3,3)
            # load world2cam
            world2cam = np.array(ann['world2cam'], np.float32) # (4,4)
            R, t = world2cam[:3, :3], world2cam[:3, 3:]  # (3,3), (3,1)
            cam_center = - R.T @ t  # (3,1)

 
            """ Load world object mesh and hand mesh and joints """
            # get world data (mesh and joints3d)
            world_data = world_data_dict[sequence_frame_name]

            if 'object_mesh' in data_dict[sequence_frame_name].keys():
                world_object_mesh = data_dict[sequence_frame_name]['object_mesh']
                world_hand_mesh = data_dict[sequence_frame_name]['hand_mesh']
                world_hand_joints = data_dict[sequence_frame_name]['hand_joints']
                world2manojoints = data_dict[sequence_frame_name]['world2manojoints']

            else:
                # world_object_mesh: (N, 3), world_hand_mesh: (778,3), world_hand_joints: (21,3), world2manojoints: (16,4,4)
                world_object_mesh = np.array(world_data['object_mesh'], dtype=np.float32)
                world_hand_mesh = np.array(world_data['vert'], dtype=np.float32)
                # joint order: https://github.com/NVlabs/dex-ycb-toolkit/blob/master/dex_ycb_toolkit/dex_ycb.py#L59-81
                world_hand_joints = np.array(world_data['joint'], dtype=np.float32)
                if np.isnan(world_hand_mesh.sum()) or np.isnan(world_hand_joints.sum()):
                    continue
                world2manojoints = np.array(world_data['world2manojoints'], dtype=np.float32)

                # check contact
                if not self.include_no_intersection_data:
                    hand_object_iou = calc_3d_iou(world_hand_mesh, world_object_mesh)
                    if  hand_object_iou < self.iou_3d_thr:
                        continue
            
            """ Load predicted cam mesh and convert to world coordinate """
            # pred mesh is predicted from each camera... so multiple for one scene
            base_rgb_path = '/'.join(img_path.split('/')[-4:])  # 'train/ABF10/rgb/0828.jpg'

            if cfg.use_pred_hand and base_rgb_path in self.pred_hand_meshes[()].keys():
                # 'cam_mesh (778,3), cam_joints (21,3), manjoints2cam transformation (16,4,4)
                pred_data = self.pred_hand_meshes[()][base_rgb_path]
                pred_cam_hand_mesh, pred_cam_hand_joints, pred_manojoints2cam = pred_data[0], pred_data[1], pred_data[2]

                # get cam gt mesh & align pred with gt
                cam_hand_mesh = (R @ world_hand_mesh.T + t).T
                cam_hand_joints = (R @ world_hand_joints.T + t).T

                """ For IHOI-NeRF """
                # get root-relative translation
                pred_manojoints2cam[:, :3, 3:] -= pred_manojoints2cam[ROOT_JOINT_IDX:ROOT_JOINT_IDX+1, :3, 3:]
                # HO3D hands are all 'right'
                # if mano_side == 'left':

                # root align
                pred_manojoints2cam[:, :3, 3:] += cam_hand_joints[ROOT_JOINT_IDX:ROOT_JOINT_IDX+1, :].T[None, :, :]
                
                # to world
                pred_cam2manojoints = np.linalg.inv(pred_manojoints2cam)
                world2cam = np.eye(4, dtype=np.float32)
                world2cam[:3, :3], world2cam[:3, 3:] = R, t
                pred_world2manojoints = pred_cam2manojoints @ world2cam[None, :, :]

                """ For MonoNHR, HandNeRF / and getting boundary """
                # get root-relative pred mesh 
                pred_cam_hand_mesh -= pred_cam_hand_joints[ROOT_JOINT_IDX:ROOT_JOINT_IDX+1]
                pred_cam_hand_joints -= pred_cam_hand_joints[ROOT_JOINT_IDX:ROOT_JOINT_IDX+1]

                # root align
                pred_cam_hand_mesh += cam_hand_joints[ROOT_JOINT_IDX:ROOT_JOINT_IDX+1]
                pred_cam_hand_joints += cam_hand_joints[ROOT_JOINT_IDX:ROOT_JOINT_IDX+1]
                
                # rigid align
                if self.do_rigid_align:
                    pred_cam_hand_mesh = rigid_align(pred_cam_hand_mesh, cam_hand_mesh)
                    pred_cam_hand_joints = rigid_align(pred_cam_hand_joints, cam_hand_joints)

                pred_world_hand_mesh = (pred_cam_hand_mesh - t.T) @ R
                pred_world_hand_joints = (pred_cam_hand_joints - t.T) @ R
            else:
                pred_world_hand_mesh = None
                pred_world_hand_joints = None
                pred_world2manojoints = None


            """ Save data to dictionary """
            # meta data
            data_dict[sequence_frame_name]['scene_identifier'] = sequence_frame_name  # for file_name to save
            data_dict[sequence_frame_name]['grabbing_object_id'] = object_id
            data_dict[sequence_frame_name]['hand_object_iou'] = hand_object_iou   # 0~1

            # the list length is the number of cameras
            data_dict[sequence_frame_name]['rgb_path_list'].append(img_path)
            data_dict[sequence_frame_name]['seg_path_list'].append(seg_path)
            data_dict[sequence_frame_name]['camera_name_list'].append(sequence_name) # ABF10
            data_dict[sequence_frame_name]['K_list'].append(K)  # [(3,3), ...]
            data_dict[sequence_frame_name]['R_list'].append(R)  # [(3,3), ...]
            data_dict[sequence_frame_name]['t_list'].append(t)  # [(3,1), ...]
            # TEMP
            # cam_center[0:2] = -cam_center[0:2] for visualization
            data_dict[sequence_frame_name]['camera_center_list'].append(cam_center)  # [(3,1), ...]
            

            # world data
            data_dict[sequence_frame_name]['pred_world_hand_mesh_list'].append(pred_world_hand_mesh)  # [(778,3), ...]
            data_dict[sequence_frame_name]['pred_world_hand_joints_list'].append(pred_world_hand_joints)  # [(778,3), ...]
            data_dict[sequence_frame_name]['pred_world2manojoints'].append(pred_world2manojoints)  # [(16,4,4), ...]
        
            data_dict[sequence_frame_name]['object_mesh'] = world_object_mesh # (num_points, 3)
            data_dict[sequence_frame_name]['hand_mesh'] = world_hand_mesh # (778,3)
            data_dict[sequence_frame_name]['hand_joints'] = world_hand_joints # (21,3)
            data_dict[sequence_frame_name]['world2manojoints'] = world2manojoints # (16,4,4)
            object_set.add(object_id)



        print("[Dataloader] Actual Number of objects: ", len(object_set))
        print("[Dataloader] Actual Objects: ", object_set)

        """ Parse the dictionary into list """
        # cam dependent keys to sort again
        cam_dependent_keys = ['rgb_path_list', 'seg_path_list', 'camera_name_list', 'K_list', 'R_list', 't_list', 'camera_center_list']
        data_list = []
        input_rgb_path_dict = {}

        for sequence_frame_name, sequence_frame_annot_dict in data_dict.items():  # key: sequence name, value: dict
            # number of views in the sequence
            num_views = len(sequence_frame_annot_dict['rgb_path_list'])
            # skip if num_views are smaller than num_input_views
            if num_views < cfg.num_input_views:
                continue
            
            camera_name_postfix_list = [x[-2:] for x in sequence_frame_annot_dict['camera_name_list']]
            sequence_frame_annot_dict['camera_name_list'] = camera_name_postfix_list 

            # skip if the selected input view is not contained
            if self.input_camera_postfix != '' and self.input_camera_postfix not in sequence_frame_annot_dict['camera_name_list']:
                continue

            # sort the camera dependent values
            sorted_indices = [i[0] for i in sorted(enumerate(sequence_frame_annot_dict['camera_name_list']), key=lambda x:x[1])]
            for cam_dep_key in cam_dependent_keys:
                sequence_frame_annot_dict[cam_dep_key] = [sequence_frame_annot_dict[cam_dep_key][i] for i in sorted_indices]

            # fixed input views, get uniformaly sampled input views
            if cfg.num_input_views == 1 and self.input_camera_postfix != '':
                input_view_index = sequence_frame_annot_dict['camera_name_list'].index(self.input_camera_postfix)
                input_view_indices = [input_view_index]     

                rgb_path = sequence_frame_annot_dict['rgb_path_list'][input_view_index]
                input_rgb_path_dict[sequence_frame_name] = '/'.join(rgb_path.split('/')[-4:])           
            else: #  not implemented
                raise NotImplementedError("Currently only supports a single input view")
                pace = num_views // cfg.num_input_views
                input_view_indices = list(range(num_views))[::pace][:cfg.num_input_views]
            sequence_frame_annot_dict['input_view_indices'] = input_view_indices 

            if self.mode == 'train':
                for j in range(num_views):
                    copied_data = copy.deepcopy(sequence_frame_annot_dict)

                    if cfg.num_render_views == 1:
                        render_view_indices = [j]
                    elif cfg.num_render_views > 1:  # uniform sample rendering views
                        render_view_indices = random.sample(list(range(num_views)), cfg.num_render_views)
                    else:
                        raise ValueError("[Dataloader] Invalid number of rendering views!")

                    # later if you want to save memory, sample other values (ex. K_list) with render_view_indices
                    copied_data['render_view_indices'] = render_view_indices

                    # use all views for input during training? (FYI, still the input to the model is a single image)
                    if cfg.use_all_input_views and cfg.num_input_views == 1:
                        for i in range(num_views):
                            copied_copied_data = copy.deepcopy(copied_data)
                            copied_copied_data['input_view_indices'] = [i]
                            self._load_helper(data_list, copied_copied_data)
                    else:
                        self._load_helper(data_list, copied_data)

            else:  # self.mode == 'test'
                if cfg.test_mode == 'render_rotate':

                    """ Get rotating camera views """
                    num_render_views = cfg.num_rotating_views

                    camera_center_list = np.array(sequence_frame_annot_dict['camera_center_list'], dtype=np.float32).reshape(-1, 3)
                    world_object_center = sequence_frame_annot_dict['object_mesh'].mean(axis=0)
                    # camera_center_list: list of camera xyz locations in the world coordinate; (N,3)
                    # world_object_center: object location in the world coordinate; (3,)

                    Rt_all, traj = generate_rendering_path_from_multiviews(
                        camera_center_list, world_object_center, num_render_views=num_render_views)
                    # Rt_all: (num_render_views, 4, 4)
                    # traj: (num_render_views, 3), new camera locations

                    # use fixed intrinsics
                    base_cam_idx = 0
                    K = sequence_frame_annot_dict['K_list'][base_cam_idx]

                    """ Debug camera locations """
                    # # world axes
                    # import open3d as o3d
                    # origin = sequence_frame_annot_dict['object_mesh'].mean(axis=0).tolist()
                    # FOR = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=origin)
                    # # original training camera locations
                    # pcd = o3d.geometry.PointCloud()
                    # pcd.points = o3d.utility.Vector3dVector(camera_center_list)
                    # # new camera locations of rotating views
                    # pcd2 = o3d.geometry.PointCloud()
                    # pcd2.points = o3d.utility.Vector3dVector(traj)
                    # colors = np.zeros_like(traj)
                    # colors[:, 0] = 1.
                    # pcd2.colors = o3d.utility.Vector3dVector(colors)
                    # o3d.visualization.draw_geometries([pcd, pcd2, FOR])
                    # import pdb; pdb.set_trace()

                    # save data
                    for idx in range(num_render_views):
                        copied_data = copy.deepcopy(sequence_frame_annot_dict)

                        # actual data for rendering rotating views
                        copied_data['rotating_K'] = K  # (3,3)
                        R, t = Rt_all[idx][:3, :3], Rt_all[idx][:3, 3:]
                        copied_data['rotating_R'] = R  # (3,3)
                        copied_data['rotating_t'] = t  # (3,1)
                        copied_data['rotating_camera_center'] = - R.T @ t  # (3,1)
                        copied_data['rotating_render_idx'] = idx

                        self._load_helper(data_list, copied_data)

                elif cfg.test_mode == 'render_dataset':
                    for idx in range(num_views):
                        copied_data = copy.deepcopy(sequence_frame_annot_dict)

                        copied_data['render_view_indices'] = [idx]

                        # use all views for input during tesing? (FYI, still the input to the model is a single image)
                        if cfg.use_all_input_views and cfg.num_input_views == 1:
                            for i in range(num_views):
                                copied_copied_data = copy.deepcopy(copied_data)
                                copied_copied_data['input_view_indices'] = [i]
                                self._load_helper(data_list, copied_copied_data)
                        else:
                            self._load_helper(data_list, copied_data)

                elif cfg.test_mode == 'recon3d':
                    self._load_helper(data_list, sequence_frame_annot_dict)

                else:
                    raise ValueError("[Dataloader] Unknown test mode!")

        if self.mode == 'test':
            target_img_list_path = osp.abspath(osp.join(self.data_dir, 'annotations', f'{cfg.test_config}_test_list.json')) 
            with open(target_img_list_path, 'w') as f:
                json.dump(input_rgb_path_dict, f)

        return data_list

    def load_3d_data(self, object_mesh, hand_mesh):
        """
        object_mesh: (num_points,3) in world, use object mesh just to get the center
        hand_mesh: (778, 3) in world

        // Returns //
        hand_object_center: (3,1) in world
        world_bounds: (2,3) in world
        world_hand_mesh_voxel_coord: (778, 3), coordinates of world mesh vertices in the descritized volume
        world_mesh_voxel_out_sh: (3,) np.int32, shape of the world mesh volume
        """

        # combine two meshes
        # using the GT object mesh for evaluation of rendering. not really necessary in pratice.
        mesh = np.concatenate([object_mesh, hand_mesh]) # (num_object_points + 778, 3)

        # get center
        hand_object_center = mesh.mean(axis=0)[:, None]  # (3,1)

        # obtain the world bounds for point sampling
        min_xyz = np.min(mesh, axis=0)
        max_xyz = np.max(mesh, axis=0)
        min_xyz -= self.bound_padding
        max_xyz += self.bound_padding
        world_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # construct the voxel coordinate in the world coordinate
        dhw = hand_mesh[:, [2, 1, 0]]
        assert dhw.shape[0] == 778, "CAUTION! Don't use object mesh for dhw!"
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array(cfg.input_mesh_voxel_size)
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

        return hand_object_center, world_bounds, world_hand_mesh_voxel_coord, world_mesh_voxel_out_sh

    def load_3d_object_data(self, world_bounds, world_mesh_voxel_out_sh, input_view_R, input_view_t, input_view_K, input_mask, downsample_ratio=1):
        """
        world_bounds: (2,3) in world
        world_mesh_voxel_out_sh: (3,) np.int32, shape of the world mesh volume, dhw
        input_view_R: (num_input_views, 3, 3)
        input_view_t: (num_input_views, 3, 1)
        input_view_K: (num_input_views, 3, 3)
        input_mask: (num_input_views, cfg.input_img_shape[0], cfg.input_img_shape[1])

        // Returns //
        world_inside_object_points: (num_object_points, 3)
        world_object_points_voxel_coord: (num_object_points, 3)
        """
        # get partially true object points from input masks
        # voxel resolution for grid point sampling
        input_voxel_size = cfg.input_mesh_voxel_size
        world_grid_points = get_grid_points_from_bounds(
            world_bounds, input_voxel_size, self.mode)  # (x_size, y_size, z_size, 3)
        world_object_inside = prepare_inside_world_object_pts(
            world_grid_points, input_view_R, input_view_t, input_view_K, input_mask)  # np.uint8, [0,1], (x_size, y_size, z_size)
        # handle the exception when there is no inside object points
        if world_object_inside.sum() == 0:
            return np.empty(0), np.empty(0)
        
        world_inside_object_points = world_grid_points[world_object_inside.astype(
            bool)]  # (num_inside_object_points, 3), xyz

        # down sample
        world_inside_object_points = world_inside_object_points[::downsample_ratio]

        # construct the voxel coordinate in the world coordinate
        dhw = world_inside_object_points[:, [2, 1, 0]]
        min_dhw = world_bounds[0, [2, 1, 0]]

        world_object_points_voxel_coord = np.round(
            (dhw - min_dhw) / input_voxel_size).astype(np.int32)

        assert world_object_points_voxel_coord[:, 0].max() < world_mesh_voxel_out_sh[0] and world_object_points_voxel_coord[:, 1].max(
        ) < world_mesh_voxel_out_sh[1] and world_object_points_voxel_coord[:, 2].max() < world_mesh_voxel_out_sh[2]

        return world_inside_object_points, world_object_points_voxel_coord

    def load_2d_input_view(self, rgb_path, seg_path, K, R, t, world_bounds, camera_center, object_center):
        rgb = load_img(rgb_path) / 255.
        seg = self.load_seg(seg_path)  # np.uint8, 240 x 320, OBJSEMANTIC or HAND_SEMANTIC
        
        mask = self.parse_mask(seg)

        assert rgb.shape[:2] == mask.shape[:2], print('rgb.shape & mask.shape ', rgb.shape, mask.shape)

        # affine transform for feature extraction
        img, mask, trans, bbox = self.affine_transform_and_masking(
            rgb, mask, cfg.input_img_shape, expand_ratio=cfg.input_bbox_expand_ratio, masking=True)  
        new_K = K.copy()
        new_K[0, 0] = K[0, 0] * cfg.input_img_shape[1] / bbox[2]
        new_K[1, 1] = K[1, 1] * cfg.input_img_shape[0] / bbox[3]
        new_K[0, 2] = (K[0, 2] - bbox[0]) * cfg.input_img_shape[1] / bbox[2]
        new_K[1, 2] = (K[1, 2] - bbox[1]) * cfg.input_img_shape[0] / bbox[3]

        if cfg.debug:
            pose = np.concatenate([R, t], axis=1)
            bound_mask = get_bound_2d_mask(
                world_bounds, new_K, pose, img.shape[0], img.shape[1])
            cv2.imshow("[input] bound_mask ", bound_mask * 255.)
            cv2.waitKey(0)

        return img, mask, new_K

    def affine_transform_and_masking(self, img, mask, out_shape, expand_ratio, masking=True):
        bbox = cv2.boundingRect(mask.astype(np.uint8))  # x, y, w, h
        bbox = process_bbox(
            bbox, img.shape[1], img.shape[0], out_shape, expand_ratio)

        trans = get_affine_trans_mat(bbox, out_shape)

        img = cv2.warpAffine(img, trans, (int(out_shape[1]), int(
            out_shape[0])), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, trans, (int(out_shape[1]), int(
            out_shape[0])), flags=cv2.INTER_NEAREST)
        # masking in input can be False, not critical
        if masking:
            img[mask == 0] = 0

        return img, mask, trans, bbox

    def load_seg(self, seg_path):
        # get segmentation
        seg = cv2.imread(seg_path)  # 240 x 320 x 3
        # blue: hand, green: object
        hand_seg = seg[:, :, 0] == 255
        obj_seg = seg[:, :, 1] == 255

        hand_seg = hand_seg.astype(np.uint8) * HAND_SEG_IDX
        obj_seg = obj_seg.astype(np.uint8) * OBJ_SEG_IDX

        seg = hand_seg + obj_seg
        
        return seg
    
    def parse_mask(self, seg,  only_object=False):
        # resize to RGB size
        mask = cv2.resize(seg, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)

        if only_object:
            mask[mask == HAND_SEG_IDX] = 0
        
        return mask

    def load_2d_render_view(self, world_mesh, rgb_path, seg_path, world_bounds, R, t, K, camera_center, object_center, render_whole):
        """
        camera_center: (3,1)
        object_center: (3,1)
        """
        img = load_img(rgb_path) / 255.  # 0~1.
        seg = self.load_seg(seg_path)  # np.uint8, 240 x 320, OBJSEMANTIC or HAND_SEMANTIC
        mask = self.parse_mask(seg)

        assert img.shape[:2] == mask.shape[:2], print(
            'img.shape & mask.shape ', img.shape, mask.shape)

        # render the whole image?
        if self.mode == 'train' and render_whole:
            render_img_shape = cfg.clip_render_img_shape  # h, w
        else:
            render_img_shape = cfg.render_img_shape   # h, w

        # affine transform
        if self.mode == 'test' and cfg.test_mode == 'render_rotate':
            bbox = [0, 0, img.shape[1], img.shape[0]]
            K[0, 2] = img.shape[1] / 2.
            K[1, 2] = img.shape[0] / 2.
            # adjust scale
            scale_factor = 1.5
            K[:2, :2] = scale_factor * K[:2, :2].copy()

            img = cv2.resize(
                img, (render_img_shape[1], render_img_shape[0]), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(
                mask, (render_img_shape[1], render_img_shape[0]), interpolation=cv2.INTER_NEAREST)

        else:
            img, mask, trans, bbox = self.affine_transform_and_masking(
                img, mask, render_img_shape, expand_ratio=cfg.render_bbox_expand_ratio)
            if mask.sum() == 0:
                raise ValueError("Invalid mask for rendering images!")

        new_K = K.copy()
        new_K[0, 0] = K[0, 0] * render_img_shape[1] / bbox[2]
        new_K[1, 1] = K[1, 1] * render_img_shape[0] / bbox[3]
        new_K[0, 2] = (K[0, 2] - bbox[0]) * render_img_shape[1] / bbox[2]
        new_K[1, 2] = (K[1, 2] - bbox[1]) * render_img_shape[0] / bbox[3]

        rgb, fg_mask, ray_o, ray_d, near, far, coord_, mask_at_box = sample_ray(
            img, mask, new_K, R, t, world_bounds, cfg.N_rand, self.mode, render_whole, camera_center[:, 0], object_center[:, 0], from_world_bounds=self.sample_rays_from_world_bounds, img_path=rgb_path)

        # org_mask is for evaluating object and hand separately
        org_mask = fg_mask.copy()

        # Make sure fg_mask contains [0,1]
        fg_mask[fg_mask != 0] = 1.

        if cfg.debug:
            pose = np.concatenate([R, t], axis=1)
            # bound_mask = get_bound_2d_mask(
            #     world_bounds, new_K, pose, img.shape[0], img.shape[1])
            # cv2.imshow("[render] bound_mask ", bound_mask * 255.)
            # cv2.waitKey(0)
            img_points = project(world_mesh.copy(), new_K, pose)
            new_img = vis_mesh(img, img_points)
            cv2.imshow(f"[render]", new_img[:, :, ::-1])
            cv2.waitKey(0)
            print(f"[Dataloader] rendering {rgb_path}")

        return rgb, fg_mask, org_mask, ray_o, ray_d, near, far, coord_, mask_at_box

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        # length: 1
        # for segmentation label
        grabbing_object_id = data['grabbing_object_id']  # int
        world_object_mesh = data['object_mesh']  # (N,3)
        world_hand_mesh = data['hand_mesh']  # (778,3)
        world_hand_joints = data['hand_joints']  # (21,3)
        # (15,4,4), exclude wrist joint
        world2manojoints = data['world2manojoints'][1:]
        if cfg.use_pred_hand and self.mode == 'test':
            if data['pred_world_hand_mesh_list'][0] is None:
                print("No prediction for ", idx)
            else:
                # pick the first one. anyway rightnow I only have first one
                pred_world_hand_mesh_list = data['pred_world_hand_mesh_list']  # [(778,3), ..]
                world_hand_mesh = pred_world_hand_mesh_list[0]
                pred_world_hand_joints_list = data['pred_world_hand_joints_list']  # [(21,3), ..]
                world_hand_joints = pred_world_hand_joints_list[0]
                pred_world2manojoints = data['pred_world2manojoints']
                world2manojoints = pred_world2manojoints[0][1:] # (15,4,4), exclude wrist joint            
        
        mano_side = 'right' #data['mano_side']
        # length: number of cameras
        rgb_path_list = data['rgb_path_list']
        seg_path_list = data['seg_path_list']
        R_list = data['R_list']
        t_list = data['t_list']
        K_list = data['K_list']
        world_camera_center_list = data['camera_center_list']
        # selecting indices
        input_view_indices = data['input_view_indices']
        render_view_indices = data['render_view_indices']

        """ Load world 3d data """
        world_hand_object_center, world_bounds, world_hand_mesh_voxel_coord, world_mesh_voxel_out_sh = self.load_3d_data(
            world_object_mesh, world_hand_mesh)
        # world_mesh_voxel_out_sh: output shape that should include BOTH hand and object

        """ Load input view """
        if cfg.num_input_views > 0 and input_view_indices is not None:
            input_rgb_path_list = [x for i, x in enumerate(
                rgb_path_list) if i in input_view_indices]
            input_seg_path_list = [x for i, x in enumerate(
                seg_path_list) if i in input_view_indices]
            input_view_R_list = [x for i, x in enumerate(
                R_list) if i in input_view_indices]
            input_view_t_list = [x for i, x in enumerate(
                t_list) if i in input_view_indices]

            input_view_K_list = []
            input_img_list = []
            input_mask_list = []

            for input_idx in input_view_indices:
                input_img, input_mask, input_new_K = self.load_2d_input_view(rgb_path_list[input_idx], seg_path_list[input_idx], K_list[input_idx],
                                                                             R_list[input_idx], t_list[input_idx], world_bounds, world_camera_center_list[input_idx], world_hand_object_center)

                if cfg.debug:
                    print("input: ", rgb_path_list[input_idx])
                    # Visualize projected points
                    Rt = np.concatenate(
                        [R_list[input_idx], t_list[input_idx]], axis=1)
                    img_points = project(
                        world_hand_mesh.copy(), input_new_K, Rt)
                    new_img = vis_mesh(input_img, img_points)
                    # cv2.imwrite("test.jpg", new_img)
                    cv2.imshow("[input] mesh projected", new_img[:, :, ::-1])
                    cv2.waitKey(0)
                    import pdb
                    pdb.set_trace()

                input_view_K_list.append(input_new_K)
                input_img_list.append(input_img)
                input_mask_list.append(input_mask)

            # stack
            input_view_R = np.stack(input_view_R_list)
            input_view_t = np.stack(input_view_t_list)
            input_view_K = np.stack(input_view_K_list)
            input_img = np.stack(input_img_list)
            input_mask = np.stack(input_mask_list)

            if 'mononhr' in cfg.nerf_mode or 'noobjectpixelnerf' in cfg.nerf_mode:
                world_inside_object_points, world_object_points_voxel_coord = np.empty(
                    0), np.empty(0)
            else:
                # get partial 3D object points from 2D input view
                world_inside_object_points, world_object_points_voxel_coord = self.load_3d_object_data(
                    world_bounds, world_mesh_voxel_out_sh, input_view_R, input_view_t, input_view_K, input_mask)

        else:
            # dummy
            input_view_R = np.ones(1)
            input_view_t = np.ones(1)
            input_view_K = np.ones(1)
            input_img = np.ones(1)

        """ Prepare reconstruction OR render view data """
        if self.mode == 'test' and cfg.test_mode == 'recon3d':
            voxel_size = cfg.mc_voxel_size  # voxel resolution for marching cube
            world_grid_points = get_grid_points_from_bounds(
                world_bounds, voxel_size, self.mode)
            print("Reconstructing... Input world points shape: ", world_grid_points.shape)
            
            if not cfg.use_multiview_masks_for_recon:  
                inp = 0
                world_inside = prepare_inside_world_pts(world_grid_points, np.stack([R_list[inp]]), np.stack(
                    [t_list[inp]]), np.stack([K_list[inp]]), [seg_path_list[inp]], -1, 'HO3D')
            else:
                world_inside = prepare_inside_world_pts(world_grid_points, np.stack(R_list), np.stack(
                    t_list), np.stack(K_list), seg_path_list, -1, 'HO3D')

            print("Reconstructing... Input world points shape: ", world_grid_points.shape)

            # dummy
            rgb, fg_mask, org_mask, ray_o, ray_d, near, far, mask_at_box = np.empty(0), np.empty(
                0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0)

        else:
            render_rgb_path_list = [x for i, x in enumerate(
                rgb_path_list) if i in render_view_indices]
            # right now, the depth is just used for evaluation
          
            render_seg_path_list = [x for i, x in enumerate(
                seg_path_list) if i in render_view_indices]
            render_R_list = [x for i, x in enumerate(
                R_list) if i in render_view_indices]
            render_t_list = [x for i, x in enumerate(
                t_list) if i in render_view_indices]
            render_K_list = [x for i, x in enumerate(
                K_list) if i in render_view_indices]
            render_world_camera_center_list = [x for i, x in enumerate(
                world_camera_center_list) if i in render_view_indices]
            if self.mode == 'test' and cfg.test_mode == 'render_rotate':
                # dummy
                render_rgb_path_list = rgb_path_list[0:1]
                render_seg_path_list = seg_path_list[0:1]
                # actual data for rendering rotating views
                render_R_list = [data['rotating_R']]
                render_t_list = [data['rotating_t']]
                render_K_list = [data['rotating_K']]
                render_world_camera_center_list = [
                    data['rotating_camera_center']]

            # follow the NueralBody code
            rgb_list, fg_mask_list, org_mask_list, ray_o_list, ray_d_list, near_list, far_list, mask_at_box_list = [], [], [], [], [], [], [], []
            for rgb_path, seg_path, R, t, K, world_camera_center in zip(render_rgb_path_list, render_seg_path_list, render_R_list, render_t_list, render_K_list, render_world_camera_center_list):
                render_whole = self.render_whole 
                rgb, fg_mask, org_mask, ray_o, ray_d, near, far, _, mask_at_box = \
                    self.load_2d_render_view(world_object_mesh, rgb_path, seg_path, 
                                             world_bounds, R, t, K, world_camera_center, world_hand_object_center, render_whole=render_whole)

                rgb_list.append(rgb)
                fg_mask_list.append(fg_mask)
                org_mask_list.append(org_mask)
                ray_o_list.append(ray_o)
                ray_d_list.append(ray_d)
                near_list.append(near)
                far_list.append(far)
                mask_at_box_list.append(mask_at_box)

            rgb, fg_mask, org_mask, ray_o, ray_d, near, far, mask_at_box = np.concatenate(rgb_list), np.concatenate(fg_mask_list), np.concatenate(
                org_mask_list), np.concatenate(ray_o_list), np.concatenate(ray_d_list), np.concatenate(near_list), np.concatenate(far_list), np.concatenate(mask_at_box_list)


        """ Prepare data dictionaries """
        input_view_paramters = {
            'input_view_R': input_view_R, 'input_view_t': input_view_t, 'input_view_K': input_view_K,
        }
        world_3d_data = {
            'world_bounds': world_bounds, 'world_hand_joints': world_hand_joints, 'world2manojoints': world2manojoints, 'world_hand_mesh': world_hand_mesh, 'world_hand_mesh_voxel_coord': world_hand_mesh_voxel_coord, 'world_mesh_voxel_out_sh': world_mesh_voxel_out_sh, 'mano_side': mano_side,
            'world_object_points': world_inside_object_points, 'world_object_points_voxel_coord': world_object_points_voxel_coord,
            'world_object_mesh': world_object_mesh  # for evaluation
        }
        rendering_rays = {
            'rgb': rgb, 'fg_mask': fg_mask, 'org_mask': org_mask, 'ray_o': ray_o, 'ray_d': ray_d, 'near': near, 'far': far, 'mask_at_box': mask_at_box
        }
        meta_info = {
            'scene_identifier': data['scene_identifier'],
            'input_img_path_list': input_rgb_path_list if 'pixel' in cfg.nerf_mode or 'hand' in cfg.nerf_mode else '',
            'render_img_path_list': render_rgb_path_list if 'render' in cfg.test_mode else ''
        }

        if self.mode != 'train':
            if cfg.test_mode == 'render_rotate':
                
                meta_info['rotating_render_idx'] = data['rotating_render_idx'] # int

            if cfg.test_mode == 'recon3d':
                world_3d_data['inside'] = world_inside
                world_3d_data['pts'] = world_grid_points.astype(np.float32)
                # transform the output 3d to the exact input camera coordinate system
                meta_info['inv_aug_R'] = data['inv_aug_R']
                meta_info['inv_aug_t'] = data['inv_aug_t']

        return input_img, input_view_paramters, world_3d_data, rendering_rays, meta_info
