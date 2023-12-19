import os.path as osp
import glob
import json
import pickle
import copy
import numpy as np
import torch
import cv2
import random
import smplx

from collections import defaultdict

from config import cfg, ManoLayer, manopth_dir

from utils.preprocessing import load_img, process_bbox, get_affine_trans_mat, get_intersection, rigid_align
from utils.geometry3d_utils import calc_3d_iou, get_grid_points_from_bounds, prepare_inside_world_pts, prepare_inside_world_object_pts, project, Rx, Ry, Rz
from utils.ray_sampling import sample_ray, sample_object_ray, get_bound_2d_mask
from utils.vis import vis_mesh, generate_rendering_path_from_multiviews
from utils.joint_mapper import JointMapper, smpl_to_openpose
from constant import OBJ_SEG_IDX, HAND_SEG_IDX, ROOT_JOINT_IDX

class Custom(torch.utils.data.Dataset):
    def __init__(self, mode, target_objects=[], target_sequence_frame=''):
        
        self.mode = mode
        self.split = 'train' if mode == 'train' else 'test'

        self.data_dir = osp.join(cfg.data_dir, 'Custom', 'data')
        self.cam_data_path = osp.join(self.data_dir, 'cam_params_final.json')
        # self.smplifyx_results_dir_list = [f'/home/hongsuk.c/Projects/smplify-x/output/{x}/results' for x in subset_list]
        # /frame/result.pkl
        self.data_dir_list = sorted(glob.glob(osp.join(self.data_dir, self.split, 'handnerf_*'))) 
        
        self.target_objects = target_objects  
        self.target_subjects: list = [] 
        self.target_sequence_frame: str = target_sequence_frame  

        # hardcoding # 

        self.num_cams = 7
        self.wrist_cam_idx = 0

        # redefine the mano model following the smplify-x; use_pca=False, flat_hand_mean=False
        self.mano_model = ManoLayer(
                             flat_hand_mean=False,
                             ncomps=45,
                             side='right',
                             mano_root=osp.join(manopth_dir, 'mano/models'),
                             use_pca=False)
        
        # self.mano_model_path = '/labdata/hongsuk/human_model_files'
        # self.mano_to_openpose = smpl_to_openpose('mano')
        # self.use_pca = False
        # self.flat_hand_mean=False
        # self.is_rhand = True
        # self.gender = 'male'
        #smplx.create(model_type='mano', model_path=self.mano_model_path, joint_mapper=JointMapper(self.mano_to_openpose),
        # gender=self.gender, use_pca=self.use_pca, flat_hand_mean=self.flat_hand_mean)

        self.render_whole = False
        self.sample_rays_from_world_bounds = True

        self.bound_padding = 0.25
        # hardcoding #

        self.cam_info = self._load_cam()
        self.datalist = self._load_data()

    def _load_cam(self):
        # load intrinsics and extrinsics
        with open(self.cam_data_path, 'r') as f:
            cam_data = json.load(f)

        camera_dict = {}
        for cam_idx in sorted(cam_data.keys(), key=lambda x: int(x)):
            
            intrinsic = np.array([[cam_data[cam_idx]['fx'], 0., cam_data[cam_idx]['cx']], [
                                 0., cam_data[cam_idx]['fy'], cam_data[cam_idx]['cy']], [0., 0., 1.]], dtype=np.float32)


            rotation, _ = cv2.Rodrigues(np.array(cam_data[cam_idx]['rvec'], dtype=np.float32))
            translation = np.array(cam_data[cam_idx]['tvec'], dtype=np.float32).reshape(3, 1) / 1000.  # mm -> m
            # world to camera
            extrinsic = np.concatenate([rotation, translation], axis=1)  # (3, 4)

            camera_dict[int(cam_idx)] = {
                'intrinsic': intrinsic,
                'extrinsic': extrinsic
            }

        return camera_dict

    def _load_helper(self, datalist, data, do_world2inputcam=True):
        # parser function that changes each idx data's coordinate frame
        data['inv_aug_R'] = np.eye(3, dtype=np.float32)
        data['inv_aug_t'] = np.zeros((3, 1), dtype=np.float32)
        if do_world2inputcam:

            ref_cam_idx = self.wrist_cam_idx
            ref_cam_R = copy.deepcopy(data['R_list'][ref_cam_idx])  # (3,3)
            ref_cam_t = copy.deepcopy(data['t_list'][ref_cam_idx])  # (3,1)
            """ aug to prevent 3D conv overfitting """
            rotate_degrees = [np.pi/10, np.pi/10, np.pi/10]
            # augmenting rotation matrix that rotates coordinate axes of the input camera
            aug_R = Rz(rotate_degrees[2]) @ Ry(rotate_degrees[1]) @ Rx(rotate_degrees[0])

            # get camera orientation and location
            # TEMP
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

            """ map world coordinate to inpt cam coordinate """
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

        point_keys = ['hand_mesh', 'hand_joints']
        if cfg.use_pred_hand:
            point_keys.extend(['pred_hand_mesh', 'pred_hand_joints'])
        for key in point_keys:
            original = data[key]  # (N, 3)
            new = (R @ original.T + t).T       
            data[key] = new

        # for i, world2manojoint in enumerate(data['world2manojoints']):
        #     R_, t_ = world2manojoint[:3, :3], world2manojoint[:3, 3:]
        #     new_R_ = R_ @ R.T
        #     new_t_ = t_ - new_R_ @ t
        #     data['world2manojoints'][i][:3, :3] = new_R_
        #     data['world2manojoints'][i][:3, 3:] = new_t_

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

    
    def _get_smplifyx(self, path):
        # load pickle file
        with open(path, 'rb') as f:
            smplifyx_data = pickle.load(f)
        # 'betas', 'global_orient', 'hand_pose', 'global_hand_translation', 'hand_scale'

        mano_pose_global = torch.tensor(smplifyx_data['global_orient'])  # (1,3)
        mano_pose_local = torch.tensor(smplifyx_data['hand_pose'])  # (1,45)
        mano_shape = torch.tensor(smplifyx_data['betas'])  # (1,10)
        hand_scale = smplifyx_data['hand_scale'][0]
        hand_translation = torch.tensor(smplifyx_data['global_hand_translation']) # (3)

        with torch.no_grad():
            mano_pose = torch.cat([mano_pose_global, mano_pose_local], dim=1)
            vertices, joints, _ = self.mano_model(th_pose_coeffs=mano_pose, th_betas=mano_shape)
            vertices, joints = vertices[0]  / 1000., joints[0] / 1000.

            hand_mesh = hand_scale * vertices + hand_translation
            hand_joints = hand_scale * joints + hand_translation
        return hand_mesh.detach().numpy(), hand_joints.detach().numpy()
    
    def _get_handoccnet(self, path):
        # load json file
        with open(path, 'r') as f:
            handoccnet_data = json.load(f)
        
        # 'hand_scale', 'hand_translation', 'mano_pose', 'mano_shape'
     
        mano_pose = torch.tensor(handoccnet_data['mano_pose']).unsqueeze(0)  # (1,48)
        mano_shape = torch.tensor(handoccnet_data['mano_shape']).unsqueeze(0)  # (1,10)
        hand_scale = handoccnet_data['hand_scale'][0]
        hand_translation = torch.tensor(handoccnet_data['hand_translation']) # (3)
        
        with torch.no_grad():
            mano_output = self.mano_model(betas=mano_shape, global_orient=mano_pose[:, :3],
                                          hand_pose=mano_pose[:, 3:], return_verts=True, return_full_pose=False)
            vertices, joints = mano_output.vertices[0], mano_output.joints[0]
            hand_mesh = hand_scale * vertices + hand_translation
            hand_joints = hand_scale * joints + hand_translation
        return hand_mesh.detach().numpy(), hand_joints.detach().numpy()

    # load datalist
    def _load_data(self):
        
        """ Organize the data by scene (one scene has 7 camera images) """
        data_dict = defaultdict(lambda: defaultdict(list))
        for data_dir_idx, data_dir in enumerate(self.data_dir_list):
            sequence_name = osp.basename(data_dir) # handnerf_training_1

            smplifyx_result_dir = osp.join(osp.dirname(data_dir), sequence_name + '_handmesh', 
                                           'results') 
            # '/home/hongsuk.c/Projects/smplify-x/output/handnerf_training_1/results'
            # assert sequence_name == smplifyx_result_dir.split('/')[-2]

            frame_dir_list =  sorted(glob.glob(osp.join(smplifyx_result_dir, '*')))
            frame_name_list = [osp.basename(x) for x in frame_dir_list]  # [0000, 0001, ...]

            for frame_name in frame_name_list:
                sequence_frame_name = '_'.join([sequence_name, frame_name])

                smplifyx_path = osp.join(smplifyx_result_dir, frame_name, 'result.pkl')
                handoccnet_path = osp.join(
                    data_dir, f'cam_{self.wrist_cam_idx}_handoccnet', f'{self.wrist_cam_idx}_{frame_name}_3dmesh.json')

                if self.target_sequence_frame != '' and sequence_frame_name != self.target_sequence_frame:
                    continue
                # if self.mode != 'train' and sequence_frame_name not in self.test_sequence_frames:
                #     continue
                # elif self.mode == 'train' and sequence_frame_name in self.test_sequence_frames:
                #     continue

                # meta data
                # for file_name to save
                data_dict[sequence_frame_name]['scene_identifier'] = sequence_frame_name
                # for segmentation label
                data_dict[sequence_frame_name]['grabbing_object_id'] = 1
                data_dict[sequence_frame_name]['mano_side'] = 'right'

                # In world coordinate
                hand_mesh, hand_joints = self._get_smplifyx(smplifyx_path)  
                data_dict[sequence_frame_name]['hand_mesh'] = hand_mesh # (778,3)
                data_dict[sequence_frame_name]['hand_joints'] = hand_joints  # (21,3)
                # data_dict[sequence_frame_name]['world2manojoints'] = world2manojoints # (16,4,4)
                
                # Already in input camera coordiante. convert it to the world for code consistency
                if cfg.use_pred_hand:
                    pred_hand_mesh, pred_hand_joints = self._get_handoccnet(handoccnet_path)
                    Rt = self.cam_info[self.wrist_cam_idx]['extrinsic']
                    R, t = Rt[:3, :3], Rt[:3, 3:]
                    R = R.T
                    t = - R @ t
                    pred_hand_mesh = (R @ pred_hand_mesh.T).T + t.T
                    pred_hand_joints = (R @ pred_hand_joints.T).T + t.T
                    # # TEMP; root adjustment
                    # pred_hand_mesh = pred_hand_mesh - pred_hand_joints[0:1] + hand_joints[0:1]
                    # pred_hand_joints = pred_hand_joints - pred_hand_joints[0:1] + hand_joints[0:1] 

                    data_dict[sequence_frame_name]['pred_hand_mesh'] = pred_hand_mesh
                    data_dict[sequence_frame_name]['pred_hand_joints'] = pred_hand_joints
                # data_dict[sequence_frame_name]['pred_world2manojoints'] =  # [(16,4,4), ...]         

                # 0~6
                for cam_idx in sorted(self.cam_info.keys()):
                    
                    rgb_path = osp.join(data_dir, f'cam_{cam_idx}', f'{cam_idx}_{frame_name}.jpg')
                    if cam_idx == self.wrist_cam_idx:
                        depth_path = osp.join(data_dir, f'cam_{cam_idx}_depth', f'{cam_idx}_{frame_name}.png')
                    else:
                        depth_path = None
                    obj_seg_path =  osp.join(data_dir, f'cam_{cam_idx}_segmentation', f'{cam_idx}_{frame_name}_obj_seg.npy')
                    hand_seg_path =  osp.join(data_dir, f'cam_{cam_idx}_segmentation', f'{cam_idx}_{frame_name}_hand_seg.npy')
                    bbox_path = osp.join(data_dir, f'cam_{cam_idx}', f'{cam_idx}_{frame_name}.json')

                    K = self.cam_info[cam_idx]['intrinsic']
                    Rt = self.cam_info[cam_idx]['extrinsic']
                    R, t = Rt[:3, :3], Rt[:3, 3:] 
                    cam_center = - R.T @ t  # (3,1)

                    """ Save data to dictionary """
                    # the list length is the number of cameras
                    data_dict[sequence_frame_name]['rgb_path_list'].append(rgb_path)
                    data_dict[sequence_frame_name]['depth_path_list'].append(depth_path)
                    data_dict[sequence_frame_name]['seg_path_list'].append((hand_seg_path, obj_seg_path))
                    data_dict[sequence_frame_name]['K_list'].append(K)  # [(3,3), ...]
                    data_dict[sequence_frame_name]['R_list'].append(R)  # [(3,3), ...]
                    data_dict[sequence_frame_name]['t_list'].append(t)  # [(3,1), ...]
                    data_dict[sequence_frame_name]['camera_center_list'].append(cam_center)  # [(3,1), ...]

        """ Parse the dictionary into list """
        # cam dependent keys to sort again
        # cam_dependent_keys = ['rgb_path_list', 'depth_path_list', 'seg_path_list',
        #                      'K_list', 'R_list', 't_list', 'camera_center_list']
        data_list = []
        for sequence_frame_name, sequence_frame_annot_dict in data_dict.items():  # key: sequence name, value: dict
            input_view_indices = [self.wrist_cam_idx] 
            sequence_frame_annot_dict['input_view_indices'] = input_view_indices

            if self.mode == 'train':
                for j in range(self.num_cams):
                    copied_data = copy.deepcopy(sequence_frame_annot_dict)

                    if cfg.num_render_views == 1:
                        render_view_indices = [j]
                    elif cfg.num_render_views > 1:  # uniform sample rendering views
                        render_view_indices = random.sample(
                            list(range(self.num_cams)), cfg.num_render_views)
                    else:
                        raise ValueError(
                            "[Dataloader] Invalid number of rendering views!")

                    # later if you want to save memory, sample other values (ex. K_list) with render_view_indices
                    copied_data['render_view_indices'] = render_view_indices

                    # use all views for input during training? (FYI, still the input to the model is a single image)
                    if cfg.use_all_input_views and cfg.num_input_views == 1:
                        for i in range(self.num_cams):
                            copied_copied_data = copy.deepcopy(copied_data)
                            copied_copied_data['input_view_indices'] = [i]
                            self._load_helper(data_list, copied_copied_data)
                    else:
                        self._load_helper(data_list, copied_data)

            else:  # self.mode == 'test'
                if cfg.test_mode == 'render_rotate':

                    """ Get rotating camera views """
                    num_render_views = cfg.num_rotating_views

                    camera_center_list = np.array(
                        sequence_frame_annot_dict['camera_center_list'], dtype=np.float32).reshape(-1, 3)
                    world_object_center = sequence_frame_annot_dict['object_mesh'].mean(
                        axis=0)
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
                        copied_data['rotating_camera_center'] = - \
                            R.T @ t  # (3,1)
                        copied_data['rotating_render_idx'] = idx

                        self._load_helper(data_list, copied_data)

                elif cfg.test_mode == 'render_dataset':
                    for idx in range(self.num_cams):
                        copied_data = copy.deepcopy(sequence_frame_annot_dict)

                        copied_data['render_view_indices'] = [idx]

                        # use all views for input during tesing? (FYI, still the input to the model is a single image)
                        if cfg.use_all_input_views and cfg.num_input_views == 1:
                            for i in range(self.num_cams):
                                copied_copied_data = copy.deepcopy(copied_data)
                                copied_copied_data['input_view_indices'] = [i]
                                self._load_helper(
                                    data_list, copied_copied_data)
                        else:
                            self._load_helper(data_list, copied_data)

                elif cfg.test_mode == 'recon3d':
                    self._load_helper(data_list, sequence_frame_annot_dict)

                else:
                    raise ValueError("[Dataloader] Unknown test mode!")

        return data_list

    # load hand mesh and get boundary
    def load_3d_data(self, hand_mesh):
        """
        hand_mesh: (778, 3) in world
        # pred_hand_mesh: (778, 3) in world, estimated by off-the-shelf. 

        // Returns //
        hand_center: (3,1) in world
        world_bounds: (2,3) in world
        world_hand_mesh_voxel_coord: (778, 3), coordinates of world mesh vertices in the descritized volume
        world_mesh_voxel_out_sh: (3,) np.int32, shape of the world mesh volume
        """

        # (778, 3)
        mesh = np.concatenate([hand_mesh])

        # get center
        hand_center = mesh.mean(axis=0)[:, None]  # (3,1)

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

        return hand_center, world_bounds, world_hand_mesh_voxel_coord, world_mesh_voxel_out_sh


    # unproject object pixels to the input camera's 3D space
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
        input_voxel_size = cfg.input_mesh_voxel_size  # voxel resolution for grid point sampling
        world_grid_points = get_grid_points_from_bounds(world_bounds, input_voxel_size, self.mode)  # (x_size, y_size, z_size, 3)
        input_mask_copy = input_mask.copy()
        world_object_inside = prepare_inside_world_object_pts(world_grid_points, input_view_R, input_view_t, input_view_K, input_mask_copy)  # np.uint8, [0,1], (x_size, y_size, z_size)
        # handle the exception when there is no inside object points
        if world_object_inside.sum() == 0:
            return np.empty(0), np.empty(0)
        
        world_inside_object_points = world_grid_points[world_object_inside.astype(bool)] # (num_inside_object_points, 3), xyz

        # down sample
        world_inside_object_points = world_inside_object_points[::downsample_ratio]

        # construct the voxel coordinate in the world coordinate
        dhw = world_inside_object_points[:, [2, 1, 0]]
        min_dhw = world_bounds[0, [2, 1, 0]]
        
        world_object_points_voxel_coord = np.round((dhw - min_dhw) / input_voxel_size).astype(np.int32)

        assert world_object_points_voxel_coord[:, 0].max() < world_mesh_voxel_out_sh[0] and world_object_points_voxel_coord[:, 1].max(
        ) < world_mesh_voxel_out_sh[1] and world_object_points_voxel_coord[:, 2].max() < world_mesh_voxel_out_sh[2], f"World mesh voxel out sh: {world_mesh_voxel_out_sh}, object voxel coord: {world_object_points_voxel_coord.max(dim=1)}"

        return world_inside_object_points, world_object_points_voxel_coord

    # combine hand/obj seg and label them
    def parse_mask(self, hand_seg, obj_seg, only_object=False):
        # no overlap
        hand_seg[obj_seg] = False
        # re-organize the mask element values
        obj_seg = obj_seg.astype(np.uint8) * OBJ_SEG_IDX
        hand_seg = hand_seg.astype(np.uint8) * HAND_SEG_IDX
        if only_object:
            mask = obj_seg
        else:
            mask = (obj_seg + hand_seg)

        return mask
    
    # preprocess the input image and superivsion images
    def affine_transform_and_masking(self, img, depth, mask, out_shape, expand_ratio, masking=True):
        bbox = cv2.boundingRect(mask.astype(np.uint8))  # x, y, w, h
        bbox = process_bbox(
            bbox, img.shape[1], img.shape[0], out_shape, expand_ratio)

        trans = get_affine_trans_mat(bbox, out_shape)

        img = cv2.warpAffine(img, trans, (int(out_shape[1]), int(
            out_shape[0])), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, trans, (int(out_shape[1]), int(
            out_shape[0])), flags=cv2.INTER_NEAREST)
        if masking:
            img[mask == 0] = 0

        if depth is not None:
            depth = cv2.warpAffine(depth, trans, (int(out_shape[1]), int(
                out_shape[0])), flags=cv2.INTER_NEAREST)
            depth[mask == 0] = 0

        return img, depth, mask, trans, bbox

    # load input view's image
    def load_2d_input_view(self, rgb_path, seg_path, K, R, t, world_bounds):
        rgb = load_img(rgb_path) / 255.
        hand_seg = np.load(seg_path[0], allow_pickle=True)[()]['seg']   # bool, 
        obj_seg = np.load(seg_path[1], allow_pickle=True)[()]['seg']
        mask = self.parse_mask(hand_seg, obj_seg)

        assert rgb.shape[:2] == mask.shape[:2], print(
            'rgb.shape & mask.shape ', rgb.shape, mask.shape)

        # bbox augmentation # 
        if self.mode == 'train':
            expand_ratio = np.random.uniform(cfg.input_bbox_expand_ratio-0.2, cfg.input_bbox_expand_ratio+0.2)
        else:
            expand_ratio = cfg.input_bbox_expand_ratio
        # affine transform for feature extraction
        img, depth, mask, trans, bbox = self.affine_transform_and_masking(
            rgb, None, mask, cfg.input_img_shape, expand_ratio=expand_ratio, masking=True) 

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

    # get rays for rendering
    def load_2d_render_view(self, rgb_path, depth_path, seg_path, world_bounds, R, t, K, camera_center, object_center, render_whole):
        """
        camera_center: (3,1)
        object_center: (3,1)
        """
        img = load_img(rgb_path) / 255.  # 0~1.
        # depth = load_img(depth_path)[0] / 255.  # 0~1.  # CAUTION; Yet I don't know how it is scaled
        depth = None

        hand_seg = np.load(seg_path[0], allow_pickle=True)[()]['seg']   # bool, 
        obj_seg = np.load(seg_path[1], allow_pickle=True)[()]['seg']
        mask = self.parse_mask(hand_seg, obj_seg)

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
            scale_factor = 2.5
            K[:2, :2] = scale_factor * K[:2, :2].copy()

            img = cv2.resize(
                img, (render_img_shape[1], render_img_shape[0]), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(
                mask, (render_img_shape[1], render_img_shape[0]), interpolation=cv2.INTER_NEAREST)
            # depth = cv2.resize(
            #     depth, (render_img_shape[1], render_img_shape[0]), interpolation=cv2.INTER_NEAREST)

        else:
            img, depth, mask, trans, bbox = self.affine_transform_and_masking(
                img, depth, mask, render_img_shape, expand_ratio=cfg.render_bbox_expand_ratio)
            if mask.sum() == 0:
                mask[:] == 1
                # raise ValueError("Invalid mask for rendering images!")

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
            bound_mask = get_bound_2d_mask(
                world_bounds, new_K, pose, img.shape[0], img.shape[1])
            cv2.imshow("[render] bound_mask ", bound_mask * 255.)
            cv2.waitKey(0)

        return rgb, depth, fg_mask, org_mask, ray_o, ray_d, near, far, coord_, mask_at_box


    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        # length: 1
        # for segmentation label
        grabbing_object_id = data['grabbing_object_id']  # int
        world_hand_mesh = data['hand_mesh']  # (778,3)
        world_hand_joints = data['hand_joints']  # (21,3)
        # (15,4,4), exclude wrist joint
        # world2manojoints = data['world2manojoints'][1:]
        # pred_world2manojoints = data['pred_world2manojoints']
        if cfg.use_pred_hand:
            pred_world_hand_mesh = data['pred_hand_mesh']
            pred_world_hand_joints = data['pred_hand_joints']
            world_hand_mesh = pred_world_hand_mesh
            world_hand_joints = pred_world_hand_joints

        mano_side = data['mano_side']
        # length: number of cameras
        rgb_path_list = data['rgb_path_list']
        depth_path_list = data['depth_path_list']
        seg_path_list = data['seg_path_list'] # list of hand/object tuples
        R_list = data['R_list']
        t_list = data['t_list']
        K_list = data['K_list']
        world_camera_center_list = data['camera_center_list']
        # selecting indices
        input_view_indices = data['input_view_indices']
        render_view_indices = data['render_view_indices']

        """ Load world 3d data """
        world_center, world_bounds, world_hand_mesh_voxel_coord, world_mesh_voxel_out_sh = self.load_3d_data(world_hand_mesh)
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
                input_img, input_mask, input_new_K = self.load_2d_input_view(rgb_path_list[input_idx], seg_path_list[input_idx],  K_list[input_idx],
                                                                             R_list[input_idx], t_list[input_idx], world_bounds)

                if cfg.debug:
                    print("input: ", rgb_path_list[input_idx])
                    # Visualize projected points
                    Rt = np.concatenate(
                        [R_list[input_idx], t_list[input_idx]], axis=1)
                    img_points = project(
                        world_hand_mesh.copy(), input_new_K, Rt)
                    new_img = (input_img * 255).astype(np.uint8) 
                    new_img = vis_mesh(new_img , img_points)
                    # cv2.imwrite("rgb_projected.jpg", new_img[:, :, ::-1])
                    # white_img = np.ones_like(input_img) * 255
                    # new_img2 = vis_mesh(white_img, img_points)
                    # cv2.imwrite("white_projected.jpg", new_img2[:, :, ::-1])

                    cv2.imshow("[input] mesh projected", new_img[:, :, ::-1])
                    cv2.waitKey(0)


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
                    world_bounds, world_mesh_voxel_out_sh, input_view_R, input_view_t, input_view_K, input_mask)\

                if cfg.debug:        
                    # TEMP
                    img_points = project(world_inside_object_points.copy(), input_new_K, Rt)
                    new_img = (input_img[0] * 255).astype(np.uint8) 
                    new_img = vis_mesh(new_img, img_points)
                    cv2.imshow("[input] object points projected", new_img[:, :, ::-1])
                    cv2.waitKey(0)

        else:
            # dummy
            input_view_R = np.ones(1)
            input_view_t = np.ones(1)
            input_view_K = np.ones(1)
            input_img = np.ones(1)

        """ Prepare reconstruction OR render view data """
        if self.mode != 'train' and cfg.test_mode == 'recon3d':
            voxel_size = cfg.mc_voxel_size  # voxel resolution for marching cube
            world_grid_points = get_grid_points_from_bounds(
                world_bounds, voxel_size, self.mode)
            if cfg.use_pred_hand:
                inp = self.wrist_cam_idx
                world_inside = prepare_inside_world_pts(world_grid_points, np.stack([R_list[inp]]), np.stack(
                    [t_list[inp]]), np.stack([K_list[inp]]), [seg_path_list[inp]], grabbing_object_id, 'Custom')
            else:
                world_inside = prepare_inside_world_pts(world_grid_points, np.stack(R_list), np.stack(
                    t_list), np.stack(K_list), seg_path_list, grabbing_object_id, 'Custom')

            print("Reconstructing... Input world points shape: ",
                  world_grid_points.shape)

            # dummy
            rgb, fg_mask, org_mask, ray_o, ray_d, near, far, mask_at_box = np.empty(0), np.empty(
                0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0)

        else:
            render_rgb_path_list = [x for i, x in enumerate(
                rgb_path_list) if i in render_view_indices]
            # right now, the depth is just used for evaluation
            render_depth_path_list = [x for i, x in enumerate(
                depth_path_list) if i in render_view_indices]
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
                render_depth_path_list = depth_path_list[0:1]
                render_seg_path_list = seg_path_list[0:1]
                # actual data for rendering rotating views
                render_R_list = [data['rotating_R']]
                render_t_list = [data['rotating_t']]
                render_K_list = [data['rotating_K']]
                render_world_camera_center_list = [
                    data['rotating_camera_center']]

            # follow the NueralBody code
            rgb_list, depth_list, fg_mask_list, org_mask_list, ray_o_list, ray_d_list, near_list, far_list, mask_at_box_list = [
            ], [], [], [], [], [], [], [], []

            for rgb_path, depth_path, seg_path, R, t, K, world_camera_center in zip(render_rgb_path_list, render_depth_path_list, render_seg_path_list, render_R_list, render_t_list, render_K_list, render_world_camera_center_list):
                render_whole = self.render_whole
                rgb, depth, fg_mask, org_mask, ray_o, ray_d, near, far, _, mask_at_box = \
                    self.load_2d_render_view(rgb_path, depth_path, seg_path,
                                             world_bounds, R, t, K, world_camera_center, world_center, render_whole=render_whole)

                # same resolution with th einput 

                rgb_list.append(rgb)
                depth_list.append(depth)
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
        input_view_parameters = {
            'input_view_R': input_view_R, 'input_view_t': input_view_t, 'input_view_K': input_view_K,
            'input_view_mask': input_mask
        }
        world_3d_data = {
            'world_bounds': world_bounds,'world_hand_mesh': world_hand_mesh, 'world_hand_mesh_voxel_coord': world_hand_mesh_voxel_coord, 'world_mesh_voxel_out_sh': world_mesh_voxel_out_sh, 'mano_side': mano_side,
            'world_object_points': world_inside_object_points, 'world_object_points_voxel_coord': world_object_points_voxel_coord,
            'world_hand_joints': world_hand_joints, #'world2manojoints': world2manojoints,
            # 'world_object_mesh': world_object_mesh  # for evaluation
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
                # int
                meta_info['rotating_render_idx'] = data['rotating_render_idx']

            if cfg.test_mode == 'recon3d':
                world_3d_data['inside'] = world_inside
                world_3d_data['pts'] = world_grid_points.astype(np.float32)
                # transform the output 3d to the exact input camera coordinate system
                meta_info['inv_aug_R'] = data['inv_aug_R']
                meta_info['inv_aug_t'] = data['inv_aug_t']

        # convert back to float32
        for dictdata in [input_view_parameters, world_3d_data, rendering_rays, meta_info]:
            for k, v in dictdata.items():
                if type(v) == np.ndarray and v.dtype == np.float32:
                    dictdata[k] = v.astype(np.float32)

        return input_img, input_view_parameters, world_3d_data, rendering_rays, meta_info
