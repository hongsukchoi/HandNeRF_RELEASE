import os.path as osp
import glob
import json
import yaml
import copy
import random
import numpy as np
import torch
import cv2
import pycocotools.mask

from typing import List
from collections import defaultdict
from tqdm import tqdm

from config import cfg
from utils.preprocessing import load_img, process_bbox, get_affine_trans_mat, get_intersection, rigid_align
from utils.geometry3d_utils import calc_3d_iou, get_grid_points_from_bounds, prepare_inside_world_pts, prepare_inside_world_object_pts, prepare_inside_world_hand_pts, project, Rx, Ry, Rz, get_pointcloud_from_rgbd
from utils.ray_sampling import sample_ray, sample_object_ray, get_bound_2d_mask
from utils.vis import vis_mesh, generate_rendering_path_from_multiviews, save_obj
from constant import OBJ_SEG_IDX, HAND_SEG_IDX, ANNOT_HAND_SEG_IDX, ROOT_JOINT_IDX, DEXYCB_OBJECT_CLASSES


# DexYCB object id list -> https://github.com/NVlabs/dex-ycb-toolkit/blob/64551b001d360ad83bc383157a559ec248fb9100/dex_ycb_toolkit/dex_ycb.py#L36
class DexYCB(torch.utils.data.Dataset):
    def __init__(self, mode, target_objects=[], target_sequence_frame=''):
        super(DexYCB, self).__init__()
        
        self.mode = mode  # 'train', 'test'

        self.category = 's0'  # s0, s1, s2, s3 // unseen grasping sequences, unseen subjects, unseen views, unseen grasping objects respecitvely
        self.subset = 'train' if self.mode == 'train' else 'test' # 'train', 'val', 'test'
        self.target_sequence_frame = target_sequence_frame  # ex) '1_20200813_145612_000072'
        self.target_objects = target_objects  # [-1]  # ex) 1 master_chef can 10 banana 15 power drill 16 wood block 18 large marker
        if cfg.test_config == 'novel_object' and self.mode == 'train':
            self.target_objects =  [x for x in list(range(1,21+1)) if x not in self.target_objects]
        
        self.target_subjects = ['20200813-subject-02', '20200709-subject-01'] # limit the subjects for faster experiments
        self.include_no_intersection_data = False  
        self.render_whole = False
        self.sample_rays_from_world_bounds = True  
        # we use the object GT mesh's coordinates to get the world bound for the quantitative evaluation of the rendering and reconstruction, 
        # but this doesn't mean the method requires the object ground truth. The world bound can be approximated with the estimated hand mesh's coordinates
        # refer to Custom.py

        # directories and pathes
        self.data_dir = osp.join(cfg.data_dir, 'DexYCB', 'data')
        self.annot_path = osp.join(self.data_dir, 'annotation', f'{self.category}_{self.subset}.json')
        self.world_data_path = osp.join(self.data_dir, 'annotation', f'{self.category}_{self.subset}_world_data.json')
        self.calibration_dir = osp.join(self.data_dir, 'calibration')
        if self.mode == 'test'  and cfg.test_mode == 'render_rotate' and self.target_sequence_frame == '':
            raise ValueError("[Dataloader] We recommend you to select one sequence_frame when testing 'render_rotate', for accurate depth denormalization and better visualization.")
            # raise ValueError("[Dataloader] When testing 'render_rotate', you need to select one sequence_frame!")

        self.pred_hand_meshes_path = osp.join(self.data_dir, 'annotation', f'DexYCB_HandNeRF_{cfg.test_config}_testset_HandOccNet_pred.npy')
        if cfg.use_pred_hand:
            self.pred_hand_meshes = np.load(self.pred_hand_meshes_path, allow_pickle=True)
            # {
            # '20201022-subject-10/20201022_113909/836212060125/color_000060.jpg': [mesh_coord_cam, joints_coord_cam, mano_joints2cam, mano_pose_aa],
            # ...
            # }
            self.do_rigid_align = False


        self.subsample_ratio = 2#1 #3 #30 if self.subset != 'train' else 10 
        self.bound_padding = 0.03 # 0.001 == 1mm. should be enough to include object around the hand
        self.iou_3d_thr = 0.1  # contact threshold using 3d iou of hand and object meshes
        self.input_camera = '836212060125' # the first view of the dataset. designate this when a single image input

        # becareful not to overwrite these
        self.intr_cam_info = self._load_intr_cam()
        self.extr_cam_info = self._load_extr_cam()
        self.datalist = self._load_data()

    def _load_intr_cam(self):
        intr_cam_data = {}
        intr_cam_file_path_list = glob.glob(self.calibration_dir + '/intrinsics/*.yml')
        for file_path in intr_cam_file_path_list:
            cam_name = osp.basename(file_path).split('_')[0]
            with open(file_path, 'r') as f:
                intr = yaml.load(f, Loader=yaml.FullLoader)
            ffpp = intr['color']
            K = np.array([[ffpp['fx'], 0., ffpp['ppx']], [0., ffpp['fy'], ffpp['ppy']], [0., 0., 1.]], dtype=np.float32)

            intr_cam_data[cam_name] = K
        
        return intr_cam_data

    def _load_extr_cam(self):
        extr_cam_data = {}
        extr_cam_file_path_list = glob.glob(self.calibration_dir + '/extrinsics_*/extrinsics.yml')
        for file_path in extr_cam_file_path_list:
            with open(file_path, 'r') as f:
                extr = yaml.load(f, Loader=yaml.FullLoader)
            T = extr['extrinsics']
            T = {
                cam: np.array(T[cam], dtype=np.float32).reshape(3, 4) for cam in T
            }
            sequence = file_path.split('/')[-2][11:] # '20201014_215638'
            extr_cam_data[sequence] = T

        return extr_cam_data

    def _load_world2cam(self, meta_file, camera):
        # load cam2world
        with open(meta_file, 'r') as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)

        cam2world = copy.deepcopy(self.extr_cam_info[meta['extrinsics']][camera]) # (3,4) # may have bug // meta['extrinsics'] and sequence could be different,which is correct. Do different sequences have the same cam_info?
        # format extrinsics
        R = cam2world[:3, :3]
        t = cam2world[:, 3:] 

        # cam2world -> world2cam
        R_inv = R.T
        t_inv = - R.T @ t

        return R_inv, t_inv

    # append data to the datalist
    def _load_helper(self, datalist, data, do_world2inputcam=True):
        data['inv_aug_R'] = np.eye(3, dtype=np.float32)
        data['inv_aug_t'] = np.zeros((3,1), dtype=np.float32)
        if do_world2inputcam:

            ref_cam_idx = data['input_view_indices'][0]
            ref_cam_R = copy.deepcopy(data['R_list'][ref_cam_idx])  # (3,3)
            ref_cam_t = copy.deepcopy(data['t_list'][ref_cam_idx])  # (3,1)

            """ aug to prevent 3D conv overfitting """
            rotate_degrees = [np.pi/10, np.pi/10, np.pi/10]
            # augmenting rotation matrix that rotates coordinate axes of the input camera
            aug_R = Rz(rotate_degrees[2]) @ Ry(rotate_degrees[1]) @ Rx(rotate_degrees[0])
        
            # get camera orientation and location
            ref_cam_ori = ref_cam_R.T @ aug_R # multiply from right! to make the ref_cam eventually be aug_R
            ref_cam_center = - ref_cam_R.T @ ref_cam_t
            # ref_cam_ori = aug_R @ ref_cam_R.T  #small mistake; 
            # ref_cam_center = - ref_cam_R.T @ ref_cam_t
            
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

    # parse every data in the world coordinate to the (first) input camera coordinate
    # do the inplace replacement
    def _transform_dictdata_world2inputcam(self, data, R, t):
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
                    R_, t_ = pred_world2manojoint[:, :3, :3], pred_world2manojoint[:, :3, 3:]
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
        
    def _parse_label_path(self, label_path):
        label_path_parts = label_path.split('/')
        subject = label_path_parts[-4]  # '20200820-subject-03'
        sequence = label_path_parts[-3]  # '20200820_135508'
        camera = label_path_parts[-2]  # '836212060125'
        frame = label_path_parts[-1].split('_')[-1][:-4]  # '000000'

        return label_path_parts, subject, sequence, camera, frame

    def _load_data(self):
        with open(self.annot_path, 'r') as f:
            annot_list = json.load(f)

        with open(self.world_data_path, 'r') as f:
            world_data_dict = json.load(f)  # hierachical dictionary; subject - sequences - frame

        """ Organize the data by scene (one scene has 8 camera images) """
        object_set = set()
        subject_set = set()
        data_dict = defaultdict(lambda: defaultdict(list))
        for idx, annot in enumerate(tqdm(annot_list)):
            rgb_path = annot['color_file']  # RGB image '/home/hongsuk/Data/DexYCB/20200820-subject-03/20200820_135508/836212060125/color_000000.jpg'
            depth_path = annot['depth_file']  # 3 channel depth image 0~255 '/home/hongsuk/Data/DexYCB/20200709-subject-01/20200709_142123/836212060125/aligned_depth_to_color_000000.png'
            label_path = annot['label_file']  # '/home/hongsuk/Data/DexYCB/20200820-subject-03/20200820_135508/836212060125/label_000000.npz'
            # Replace the absolute directory
            # https://github.com/NVlabs/dex-ycb-toolkit
            rgb_path = rgb_path.replace('/home/hongsuk/Data/DexYCB', self.data_dir)
            depth_path = depth_path.replace('/home/hongsuk/Data/DexYCB', self.data_dir)
            label_path = label_path.replace('/home/hongsuk/Data/DexYCB', self.data_dir)
            label_path_parts, subject, sequence, camera, frame = self._parse_label_path(label_path)
            grabbing_object_id = annot['ycb_ids'][annot['ycb_grasp_ind']] # find target object
            sequence_frame_name = str(grabbing_object_id) + '_' + sequence + '_' + frame

            """ Check skip """
            if self.target_sequence_frame != '' and sequence_frame_name != self.target_sequence_frame:
                continue
            if self.target_objects[0] != -1 and grabbing_object_id not in self.target_objects:
                continue
            if self.mode == 'test':
                pass
            elif self.target_subjects != '' and subject not in self.target_subjects:
                continue

            if int(frame) % self.subsample_ratio != 0:
                continue

            """ Load camera parameters """
            # load K
            K = copy.deepcopy(self.intr_cam_info[camera])  # (3,3)
            # load world2cam
            meta_file = osp.join('/'.join(label_path_parts[:-2]), 'meta.yml')
            R, t = self._load_world2cam(meta_file, camera)  # (3,3), (3,1)
            cam_center = - R.T @ t  # (3,1)

            """ Load world object mesh and hand mesh and joints """
            try:
                world_data = world_data_dict[subject][sequence][frame]  # get world data (mesh and joints3d)
            except:
                continue  # if no mano label, skip
                
            
            if 'object_mesh' in data_dict[sequence_frame_name].keys():
                world_object_mesh = data_dict[sequence_frame_name]['object_mesh']
                world_hand_mesh = data_dict[sequence_frame_name]['hand_mesh']
                world_hand_joints = data_dict[sequence_frame_name]['hand_joints']
                world2manojoints = data_dict[sequence_frame_name]['world2manojoints']
                mano_side = data_dict[sequence_frame_name]['mano_side']
                hand_object_iou = data_dict[sequence_frame_name]['hand_object_iou']
                object2cam_object_pose = data_dict[sequence_frame_name]['object2cam_object_pose']

            else:
                # world_object_mesh: (N, 3), world_hand_mesh: (778,3), world_hand_joints: (21,3), world2manojoints: (16,4,4)
                world_object_mesh = np.array(world_data['object_mesh'], dtype=np.float32)
                world_hand_mesh = np.array(world_data['vert'], dtype=np.float32)
                world_hand_joints = np.array(world_data['joint'], dtype=np.float32)  # joint order: https://github.com/NVlabs/dex-ycb-toolkit/blob/master/dex_ycb_toolkit/dex_ycb.py#L59-81
                if np.isnan(world_hand_mesh.sum()) or np.isnan(world_hand_joints.sum()):
                    continue
                # Note: The number and orders of joints in world_hand_joints and world2manojoints do not match
                world2manojoints = np.array(world_data['world2manojoints'], dtype=np.float32)

                mano_side = annot['mano_side']

                # check contact
                if not self.include_no_intersection_data:
                    hand_object_iou = calc_3d_iou(world_hand_mesh, world_object_mesh)
                    if  hand_object_iou < self.iou_3d_thr:
                        continue
                
                object2cam_object_pose = np.load(label_path)['pose_y'][annot['ycb_grasp_ind']]  # 3x4


            """ Load predicted cam mesh and convert to world coordinate """
            # pred mesh is predicted from each camera... so multiple for one scene
            base_rgb_path = '/'.join(rgb_path.split('/')[-4:])

            if cfg.use_pred_hand and base_rgb_path in self.pred_hand_meshes[()].keys():  
                pred_data = self.pred_hand_meshes[()][base_rgb_path]  # 'cam_mesh (778,3), cam_joints (21,3), manjoints2cam transformation (16,4,4) 
                pred_cam_hand_mesh, pred_cam_hand_joints, pred_manojoints2cam = pred_data[0], pred_data[1], pred_data[2]

                # get cam gt mesh & align pred with gt
                cam_hand_mesh = (R @ world_hand_mesh.T + t).T
                cam_hand_joints = (R @ world_hand_joints.T + t).T

                """ For IHOI-NeRF """
                # get root-relative translation
                pred_manojoints2cam[:, :3, 3:] -= pred_manojoints2cam[ROOT_JOINT_IDX:ROOT_JOINT_IDX+1, :3, 3:]
                if mano_side == 'left':
                    # print('do left flipping')
                    # flip x translation
                    pred_manojoints2cam[:, :1, 3:] *= -1
                    # follow flipping axis angle. aa[:, 1:] *= -1 
                    # Rotation matrix from axis and angle - https://en.wikipedia.org/wiki/Rotation_matrix
                    pred_manojoints2cam[:, :3, 0] *= -1
                    pred_manojoints2cam[:, 0, :3] *= -1


                # root align
                pred_manojoints2cam[:, :3, 3:] += cam_hand_joints[ROOT_JOINT_IDX:ROOT_JOINT_IDX+1, :].T[None, :, :]
                # to world
                pred_cam2manojoints = np.linalg.inv(pred_manojoints2cam)
                world2cam = np.eye(4, dtype=np.float32)
                world2cam[:3, :3], world2cam[:3, 3:] = R, t
                pred_world2manojoints = pred_cam2manojoints @ world2cam[None, :, :]

                """ For MonoNHR, HandNeRF / and getting boundary """
                # get root-relative pred mesh and root align
                pred_cam_hand_mesh -= pred_cam_hand_joints[ROOT_JOINT_IDX:ROOT_JOINT_IDX+1]
                pred_cam_hand_joints -= pred_cam_hand_joints[ROOT_JOINT_IDX:ROOT_JOINT_IDX+1]

                # flip back to left hand
                if mano_side == 'left':
                    pred_cam_hand_mesh[:, 0] *= -1
                    pred_cam_hand_joints[:, 0] *= -1

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
            data_dict[sequence_frame_name]['grabbing_object_id'] = grabbing_object_id  # for segmentation label 
            data_dict[sequence_frame_name]['hand_object_iou'] = hand_object_iou   # 0~1
            data_dict[sequence_frame_name]['mano_side'] = mano_side

            # the list length is the number of cameras
            data_dict[sequence_frame_name]['rgb_path_list'].append(rgb_path)
            data_dict[sequence_frame_name]['depth_path_list'].append(depth_path)
            data_dict[sequence_frame_name]['label_path_list'].append(label_path)
            data_dict[sequence_frame_name]['camera_name_list'].append(camera)
            data_dict[sequence_frame_name]['K_list'].append(K)  # [(3,3), ...]
            data_dict[sequence_frame_name]['R_list'].append(R)  # [(3,3), ...]
            data_dict[sequence_frame_name]['t_list'].append(t)  # [(3,1), ...]
            data_dict[sequence_frame_name]['camera_center_list'].append(cam_center)  # [(3,1), ...]
            
            # world data
            data_dict[sequence_frame_name]['pred_world_hand_mesh_list'].append(pred_world_hand_mesh)  # [(778,3), ...]
            data_dict[sequence_frame_name]['pred_world_hand_joints_list'].append(pred_world_hand_joints)  # [(778,3), ...]
            data_dict[sequence_frame_name]['pred_world2manojoints'].append(pred_world2manojoints)  # [(16,4,4), ...]

            data_dict[sequence_frame_name]['object_mesh'] = world_object_mesh  # (num_points, 3)
            data_dict[sequence_frame_name]['hand_mesh'] = world_hand_mesh  # (778,3)
            data_dict[sequence_frame_name]['hand_joints'] = world_hand_joints  # (21,3)
            data_dict[sequence_frame_name]['world2manojoints'] = world2manojoints # (16,4,4)
            data_dict[sequence_frame_name]['object2cam_object_pose'] = object2cam_object_pose
            object_set.add(grabbing_object_id)
            subject_set.add(subject)


        print("[Dataloader] Actual Number of objects: ", len(object_set))
        print("[Dataloader] Actual Objects: ", object_set)
        print("[Dataloader] Actual Subjects: ", subject_set)

        """ Parse the dictionary into list """
        # cam dependent keys to sort again
        cam_dependent_keys = ['rgb_path_list', 'depth_path_list', 'label_path_list', 'camera_name_list', 'K_list', 'R_list', 't_list', 'camera_center_list']
        data_list = []
        input_rgb_path_dict = {}
        for sequence_frame_name, sequence_frame_annot_dict in data_dict.items():  # key: sequence name, value: dict            
            # number of views in the sequence
            num_views = len(sequence_frame_annot_dict['rgb_path_list'])
            # skip if num_views are smaller than num_input_views -
            if num_views < cfg.num_input_views:
                continue

            # skip if the selected input view is not contained
            if self.input_camera != '' and self.input_camera not in sequence_frame_annot_dict['camera_name_list']:
                continue
            # sort the camera dependent values
            sorted_indices = [i[0] for i in sorted(enumerate(sequence_frame_annot_dict['camera_name_list']), key=lambda x:x[1])]
            for cam_dep_key in cam_dependent_keys:
                sequence_frame_annot_dict[cam_dep_key] = [sequence_frame_annot_dict[cam_dep_key][i] for i in sorted_indices]

            # fixed input views, get uniformaly sampled input views
            if cfg.num_input_views == 1 and self.input_camera != '':
                input_view_index = sequence_frame_annot_dict['camera_name_list'].index(self.input_camera)
                input_view_indices = [input_view_index]  

                rgb_path = sequence_frame_annot_dict['rgb_path_list'][input_view_index]
                input_rgb_path_dict[sequence_frame_name] = '/'.join(rgb_path.split('/')[-4:])
            else:  # not implemented
                raise NotImplementedError("Currently only supports a single input view")
                pace = num_views // cfg.num_input_views
                input_view_indices = list(range(num_views))[::pace][:cfg.num_input_views]
            sequence_frame_annot_dict['input_view_indices'] = input_view_indices 

            # save the input image name 

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
            target_img_list_path = osp.abspath(osp.join(self.data_dir, 'annotation', f'{cfg.test_config}_test_list.json'))
            with open(target_img_list_path, 'w') as f:
                json.dump(input_rgb_path_dict, f)
            # import pdb; pdb.set_trace()

        return data_list

    def load_3d_data(self, object_mesh, hand_mesh):
        """
        object_mesh: (num_points,3) in world, use object mesh just to get the center
        hand_mesh: (778, 3) in world
        # pred_hand_mesh: (778, 3) in world, estimated by off-the-shelf. 

        // Returns //
        hand_object_center: (3,1) in world
        world_bounds: (2,3) in world
        world_hand_mesh_voxel_coord: (778, 3), coordinates of world mesh vertices in the descritized volume
        world_mesh_voxel_out_sh: (3,) np.int32, shape of the world mesh volume
        """
        
        # combine two meshes
        # using the GT object mesh for evaluation of rendering. not really necessary in pratice.
        mesh = np.concatenate([object_mesh, hand_mesh])  # (num_object_points + 778, 3)

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
        world_hand_mesh_voxel_coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape; that includes BOTH hand and object
        world_mesh_voxel_out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
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
        input_voxel_size = cfg.input_mesh_voxel_size  # voxel resolution for grid point sampling
        world_grid_points = get_grid_points_from_bounds(world_bounds, input_voxel_size, self.mode)  # (x_size, y_size, z_size, 3)
        world_object_inside = prepare_inside_world_object_pts(world_grid_points, input_view_R, input_view_t, input_view_K, input_mask)  # np.uint8, [0,1], (x_size, y_size, z_size)
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

    def load_2d_input_view(self, rgb_path, label_path, grabbing_object_id, K, R, t, world_bounds, camera_center, object_center):
        rgb = load_img(rgb_path) / 255.
        seg = np.load(label_path)['seg']   # np.uint8, 640 x 480, 0~255

        mask = self.parse_mask(seg, grabbing_object_id)

        assert rgb.shape[:2] == mask.shape[:2], print('rgb.shape & mask.shape ', rgb.shape, mask.shape)

        # affine transform for feature extraction
        img, depth, mask, trans, bbox = self.affine_transform_and_masking(rgb, None, mask, cfg.input_img_shape, expand_ratio=cfg.input_bbox_expand_ratio, masking=True) 

        new_K = K.copy()
        new_K[0, 0] = K[0, 0] * cfg.input_img_shape[1] / bbox[2]
        new_K[1, 1] = K[1, 1] * cfg.input_img_shape[0] / bbox[3]
        new_K[0, 2] = (K[0, 2] - bbox[0]) * cfg.input_img_shape[1] / bbox[2]
        new_K[1, 2] = (K[1, 2] - bbox[1]) * cfg.input_img_shape[0] / bbox[3]


        if cfg.debug:
            pose = np.concatenate([R, t], axis=1)
            bound_mask = get_bound_2d_mask(world_bounds, new_K, pose, img.shape[0], img.shape[1])
            cv2.imshow("[input] bound_mask ", bound_mask * 255. )
            cv2.waitKey(0)

        return img, mask, new_K

    def affine_transform_and_masking(self, img, depth, mask, out_shape, expand_ratio, masking=True):
        bbox = cv2.boundingRect(mask.astype(np.uint8))  # x, y, w, h
        bbox = process_bbox(bbox, img.shape[1], img.shape[0], out_shape, expand_ratio)

        trans = get_affine_trans_mat(bbox, out_shape)

        img = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_NEAREST)

        # masking in input can be False, not critical
        if masking:
            img[mask == 0] = 0

        if depth is not None:
            depth = cv2.warpAffine(depth, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_NEAREST)
            depth[mask == 0] = 0

        return img, depth, mask, trans, bbox

    def parse_mask(self, seg, grabbing_object_id, only_object=False):
        # re-organize the mask element values
        obj_seg = (seg == grabbing_object_id).astype(np.uint8) * OBJ_SEG_IDX
        hand_seg = (seg == ANNOT_HAND_SEG_IDX).astype(np.uint8) * HAND_SEG_IDX
        if only_object:
            mask = obj_seg
        else:
            mask = (obj_seg + hand_seg)

        return mask

    def load_2d_render_view(self, world_mesh, rgb_path, depth_path, label_path, grabbing_object_id, world_bounds, R, t, K, camera_center, object_center, render_whole):
        """
        camera_center: (3,1)
        object_center: (3,1)
        """
        img = load_img(rgb_path) / 255.  # 0~1.
        depth = load_img(depth_path)[0] / 255.  # 0~1.  # not in use. even this read function is wrong

        seg = np.load(label_path)['seg']   # np.uint8, 640 x 480, 0~255

        mask = self.parse_mask(seg, grabbing_object_id, only_object=False)
        assert img.shape[:2] == mask.shape[:2], print('img.shape & mask.shape ', img.shape, mask.shape)

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

            img = cv2.resize(img, (render_img_shape[1], render_img_shape[0]), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (render_img_shape[1], render_img_shape[0]), interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(depth, (render_img_shape[1], render_img_shape[0]), interpolation=cv2.INTER_NEAREST)

        else:
            img, depth, mask, trans, bbox = self.affine_transform_and_masking(img, depth, mask, render_img_shape, expand_ratio=cfg.render_bbox_expand_ratio)
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
            # bound_mask = get_bound_2d_mask(
            #     world_bounds, new_K, pose, img.shape[0], img.shape[1])
            # cv2.imshow("[render] bound_mask ", bound_mask * 255.)
            # cv2.waitKey(0)
            img_points = project(world_mesh.copy(), new_K, pose)
            new_img = vis_mesh(img, img_points)
            cv2.imshow(f"[render]", new_img[:, :, ::-1])
            cv2.waitKey(0)
            print(f"[Dataloader] rendering {rgb_path}")

        return rgb, depth, fg_mask, org_mask, ray_o, ray_d, near, far, coord_, mask_at_box

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        # length: 1
        # for segmentation label
        grabbing_object_id = data['grabbing_object_id'] # int
        world_object_mesh = data['object_mesh']  # (N,3)
        world_hand_mesh = data['hand_mesh']  # (778,3)
      
        world_hand_joints = data['hand_joints']  # (21,3)
        world2manojoints = data['world2manojoints'][1:]  # (15,4,4), exclude wrist joint
        if cfg.use_pred_hand:
            # pick the first one. anyway rightnow I only have first one
            pred_world_hand_mesh_list = data['pred_world_hand_mesh_list']  # [(778,3), ..]
            world_hand_mesh = pred_world_hand_mesh_list[0]
            pred_world_hand_joints_list = data['pred_world_hand_joints_list']  # [(21,3), ..]
            world_hand_joints = pred_world_hand_joints_list[0]
            pred_world2manojoints = data['pred_world2manojoints']
            world2manojoints = pred_world2manojoints[0][1:] # (15,4,4), exclude wrist joint            
            
        mano_side = data['mano_side']
        # length: number of cameras
        rgb_path_list = data['rgb_path_list']
        depth_path_list = data['depth_path_list']
        label_path_list = data['label_path_list']
        R_list = data['R_list']
        t_list = data['t_list']
        K_list = data['K_list']
        world_camera_center_list = data['camera_center_list']
        # selecting indices
        input_view_indices = data['input_view_indices']
        render_view_indices = data['render_view_indices']

        """ Load world 3d data """
        world_hand_object_center, world_bounds, world_hand_mesh_voxel_coord, world_mesh_voxel_out_sh = self.load_3d_data(world_object_mesh, world_hand_mesh)
        # world_mesh_voxel_out_sh: output shape that should include BOTH hand and object
        

        """ Load input view """
        if cfg.num_input_views > 0 and input_view_indices is not None:
            input_rgb_path_list = [x for i, x in enumerate(rgb_path_list) if i in input_view_indices] 
            input_label_path_list = [x for i, x in enumerate(label_path_list) if i in input_view_indices]  
            input_view_R_list = [x for i, x in enumerate(R_list) if i in input_view_indices]  
            input_view_t_list = [x for i, x in enumerate(t_list) if i in input_view_indices]  

            input_view_K_list = []
            input_img_list = []
            input_mask_list = []
            input_view_pc_list = []

            for input_idx in input_view_indices:
                input_img, input_mask, input_new_K= self.load_2d_input_view(rgb_path_list[input_idx], label_path_list[input_idx], grabbing_object_id, K_list[input_idx],
                                                                 R_list[input_idx], t_list[input_idx], world_bounds, world_camera_center_list[input_idx], world_hand_object_center)
                input_view_pc = get_pointcloud_from_rgbd(rgb_path_list[input_idx], label_path_list[input_idx], depth_path_list[input_idx], K_list[input_idx], grabbing_object_id)


                if cfg.debug:
                    print("input: ", rgb_path_list[input_idx])
                    # Visualize projected points
                    Rt = np.concatenate([R_list[input_idx], t_list[input_idx]], axis=1)
                    img_points = project(world_hand_mesh.copy(), input_new_K, Rt)
                    new_img = vis_mesh(input_img * 255, img_points)
                    cv2.imwrite("rgb_projected.jpg", new_img[:, :, ::-1])
                    white_img = np.ones_like(input_img) * 255
                    new_img2 = vis_mesh(white_img, img_points)
                    cv2.imwrite("white_projected.jpg", new_img2[:, :, ::-1])


                    # cv2.imshow("[input] mesh projected", new_img[:, :, ::-1])
                    # cv2.waitKey(0)
                    import pdb; pdb.set_trace()

                input_view_K_list.append(input_new_K)
                input_img_list.append(input_img)
                input_mask_list.append(input_mask)
                input_view_pc_list.append(input_view_pc)

            # stack
            input_view_R = np.stack(input_view_R_list)
            input_view_t = np.stack(input_view_t_list)
            input_view_K = np.stack(input_view_K_list)
            input_img = np.stack(input_img_list)
            input_mask = np.stack(input_mask_list)
            input_view_pc = np.stack(input_view_pc_list)

            if 'mononhr' in cfg.nerf_mode or 'noobjectpixelnerf' in cfg.nerf_mode:
                world_inside_object_points, world_object_points_voxel_coord = np.empty(0), np.empty(0)
            else:
                # get partial 3D object points from 2D input view
                world_inside_object_points, world_object_points_voxel_coord = self.load_3d_object_data(world_bounds, world_mesh_voxel_out_sh, input_view_R, input_view_t, input_view_K, input_mask)

        else:
            # dummy
            input_view_R = np.ones(1)
            input_view_t = np.ones(1)  
            input_view_K = np.ones(1)  
            input_img = np.ones(1)



        """ Prepare reconstruction OR render view data """
        if self.mode == 'test' and cfg.test_mode == 'recon3d':
            voxel_size = cfg.mc_voxel_size  # voxel resolution for marching cube
            world_grid_points = get_grid_points_from_bounds(world_bounds, voxel_size, self.mode) 
            print("Reconstructing... Input world points shape: ", world_grid_points.shape)
            
            if not cfg.use_multiview_masks_for_recon:  
                inp = 0
                world_inside = prepare_inside_world_pts(world_grid_points, np.stack([R_list[inp]]), np.stack(
                    [t_list[inp]]), np.stack([K_list[inp]]), [label_path_list[inp]], grabbing_object_id, 'DexYCB')
            else:
                world_inside = prepare_inside_world_pts(world_grid_points, np.stack(R_list), np.stack(
                    t_list), np.stack(K_list), label_path_list, grabbing_object_id, 'DexYCB')

            # dummy
            rgb, fg_mask, org_mask, ray_o, ray_d, near, far, mask_at_box = np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0)

        else:
            render_rgb_path_list = [x for i, x in enumerate(rgb_path_list) if i in render_view_indices] 
            # right now, the depth is just used for evaluation
            render_depth_path_list = [x for i, x in enumerate(depth_path_list) if i in render_view_indices]
            render_label_path_list = [x for i, x in enumerate(label_path_list) if i in render_view_indices]  
            render_R_list = [x for i, x in enumerate(R_list) if i in render_view_indices]  
            render_t_list = [x for i, x in enumerate(t_list) if i in render_view_indices] 
            render_K_list = [x for i, x in enumerate(K_list) if i in render_view_indices]  
            render_world_camera_center_list = [x for i, x in enumerate(world_camera_center_list) if i in render_view_indices] 
            if self.mode == 'test' and cfg.test_mode == 'render_rotate':
                # dummy
                render_rgb_path_list = rgb_path_list[0:1]
                render_depth_path_list = depth_path_list[0:1]
                render_label_path_list = label_path_list[0:1]
                # actual data for rendering rotating views
                render_R_list = [data['rotating_R']]
                render_t_list = [data['rotating_t']]
                render_K_list = [data['rotating_K']]
                render_world_camera_center_list = [data['rotating_camera_center']]

            # follow the NueralBody code
            rgb_list, depth_list, fg_mask_list, org_mask_list, ray_o_list, ray_d_list, near_list, far_list, mask_at_box_list = [], [], [], [], [], [], [], [], []
            for rgb_path, depth_path, label_path, R, t, K, world_camera_center in zip(render_rgb_path_list, render_depth_path_list, render_label_path_list, render_R_list, render_t_list, render_K_list, render_world_camera_center_list):
                render_whole = self.render_whole
                rgb, depth, fg_mask, org_mask, ray_o, ray_d, near, far, _, mask_at_box = \
                    self.load_2d_render_view(world_object_mesh, rgb_path, depth_path, label_path, grabbing_object_id, world_bounds, R, t, K, world_camera_center, world_hand_object_center, render_whole=render_whole)

                rgb_list.append(rgb)
                depth_list.append(depth)
                fg_mask_list.append(fg_mask)
                org_mask_list.append(org_mask)
                ray_o_list.append(ray_o)
                ray_d_list.append(ray_d)
                near_list.append(near)
                far_list.append(far)
                mask_at_box_list.append(mask_at_box)

            rgb, depth, fg_mask, org_mask, ray_o, ray_d, near, far, mask_at_box = np.concatenate(rgb_list), np.concatenate(depth_list), np.concatenate(fg_mask_list), np.concatenate(org_mask_list), np.concatenate(ray_o_list), np.concatenate(ray_d_list), np.concatenate(near_list), np.concatenate(far_list), np.concatenate(mask_at_box_list)    


        """ Prepare data dictionaries """
        input_view_parameters = {
            'input_view_R': input_view_R, 'input_view_t': input_view_t, 'input_view_K': input_view_K, 'input_view_mask': input_mask,
            # just for saving
            'input_view_pointcloud': input_view_pc
        }
        world_3d_data = {
            'world_bounds': world_bounds, 'world_hand_joints': world_hand_joints, 'world2manojoints': world2manojoints, 'world_hand_mesh': world_hand_mesh, 'world_hand_mesh_voxel_coord': world_hand_mesh_voxel_coord, 'world_mesh_voxel_out_sh': world_mesh_voxel_out_sh, 'mano_side': mano_side,
            'world_object_points': world_inside_object_points, 'world_object_points_voxel_coord': world_object_points_voxel_coord,
            'world_object_mesh': world_object_mesh # for evaluation
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
                meta_info['rotating_render_idx'] = data['rotating_render_idx']  # int
            
            if cfg.test_mode == 'recon3d':
                world_3d_data['inside'] = world_inside
                world_3d_data['pts'] = world_grid_points.astype(np.float32)
                # transform the output 3d to the exact input camera coordinate system
                meta_info['inv_aug_R'] = data['inv_aug_R']
                meta_info['inv_aug_t'] = data['inv_aug_t']

                # some effort to get separate mesh for IHOINeRF; doesn't help
                if False and 'handjoint' in cfg.nerf_mode:
                    label_path_list = data['label_path_list']
                    mask_list = []
                    for label_path in label_path_list:
                        seg= np.load(label_path)['seg']
                        mask = self.parse_mask(seg, grabbing_object_id)
                        mask_list.append(mask)

                    # get object mesh
                    world_hand_inside = prepare_inside_world_hand_pts(world_grid_points, np.stack(R_list), np.stack(t_list), np.stack(K_list), np.stack(mask_list))  # np.uint8, [0,1], (x_size, y_size, z_size)
                    not_world_hand_inside = ~world_hand_inside.astype(bool)
                    world_inside = world_inside * not_world_hand_inside.astype(int)

                    # get hand mesh
                    # world_object_inside = prepare_inside_world_object_pts(world_grid_points, np.stack(R_list), np.stack(t_list), np.stack(K_list), np.stack(mask_list))  # np.uint8, [0,1], (x_size, y_size, z_size)
                    # not_world_object_inside = ~world_object_inside.astype(bool)
                    # world_inside = world_inside * not_world_object_inside.astype(int)
                    
                    world_3d_data['inside'] = world_inside

                # pass camera coordinate object detailed mesh with face, texture
                class_name = DEXYCB_OBJECT_CLASSES[grabbing_object_id] #grasping_object_pose 
                object_model_path = osp.join(self.data_dir, 'models', class_name, 'textured_simple.obj')
                meta_info['object_model_name'] = class_name
                meta_info['object_model_path'] = object_model_path
                meta_info['object2cam_object_pose'] = data['object2cam_object_pose']

        # convert back to float32
        for dictdata in [input_view_parameters, world_3d_data, rendering_rays, meta_info]:
            for k, v in dictdata.items():
                if type(v) == np.ndarray and v.dtype == np.float32:
                    dictdata[k] = v.astype(np.float32)

        return input_img, input_view_parameters, world_3d_data, rendering_rays, meta_info



if __name__ == '__main__':
    split = 'train'
    dataset = DexYCB(split)

    for idx in range(len(dataset)):
        dataset.__getitem__(idx)