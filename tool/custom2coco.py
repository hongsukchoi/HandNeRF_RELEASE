# For HandOccNet
import cv2
from tqdm import tqdm
import numpy as np
import torch

import sys
import os.path as osp
import argparse
import glob
import json
import pickle
sys.path.insert(0, '../common')
from utils.joint_mapper import JointMapper, smpl_to_openpose
from utils.vis import vis_2d_keypoints

import smplx
# Add this to L1683 of `envs/handnerf/lib/python3.7/site-packages/smplx/body_models.py`.   
# ```     
# if self.is_rhand:
#     tips = vertices[:, [745, 317, 444, 556, 673]]
# else:
#     tips = vertices[:, [745, 317, 445, 556, 673]]
#     joints = torch.cat([joints, tips], dim=1)
# ```
mano_model_path = '/labdata/hongsuk/human_model_files'
mano_model = smplx.create(model_type='mano', model_path=mano_model_path, joint_mapper=JointMapper(smpl_to_openpose('mano')),
            gender='male', use_pca=False, flat_hand_mean=False)


def get_smplifyx(path, cam_R, cam_t):
    # cam_R: (3, 3), cam_t: (3,1)

    # load pickle file
    with open(path, 'rb') as f:
        smplifyx_data = pickle.load(f)
    # 'betas', 'global_orient', 'hand_pose', 'global_hand_translation', 'hand_scale'

    mano_pose_global = torch.tensor(smplifyx_data['global_orient'])  # (1,3)
    
    # world to cam                
    root_pose, _ = cv2.Rodrigues(smplifyx_data['global_orient'][0])
    root_pose, _ = cv2.Rodrigues(cam_R @ root_pose)
    mano_pose_global = torch.tensor(root_pose.reshape(-1, 3))


    mano_pose_local = torch.tensor(smplifyx_data['hand_pose'])  # (1,45)
    mano_shape = torch.tensor(smplifyx_data['betas'])  # (1,10)
    hand_scale = smplifyx_data['hand_scale'][0]
    hand_translation = torch.tensor(smplifyx_data['global_hand_translation']) # (3)


    with torch.no_grad():
        mano_output = mano_model(betas=mano_shape, global_orient=mano_pose_global, hand_pose=mano_pose_local, return_verts=True, return_full_pose=False)
        vertices, joints = mano_output.vertices[0], mano_output.joints[0]

        # world to cam  
        hand_translation = torch.tensor(cam_R) @ hand_translation + torch.tensor(cam_t.T)
        root_joint_coord = joints[0:1]  # (1,3)
        hand_translation = hand_translation - root_joint_coord + \
        (torch.tensor(cam_R) @ root_joint_coord.transpose(0,1)).transpose(0,1) 
        # fortunately hand_scale is 1

        hand_mesh = hand_scale * vertices + hand_translation
        hand_joints = hand_scale * joints + hand_translation

    mano_pose = torch.cat([mano_pose_global, mano_pose_local], dim=-1).numpy()
    mano_shape = mano_shape.numpy()

    return mano_pose, mano_shape, hand_joints.detach().numpy()
    

def load_cam(cam_data_path):
    # load intrinsics and extrinsics
    with open(cam_data_path, 'r') as f:
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom_dir', type=str, default='')

    args = parser.parse_args()

    return args
    
if __name__ == '__main__':
    args = parse_args()
    root_path = args.custom_dir #

    cam_info_dict = load_cam(osp.join(root_path), 'cam_params_final.json')

    set_list = ['train', 'test']
    subsample_ratio = 1

    for set_split in set_list:

        set_dir = osp.join(root_path, set_split)

        img_id = 0
        annot_id = 0
        images = [] 
        annotations = []
        
        subset_dir_list =sorted([x for x in glob.glob(set_dir + '/handnerf_*') if 'handmesh' not in x])

        for subset_dir in subset_dir_list:
            cam_dir_list = sorted([x for x in glob.glob(subset_dir + '/cam_*') if len(osp.basename(x)) == 5])
            smplifyx_result_dir = osp.join(subset_dir + '_handmesh', 'results') 

            print("Processing: ", osp.basename(subset_dir))
            for cam_dir in cam_dir_list:    
                img_path_list = sorted(glob.glob(cam_dir + '/*.jpg'))

                cam_idx = int(cam_dir.split('_')[-1])
                cam_data = cam_info_dict[cam_idx] # intrinsic, extrinsic
    
                for img_path in img_path_list:           
                    img = cv2.imread(img_path)
                    height, width = img.shape[:2]
                    file_name = '/'.join(img_path.split('/')[-4:])
                    frame_idx_str = osp.basename(img_path).split('_')[1][:-4]

                    ### Parse mesh annotation ###
                    smplifyx_path = osp.join(smplifyx_result_dir, frame_idx_str, 'result.pkl')

                    cam_R, cam_t = cam_data['extrinsic'][:3, :3], cam_data['extrinsic'][:3, 3:]
                    mano_pose, mano_shape, cam_joints = get_smplifyx(smplifyx_path, cam_R, cam_t)  

                    # hand_joints = (cam_R @ hand_joints.T).T + cam_t.T
                    K = cam_data['intrinsic']
                    img_joints = cam_joints @ K.T
                    img_joints  = img_joints[:, :2] / img_joints[:, 2:]
                    # newimg = vis_2d_keypoints(img, img_joints)
                    # cv2.imshow('check', newimg)
                    # cv2.waitKey(0)
                    # import pdb; pdb.set_trace()

                    ### Parse bounding box ###
                    bbox_annot_path = img_path[:-4] + '.json'
                    with open(bbox_annot_path, 'r') as f:
                        bbox_data = json.load(f)
                        hand_bbox, object_bbox = None, None
                        for bbox_label in bbox_data['shapes']:
                            if bbox_label['label'] == 'right_hand':
                                hand_bbox = np.array(bbox_label['points']).reshape(-1)
                                # xyxy -> xywh
                                hand_bbox[2], hand_bbox[3] = hand_bbox[2] - hand_bbox[0], hand_bbox[3] - hand_bbox[1]
                                hand_bbox = hand_bbox.tolist()
                            elif bbox_label['label'] == 'target_object':
                                object_bbox = np.array(bbox_label['points']).reshape(-1)
                                # xyxy -> xywh
                                object_bbox[2], object_bbox[3] = object_bbox[2] - object_bbox[0], object_bbox[3] - object_bbox[1]
                                object_bbox = object_bbox.tolist()

                    img_dict = {}
                    annot_dict = {}

                    img_dict['width'] = width
                    img_dict['height'] = height
                    img_dict['file_name'] = file_name
                    img_dict['id'] = img_id

                    annot_dict['id'] = annot_id
                    annot_dict['image_id'] = img_id
                    annot_dict['joints_coord_cam'] = cam_joints.tolist()
                    annot_dict['joints_img'] = img_joints.tolist()
                    annot_dict['hand_type'] = 'right'
                    annot_dict['cam_param'] = {'R': cam_R.tolist(), 't': cam_t.tolist(), 'K': K.tolist()}
                    annot_dict['mano_param'] = {'mano_pose': mano_pose.tolist(), 'mano_shape': mano_shape.tolist()}
                    annot_dict['hand_bbox'] = hand_bbox
                    annot_dict['object_bbox'] = object_bbox

                    img_id += 1
                    annot_id += 1

                    images.append(img_dict)
                    annotations.append(annot_dict)

        print(f"Total {len(images)} images, {len(annotations)} annotations")
        output = {'images': images, 'annotations': annotations}
        output_path = osp.join(root_path, f'custom_{set_split}_data.json')
        with open(output_path, 'w') as f:
            json.dump(output, f)
        print('Saved at ' + output_path)