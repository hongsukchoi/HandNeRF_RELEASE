import os.path as osp
import glob
import argparse
import sys
import json
import yaml
import numpy as np
import torch
import cv2


from tqdm import tqdm

# DexYCB object id list -> https://github.com/NVlabs/dex-ycb-toolkit/blob/64551b001d360ad83bc383157a559ec248fb9100/dex_ycb_toolkit/dex_ycb.py#L36
_YCB_CLASSES = {
    1: '002_master_chef_can',
    2: '003_cracker_box',
    3: '004_sugar_box',
    4: '005_tomato_soup_can',
    5: '006_mustard_bottle',
    6: '007_tuna_fish_can',
    7: '008_pudding_box',
    8: '009_gelatin_box',
    9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    19: '051_large_clamp',
    20: '052_extra_large_clamp',
    21: '061_foam_brick',
}


def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()

def read_xyz(file_name='points.xyz'):
    vertices = []
    with open(file_name, 'r') as f:
        for line in f:
            xyz_parts = line.split(' ')
            xyz = [float(xyz_parts[0]), float(xyz_parts[1]), float(xyz_parts[2][:-1])]
            vertices.append(xyz)
    return vertices



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dexycb_dir', type=str, default='')

    args = parser.parse_args()

    return args
    
if __name__ == '__main__':
    args = parse_args()
    root_path = args.dexycb_dir # '/home/hongsuk.c/Data/DexYCB' #'/data/hongsuk/DexYCB'

    # Load object meshes 
    object_mesh_xyz_path_list = glob.glob(osp.join(root_path, 'models') + '/*/points.xyz')
    object_mesh_dict = {}  # '002_master_chef_can': [[]]
    for path in object_mesh_xyz_path_list:
        mesh_name = path.split('/')[-2]
        mesh = read_xyz(path)
        object_mesh_dict[mesh_name] = mesh

    sys.path.insert(0, './manopth')
    from manopth.manolayer import ManoLayer
    # Load MANO layer.
    mano_right_layer = ManoLayer(flat_hand_mean=False,
                            ncomps=45,
                            side='right',
                            mano_root='manopth/mano/models',
                            use_pca=True)
    mano_left_layer = ManoLayer(flat_hand_mean=False,
                            ncomps=45,
                            side='left',
                            mano_root='manopth/mano/models',
                            use_pca=True)
    mano_layer = {'right': mano_right_layer, 'left': mano_left_layer}

    category_list = ['s0']
    set_list = ['train', 'val', 'test']
    subsample_ratio = 1


    for category_split in category_list:
        for set_split in set_list:
            world_data = {}
            json_path = osp.join(root_path, 'annotation', f'{category_split}_{set_split}.json')


            with open(json_path, 'r') as f:
                annot_list = json.load(f)

            print(f"Parsing world data of split {category_split} {set_split} set ...")
            for annot in tqdm(annot_list):
                label_path = annot['label_file']  # '/home/hongsuk/Data/DexYCB/20200820-subject-03/20200820_135508/836212060125/labels_000000.npz'            
                # TEMP
                label_path = label_path.replace('/home/hongsuk/Data/DexYCB', root_path)
                mano_side = annot['mano_side']
                mano_betas = torch.tensor(annot['mano_betas'], dtype=torch.float32).unsqueeze(0)

                label_path_parts = label_path.split('/')
                subject = label_path_parts[-4]
                scenario = label_path_parts[-3]
                camera = label_path_parts[-2]
                frame = label_path_parts[-1].split('_')[-1][:-4]  # '000000'

                if subject not in world_data:
                    world_data[subject] = {}
                if scenario not in world_data[subject]:
                    world_data[subject][scenario] = {}
                if frame not in world_data[subject][scenario]:
                    world_data[subject][scenario][frame] = {
                        'vert': None, 
                        'joint': None, 
                        'world2manojoints': None,
                        'object_mesh': None
                    }
                else:
                    continue

                if int(frame) % subsample_ratio != 0:
                    continue

                label = np.load(label_path)
                """ load object mesh """
                object_poses = label['pose_y']

                ycb_grasp_ind = annot['ycb_grasp_ind']
                target_object_pose = object_poses[ycb_grasp_ind]
                target_object_id = annot['ycb_ids'][ycb_grasp_ind]
                target_object_class = _YCB_CLASSES[target_object_id]
                object_mesh = object_mesh_dict[target_object_class] 
                object_mesh = torch.Tensor(object_mesh)

                target_object_pose_R = torch.Tensor(target_object_pose[:3, :3])
                target_object_pose_t = torch.Tensor(target_object_pose[:3, 3:])
                camera_object_mesh =  torch.addmm(target_object_pose_t, target_object_pose_R, object_mesh.transpose(1,0))
                camera_object_mesh =camera_object_mesh.transpose(1,0)

                """ load mano parameters and joints3d in the camera coordinate """
                mano_poses = label['pose_m']  # [1, 51]
                mano_joints3d = label['joint_3d'] # [1, 21, 3]

                # filter no annotation data point
                if np.all(mano_poses == 0.0):
                    continue
                
                mano_pose = torch.from_numpy(mano_poses)
                vert, tmp_joint, manojoints2cam = mano_layer[mano_side](mano_pose[:, 0:48], mano_betas, mano_pose[:, 48:51])\
                # manojoints2cam: (1, 16, 4, 4)
                vert /= 1000
                vert = vert.view(778, 3)
                # vert = vert.numpy()
                
                # TEMP: save meshes in the camera coordinate
                # save_obj(vert.numpy(), mano_layer[mano_side].th_faces.numpy(), file_name=f'{camera}_hand.obj')
                # # object mesh face is dummy
                # save_obj(camera_object_mesh.numpy(), mano_layer[mano_side].th_faces.numpy(), file_name=f'{camera}_object.obj')

                """ transform to the world coordinate. # from dex-ycb repo """
                meta_file = osp.join('/'.join(label_path_parts[:-2]), 'meta.yml')
                with open(meta_file, 'r') as f:
                    meta = yaml.load(f, Loader=yaml.FullLoader)

                # Load extrinsics.  # from dex-ycb repo
                extr_file = root_path + "/calibration/extrinsics_" + meta[
                    'extrinsics'] + "/extrinsics.yml"
                with open(extr_file, 'r') as f:
                    extr = yaml.load(f, Loader=yaml.FullLoader)
                T = extr['extrinsics']
                T = {
                    cam: torch.tensor(T[cam], dtype=torch.float32,
                                    device='cpu').view(3, 4) for cam in T
                }

                cam2world = T[camera]
                
                """ camera to world transformation """
                R = cam2world[:, :3] 
                t = cam2world[:, 3] 

                # transform object
                object_mesh_world = torch.addmm(t.unsqueeze(1), R, camera_object_mesh.transpose(1,0))
                object_mesh_world = object_mesh_world.transpose(1, 0)

                world_data[subject][scenario][frame]['object_mesh'] = object_mesh_world.tolist()


                # transform hand
                vert_world = torch.addmm(t.unsqueeze(1), R, vert.transpose(1,0))
                vert_world = vert_world.transpose(1,0)
                
                # save_obj(vert_world.numpy(), mano_layer[mano_side].th_faces.numpy(), file_name=f'{camera}_world_hand.obj')
                # save_obj(object_mesh_world.numpy(), mano_layer[mano_side].th_faces.numpy(), file_name=f'{camera}_world_object.obj')

                joint_world = torch.addmm(t.unsqueeze(1), R, torch.tensor(mano_joints3d[0]).transpose(1,0))
                joint_world = joint_world.transpose(1,0)

                world_data[subject][scenario][frame]['vert'] = vert_world.tolist()
                world_data[subject][scenario][frame]['joint'] = joint_world.tolist()

                # get world to joint coordinates
                manojoints2cam = manojoints2cam[0] #(16, 4, 4)
                manojoints2cam[:, :3, 3] = manojoints2cam[:, :3, 3] + mano_pose[:, 48:51]  # add the translation 
                cam2manojoints = np.linalg.inv(manojoints2cam)
                cam2world_4by4 = np.concatenate([cam2world, [[0,0,0,1]]],axis=0)
                world2cam = np.linalg.inv(cam2world_4by4)
                world2manojoints = cam2manojoints @ world2cam[None, :, :]

                world_data[subject][scenario][frame]['world2manojoints'] = world2manojoints.tolist()

            dump_path = osp.join(root_path, 'annotation', f'{category_split}_{set_split}_world_data.json')
            with open(dump_path,'w') as f:
                json.dump(world_data, f)

            



