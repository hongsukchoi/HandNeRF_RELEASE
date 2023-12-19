import os
import os.path as osp
import argparse
import json
import random
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import trimesh

from chamferdist import ChamferDistance
from tqdm import tqdm

from config import cfg, mano_layer
from loss import calc_3d_metrics
from utils.vis import save_obj, load_obj

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--exp_dir', type=str, default='')
    parser.add_argument('--cfg', type=str, default='', help='experiment configure file name')
    parser.add_argument('--obj', type=str, dest='test_objects', default='',  help='target objects to test')
    parser.add_argument('--scene', type=str,dest='test_scene', default='', help='target scene"s sequence frame name')
    parser.add_argument('--nerf_mode', type=str, default='semantic_handmeshpixelnerf_fine',  help='model to train/test')

    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():
    args = parse_args()
    cfg.set_args(args.gpu_ids, is_test=True, exp_dir=args.exp_dir)
    cudnn.benchmark = True
    if args.cfg:
        cfg.update(args.cfg)
    
    # set model
    cfg.nerf_mode = args.nerf_mode

    # Set test mode
    cfg.test_mode = 'recon3d'
    chamfer_distance = ChamferDistance()
    dummy_cd = 10 # mm. in case of NaN
    th_list = [.5/100, 1/100]  # F-score at threshold 5mm, F-score at threshold 10mm
    total_metrics = {'cd': [], 'f-5': [], 'f-10': []}
    hand_metrics = {'cd': [], 'f-5': [], 'f-10': []}
    object_metrics = {'cd': [], 'f-5': [], 'f-10': []}
    object_models = {}

    # import modules after setting gpu
    from base import Tester
    from utils.utils import make_folder
    
    tester = Tester(args.test_epoch)
    if args.test_objects != '':
        tester.test_objects = [int(x) for x in args.test_objects.split(',')]
    if args.test_scene != '':
        tester.test_scene = args.test_scene
    tester.make_batch_generator()
    tester.make_model()

    for itr, (input_img, input_view_paramters, world_3d_data, rendering_rays, meta_info) in tqdm(enumerate(tester.batch_generator)):
        # For saving
        scene_identifier = meta_info['scene_identifier'][0]  # assuming batch size 1
        
        # transform 3d points to the exact input camera coorindate
        inv_aug_R = meta_info['inv_aug_R'] # (1, 3, 3) 
        inv_aug_t = meta_info['inv_aug_t'] # (1, 3, 1)
        row = torch.Tensor([0, 0, 0, 1]).reshape(1,1,4)
        tmp = torch.cat([inv_aug_R, inv_aug_t], dim=-1)
        Rt_4by4 = torch.cat([tmp, row], dim=1)
        if not cfg.eval_only or cfg.testset in ['Custom']:
            # save input hand mesh
            world_hand_mesh, mano_side = world_3d_data['world_hand_mesh'][0].cpu().numpy(), world_3d_data['mano_side'][0]
            R, t = Rt_4by4[0][:3, :3].cpu().numpy(), Rt_4by4[0][:3, 3:].cpu().numpy()
            world_hand_mesh = (R @ world_hand_mesh.T).T + t.T

            midfix = 'PREDinput' if cfg.use_pred_hand else 'GTinput' 
            save_dir = osp.join(cfg.vis_dir, f'{scene_identifier}_{midfix}_3D_reconstruction_epoch{args.test_epoch}')
            make_folder(save_dir)
            file_path = osp.join(save_dir, f'input_hand.obj')
            save_obj(world_hand_mesh, f=mano_layer[mano_side].th_faces.numpy(), file_name=file_path)

            # save input image
            file_path = osp.join(save_dir, f'input_img.png')
            cv2.imwrite(file_path, input_img[0][0].cpu().numpy()[:, :, ::-1] * 255)

            # save the ground truth object mesh 
            if 'object_model_path' in meta_info:                 
                object_model_name = meta_info['object_model_name'][0]
                object_model_path = meta_info['object_model_path'][0]
                object2cam_object_pose = meta_info['object2cam_object_pose'][0].cpu().numpy()

                if object_model_name not in object_models:
                    vertices, rgb, faces = load_obj(object_model_path)
                    object_models[object_model_name] = (vertices, rgb, faces)
                else:
                    vertices, rgb, faces = object_models[object_model_name]
                
                file_path = osp.join(save_dir, f'gt_object_mesh.obj')
                object2cam_R, object2cam_t = object2cam_object_pose[:3, :3], object2cam_object_pose[:3, 3:]
                vertices = np.array(vertices)
                vertices = (object2cam_R @ vertices.T).T + object2cam_t.T
                save_obj(vertices, c=rgb, f=faces, file_name=file_path)

            # save input view's pointcloud for reference. Note that The model input does not include depth or pointcloud. 
            if 'input_view_pointcloud' in input_view_paramters:
                file_path = osp.join(save_dir, f'input_view_pointcloud.obj')
                pointcloud = input_view_paramters['input_view_pointcloud'][0][0].cpu().numpy()# 7 channels xyz color label
                save_obj(pointcloud[:, :3], c=pointcloud[:, 3:6], file_name=file_path)
                file_path = osp.join(save_dir, f'input_view_pointcloud.npy')
                np.save(file_path, pointcloud)

        # Get GTs
        if cfg.testset not in ['Custom']:
            world_object_points, world_hand_points = world_3d_data['world_object_mesh'], world_3d_data['world_hand_mesh']
            world_scene_points = torch.cat([world_object_points, world_hand_points], dim=1)  # (1, N+778, 3)
            gt_pc_dict = {'total': world_scene_points, 'hand': world_hand_points, 'object': world_object_points}

        tester.gpu_timer.tic()
        # Forward
        with torch.no_grad():
            output = tester.model(input_img, input_view_paramters, world_3d_data, rendering_rays)

        tester.gpu_timer.toc()
        if output is None:
            tester.logger.info(f'Skipping {scene_identifier}, since the result is None. Assigning dummy values for CD and F-scores.')
            total_metrics['cd'].append(dummy_cd); total_metrics['f-5'].append(0); total_metrics['f-10'].append(0)
            hand_metrics['cd'].append(dummy_cd); hand_metrics['f-5'].append(0); hand_metrics['f-10'].append(0)
            object_metrics['cd'].append(dummy_cd); object_metrics['f-5'].append(0); object_metrics['f-10'].append(0)
            continue

        # Go through 'total', 'hand', and 'object' meshes
        for mesh_name, mesh in output.items(): # Trimesh mesh object
            mesh.apply_transform(Rt_4by4[0].numpy())
            # evalute metrics
            if cfg.testset not in ['Custom']:
            
                # exception handling
                if mesh.vertices.size == 0:
                    tester.logger.info(f"No valid mesh for {mesh_name} of {scene_identifier}")

                    # cdscore = [np.NAN]
                    # fscore = [[np.NAN], [np.NAN]]
                    cdscore = [dummy_cd]  # dummy for compensate NaN
                    fscore = [[0], [0]]

                # compute Chamfer distance and F-scores
                else:
                    gt_pc = gt_pc_dict[mesh_name]  # cpu tensor
                    gt_pc = (inv_aug_R @ gt_pc.transpose(1, 2)).transpose(1,2) + inv_aug_t.transpose(1,2)
                   
                    # sample points from mesh
                    pred_pc, _ = trimesh.sample.sample_surface(mesh, gt_pc.shape[1]) # TrackedArray object
                    pred_pc = torch.Tensor(pred_pc.view(np.ndarray))[None, :, :].to(gt_pc)  # cpu tensor, (1, num_points, 3)
                    # compuate evaluation metrics            
                    fscore, cdscore = calc_3d_metrics(chamfer_distance, pred_pc.cuda(), gt_pc.cuda(), num_samples=gt_pc.shape[1], th=th_list)

                # Log metrics; assuming batch_size 1
                if mesh_name == 'total':
                    total_metrics['cd'].append(cdscore[0]); total_metrics['f-5'].append(fscore[0][0]); total_metrics['f-10'].append(fscore[1][0])
                elif mesh_name == 'hand':
                    hand_metrics['cd'].append(cdscore[0]); hand_metrics['f-5'].append(fscore[0][0]); hand_metrics['f-10'].append(fscore[1][0])
                elif mesh_name == 'object':
                    object_metrics['cd'].append(cdscore[0]); object_metrics['f-5'].append(fscore[0][0]); object_metrics['f-10'].append(fscore[1][0])
                else:
                    raise ValueError("We only measure metrics for 'total', 'hand', and 'object'!")
            
            if not cfg.eval_only or cfg.testset in ['Custom']:
                file_path = osp.join(save_dir, f'voxelsize_{cfg.mc_voxel_size[0]}_thr{cfg.mesh_thr}_{mesh_name}.ply')

                mesh.export(file_path)
                tester.logger.info(f"Saved mesh to {file_path}")

    tester.logger.info(f"Average inference time of model: {tester.gpu_timer.average_time}s/itr")

    eval_result = {
        'total_metrics': total_metrics,
        'object_metrics': object_metrics,
        'hand_metrics': hand_metrics
    }
    eval_save_file_path = osp.join(cfg.result_dir, 'evaluation_3d_metrics.json')
    with open(eval_save_file_path, 'w') as f:
        json.dump(eval_result, f)
    tester.logger.info(f"Saved evaluation results into {eval_save_file_path}")
    
    tester.logger.info("\n---Evaluation result---\n")
    tester.logger.info("Evaluation summary for total")
    tester.logger.info(
        f"\nWhole scene's F-5mm: {np.mean(total_metrics['f-5']):.2f} F-10mm: {np.mean(total_metrics['f-10']):.2f} CD: {np.mean(total_metrics['cd']) * 1000:.2f}mm \
          \nObject's F-5mm: {np.mean(object_metrics['f-5']):.2f} F-10mm: {np.mean(object_metrics['f-10']):.2f} CD: {np.mean(object_metrics['cd']) * 1000:.2f}mm \
          \nHand's F-5mm: {np.mean(hand_metrics['f-5']):.2f} F-10mm: {np.mean(hand_metrics['f-10']):.2f} CD: {np.mean(hand_metrics['cd']) * 1000:.2f}mm")

if __name__ == "__main__":
    main()