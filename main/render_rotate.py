import os.path as osp
import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import cfg


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
    cfg.test_mode = 'render_rotate'

    # import modules after setting gpu
    from base import Tester
    from constant import MAX_NUM_CLASSES
    from utils.vis import make_depth_image
    from utils.utils import make_folder

    tester = Tester(args.test_epoch)
    if args.test_objects != '':
        tester.test_objects = [int(x) for x in args.test_objects.split(',')]
    if args.test_scene != '':
        tester.test_scene = args.test_scene
    tester.make_batch_generator()
    tester.make_model()

    # Set color for segmentation
    np.random.seed(1)
    lut = np.random.rand(MAX_NUM_CLASSES, 3)
    depth_pred_list, mask_pred_list = [], []

    for itr, (input_img, input_view_paramters, world_3d_data, rendering_rays, meta_info) in tqdm(enumerate(tester.batch_generator)):

        # forward
        with torch.no_grad():
            output = tester.model(input_img, input_view_paramters, world_3d_data, rendering_rays)

        # batch size is always 1
        _rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
        mask_at_box = rendering_rays['mask_at_box'][0].detach().cpu().numpy()
        _depth_pred = output['depth_map'][0].detach().cpu().numpy()
        _mask_pred = output['mask_map'][0].detach().cpu().numpy()

        # parse rgb map
        H, W = cfg.render_img_shape
        mask_at_box = mask_at_box.reshape(H, W)

        rgb_pred = np.zeros((H, W, 3)) 
        rgb_pred[mask_at_box] = _rgb_pred

        # parse depth map
        depth_pred = np.zeros((H, W, 1))
        mask_pred = np.zeros((H, W, 1))
        depth_pred[mask_at_box] = _depth_pred #+ 1e-5) 
        mask_pred[mask_at_box] = _mask_pred
        depth_pred_list.append(depth_pred)
        mask_pred_list.append(mask_pred)

        # save prefixes
        scene_identifier = meta_info['scene_identifier'][0]  # assuming batch size 1
        save_dir = osp.join(cfg.vis_dir, f'{scene_identifier}_rotating_epoch{args.test_epoch}')
        make_folder(save_dir)
        rotating_render_idx = meta_info['rotating_render_idx']  # scalar

        """ save RGB """
        rgb_pred = rgb_pred[:, :, ::-1] * 255
        file_path = osp.join(save_dir, f'Rotating_RGB_{int(rotating_render_idx):06d}.jpg')
        cv2.imwrite(file_path, rgb_pred)
        print("Saved RGB to ", file_path)

        """ save Segmentation """
        if 'semantic' in cfg.nerf_mode:
            _object_rgb_pred = output['object_rgb_map'][0].detach().cpu().numpy()
            object_rgb_pred = np.zeros((H, W, 3)) 
            object_rgb_pred[mask_at_box] = _object_rgb_pred

            _semantic_pred = output['semantic_map'][0].detach().cpu().numpy()  # (H*W - n, 3)
            semantic_pred = np.zeros((H, W, 3))
            semantic_pred[mask_at_box] = _semantic_pred
            semantic_pred = np.argmax(semantic_pred, axis=2) # (H,W)

            object_rgb_pred = object_rgb_pred[:, :, ::-1] * 255
            semantic_pred_vis = lut[semantic_pred.astype(np.int32), :] * 255.
            file_path = osp.join(save_dir, f'Rotating_RGB_Object_{int(rotating_render_idx):06d}.jpg')
            cv2.imwrite(file_path, object_rgb_pred)
            file_path = osp.join(save_dir, f'Rotating_SEG_{int(rotating_render_idx):06d}.jpg')
            cv2.imwrite(file_path, semantic_pred_vis)

    # save depth video
    depth_pred_list, mask_pred_list = np.stack(depth_pred_list), np.stack(mask_pred_list) 
    depth_pred_list = make_depth_image(depth_pred_list, mask_pred_list, max_quantile = 0.98, min_quantile = 0.02)
    # depth_pred_list: (N,H,W,1)

    depth_pred_list = depth_pred_list * 255
    vmin, vmax = depth_pred_list.max() / 5, depth_pred_list.max()

    for idx in range(len(depth_pred_list)):
        file_path = osp.join(save_dir, f'Rotating_Depth_{int(idx):06d}.jpg')
        # assume that distance of min, max is not that big, for better visualization
        # plt.imshow(depth_pred_list[idx,:,:,0], cmap='jet', interpolation='bilinear', vmin=vmin, vmax=vmax)  # cmap: gray gray_r, jet
        plt.axis('off') # plt.colorbar(); 
        plt.imsave(fname=file_path, arr=depth_pred_list[idx,:,:,0], cmap='jet',  vmin=vmin, vmax=vmax, format='png')
    plt.close()



    

if __name__ == "__main__":
    main()