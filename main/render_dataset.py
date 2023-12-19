import os.path as osp
import json
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
    parser.add_argument('--cfg', type=str, default='',  help='experiment configure file name')
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
    cfg.test_mode = 'render_dataset'
    cfg.render_img_shape = (64, 64)  # heigth, width
    cfg.chunk = 1024

    # import modules after setting gpu
    from constant import OBJ_SEG_IDX, HAND_SEG_IDX, MAX_NUM_CLASSES
    from base import Tester
    from loss import calc_mse, calc_psnr, calc_lpips, calc_ssim, iou_binary
    from utils.vis import make_depth_image
    from utils.utils import make_folder

    tester = Tester(args.test_epoch)
    if args.test_objects != '':
        tester.test_objects = [int(x) for x in args.test_objects.split(',')]
    if args.test_scene != '':
        tester.test_scene = args.test_scene

    tester.make_batch_generator()
    tester.make_model()

    # Quantitative numbers
    total_mse_list, total_psnr_list, total_lpips_list, total_ssim_list = [], [], [], []
    total_semantic_iou_list = []  # to evaluate the hand object spatial relationship

    input_view_img_object_mse_list, noninput_view_img_object_mse_list = [], []
    input_view_img_object_psnr_list, noninput_view_img_object_psnr_list = [], []
    input_view_img_hand_mse_list, noninput_view_img_hand_mse_list = [], []
    input_view_img_hand_psnr_list, noninput_view_img_hand_psnr_list = [], []
    #
    input_view_img_psnr_list, noninput_view_img_psnr_list = [], []
    input_view_img_lpips_list, noninput_view_img_lpips_list = [], []
    input_view_img_ssim_list, noninput_view_img_ssim_list = [], []
    input_view_img_semantic_iou_list, noninput_view_img_semantic_iou_list = [], []
    # Set color for segmentation
    np.random.seed(1)
    lut = np.random.rand(MAX_NUM_CLASSES, 3)
    for itr, (input_img, input_view_paramters,  world_3d_data, rendering_rays, meta_info) in enumerate(tqdm(tester.batch_generator)):
        # get meta info
        input_img_path_list = meta_info['input_img_path_list'][0] # assuming batch size 1
        render_img_path_list = meta_info['render_img_path_list'][0] # assuming batch size 1
        render_img_path = render_img_path_list[0] # always render one image at a time
        
        tester.gpu_timer.tic()
        """ Forward Pass """
        with torch.no_grad():
            output = tester.model(input_img, input_view_paramters, world_3d_data, rendering_rays)
        tester.gpu_timer.toc()

        """ Quantitative Evaluation; Compute MSE, PSNR, LPIPS, SSIM, Semantic segmentation IoU"""
        _rgb_pred = output['rgb_map'][0]  # (H*W - n, 3)
        _rgb_gt = rendering_rays['rgb'][0].to(_rgb_pred) # (H*W - n, 3)
        mask_at_box = rendering_rays['mask_at_box'][0].to(_rgb_pred)  # (H*W,)
        org_mask = rendering_rays['org_mask'][0].to(_rgb_pred)  # (H*W - n, )

        # PSNR, MSE calculation
        object_rgb_pred, hand_rgb_pred = _rgb_pred[org_mask == OBJ_SEG_IDX], _rgb_pred[org_mask == HAND_SEG_IDX]
        object_rgb_gt, hand_rgb_gt = _rgb_gt[org_mask == OBJ_SEG_IDX], _rgb_gt[org_mask == HAND_SEG_IDX]
        with torch.no_grad():
            total_mse, object_mse, hand_mse = calc_mse(_rgb_pred, _rgb_gt), calc_mse(object_rgb_pred, object_rgb_gt), calc_mse(hand_rgb_pred, hand_rgb_gt)
            total_psnr, object_psnr, hand_psnr = calc_psnr(total_mse), calc_psnr(object_mse), calc_psnr(hand_mse)
            # handle NaN
            if object_rgb_pred.sum() == 0:
                object_mse, object_psnr = torch.tensor(1), torch.tensor(0)
            if hand_rgb_pred.sum() == 0:
                hand_mse, hand_psnr = torch.tensor(1), torch.tensor(0)
            # not quite right... just for comparison
            if object_rgb_gt.sum() == 0:
                object_mse, object_psnr = torch.tensor(1), torch.tensor(0)
            if hand_rgb_gt.sum() == 0:
                hand_mse, hand_psnr = torch.tensor(1), torch.tensor(0)
                
        # LPIPS & SSIM calculation; image should be grid RGB, IMPORTANT; normalized to [-1,1]
        H, W = cfg.render_img_shape
        mask_at_box = mask_at_box.reshape(H, W).bool()

        rgb_gt = torch.zeros(1, 3, H, W).to(_rgb_pred)  # black background
        rgb_pred = torch.zeros(1, 3, H, W).to(_rgb_pred)

        rgb_gt[:, :, mask_at_box] = _rgb_gt.transpose(0,1)
        rgb_pred[:, :, mask_at_box] = _rgb_pred.transpose(0, 1)
        # normalize from [0,1] to [-1,1]
        rgb_gt = (rgb_gt - 0.5) * 2
        rgb_pred = (rgb_pred - 0.5) * 2

        with torch.no_grad():
            lpips = calc_lpips(rgb_gt, rgb_pred)
            ssim = calc_ssim(rgb_gt, rgb_pred)

        if render_img_path in input_img_path_list:
            input_view_img_object_mse_list.append(object_mse.item())
            input_view_img_hand_mse_list.append(hand_mse.item())
            input_view_img_object_psnr_list.append(object_psnr.item())
            input_view_img_hand_psnr_list.append(hand_psnr.item())
            #
            input_view_img_lpips_list.append(lpips.item())
            input_view_img_ssim_list.append(ssim.item())
            input_view_img_psnr_list.append(total_psnr.item())
        else:
            noninput_view_img_object_mse_list.append(object_mse.item())
            noninput_view_img_hand_mse_list.append(hand_mse.item())
            noninput_view_img_object_psnr_list.append(object_psnr.item())
            noninput_view_img_hand_psnr_list.append(hand_psnr.item())
            #
            noninput_view_img_lpips_list.append(lpips.item())
            noninput_view_img_ssim_list.append(ssim.item())
            noninput_view_img_psnr_list.append(total_psnr.item())

        # Get the numbers for the table
        total_mse_list.append(total_mse.item()); total_psnr_list.append(total_psnr.item()); total_lpips_list.append(lpips.item()); total_ssim_list.append(ssim.item())

        # Semantic segmentation IoU
        if 'semantic' in cfg.nerf_mode:
            _semantic_pred = output['semantic_map'][0]  # (H*W - n, 3)

            # always batch 1
            semantic_gt = torch.zeros(1, H, W).to(_semantic_pred)
            semantic_pred = torch.zeros(1, 3, H, W).to(_semantic_pred)  # 3 classes; background: 0, object: 1, hand: 2

            semantic_gt[:, mask_at_box] = org_mask
            semantic_gt[semantic_gt == OBJ_SEG_IDX] = 1
            semantic_gt[semantic_gt == HAND_SEG_IDX] = 2
            semantic_gt = semantic_gt[0].detach().cpu().numpy()   # (H,W)

            semantic_pred[:, :, mask_at_box] = _semantic_pred.transpose(0, 1)
            semantic_pred = torch.argmax(semantic_pred.squeeze(), dim=0).detach().cpu().numpy()  # (H,W)

            # compute iou
            num_classes = 3
            semantic_pred_list, semantic_gt_list = [], []
            for idx in range(num_classes):
                pred_binary_map, gt_binary_map = np.zeros_like(semantic_pred), np.zeros_like(semantic_pred)
                pred_binary_map[semantic_pred == idx], gt_binary_map[semantic_gt == idx] = 1, 1
                semantic_pred_list.append(pred_binary_map); semantic_gt_list.append(gt_binary_map)
            iou = iou_binary(semantic_pred_list, semantic_gt_list)
            total_semantic_iou_list.append(iou)
            if render_img_path in input_img_path_list:    
                input_view_img_semantic_iou_list.append(iou)
            else:
                noninput_view_img_semantic_iou_list.append(iou)

        """ Qualitative Evaluation; Save Predicted and Groundtruth images """
        if not cfg.eval_only:
            """ Save RGB """
            # batch size is always 1
            # reshape accordingly to opencv: (H,W,3)
            # permute causes some weired tensor -> numpy error
            rgb_gt = rgb_gt[0].permute(1, 2, 0).detach().cpu().numpy().copy()
            rgb_pred = rgb_pred[0].permute(1,2,0).detach().cpu().numpy().copy()
            # normalize from [-1,1] to [0,1]
            rgb_gt = rgb_gt / 2. + 0.5 
            rgb_pred = rgb_pred / 2. + 0.5

            # cv2.putText(rgb_gt, 'GT', (15, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 255))
            # cv2.putText(rgb_pred, 'PRED', (15, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 255))
            rgb_stack = np.concatenate([rgb_pred, rgb_gt], axis=1)
            rgb_stack = rgb_stack[:, :, ::-1] * 255

            # save
            midfix = 'PREDinput' if cfg.use_pred_hand else 'GTinput'
            scene_identifier = meta_info['scene_identifier'][0]  # assuming batch size 1
            input_camera = meta_info['input_img_path_list'][0][0].split('/')[-2]  # camera identifier, ex) '836212060125'
            save_dir = osp.join(cfg.vis_dir, f'{scene_identifier}_{midfix}_inputview{input_camera}_epoch{args.test_epoch}')
            make_folder(save_dir)

            # assuming we are rendering one train image per iteration
            if cfg.testset in ['DexYCB', 'HanCo']:
                render_img_name = '_'.join(render_img_path.split('/')[-2:])  # cameraName_frameIdx.jpg
            elif cfg.testset == 'HO3D':
                render_img_name = '_'.join(render_img_path.split('/')[-3:]) # cameraName_rgb_frameIdx.jpg
            else:  # CO3Dv1
                render_img_name = osp.basename(render_img_path) 

            file_path = osp.join(save_dir, f'Rendered_{render_img_name[:-4]}_RGB.jpg')
            cv2.imwrite(file_path, rgb_stack)
            print("Saved RGB image to ", file_path)

            """ Save Depth """
            # parse depth map
            _depth_pred = output['depth_map'][0].detach().cpu().numpy()
            _mask_pred = output['mask_map'][0].detach().cpu().numpy()

            depth_pred = np.zeros((H, W, 1))
            mask_pred = np.zeros((H, W, 1))

            depth_pred[mask_at_box.detach().cpu().numpy()] = _depth_pred #+ 1e-5) 
            mask_pred[mask_at_box.detach().cpu().numpy()] = _mask_pred

            depth_pred = make_depth_image(depth_pred[None], mask_pred[None].copy())[0]
            file_path = osp.join(save_dir, f'Rendered_{render_img_name[:-4]}_Depth.jpg')
            depth_pred = depth_pred * 255

            vmin, vmax = depth_pred.max() / 5, depth_pred.max()
            plt.imshow(depth_pred[:,:,0], cmap='jet', interpolation='bilinear', vmin=vmin, vmax=vmax)  # cmap: gray gray_r, jet
            plt.title('depth'); plt.axis('off') # plt.colorbar(); 
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()
            print("Saved Depth image to ", file_path)

            """ Save semantic segmentation """
            if 'semantic' in cfg.nerf_mode: 
                semantic_gt_vis = lut[semantic_gt.astype(np.int32), :]
                semantic_pred_vis = lut[semantic_pred.astype(np.int32), :]
                semantic_stack = np.concatenate([semantic_pred_vis, semantic_gt_vis], axis=1) * 255.
                
                file_path = osp.join(save_dir, f'Rendered_{render_img_name[:-4]}_SEG.jpg')
                cv2.imwrite(file_path, semantic_stack)
                print("Saved semantic segmentation to ", file_path)
            
            else: 
                """ Save mask """
                mask_gt = torch.zeros(1, H, W).to(org_mask)
                mask_gt[:, mask_at_box] = org_mask
                mask_gt[mask_gt != 0] = 1
                mask_gt = mask_gt[0].detach().cpu().numpy()  # (H,W)
                mask_gt_vis = lut[mask_gt.astype(np.int32), :]

                mask_pred = mask_pred[:, :, 0]
                mask_pred[mask_pred != 0] = 1
                mask_pred_vis = lut[mask_pred.astype(np.int32), :]
                mask_stack = np.concatenate([mask_pred_vis, mask_gt_vis], axis=1) * 255.

                file_path = osp.join(save_dir, f'Rendered_{render_img_name[:-4]}_MASK.jpg')
                cv2.imwrite(file_path, mask_stack)
                print("Saved mask to ", file_path)

    tester.logger.info(f"Average inference time of model: {tester.gpu_timer.average_time}s/itr")

    eval_result = {
        'total_mse_list': total_mse_list, 'total_psnr_list': total_psnr_list, 'total_lpips_list': total_lpips_list, 'total_ssim_list': total_ssim_list,
        'total_semantic_iou_list': total_semantic_iou_list,
        'input_view_img_lpips_list': input_view_img_lpips_list, 'input_view_img_ssim_list': input_view_img_ssim_list, 'input_view_img_object_mse_list': input_view_img_object_mse_list, 'input_view_img_object_psnr_list': input_view_img_object_psnr_list,
        'input_view_img_hand_mse_list': input_view_img_hand_mse_list, 'input_view_img_hand_psnr_list': input_view_img_hand_psnr_list, 
        'input_view_img_semantic_iou_list': input_view_img_semantic_iou_list, 
        'noninput_view_img_lpips_list': noninput_view_img_lpips_list, 'noninput_view_img_ssim_list': noninput_view_img_ssim_list, 'noninput_view_img_object_mse_list': noninput_view_img_object_mse_list, 'noninput_view_img_object_psnr_list': noninput_view_img_object_psnr_list,
        'noninput_view_img_hand_mse_list': noninput_view_img_hand_mse_list, 'noninput_view_img_hand_psnr_list': noninput_view_img_hand_psnr_list,
        'noninput_view_img_semantic_iou_list': noninput_view_img_semantic_iou_list, 
    }

    eval_save_file_path = osp.join(cfg.result_dir, 'evaluation_2d_metrics.json')
    with open(eval_save_file_path, 'w') as f:
        json.dump(eval_result, f)
    tester.logger.info(f"Saved evaluation results into {eval_save_file_path}")
    

    print("---Evaluation result---\n")
    print("Evaluation summary for total")
    print(f"Whole's [PSNR]: {np.mean(total_psnr_list):.2f} [IoU]: {np.mean(total_semantic_iou_list):.2f}  [SSIM]: {np.mean(total_ssim_list):.2f} [LPIPS]: {np.mean(total_lpips_list):.2f}")

    print()
    print("Input view' metrics")
    print(f"Whole's [PSNR]: {np.mean(input_view_img_psnr_list):.2f} [IoU]: {np.mean(input_view_img_semantic_iou_list):.2f} [SSIM]: {np.mean(input_view_img_ssim_list):.2f} [LPIPS]: {np.mean(input_view_img_lpips_list):.2f} ")
    print(f"Object's [MSE]: {np.mean(input_view_img_object_mse_list):.2f} [PSNR]: {np.mean(input_view_img_object_psnr_list):.2f}")
    print(f"Hand's [MSE]: {np.mean(input_view_img_hand_mse_list):.2f} [PSNR]: {np.mean(input_view_img_hand_psnr_list):.2f}")
    print()
    print("None-input view' metrics")
    print(f"Whole's [PSNR]: {np.mean(noninput_view_img_psnr_list):.2f}  [IoU]: {np.mean(noninput_view_img_semantic_iou_list):.2f} [SSIM]: {np.mean(noninput_view_img_ssim_list):.2f} [LPIPS]: {np.mean(noninput_view_img_lpips_list):.2f} ")
    print(f"Object's [MSE]: {np.mean(noninput_view_img_object_mse_list):.2f} [PSNR]: {np.mean(noninput_view_img_object_psnr_list):.2f}")
    print(f"Hand's [MSE]: {np.mean(noninput_view_img_hand_mse_list):.2f} [PSNR]: {np.mean(noninput_view_img_hand_psnr_list):.2f}")

if __name__ == "__main__":
    main()
