import os.path as osp
import argparse
import shutil
import numpy as np
import torch
torch.manual_seed(42)
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from config import cfg
from utils.utils import increase_object_ray_ratio_grad


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    parser.add_argument('--exp_dir', type=str, default='', help='training experiment to continue')
    parser.add_argument('--cfg', type=str, default='', help='experiment configure file name')
    parser.add_argument('--nerf_mode', type=str, default='semantic_handmeshpixelnerf_fine',  help='model to train/test')

    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"
 
    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def main():
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.gpu_ids, continue_train=args.continue_train, exp_dir=args.exp_dir)
    cudnn.benchmark = True
    if args.cfg:
        cfg.update(args.cfg)
    shutil.copyfile('config.py', cfg.log_dir + '/config.py')

    # set model
    cfg.nerf_mode = args.nerf_mode

    # import modules after setting gpu
    from base import Trainer
    from logger import AverageMeter
    from loss import calc_psnr

    trainer = Trainer()
    trainer.make_batch_generator()
    trainer.make_model()

    # tensorboard writer
    writer = SummaryWriter(log_dir=cfg.log_dir)
    global_iter = 0
    
    # train
    trainer.logger.info(f'[Training] GPU: {args.gpu_ids}, Experiment home: {osp.basename(cfg.output_dir)} NeRF-mode: {cfg.nerf_mode}')
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        np.random.seed() 

        trainer.logger.info(f'Epoch {epoch} itr 0; Object sample ratio: {cfg.object_sample_ratio}')
        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        for itr, (input_img, input_view_paramters, world_3d_data, rendering_rays, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()

            # Control the sampling ratio of object in the input image
            if cfg.final_object_ratio > 0:
                # caution; this function directly (in-place) replaces cfg.object_sample_ratio
                cfg_changed = increase_object_ray_ratio_grad(global_iter, steps=cfg.object_sample_ratio_increase_steps)
                if cfg_changed:
                    trainer.logger.info(f'Epoch {epoch} itr {itr}; Object sample ratio: {cfg.object_sample_ratio}')

            """ Forward and Backward passes """
            # Forward
            trainer.gpu_timer.tic()
            trainer.optimizer.zero_grad()
            loss = trainer.model(input_img, input_view_paramters, world_3d_data, rendering_rays, meta_info, global_iter)
            # Skip iteration with invalid training data
            if loss is None:
                trainer.logger.info(f'Skip invalid training data; epoch/iteration: {epoch}/{itr}')
                continue
            loss = {k: loss[k].mean() for k in loss}
            _loss = sum(loss[k] for k in loss)

            # Weaken the loss when input==render
            if cfg.num_render_views == 1 and cfg.num_input_views == 1 and meta_info['input_img_path_list'][0][0] == meta_info['render_img_path_list'][0][0]:
                _loss = cfg.input_view_loss_weight * _loss

            # Backward
            _loss.backward()
            trainer.optimizer.step()
            trainer.gpu_timer.toc()

            # calculate PSNR
            if 'img_mse' in loss:
                psnr = calc_psnr(loss['img_mse'].detach())
                loss['psnr'] = psnr

            """ Log metrics """
            for key in loss:
                new_key = f'loss_{key}' if key != 'psnr' else key
                # log on tensorboard and AverageMeter
                writer.add_scalar(f'Train/{new_key}', loss[key], global_iter)
                # Follow Pytorch3d; save average metrics
                if new_key not in trainer.stats:
                    trainer.stats[new_key] = AverageMeter()
                trainer.stats[new_key].update(loss[key].item(), epoch=epoch, n=1)

            global_iter += 1
            
            # Print
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                'lr: %g' % (trainer.get_lr()),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                ]
            # Follow Pytorch3d; print average metrics
            screen += ['%s: %.4f' % (k, v.avg) for k,v in trainer.stats.items()]
            trainer.logger.info(' '.join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()
        
        if epoch % cfg.save_epoch_period == 0:
            trainer.save_model({
                'epoch': epoch,
                'network': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
            }, epoch)

    trainer.save_model({
        'epoch': epoch,
        'network': trainer.model.state_dict(),
        'optimizer': trainer.optimizer.state_dict(),
    }, epoch)
    writer.flush()
    writer.close()

if __name__ == "__main__":
    main()