import os
import os.path as osp
import sys
import datetime
import glob
import shutil
import yaml
from easydict import EasyDict as edict

class Config:
    
    debug = False

    """ NeRF mode options """
    # ('semantic_' +) # Estimate the semantic label of each query point

    # SOTA
    # 'pixelnerf' # PixelNeRF (https://alexyu.net/pixelnerf/) or M2
    # 'handjointpixelnerf' # IHOINeRF; IHOI (https://github.com/JudyYe/ihoi) adapted to multi-view image supervision. it's not compatible with semantic_ ... 
    # 'mononhrhandmeshpixelnerf' # MonoNHR (https://arxiv.org/abs/2210.00627) without mesh inpainter
    # 'handmeshpixelnerf' # HandNeRF 
    
    # Ablation
    # M1: 'handmeshtransformerpixelnerf'  # Instead of 3DCNN for correlation, use transformer
    # M2: 'pixelnerf'
    # M3: 'handmeshnnerf'  #  HandNeRF without pixel-aligned image feature
    # M4: 'noobjectpixelnerf' # Similar to MonoNHR, no explicit use of Object features
    # M5: 'handmeshpixelnerf' # Ours
    # 'latenthandmeshnerf' # HandNeRF without image features; not in paper, but interesting

    # (+ '_fine') # Do the hierachical sampling of NeRF
    nerf_mode = 'semantic_handmeshpixelnerf_fine'

    """ Data """
    trainset = ['DexYCB']
    testset = 'DexYCB' # 'HO3D'
    test_config = 'novel_object'  # 'novel_grasp'
    trainset_make_same_len = False  # if true, multiple datasets feed the same number of samples per one training epoch
    target_objects = [1, 2, 10, 15]  

    """ Testing options """
    
    is_test = False # Will be updated later if you are not training.
    eval_only = False  # if true, just compute metrics. no saving images or meshes 
    deterministic_eval = False  # if true, becomes slower. related to sparse 3D conv. negligible
    use_multiview_masks_for_recon = True # if false, use deprojection of only the input view mask. becomes very slow due to many points. for the quantitative evaluation purpose, set this True
    use_pred_hand = False # use hand meshes from HandOccNet
    test_mode = 'render_rotate'  # 'recon3d': generate hand and object meshes with Marching cube  # 'render_*': synthesize novel view images of RGB, depth, and semantic segmentation
    num_rotating_views = 40  # when 'render_rotate', number of 360 degree views
    mesh_thr = 10  # marching cube level-set threshold; high -> denser reconstruction. low -> sparser reconstruction. refer to https://github.com/pmneila/PyMCubes
    mc_voxel_size = [0.002, 0.002, 0.002]
    # dhw marching cube resolution; [0.1, 0.1, 0.1]  # [0.005, 0.005, 0.005]  #[0.008, 0.008, 0.008] # #[0.002, 0.002, 0.002]
    # high resolution mesh = small voxel size -> smoother surface, more memory consumption
   
    """ Training options """
    num_input_views = 1  # fixed. increasing this is left for future research
    num_render_views = 1 # fixed. most efficient and increasing it has no difference. following Pytorch NeRF implementation
    use_all_input_views = False  #  make this true if you want achieve generalization over input view
    input_view_loss_weight = 0.8  # lessen the input view image's supervision to prevent overfitting
    # NeRF ray sampling
    object_sample_ratio_increase_steps = 5000 # to prevent overfitting to hand rendering, secure enough object ray rendering
    final_object_ratio = 0.5 # -1 # to prevent overfitting to hand rendering, secure enough object ray rendering 
    object_sample_ratio = 0.0 # 0.5
    hand_sample_ratio = 0.0 # 0.5
    boundary_sample_ratio = 0.  # don't use this. TO BE REMOVED
    # Data loader 
    num_gpus = 1
    train_batch_size = 1  # num_gpus and train_batch_size should be the same to make DataLoader to feed 1 sample to the network
    test_batch_size = 1
    num_thread = 0
    # Training schedule 
    lr_feature_encoder = 3e-4
    lr = 3e-4
    lr_dec_epoch = [75]
    lr_dec_factor = 10
    end_epoch = 101
    save_epoch_period = 5
    continue_train = False

    """ Model architecture """
    do_object_segmentation = False  # integrate object segmentation into the entire model
    pretrain_object_segmentation = False and do_object_segmentation
    resnet_type = 18 # 34, 50
    img_feat_dim = 64 
    handmesh_feat_dim_from_spconv = 64  
    handmesh_feat_use_as_z = False
    handmesh_feat_dim = 32
    handpose_feat_dim = 32
    # how to aggregate voxel feature and surface feature; if '', no surface feature. 'learn_weighted_sum', 'learn_weighted_sum_multi_voxel_feat', 'average', 'add', 'concat'
    handmesh_feat_aggregation = '' # removed. not significant



    """ Pre-processing and feature processing """
    # Data image augmentation and feature extraction
    input_img_shape = (256, 256)  # heigth, width
    input_mesh_voxel_size = [0.005, 0.005, 0.005]  # dhw; decide the resolution of the 3D feature volume 
    input_bbox_expand_ratio = 1.3  # crop and resize bbox ration
    render_bbox_expand_ratio = 1.3  # crop and resize bbox ration
    feature_volume_padding_mode = 'zeros'
    feature_map_padding_mode = 'zeros'
   
    # NeRF related rendering options
    N_samples = 64  # number of points per rendering ray (pixel)
    N_importance = 128
    N_rand = 1024
    N_object_rand = 1024 # sample object rays for building a volume feature
    render_img_shape = (256, 256) # heigth, width
    chunk = 2048 # ray chunk number
    raw_noise_std = 0.# 0.001
    perturb = 1
    
    # CLIP feature from DietNeRF. removed. not helpful
    clip_loss = False
    clip_loss_interval = 50
    clip_loss_weight = 0.1
    clip_loss_thr = -0.75  # if clip loss is lower than this (if the influence is significant), weaken the sueprvision by * 0.1
    clip_render_img_shape = (32, 32)  # heigth, width, different from 224. this is the outut shape of my method

    """ Directory """
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    save_folder = 'exp_' + str(datetime.datetime.now())[5:-10]
    save_folder = save_folder.replace(" ", "_")
    output_dir = osp.join(output_dir, save_folder)
    model_dir = osp.join(output_dir, 'checkpoint')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    
    def set_args(self, gpu_ids, is_test=False, continue_train=False, exp_dir=''):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
    
        if not is_test:
            self.continue_train = continue_train
            if self.continue_train and exp_dir:
                checkpoints = sorted(glob.glob(osp.join(exp_dir, 'checkpoint') + '/*.pth.tar'), key=lambda x: int(x.split('_')[-1][:-8]))
                shutil.copy(checkpoints[-1], osp.join(cfg.model_dir, checkpoints[-1].split('/')[-1]))

        elif is_test:
            self.is_test = True
            if exp_dir == '':
                raise ValueError('[Config] Specifiy the experiment directory you want to test!')
            else:
                self.output_dir = exp_dir
                print('[Config] Change output dir to ', self.output_dir)

                self.model_dir = osp.join(self.output_dir, 'checkpoint')
                self.vis_dir = osp.join(self.output_dir, 'vis')
                self.log_dir = osp.join(self.output_dir, 'log')
            self.result_dir = osp.join(self.output_dir, 'result')

        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('[Config] Using GPU: {}'.format(self.gpu_ids))

    def update(self, config_file):
        with open(config_file) as f:
            exp_config = edict(yaml.safe_load(f))
            for k, v in exp_config.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
                else:
                    raise ValueError("{} not exist in config.py".format(k))

cfg = Config()
print('[Config] output dir: ', cfg.output_dir)

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
sys.path.insert(0, osp.join(cfg.root_dir, 'main'))
sys.path.insert(0, osp.join(cfg.root_dir, 'data'))


manopth_dir = osp.abspath('../tool/manopth')
sys.path.insert(0, manopth_dir)
# Load MANO layer.
from manopth.manolayer import ManoLayer
mano_right_layer = ManoLayer(flat_hand_mean=False,
                             ncomps=45,
                             side='right',
                             mano_root=osp.join(manopth_dir, 'mano/models'),
                             use_pca=True)
mano_left_layer = ManoLayer(flat_hand_mean=False,
                            ncomps=45,
                            side='left',
                            mano_root=osp.join(manopth_dir, 'mano/models'),
                            use_pca=True)
mano_layer = {'right': mano_right_layer, 'left': mano_left_layer}


from utils.utils import make_folder

make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)

