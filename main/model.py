import torch
import torch.nn as nn
# from thop import profile, clever_format
import torchvision.transforms as T

from config import cfg
from nets.feature_encoder import FeatureEncoder
from nets.handpose import HandPoseEncoder
from nets.handobject import HandObjectCorrelationEncoder
from nets.handobject_transformer import HandObjectCorrelationEncoderTransformer
from nets.handobject_transformerCNN import HandObjectCorrelationEncoderTransformerCNN
from nets.object_segmenter import ObjectSegmenter


from nets.original_nerf import OriginalNeRF
from nets.mononhr_nerf import MonoNHRNeRF
from utils.rendering import get_pixel_value, get_alpha_value
from utils.geometry3d_utils import torch_get_object_points_and_voxels
from loss import calc_bce, calc_mse, calc_semantic_ce
from constant import OBJ_SEG_IDX




class Model(nn.Module):
    def __init__(self, imgfeature_encoder, object_segmenter, handpose_encoder, handobject_encoder, nerf, nerf_fine, mode):
        super(Model, self).__init__()

        self.mode = mode  # 'train', 'render_rotate', 'render_dataset', 'recon3d'

        # feature encoders
        self.feature_encoder = imgfeature_encoder
        self.handpose_encoder = handpose_encoder
        self.handmesh_encoder = handobject_encoder  # explicitly encodes the correlation of hand and object with hand mesh and object segmentation

        # object segmentation
        self.resize = T.Resize(size=(1*cfg.input_img_shape[1], 1*cfg.input_img_shape[0]), interpolation=T.InterpolationMode.NEAREST)
        self.object_segmenter = object_segmenter
        self.object_seg_thr = 0.5

        # nerf 
        self.nerf = nerf
        self.nerf_fine = nerf_fine
        
        # loss weights
        self.bce_loss_weight = 0.1 
        self.ce_loss_weight = 0.4 
        self.sce_loss_weight = torch.Tensor([1,2,1]) # (3,), background, object, hand 
        self.obj_seg_bce_loss_weight = 3 # 1

    def forward(self, input_img, input_view_parameters, world_3d_data, rendering_rays, meta_info=None, train_iter=-1):
        """
        input_img: (1, cfg.num_nput_views, cfg.input_img_shape[0], cfg.input_img_shape[1], 3)

        """
        # calculate computation cost
        computation_cost = {}

        """ Encoding image features """
        # feature_map: (1, cfg.img_feat_dim, cfg.input_img_shape[0], cfg.input_img_shape[1])
        feature_map = self.feature_encoder(input_img) if self.feature_encoder is not None else None
        # calculate computation cost
        # macs, params = profile(self.feature_encoder, inputs=(input_img,))
        # computation_cost['feature_encoder'] = {'macs': macs, 'params': params}

        loss = {}
        # Estimate object segmentation and deproject the object mask!
        if self.object_segmenter is not None: # ours
            # Estimate object segmentation
            # pred_obj_seg (1, 1, 1 x cfg.input_img_shape[0], 1 x cfg.input_img_shape[1])
            pred_obj_seg = self.object_segmenter(feature_map)

            if self.mode == 'train': 
                input_view_mask = self.resize(input_view_parameters['input_view_mask'])
                gt_obj_seg = input_view_mask.clone().float()
                gt_obj_seg [gt_obj_seg  != OBJ_SEG_IDX] = 0
                gt_obj_seg [gt_obj_seg  == OBJ_SEG_IDX] = 1
                
                if cfg.do_object_segmentation:
                    obj_seg_bce = calc_bce(pred_obj_seg, gt_obj_seg)
                    loss['obj_seg_bce'] = self.obj_seg_bce_loss_weight * obj_seg_bce
                    if cfg.pretrain_object_segmentation:
                        return loss

            else: # deproject the object mask
                # make prediction to binary mask
                obj_seg = pred_obj_seg.detach().clone()
                obj_seg[obj_seg > self.object_seg_thr] = 1
                obj_seg[obj_seg != 1] = 0
                world_inside_object_points, world_object_points_voxel_coord = \
                torch_get_object_points_and_voxels(world_3d_data['world_bounds'], cfg.input_mesh_voxel_size, input_view_parameters, obj_seg, self.mode, downsample_ratio=1)
                if world_inside_object_points is None:
                    return None            
                world_3d_data['world_object_points'] = world_inside_object_points
                world_3d_data['world_object_points_voxel_coord'] = world_object_points_voxel_coord

        """ Forwrad to implicit function """
        if self.mode == 'recon3d':
            """ occupancy calcuation for world point """
            nerf = self.nerf_fine if 'fine' in cfg.nerf_mode else self.nerf
            ret = get_alpha_value(nerf, self.handpose_encoder, self.handmesh_encoder, world_3d_data, input_view_parameters, feature_map) 

            return ret 

        else:
            if self.mode == 'train':
                """ Filter out invalid rays before putting into the network during trianing """
                mask_at_box = rendering_rays['mask_at_box']  # (1, 1024);
                if mask_at_box.sum() == 0:  # batch with all invalid rays
                    return None
                else:
                    # k: 'rgb', 'fg_mask', 'ray_o', 'ray_d', 'near', 'far', 'mask_at_box'
                    # v: (1, 1024, 3), (1, 1024), (1, 1024, 3), (1, 1024, 3), (1, 1024), (1, 1024), (1, 1024)
                    for k, v in rendering_rays.items():
                        if k == 'mask_at_box':
                            continue
                        new_v = []
                        for batch_idx in range(v.shape[0]):
                            slice = v[batch_idx][mask_at_box[batch_idx]]
                            new_v.append(slice)
                        new_v = torch.stack(new_v) # (1, 1024 - ?,  )

                        rendering_rays[k] = new_v
            

            """ volume rendering for each pixel """
            n_batch, n_pixel = rendering_rays['ray_o'].shape[:2]
            chunk = cfg.chunk
            ret_list = []
            for i in range(0, n_pixel, chunk):
                start_idx = i
                end_idx = min(i + chunk, n_pixel)
                rendering_rays_chunk = {
                    'ray_o': rendering_rays['ray_o'][:, start_idx:end_idx],
                    'ray_d' : rendering_rays['ray_d'][:, start_idx:end_idx],
                    'near' : rendering_rays['near'][:, start_idx:end_idx],
                    'far' : rendering_rays['far'][:, start_idx:end_idx]
                }
                pixel_value = get_pixel_value(self.nerf, self.nerf_fine, self.handpose_encoder, self.handmesh_encoder,
                    rendering_rays_chunk, input_view_parameters, world_3d_data,
                    feature_map, self.mode, train_iter, computation_cost)
                ret_list.append(pixel_value)

            keys = ret_list[0].keys()
            ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in keys}

            # calculate computation cost
            # for module in computation_cost.keys():
            #     macs, params = computation_cost[module]['macs'], computation_cost[module]['params']
            #     macs, params = clever_format([macs, params], "%.3f")                
            #     print(f"{module.upper()} MACS: {macs} PARAMS: {params}")
            
            if self.mode == 'train':
                """ rendering loss """
                img_mse = calc_mse(ret['rgb_map'], rendering_rays['rgb'], mask=None)  
                mask_bce = calc_bce(ret['mask_map'], rendering_rays['fg_mask'][:, :, None]) # rendering_rays['fg_mask'][:, :, None]  # (1, 1024, 1)
            
                loss['img_mse'] = img_mse
                loss['mask_bce'] = self.bce_loss_weight * mask_bce
                
                if 'semantic' in cfg.nerf_mode:
                    semantic_ce = calc_semantic_ce(ret['semantic_map'], rendering_rays['org_mask'].long(), self.sce_loss_weight.to(ret['semantic_map']))
                    loss.update({'semantic_ce': self.ce_loss_weight * semantic_ce,})

                
                # Importance sampling; coarse & fine NeRFs
                if cfg.N_importance > 0 and self.nerf_fine != None:
                    # Compute loss on coarse outputs too! Don't confuse:)
                    img_mse = calc_mse(ret['rgb_map_coarse'], rendering_rays['rgb'], mask=None)  # rendering_rays['fg_mask'][:, :, None]   # (1, 1024, 1)
                    mask_bce = calc_bce(ret['mask_map_coarse'], rendering_rays['fg_mask'][:, :, None])

                    loss_coarse = {
                        'img_mse_coarse': img_mse,
                        'mask_bce_coarse': self.bce_loss_weight * mask_bce,
                    }

                    if 'semantic' in cfg.nerf_mode:
                        semantic_ce = calc_semantic_ce(ret['semantic_map_coarse'], rendering_rays['org_mask'].long(), self.sce_loss_weight.to(ret['semantic_map']))
                        loss_coarse.update({'semantic_ce_coarse': self.ce_loss_weight * semantic_ce})

                    loss.update(loss_coarse)

                """ object segmentaiton loss """
                if self.object_segmenter is not None and 'obj_seg_bce' not in loss:
                    obj_seg_bce = calc_bce(pred_obj_seg, gt_obj_seg)
                    loss['obj_seg_bce'] = self.obj_seg_bce_loss_weight * obj_seg_bce

                return loss
            
            elif self.mode == 'render_dataset' or self.mode == 'render_rotate':
                
                return ret


def get_model(mode='train'):
    
    if 'mononhr' not in cfg.nerf_mode:
       nerf = OriginalNeRF()
       nerf_fine = OriginalNeRF() if 'fine' in cfg.nerf_mode else None
    else:
        nerf = MonoNHRNeRF()
        nerf_fine = MonoNHRNeRF() if 'fine' in cfg.nerf_mode else None
        
    imgfeature_encoder = None #FeatureEncoder()
    handpose_encoder = None # HandPoseEncoder()
    handobject_encoder = None # HandMeshEncoder()
    object_segmenter = None 
    
    
    if 'pixel' in cfg.nerf_mode or ('handmesh' in cfg.nerf_mode and 'latenthandmesh' not in cfg.nerf_mode):
        imgfeature_encoder = FeatureEncoder()
        
    if 'handjoint' in cfg.nerf_mode:  
        handpose_encoder = HandPoseEncoder()  # IHOI hand pose encoder; https://judyye.github.io/ihoi/
        
    if 'handmesh' in cfg.nerf_mode: # for MonoNHR and HandNeRF
        handobject_encoder = HandObjectCorrelationEncoder() # MonoNHR only uses the hand mesh.
        if 'transformer' in cfg.nerf_mode:
            handobject_encoder = HandObjectCorrelationEncoderTransformer()
        # HandObjectCorrelationEncoderTransformerCNN()
        if '_handmeshpixelnerf_' in cfg.nerf_mode and cfg.do_object_segmentation:
            object_segmenter = ObjectSegmenter()

    model = Model(imgfeature_encoder, object_segmenter, handpose_encoder, handobject_encoder, nerf, nerf_fine, mode)
    return model