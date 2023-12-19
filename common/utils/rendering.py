import numpy as np
import torch
import torch.nn.functional as F

from config import cfg
from constant import OBJ_SEMANTIC_IDX, HAND_SEMANTIC_IDX
from utils.rendering_utils import batchify_rays, get_sampling_points, raw2outputs, sample_pdf, get_2d_features, get_3d_features, get_mesh_from_marching_cube, remove_dust_from_voxel_cube

# For 3D reconstruction using marching cube
def get_alpha_value(nerf, handpose_encoder, handmesh_encoder, world_3d_data, input_view_parameters, feature_map):
    """
    Returns Trimesh meshes of total scene, hand, and objects
    """

    # code from neuralbody; https://github.com/zju3dv/neuralbody/blob/master/lib/networks/renderer/if_mesh_renderer.py
    inside = world_3d_data['inside'][0].bool()  # (33, 40, 26)
    world_points = world_3d_data['pts']  # (1, 33, 40, 26, 3)
    sh = world_points.shape
    world_points = world_points[0][inside][None]  # (1, 12020, 3)    
    
    """ Encode hand pose information """
    # encode hand pose?
    handpose_feature = handpose_encoder(world_points[:, :, None, :], world_3d_data) if handpose_encoder is not None else None
    # encode hand mesh?
    handmesh_feature_volume_list = handmesh_encoder(world_3d_data, input_view_parameters, feature_map, 'test') if handmesh_encoder is not None else None

    render_viewdir = None
    alpha_decoder = lambda x_point, x_feature: calculate_density_color(x_point, render_viewdir, nerf, input_view_parameters, world_3d_data, feature_map, x_feature, handmesh_feature_volume_list, {} )

    """ Forward and get the semantic and alpha """
    raw = batchify_rays(world_points, handpose_feature, alpha_decoder, cfg.chunk * cfg.N_samples)
    if raw is None:
        return None
        
    """ Make meshes of total, hand, and, object """
    inside = inside.detach().cpu().numpy()   # (33, 40, 26)
    world_min_xyz = world_3d_data['world_bounds'][0, 0].detach().cpu().numpy()  # min_xyz
    if 'semantic' in cfg.nerf_mode:
        # raw: (1, 12020, 3+1)
        # separate alpha and semantic
        raw_semantic, alpha = raw[:, :, :3], raw[:, :, 3:]
        semantic_class = torch.argmax(raw_semantic, dim=-1)[0] # (12020,)

        # alpha: (1, 12020, 1)
        alpha = alpha[0, :, 0].detach().cpu().numpy()
        semantic_class = semantic_class.detach().cpu().numpy() 

        total_cube = np.zeros(sh[1:-1])
        hand_cube = np.zeros(sh[1:-1])
        object_cube = np.zeros(sh[1:-1])

        hand_alpha = alpha.copy()
        object_alpha = alpha.copy()
        hand_alpha[semantic_class != HAND_SEMANTIC_IDX] = 0.
        
        # object_alpha[semantic_class == HAND_SEMANTIC_IDX] = 0.
        object_alpha[semantic_class != OBJ_SEMANTIC_IDX] = 0.
        
        total_cube[inside == 1] = alpha
        hand_cube[inside == 1] = hand_alpha
        object_cube[inside == 1] = object_alpha

        # Dust removal; eliminate the influence of noise alpha and mis segmentation
        # Becarefull for total scene, when hand and object are aparted in image
        total_cube = remove_dust_from_voxel_cube(total_cube, thr_ratio=0.1)
        hand_cube = remove_dust_from_voxel_cube(hand_cube, thr_ratio=0.1)
        object_cube = remove_dust_from_voxel_cube(object_cube, thr_ratio=0.1)

        total_mesh = get_mesh_from_marching_cube(total_cube, world_min_xyz)
        hand_mesh = get_mesh_from_marching_cube(hand_cube, world_min_xyz)
        object_mesh = get_mesh_from_marching_cube(object_cube, world_min_xyz)

        ret_meshes = {
            'total': total_mesh,
            'hand': hand_mesh,
            'object': object_mesh
        }
    else:
        alpha = raw
        alpha = alpha[0, :, 0].detach().cpu().numpy()

        total_cube = np.zeros(sh[1:-1])
        total_cube[inside == 1] = alpha
        total_cube = remove_dust_from_voxel_cube(total_cube, thr_ratio=0.1)
        total_mesh = get_mesh_from_marching_cube(total_cube, world_min_xyz)

        ret_meshes = {
            'total': total_mesh,
            # 'object': total_mesh

        }

    return ret_meshes


# For novel view synthesis
def get_pixel_value(nerf, nerf_fine, handpose_encoder, handmesh_encoder, rendering_rays_chunk, input_view_parameters, world_3d_data, feature_map, mode, train_iter=-1, computation_cost={}):
    """
    nerf, nerf_fine: implicit functions
    handpose_encoder, handmesh_encoder: encode hand information 
    rendering_rays_chunk: ('ray_o', 'ray_d', 'near', 'far'); (1, num_rays, 3), (1, num_rays, 3), (1, num_rays), (1, num_rays)
    input_view_parameters:
    world_3d_data: ('world_bounds', ); (1, 2, 3), s
    feature_volume_list:
    feature_map:
    mode: one of ('train', 'render_rotate', 'render_dataset', 'recon3d')
    
    """
    # sampling points along camera rays 
    world_points, world_z_vals = get_sampling_points(rendering_rays_chunk['ray_o'], rendering_rays_chunk['ray_d'], rendering_rays_chunk['near'], rendering_rays_chunk['far'], mode)
    # world_points: (1, num_rays, cfg.N_samples, 3)
    # world_z_vals: (1, num_rays, cfg.N_samples)
    batch_size, num_rays, num_points_per_ray = world_points.shape[:3]  # num_points_per_ray == cfg.N_samples
    # normalize rendering viewing direction 
    rendering_viewdir = rendering_rays_chunk['ray_d'] / torch.norm(rendering_rays_chunk['ray_d'], dim=2, keepdim=True)
    # rendering_viewdir: (1, num_rays, 3)

    """ Encode hand information """
    # encode hand pose
    handpose_feature = handpose_encoder(world_points, world_3d_data) if handpose_encoder is not None else None
    # encode hand mesh?
    handmesh_feature_volume_list = handmesh_encoder(world_3d_data, input_view_parameters, feature_map, mode) if handmesh_encoder is not None else None

    # Track computation
    # macs, params = profile(handmesh_encoder, inputs=(world_3d_data, input_view_parameters, feature_map, mode))
    # computation_cost['handpose_encoder'] = {
    #     'macs': macs,
    #     'params': params
    # }

    # caculate density and color per world point
    # >> nerf input <<
    world_points = world_points[:, :, None, :, :]  # (1, num_rays, 1, 64, 3)
    rendering_viewdir = rendering_viewdir[:, :, None, :]  # (1, num_rays, 1, 3)

    raw_output = calculate_density_color(world_points, rendering_viewdir, nerf, input_view_parameters, world_3d_data, feature_map, handpose_feature, handmesh_feature_volume_list, computation_cost )

    # volume rendering 
    output_ch_dim = 7 if 'semantic' in cfg.nerf_mode else 4
    raw_output_reshaped = raw_output.reshape(-1, num_points_per_ray, output_ch_dim)  # )num_rays, num_points_per_ray, output_ch_dim)
    world_z_vals = world_z_vals.view(-1, num_points_per_ray)
    rgb_semantic_map, depth_map, mask_map, weights = raw2outputs(raw_output_reshaped[:, :, -1:], raw_output_reshaped[:, :, :-1], world_z_vals, background_opacity=0.)

    rgb_map = rgb_semantic_map[:, :3]
    # rgb_semantic_map: (num_rays, 6)
    # depth_map: (num_rays, 1)
    # mask_map: (num_rays, 1)
    # weights: (num_rays, num_points_per_ray)

    ret = {
        'rgb_map': rgb_map.view(batch_size, num_rays, -1),
        'depth_map': depth_map.view(batch_size, num_rays, -1),
        'mask_map': mask_map.view(batch_size, num_rays, -1),
        'weights': weights.view(batch_size, num_rays, -1),  # (batch_size, num_rays, cfg.N_samples)
        'alphas': raw_output_reshaped[:, :, -1].view(batch_size, num_rays, -1)   # (batch_size, num_rays, cfg.N_samples)
    }
    if 'semantic' in cfg.nerf_mode:
        semantic_map = rgb_semantic_map[:, 3:]
        ret['semantic_map'] = semantic_map.view(batch_size, num_rays, -1)
    
        # render image only with object
        semantic = raw_output_reshaped[:,:,3:6]
        semantic_class = torch.argmax(semantic, dim=-1)
        raw_output_reshaped = raw_output_reshaped * (semantic_class == OBJ_SEMANTIC_IDX)[:, :, None]
        rgb_semantic_map, _, _, _ = raw2outputs(raw_output_reshaped[:, :, -1:], raw_output_reshaped[:, :, :-1], world_z_vals, background_opacity=0.)
        object_rgb_map = rgb_semantic_map[:, :3]
        ret['object_rgb_map'] = object_rgb_map.view(batch_size, num_rays, -1)

    if cfg.N_importance > 0 and nerf_fine != None:
        # sample important z_vals
        world_z_vals_mid = .5 * (world_z_vals[..., 1:] + world_z_vals[..., :-1])
        world_z_samples = sample_pdf(world_z_vals_mid, weights[..., 1:-1], cfg.N_importance, det=(cfg.perturb == 0.))
        world_z_samples = world_z_samples.detach()
        world_z_vals, _ = torch.sort(torch.cat([world_z_vals, world_z_samples], -1), -1)
        world_points = rendering_rays_chunk['ray_o'][..., None, :] + rendering_rays_chunk['ray_d'][..., None, :] * world_z_vals[..., :, None]  
        # [1, N_rays, N_samples + N_importance, 3]
        num_points_per_ray = world_points.shape[2]

        """ Encode hand pose information """
        # encode hand pose
        handpose_feature = handpose_encoder(world_points, world_3d_data) if handpose_encoder is not None else None
        # encode hand mesh?
        # handmesh_feature_volume_list = handmesh_encoder(world_3d_data, input_view_parameters, feature_map, mode) if handmesh_encoder is not None else None

        # caculate density and color per world point
        # >> nerf input <<
        world_points = world_points[:, :, None, :, :]  # (1, num_rays, 1, 64, 3)
        # Already expanded; 
        # rendering_viewdir = rendering_viewdir[:, :, None,:] # (1, num_rays, 1, 3)
        raw_output = calculate_density_color(world_points, rendering_viewdir, nerf_fine, input_view_parameters, world_3d_data, feature_map, handpose_feature, handmesh_feature_volume_list, computation_cost)

        # volume rendering
        # output_ch_dim = 7 if 'semantic' in cfg.nerf_mode else 4
        raw_output_reshaped = raw_output.reshape(-1, num_points_per_ray, output_ch_dim)
        world_z_vals = world_z_vals.view(-1, num_points_per_ray)
        rgb_semantic_map, depth_map, mask_map, weights = raw2outputs(raw_output_reshaped[:, :, -1:], raw_output_reshaped[:, :, :-1], world_z_vals, background_opacity=0.)
        
        rgb_map = rgb_semantic_map[:, :3]
        # rgb_semantic_map: (num_rays, 6)
        # depth_map: (num_rays, 1)
        # mask_map: (num_rays, 1)
        # weights: (num_rays, num_points_per_ray)

        
        ret_fine = {
            'rgb_map': rgb_map.view(batch_size, num_rays, -1),
            'depth_map': depth_map.view(batch_size, num_rays, -1),
            'mask_map': mask_map.view(batch_size, num_rays, -1),
            'weights': weights.view(batch_size, num_rays, -1),  # (batch_size, num_rays, cfg.N_samples)
            'alphas': raw_output_reshaped[:, :, -1].view(batch_size, num_rays, -1)  # (batch_size, num_rays, cfg.N_samples)
        }
        if 'semantic' in cfg.nerf_mode:
            semantic_map = rgb_semantic_map[:, 3:]
            
            ret_fine['semantic_map'] = semantic_map.view(batch_size, num_rays, -1)

            # render image only with object
            semantic = raw_output_reshaped[:, :, 3:6]
            semantic_class = torch.argmax(semantic, dim=-1)
            raw_output_reshaped = raw_output_reshaped * (semantic_class == OBJ_SEMANTIC_IDX)[:, :, None]
            rgb_semantic_map, _, _, _ = raw2outputs(raw_output_reshaped[:, :, -1:], raw_output_reshaped[:, :, :-1], world_z_vals, background_opacity=0.)
            object_rgb_map = rgb_semantic_map[:, :3]
            ret_fine['object_rgb_map'] = object_rgb_map.view(batch_size, num_rays, -1)

        # change return dictionary
        for k, v in ret.items():
            ret_fine[k + '_coarse'] = v
        ret = ret_fine

    return ret


# use image feature or not, forward points to implicit functions and get density and color per point
def calculate_density_color(world_points, rendering_viewdir, nerf, input_view_parameters, world_3d_data, feature_map, points_handpose_feat, handmesh_feature_volume_list, computation_cost):
    """
    world_points: (1, num_rays, 1, num_points_per_ray, 3)
    rendering_viewdir: (1, num_rays, 1, 3)
    nerf: nerf network
    input_view_parameters:
    world_3d_data:
    feature_map: (1, C, H, W)
    points_handpose_feat: (1, num_rays, num_points_per_ray, cfg.handpose_feat_dim)
    handmesh_feature_volume_list: list, (_4_, 32/32/64/64, D, H, W), D,H,W is accordingly defined by sp_out_sh
    """
 
    points_img_feat = get_2d_features(world_points, input_view_parameters, feature_map) if feature_map is not None and 'pixel' in cfg.nerf_mode else None
    # points_img_feat: (cfg.num_input_views, num_rays, 1, num_points_per_ray, C)
    points_handmesh_feat = get_3d_features(world_points, world_3d_data, handmesh_feature_volume_list) if handmesh_feature_volume_list is not None else None
   
    # normalize the hand mesh translation respect to the wrist joint
    world_hand_joints = world_3d_data['world_hand_joints'].to(torch.float32)  # (1, 21, 3)
    world_points = world_points - world_hand_joints[:, 0:1, :]

    alpha, semantics, rgb = nerf(world_points, rendering_viewdir, points_img_feat, points_handpose_feat, points_handmesh_feat)
    # alpha: (1, num_points, 1), semantics: (1, num_points, 3) rgb: (1, , num_points, 3),

    # Track computation
    # macs, params =  profile(nerf, inputs=(world_points, rendering_viewdir, points_img_feat, points_handpose_feat, points_handmesh_feat))
    # if 'nerf' in computation_cost:
    #     computation_cost['nerf_fine'] = {
    #         'macs': macs,
    #         'params': params
    #     }
    # else:
    #     computation_cost['nerf'] = {
    #         'macs': macs,
    #         'params': params
    #     }

    raw = alpha
    if semantics is not None:
        raw = torch.cat((semantics, raw), dim=-1)

    if rgb is not None:
        raw = torch.cat((rgb, raw), dim=-1)

    return raw




