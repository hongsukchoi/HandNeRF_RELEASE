import numpy as np
import cc3d
import mcubes
import trimesh
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional

from config import cfg, mano_layer
from utils.geometry3d_utils import tensor_get_surfaces, tensor_get_shortest_distance_from_surfaces, tensor_get_projection


""" For Deterministic Evaluation """
def stable_argsort(arr, dim=-1, descending=False):
    arr_np = arr.detach().cpu().numpy()
    if descending:
        indices = np.argsort(-arr_np, axis=dim, kind='stable')
    else:
        indices = np.argsort(arr_np, axis=dim, kind='stable')
    return torch.from_numpy(indices).long().to(arr.device)
        
def deterministic_unique(x, dim=0):
    unique, inverse, counts = torch.unique(x, dim=dim, return_inverse=True, return_counts=True)
    inv_sorted = stable_argsort(inverse, dim=0, descending=False)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    index = index.sort().values
    return unique, inverse, counts, index
        

# use connected component graph theory
# the output mesh is very unlikely to splitted into half meshes. so component with less than half points is dust
# refer to https://pypi.org/project/connected-components-3d/
def remove_dust_from_voxel_cube(cube: np.ndarray, thr_ratio: Optional[float] = 0.5) -> np.ndarray:
    """
    cube: (w, h, d)
    thr_ratio: give number of occupied voxels, percentage threshold for checking valid component
    """
    cube_mask = cube.copy()
    cube_mask[cube_mask > 0] = 1
    cube_mask = cube_mask.astype(np.int32)
    threshold = int(cube.nonzero()[0].shape[0] * thr_ratio + 0.5)  #cube.nonzero()[0].shape[0] // 2
    cube_mask_out = cc3d.dust(cube_mask, threshold=threshold, connectivity=26, in_place=False)
    cube = cube * (cube_mask_out > 0)
    return cube

def get_mesh_from_marching_cube(cube: np.ndarray, world_min_xyz: np.ndarray) -> trimesh.Trimesh:
    """
    cube: (w, h, d)
    world_min_xyz: (3,)
    """
    cube = np.pad(cube, 10, mode='constant')
    vertices, triangles = mcubes.marching_cubes(cube, cfg.mesh_thr)
    
    vertices = (vertices - 10) * cfg.mc_voxel_size[0]
    vertices = vertices + world_min_xyz
    # vertices = vertices - vertices.mean(axis=0)[None, :] 
    mesh = trimesh.Trimesh(vertices, triangles)

    return mesh

# Render rays in smaller minibatches to avoid OOM.
def batchify_rays(world_points: torch.Tensor, handpose_feature: torch.Tensor, alpha_decoder: object, chunk: int = 1024 * 32) -> torch.Tensor:
    """
    world_points: (batch_size, num_points, 3)
    handpose_feature: (batch_size, num_points, 1, C)
    alpha_decoder: nerf related function
    chunk: number of points to render/recon per forward
    
    // Returns //
    all_re: (batch_size, num_points, 1)
    """

  
    n_batch, n_point = world_points.shape[:2]
    all_ret = []
    # >> nerf input <<
    # rays_points_world: (1, num_rays, 1, 64, 3)
    for i in range(0, n_point, chunk):
        chunked_world_points = world_points[:, i:i + chunk, None, None, :]
        chunked_handpose_feature = handpose_feature[:, i:i + chunk, :, :] if handpose_feature is not None else None

        ret = alpha_decoder(chunked_world_points, chunked_handpose_feature)
        all_ret.append(ret)
    if len(all_ret) > 0:
        all_ret = torch.cat(all_ret, 1)
        return all_ret
    else:
        return None
    
# from https://github.com/zju3dv/neuralbody/blob/caa8b8dfffe49d69f636ef9958ffd47844505589/lib/networks/renderer/nerf_net_utils.py#L55
def raw2outputs(
    rays_densities: torch.Tensor,
    rays_features: torch.Tensor,
    ray_lengths: torch.Tensor,
    density_noise_std: float = 0.0,
    surface_thickness: int = 1,
    background_opacity: float = 1e10,
    blend_output: bool = False,
    _bg_color: float = 0.
):
    """
        Args:
            rays_densities: Per-ray density values represented with a tensor
                of shape `(..., n_points_per_ray, 1)`.
            rays_features: Per-ray feature values represented with a tensor
                of shape `(..., n_points_per_ray, feature_dim)`.
            aux: a dictionary with extra information.
            ray_lengths: Per-ray depth values represented with a tensor
                of shape `(..., n_points_per_ray, feature_dim)`.
            density_noise_std: the magnitude of the noise added to densities.

        Returns:
            features: A tensor of shape `(..., feature_dim)` containing
                the rendered features for each ray.
            depth: A tensor of shape `(..., 1)` containing estimated depth.
            opacities: A tensor of shape `(..., 1)` containing rendered opacities.
            weights: A tensor of shape `(..., n_points_per_ray)` containing
                the ray-specific non-negative opacity weights. In general, they
                don't sum to 1 but do not overcome it, i.e.
                `(weights.sum(dim=-1) <= 1.0).all()` holds.
        """

    def _capping_function(x): return 1.0 - torch.exp(-x)
    ray_lengths_diffs = ray_lengths[..., 1:] - ray_lengths[..., :-1]

    last_interval = torch.full_like(
        ray_lengths[..., :1], background_opacity
    )
    deltas = torch.cat((ray_lengths_diffs, last_interval), dim=-1)

    rays_densities = rays_densities[..., 0]

    if density_noise_std > 0.0:
        noise = torch.randn_like(
            rays_densities).mul(density_noise_std)
        rays_densities = rays_densities + noise

    rays_densities = torch.relu(rays_densities)

    weighted_densities = deltas * rays_densities
    capped_densities = _capping_function(
        weighted_densities)  # pyre-ignore: 29

    rays_opacities = _capping_function(  # pyre-ignore: 29
        torch.cumsum(weighted_densities, dim=-1)
    )
    opacities = rays_opacities[..., -1:]
    absorption_shifted = (-rays_opacities + 1.0).roll(
        surface_thickness, dims=-1
    )
    absorption_shifted[..., : surface_thickness] = 1.0

    weights = capped_densities * absorption_shifted


    features = (weights[..., None] * rays_features).sum(dim=-2)
    depth = (weights * ray_lengths)[..., None].sum(dim=-2)

    alpha = opacities if blend_output else 1

    features = alpha * features + (1 - opacities) * _bg_color

    return features, depth, opacities, weights

# from https://github.com/zju3dv/neuralbody/blob/caa8b8dfffe49d69f636ef9958ffd47844505589/lib/networks/renderer/nerf_net_utils.py#L55
# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False):
    # from torchsearchsorted import searchsorted

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf],
                    -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples).to(cdf)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples]).to(cdf)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


# sample points along each ray
def get_sampling_points(ray_o, ray_d, near, far, mode, steps=cfg.N_samples):
    # calculate the steps for each ray
    t_vals = torch.linspace(0., 1., steps=steps).to(near)
    z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

    if cfg.perturb > 0. and mode == 'train':
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(upper)
        z_vals = lower + (upper - lower) * t_rand

    pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

    return pts, z_vals



def get_2d_features(world_points: torch.Tensor, input_view_parameters: Dict[str, torch.Tensor], feature_map: torch.Tensor) -> torch.Tensor:
    # avoid quantization error
    # world_points = world_points.to(torch.float32)
    # feature_map = feature_map.to(torch.float32)

    # When reconstructing, num_rays could be total_num_points and num_points_per_ray could be just 1
    num_rays, _, num_points_per_ray = world_points.shape[1:4]

    input_view_K = input_view_parameters['input_view_K'][0] # (cfg.num_input_views, 3, 3)
    input_view_R = input_view_parameters['input_view_R'][0] # (cfg.num_input_views, 3, 3)
    input_view_t = input_view_parameters['input_view_t'][0] # (cfg.num_input_views, 3, 1)

    """ interpolate features """
    # transform 3d points from the world to the input cameras
    input_camera_points = torch.bmm(input_view_R, world_points.view(1, -1, 3).transpose(1,2).expand(input_view_R.shape[0], -1, -1)).transpose(1,2) + input_view_t.view(-1, 1, 3)
    # input_camera_points: (cfg.num_input_views, num_rays * 1 * num_points_per_ray, 3)

    input_img_xyz = torch.bmm(input_camera_points, input_view_K.transpose(1,2))
    input_img_points = input_img_xyz[:, :, :2] / input_img_xyz[:, :, 2:]
    # input_img_points: (cfg.num_input_views, num_rays * 1 * num_points_per_ray, 2)

    # normalize to [-1,1]
    normalized_input_img_points = input_img_points / torch.Tensor([cfg.input_img_shape[1]-1, cfg.input_img_shape[0]-1])[None, None, :].to(input_img_points) * 2. - 1

    points_img_feat = F.grid_sample(feature_map, normalized_input_img_points[:, :, None, :], padding_mode=cfg.feature_map_padding_mode, align_corners=True)[:, :, :, 0]
    # points_img_feat: (cfg.num_input_views, C, num_rays * 1 * num_points_per_ray)

    points_img_feat = points_img_feat.permute(0, 2, 1).view(input_view_R.shape[0], num_rays, 1, num_points_per_ray, -1)
    # points_img_feat: (cfg.num_input_views, num_rays, 1, num_points_per_ray, C)

    # convert feature back to torch.float32 from 64
    return points_img_feat.to(torch.float32)

# batch computation
def tensor_get_normalized_voxel_coords(pts, bounds, out_sh, voxel_size):
    # convert xyz to the voxel coordinate dhw
    dhw = pts[..., [2, 1, 0]]

    min_dhw = bounds[:, 0, [2, 1, 0]]
    dhw = dhw - min_dhw[:, None]
    dhw = dhw / torch.tensor(voxel_size).to(dhw)
    # convert the voxel coordinate to [-1, 1]
    out_sh = out_sh.to(dhw)
    dhw = dhw / out_sh * 2 - 1

    # convert dhw to whd, since the occupancy is indexed by dhw
    grid_coords = dhw[..., [2, 1, 0]]
    return grid_coords

# batch computation
def tensor_get_hand_mesh_projection(pts: torch.Tensor, world_3d_data: object) -> torch.Tensor:
    """
    pts: (1, num_points, 3)
    world_3d_data:

    // Returns //
    weights: (num_points, 1538)
    surface_points: (num_points, 1538, 3)
    """
    world_hand_mesh = world_3d_data['world_hand_mesh'][0]  # (778, 3), mano hand mesh
    # avoid OOM
    world_hand_mesh = world_hand_mesh.to(torch.float32)
    mano_side = world_3d_data['mano_side'][0]  # 'right' or 'left', batch size is 1
    
    faces_of_mano = mano_layer[mano_side].th_faces  # (1538, 3), indices of faces
    surfaces_vertices = world_hand_mesh[faces_of_mano]  # (1538, 3, 3)

    surfaces = tensor_get_surfaces(surfaces_vertices)  # (1538, 4), a, b, c, d

    distances, feet = tensor_get_projection(pts[0], surfaces)
    # distances: (N, 1538), feet: (N, 1538, 3)

    # get uniform k after sorting 
    _, indices = torch.sort(-distances, 1)
    pace = distances.shape[1] // cfg.handmesh_surface_topk
    selected_indices = indices[:, ::pace][:, 1:]  # (N, k)
    
    # select k distances and feet
    k_distances = torch.gather(distances, 1, selected_indices) # (N, k)
    k_feet = torch.gather(feet, 1, selected_indices[:, :, None].repeat(1,1,3))  # (N, k, 3)

    # normalize distance to [0,1]
    norm_k_distances = k_distances / k_distances.max()
    norm_k_weights = (1 / (norm_k_distances + cfg.handmesh_surface_topk))

    return norm_k_weights, k_feet

# batch computation
def tensor_get_closests_surfaces_mean_position(pts: torch.Tensor, world_3d_data: object) -> torch.Tensor:
    """
    pts: (1, num_points, 3)
    world_3d_data:

    // Returns //
    topk_weights: (num_points, k)
    topk_surface_points: (num_points, k, 3)
    topk_min_surface_indices: (num_points, k)
    surfaces_vertices: (1538, 3, 3)
    """

    world_hand_mesh = world_3d_data['world_hand_mesh'][0]  # (778, 3), mano hand mesh
    # avoid OOM
    world_hand_mesh = world_hand_mesh.to(torch.float32)
    mano_side = world_3d_data['mano_side'][0]  # 'right' or 'left', batch size is 1
    
    faces_of_mano = mano_layer[mano_side].th_faces  # (1538, 3), indices of faces
    surfaces_vertices = world_hand_mesh[faces_of_mano]  # (1538, 3, 3)

    surfaces = tensor_get_surfaces(surfaces_vertices)  # (1538, 4), a, b, c, d
    topk_min_distances, topk_min_surface_indices = tensor_get_shortest_distance_from_surfaces(pts[0], surfaces, k=cfg.handmesh_surface_topk)  # (num_points,k), (num_points,k)
    # topk_weights = F.softmax(1/topk_min_distances, dim=-1) *  (1 / (topk_min_distances + 1))
    
    # normalize distance to [0,1]
    norm_topk_min_distances = topk_min_distances / topk_min_distances.max()
    topk_weights = (1 / (norm_topk_min_distances + cfg.handmesh_surface_topk))
    # topk_weights = (1 / (topk_min_distances + cfg.handmesh_surface_topk))

    nearest_surface_vertices = surfaces_vertices[topk_min_surface_indices] # (num_points, k, 3, 3) num_points, topk, vertices, xyz
    topk_surface_points = nearest_surface_vertices.mean(dim=-2)  # (num_points, k, 3)

    return topk_weights, topk_surface_points, topk_min_surface_indices, surfaces_vertices


# batch computation
def tensor_get_approx_projection(pts: torch.Tensor, world_3d_data: object) -> torch.Tensor:
    """
    pts: (1, num_points, 3)
    world_3d_data:

    // Returns //
    weights: (num_points, 1538)
    surface_points: (num_points, 1538, 3)
    """
    world_hand_mesh = world_3d_data['world_hand_mesh'][0]  # (778, 3), mano hand mesh
    # avoid OOM
    world_hand_mesh = world_hand_mesh.to(torch.float32)
    mano_side = world_3d_data['mano_side'][0]  # 'right' or 'left', batch size is 1
    
    faces_of_mano = mano_layer[mano_side].th_faces  # (1538, 3), indices of faces
    surfaces_vertices = world_hand_mesh[faces_of_mano]  # (1538, 3, 3)

    surfaces_centers = surfaces_vertices.mean(dim=1)  # (1538, 3)

    # avoid OOM; .to(torch.float16)
    distances = torch.sqrt(torch.sum((pts.to(torch.float16).transpose(0, 1) - surfaces_centers[None, :, :].to(torch.float16)) ** 2, dim=-1))
    # distances: (N, 1538)

    # distances -> normalized weights
    inv_distances = 1 / (1e-6 + distances.to(torch.float32))  # prevent zero division
    weights = F.softmax(inv_distances, dim=1) # (N, 1538)

    # make far points (with high inv_distances) have less weight sum
    inv_distances_sums = inv_distances.sum(dim=1, keepdim=True)
    inv_distances_sums_mean = inv_distances_sums.mean()
    inv_distances_sums_norm = inv_distances_sums / inv_distances_sums_mean
    weights = inv_distances_sums_norm * weights

    return weights, surfaces_centers


# trinlinear interpolation from 3D feature volume list
def interpolate_3d_features(pts: torch.Tensor, bounds: torch.Tensor, mesh_voxel_out_sh: torch.Tensor, feature_volume_list: List[torch.Tensor]) -> torch.Tensor:
    """
    pts: (..., 3)
    bounds: (1, 2, 3)
    mesh_voxel_out_sh: (1, 3)
    feature_volumue_list: (_4_, 32/32/64/64, D, H, W), D,H,W is accordingly defined by sp_out_sh
    
    // Returns // 
    features: (1, C, num_points)     
    """

    # avoid quantization error
    # pts = pts.to(torch.float32)

    normalized_points = tensor_get_normalized_voxel_coords(pts.view(1, -1, 3), bounds, mesh_voxel_out_sh, cfg.input_mesh_voxel_size)

    feature_list = []
    for volume in feature_volume_list:
        # volume = volume.to(torch.float32)
        feature = F.grid_sample(volume, normalized_points[:, :, None, None, :], padding_mode=cfg.feature_volume_padding_mode , align_corners=True)[:, :, :, 0, 0]
        feature_list.append(feature)

    if ('mononhr' in cfg.nerf_mode) or (cfg.handmesh_feat_aggregation == 'learn_weighted_sum_multi_voxel_feat'):
        features = torch.cat(feature_list, dim=1)  # (1, C, num_points)
    else:
        features = torch.stack(feature_list, dim=1).sum(dim=1)  # (1, C, num_points)
        # features = torch.stack(feature_list, dim=1).mean(dim=1)  # (1, C, num_points)

    # convert feature back to torch.float32 from float32
    return features.to(torch.float32)

# tri-linear interpolation for features of 3D points of a neural radiance field
def get_3d_features(world_points: torch.Tensor, world_3d_data: object, feature_volume_list: List[torch.Tensor]) -> torch.Tensor:
    """
    world_points: (1, num_rays, 1, num_points_per_ray, 3)

    // Returns //
    points_handmesh_feat: (1, num_rays, num_points_per_ray, C')
    """
    world_bounds = world_3d_data['world_bounds']  # (1, 2, 3)    
    world_mesh_voxel_out_sh = world_3d_data['world_mesh_voxel_out_sh']  # (1, 3), boundary of hand+object
    
    """ sample mesh_feat from feaure volumes """
    voxel_feat = interpolate_3d_features(world_points, world_bounds, world_mesh_voxel_out_sh, feature_volume_list)
    # voxel_feat: (1, C', num_rays*num_points_per_ray)
    points_handvoxel_feat = voxel_feat.permute(0,2,1).view(1, world_points.shape[1], world_points.shape[3], -1)
    # points_handvoxel_feat: (1, num_rays, num_points_per_ray, C')
    
    if ('mononhr' in cfg.nerf_mode) or (cfg.handmesh_feat_aggregation == ''):
        points_handmesh_feat = points_handvoxel_feat

    else:
        # TEMP
        newnew = True
        new = False
        if newnew:
            # weights, surface_centers = tensor_get_hand_mesh_projection(world_points.view(1, -1, 3), world_3d_data)
            weights, surface_centers = tensor_get_approx_projection(world_points.view(1, -1, 3), world_3d_data)
            # weights: (num_points, 1538)
            # surface_points: (num_points, 1538, 3)

            surface_feat = interpolate_3d_features(surface_centers, world_bounds, world_mesh_voxel_out_sh, feature_volume_list)
            # surface_feat: (1, C', 1538)
            points_handsurface_feat = surface_feat.mean(dim=-1).expand(weights.shape[0],-1)  # (num_points, C')
            
            
        elif new:
            # get all projection
            
            # weights, surface_centers = tensor_get_hand_mesh_projection(world_points.view(1, -1, 3), world_3d_data)
            weights, surface_centers = tensor_get_approx_projection(world_points.view(1, -1, 3), world_3d_data)
            # weights: (num_points, 1538)
            # surface_points: (num_points, 1538, 3)

            surface_feat = interpolate_3d_features(surface_centers, world_bounds, world_mesh_voxel_out_sh, feature_volume_list)
            # surface_feat: (1, C', 1538)
            points_handsurface_feat = (weights @ surface_feat[0].transpose(0,1))  # (num_points, C')
            
            points_handsurface_feat = points_handsurface_feat.view(1, world_points.shape[1], world_points.shape[3], -1) 
            # points_handsurface_feat: (1, num_rays, num_points_per_ray, C')
        else:
            # nearest surface features
            topk_weights, topk_surface_points, topk_min_surface_indices, surfaces_vertices = tensor_get_closests_surfaces_mean_position(world_points.view(1, -1, 3), world_3d_data)
            # topk_weights: (num_points, k)
            # topk_surface_points: (num_points, k, 3)
            # surfaces_vertices: (1538, 3, 3)

            surface_points = surfaces_vertices.mean(dim=1)  # (1538, 3)
            surface_feat = interpolate_3d_features(surface_points, world_bounds, world_mesh_voxel_out_sh, feature_volume_list)
            # surface_feat: (1, C', 1538)

            # sample and reshape surface feat; instead of interpolating with repeatition
            surface_feat = surface_feat.transpose(0, 2) #(1538, C', 1)
            topk_surface_feat = surface_feat[topk_min_surface_indices]  # (num_points, k, C', 1)
            topk_surface_feat = topk_surface_feat.permute(3,2,0,1)  # (1, C', num_points, k)

            # weighted sum of the topk features
            weighted_topk_surface_feat = topk_surface_feat * topk_weights[None, None, :, :]
            surface_feat = weighted_topk_surface_feat.sum(-1)

            points_handsurface_feat = surface_feat.permute(0,2,1).view(1, world_points.shape[1], world_points.shape[3], -1)
            # points_handsurface_feat: (1, num_rays, num_points_per_ray, C')

        
        """ Aggregate the voxel feature and surface feature """
        # points_handmesh_feat = points_handvoxel_feat
        if cfg.handmesh_feat_aggregation == 'add':
            points_handmesh_feat = points_handsurface_feat + points_handvoxel_feat
        elif cfg.handmesh_feat_aggregation == 'average':
            points_handmesh_feat = (points_handsurface_feat + points_handvoxel_feat) / 2.
        elif cfg.handmesh_feat_aggregation == 'concat':
            points_handmesh_feat = torch.cat([points_handsurface_feat, points_handvoxel_feat], dim=-1)
        elif cfg.handmesh_feat_aggregation == 'learn_weighted_sum':
            points_handmesh_feat = torch.cat([points_handsurface_feat[:, :, :, None, :], points_handvoxel_feat[:, :, :, None, :]], dim=-2)
            # points_handsurface_feat: (1, num_rays, num_points_per_ray, 2, C')
        elif cfg.handmesh_feat_aggregation == 'learn_weighted_sum_multi_voxel_feat':
            sh = points_handvoxel_feat.shape  #  (1, num_rays, num_points_per_ray, C')
            points_handvoxel_feat = points_handvoxel_feat.reshape(sh[0], sh[1], sh[2], 4, -1)
            points_handsurface_feat = points_handsurface_feat.reshape(sh[0], sh[1], sh[2], 4, -1)

            points_handmesh_feat = torch.cat([points_handsurface_feat, points_handvoxel_feat], dim=-2)  # 8

        else:
            raise ValueError("[Model] Undefined handmesh feature aggregation!")
    
    # convert feature back to torch.float32 from float32
    return points_handmesh_feat.to(torch.float32)
