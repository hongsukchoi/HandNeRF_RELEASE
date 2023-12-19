import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg

class MonoNHRNeRF(nn.Module):
    def __init__(self):
        super(MonoNHRNeRF, self).__init__()

        img_feat_dim = cfg.img_feat_dim
        mesh_feat_dim = 4 * cfg.handmesh_feat_dim_from_spconv + 1  # 4 volume features + z

        self.alpha_0 = nn.Conv1d(img_feat_dim + mesh_feat_dim, 256, 1)
        self.alpha_1 = nn.Conv1d(256, 256, 1)
        self.alpha_2 = nn.Conv1d(256, 256, 1)
        self.alpha_out = nn.Conv1d(256, 1, 1)

        if 'semantic' in cfg.nerf_mode:
            # semantic segmentation estimation
            self.num_classes = 3  # background, hand, object
            self.semantic_out = nn.Conv1d(256, self.num_classes, 1)

        self.rgb_0 = nn.Conv1d(1, 64, 1)
        self.rgb_1 = nn.Conv1d(64 + img_feat_dim, 256, 1)
        self.rgb_2 = nn.Conv1d(256, 256, 1)
        self.rgb_3 = nn.Conv1d(256 + 3, 128, 1) # feature + ray direction
        self.rgb_out = nn.Conv1d(128, 3, 1)

        self.apply(self._init_weights)

    # def forward(self, mesh_feat, img_feat, viewdir_render=None):
    def forward(self, rays_points_world, ray_directions, rays_points_world_img_features=None, ray_points_handpose_features=None, ray_points_handmesh_features=None):
        """
        rays_points_world: (1, num_rays, 1, -1, 3)  
        # num_rays can be number of points and -1 can be 1 if reconstructing 
        # -1 is cfg.N_samples for rendering
        ray_directions: (1, num_rays, 1, 3)
        rays_points_world_img_features: (cfg.num_input_views, num_rays, 1, num_points_per_ray, C)
        ray_points_handpose_features: (1, num_rays, num_points_per_ray, cfg.handpose_feat_dim_from_spconv)
        ray_points_handmesh_features: (1, num_rays, num_points_per_ray, cfg.handmesh_feat_dim_from_spconv)
        
        // Returns //
        raw_densities: (1, num_rays * num_points_per_ray, 1)
        rays_semantics: (1, num_rays * num_points_per_ray, self.num_classes)
        rays_colors: (1, num_rays * num_points_per_ray, 3)
        
        """
        
        """ Parse tensors' shape and name to those of MonoNHR's """
        num_points_per_ray = rays_points_world.shape[-2]
        points = rays_points_world.view(1, -1, 3)
        num_points = points.shape[1]
        mesh_feat = ray_points_handmesh_features.view(1, num_points, -1)
        img_feat = rays_points_world_img_features.view(1, num_points, -1)
        # mesh_feat: (1, num_rays * num_points_per_ray, C)
        # img_feat: (1, num_rays * num_points_per_ray, C')
        
        # concat with z
        mesh_feat = torch.cat((mesh_feat, points[:, :, 2:]), dim=-1)

        # premute for convolution
        mesh_feat = mesh_feat.transpose(1,2)
        img_feat = img_feat.transpose(1,2)

        """ MonoNHR Forwarding """
        # calculate density
        comb_feat = torch.cat((mesh_feat, img_feat), 1)
        alpha_feat = F.relu(self.alpha_0(comb_feat))
        alpha_feat = F.relu(self.alpha_1(alpha_feat))
        alpha_feat = F.relu(self.alpha_2(alpha_feat))
        alpha = self.alpha_out(alpha_feat)
        # alpha: (1, 1, num_rays * num_points_per_ray)
        raw_densities = alpha.transpose(1,2)

        if 'semantic' in cfg.nerf_mode:
            semantic = self.semantic_out(alpha_feat)
            # semantic: (1, self.num_classes, num_rays * num_points_per_ray)
            rays_semantics = semantic.transpose(1,2)
        else:
            rays_semantics = None

        if ray_directions is None:
            rays_colors = None
        else:
            # calculate rgb conditioned on alpha
            rendering_viewdir = ray_directions.repeat(1, 1, num_points_per_ray,1).view(1, -1, 3)

            alpha_feat = F.relu(self.rgb_0(alpha.detach()))
            # alpha_feat = F.relu(self.rgb_0(alpha))

            rgb_feat = torch.cat((alpha_feat, img_feat), 1)
            rgb_feat = self.rgb_1(rgb_feat)
            rgb_feat = self.rgb_2(rgb_feat)
            
            rgb_feat = torch.cat((rgb_feat, rendering_viewdir.permute(0, 2, 1)), 1)
            rgb_feat = F.relu(self.rgb_3(rgb_feat))
            rgb = self.rgb_out(rgb_feat)
            # rgb: (1, 3, num_rays * num_points_per_ray)
            rays_colors = rgb.transpose(1,2)

        return raw_densities, rays_semantics, rays_colors

    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            torch.nn.init.xavier_uniform_(module.weight)
