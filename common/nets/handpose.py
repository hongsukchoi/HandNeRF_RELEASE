"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Author(s):
Hongsuk Choi (redstonepo@gmail.com)

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0/
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE file.
"""

import torch
import torch.nn as nn
from typing import Optional

from config import cfg


class HandPoseEncoder(nn.Module):
    
    last_layer_bias_init: Optional[float] = None
    last_activation: str = 'relu'
    use_xavier_init: bool = True

    def __init__(self):
        super(HandPoseEncoder, self).__init__()
        
        self.n_layers = 3  
        self.hidden_dim = cfg.handpose_feat_dim
        self.input_dim = 15 * 3
        self.output_dim = cfg.handpose_feat_dim
        self.dummy_val = 0.  # assign to point features that has no relation with hand pose (i.e. far from hand joints)

        try:
            last_activation = {
                'relu': torch.nn.ReLU(True),
                'softplus': torch.nn.Softplus(),
                'sigmoid': torch.nn.Sigmoid(),
                'identity': torch.nn.Identity(),
            }[self.last_activation]
        except KeyError as e:
            raise ValueError(
                "`last_activation` can only be `RELU`,"
                " `SOFTPLUS`, `SIGMOID` or `IDENTITY`."
            ) from e


        layers = []
        for layeri in range(self.n_layers):
            dimin = self.hidden_dim if layeri > 0 else self.input_dim
            dimout = self.hidden_dim if layeri + 1 < self.n_layers else self.output_dim

            linear = torch.nn.Linear(dimin, dimout)
            if self.use_xavier_init:
                _xavier_init(linear)
            if layeri == self.n_layers - 1 and self.last_layer_bias_init is not None:
                torch.nn.init.constant_(linear.bias, self.last_layer_bias_init)
            layers.append(
                torch.nn.Sequential(linear, torch.nn.Softplus())
                if not layeri + 1 < self.n_layers
                else torch.nn.Sequential(linear, last_activation)
            )

        # self.mlp = torch.nn.ModuleList(layers)
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, world_points, world_3d_data):
        """
        world_points: (1, num_rays, cfg.N_samples=num_points_per_ray, 3)
        world_3d_data: ('world_bounds', 'world_mean', 'world_joints', 'world2manojoints); (1, 2, 3), (1,3), (1,15,3), (1,15,4,4)
        """
        world2manojoints = world_3d_data['world2manojoints']
        _, num_rays, num_points_per_ray = world_points.shape[:3]

        # world_points = world_points.view(1, -1, 1, 3)  # (1, num_rays*num_point_per_ray, 1, 3)
        # world_joints = world_joints.view(1, 1, -1, 3)  # (1, 1, 15, 3)
        # world_points_joints_dist = world_points - world_joints # (1, num_rays*num_point_per_ray, 15, 3)
        
        world_points_reshaped = world_points.view(1, -1, 3)  # (1, num_rays*num_point_per_ray, 3)
        world_points_reshaped_xyz1 =  torch.cat([world_points_reshaped, torch.ones_like(world_points_reshaped[:, :, :1])], dim=-1) # (1, num_rays*num_point_per_ray, 4)
        world_points_reshaped_xyz1 =world_points_reshaped_xyz1[0, :, :, None]  # (num_rays*num_point_per_ray, 4, 1)

        mano_points = torch.matmul(world2manojoints[0, :, None, :, :], world_points_reshaped_xyz1)  # (15, N, 4, 1)
        # https://pytorch.org/docs/stable/generated/torch.matmul.html
        # (j x 1 x n x m) @ (k x m x p) -> (j x k x n x p)
        mano_points = mano_points.permute(3,1,0,2)[:, :, :, :3] # (1, N, 15, 3)

        # masking that is based on distance between joints
        # dist = (mano_points ** 2).sum(dim=-1).sqrt()  # (1, N, 15)
        # avg_dist = dist.mean(dim=-1)  # (1, N)
        # mano_points_mask = avg_dist < cfg.handpose_dist_thr  # (1, N)  # 0.1  # 0.001 == 1mm

        # forward
        mano_points = mano_points.reshape(1, num_rays*num_points_per_ray, -1)  # (1, N, 45)
        latent = self.mlp(mano_points)
        # latent: (1, num_rays*num_points_per_ray, hidden_dim)
        
        # masking
        # latent = latent * mano_points_mask[:, :, None]

        # reshape 
        latent = latent.view(1, num_rays, num_points_per_ray, -1)

        return latent


def _xavier_init(linear) -> None:
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    torch.nn.init.xavier_uniform_(linear.weight.data)
