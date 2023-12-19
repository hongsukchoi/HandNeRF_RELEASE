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

import os.path as osp
import torch.nn as nn
import spconv.pytorch as spconv
import torch.nn.functional as F
import torch



from config import cfg
from utils.rendering_utils import deterministic_unique

class HandObjectCorrelationEncoder(nn.Module):
    def __init__(self):
        super(HandObjectCorrelationEncoder, self).__init__()

        self.multiview_feat_aggregation = 'average'  # Method of multi-input view feature aggregation

        # For semantic latent encoding of (hand vertex | object point) features
        latent_path = osp.join(osp.abspath(osp.dirname(__file__)), 'latent', 'latent.pt')
        latents = torch.load(latent_path) # just gaussian random tensor
        self.register_buffer("hand_latent", latents['hand_latent'], persistent=True)  # (778,10)
        self.register_buffer("object_latent", latents['object_latent'], persistent=True)  # (1,10)
        self.latent_dim = 10

        if 'latenthandmesh' in cfg.nerf_mode: # replace image feature with latent vector
            latent64_path =  osp.join(osp.abspath(__file__), 'latent', 'latent_64.pt')
            latents_64 = torch.load(latent64_path)
            self.register_buffer("hand_latent64", latents_64['hand_latent'], persistent=True)  # (778,64)
            self.register_buffer("object_latent64", latents_64['object_latent'], persistent=True)  # (1,64)

        self.sparse_3d_conv = SparseConvNet(64+3+self.latent_dim)

    def forward(self, world_3d_data, input_view_parameters, feature_map, mode):
        """
        world_3d_data: 'world_bounds', 'world2manojoints', 'world_hand_mesh', 'world_hand_mesh_voxel_coord', 'world_mesh_voxel_out_sh'
        input_view_parameters: 'input_view_R', 'input_view_t', 'input_view_K'
        feature_map: (cfg.num_input_views, feat_dim, h, w), (h,w) == (cfg.input_img_shape[0] // 2, cfg.input_img_shape[1] // 2)

        // Returns //
        feature_volume_list: list, (_4_, 32/32/64/64, D, H, W), D,H,W is accordingly defined by sp_out_sh
        """

        # feature_map = feature_map.to(torch.float32)

        """" Extract hand mesh image features """
        input_view_K = input_view_parameters['input_view_K'][0]  # (cfg.num_input_views, 3, 3)
        input_view_R = input_view_parameters['input_view_R'][0]  # (cfg.num_input_views, 3, 3)
        input_view_t = input_view_parameters['input_view_t'][0]  # (cfg.num_input_views, 3, 1)

        world_hand_mesh = world_3d_data['world_hand_mesh']  # (1, 778, 3), mano hand mesh
        world_hand_joints = world_3d_data['world_hand_joints']  # (1, 21, 3), mano hand joints
        # world2manojoints = world_3d_data['world2manojoints']  # (1, 15, 4, 4)

        world_hand_mesh_voxel_coord = world_3d_data['world_hand_mesh_voxel_coord']  # (1, 778, 3), discretized world_mesh. 
        world_mesh_voxel_out_sh = world_3d_data['world_mesh_voxel_out_sh']  # (1, 3), boundary of hand+object

        if 'latenthandmesh' in cfg.nerf_mode:
            input_img_hand_mesh_feat = self.hand_latent64.transpose(0, 1)[None]
        else:
            # transform 3d hand vertices from the world to the input cameras
            input_camera_hand_vertices = torch.bmm(input_view_R, world_hand_mesh.transpose(1,2).expand(input_view_R.shape[0], -1, -1)).transpose(1,2) + input_view_t.view(-1, 1, 3)
            # input_camera_hand_vertices: (cfg.num_input_views, 778, 3)
            
            input_img_xyz = torch.bmm(input_camera_hand_vertices, input_view_K.transpose(1, 2))
            input_img_hand_vertices = input_img_xyz[:, :, :2] / input_img_xyz[:, :, 2:]
            # input_img_hand_vertices: (cfg.num_input_views,778, 2)
            
            # normalize to [-1,1]
            normalized_input_img_hand_vertices = input_img_hand_vertices / torch.Tensor([cfg.input_img_shape[1]-1, cfg.input_img_shape[0]-1])[None, None, :].to(input_img_hand_vertices) * 2. - 1

            input_img_hand_mesh_feat = F.grid_sample(feature_map, normalized_input_img_hand_vertices[:, :, None, :], padding_mode=cfg.feature_map_padding_mode, align_corners=True)[:, :, :, 0]
            # input_img_hand_mesh_feat: (cfg.num_input_views, C, 778)
    
        input_img_hand_mesh_feat = input_img_hand_mesh_feat.permute(0,2,1)  # (cfg.num_input_views, 778, C)
        if self.multiview_feat_aggregation == 'average':
            hand_mesh_feat = input_img_hand_mesh_feat.mean(dim=0, keepdim=True)
        else:
            raise ValueError("No other mesh feature aggregation type defined!")

        """ Get 3D object ray points and image features """
        world_object_points = world_3d_data['world_object_points']  # (1, -1 , 3), 3D object points inside the 2D masks
        world_object_points_voxel_coord = world_3d_data['world_object_points_voxel_coord']  # (1, -1, 3), discretized voxel coord. dhw
        
        if world_object_points.sum() != 0:
            if 'latenthandmesh' in cfg.nerf_mode:
                input_img_object_points_feat = self.object_latent64.transpose(0, 1)[None].repeat(1, 1,world_object_points.shape[1])
            else:
                input_camera_object_points = torch.bmm(input_view_R, world_object_points.transpose(1,2).expand(input_view_R.shape[0], -1, -1)).transpose(1,2) + input_view_t.view(-1, 1, 3)
                # input_camera_object_points: (cfg.num_input_views, -1, 3)
                
                input_img_xyz = torch.bmm(input_camera_object_points, input_view_K.transpose(1, 2))
                input_img_object_points= input_img_xyz[:, :, :2] / input_img_xyz[:, :, 2:]
                # input_img_object_points: (cfg.num_input_views, -1, 2)
                
                # normalize to [-1,1]
                normalized_input_img_object_points = input_img_object_points/ torch.Tensor([cfg.input_img_shape[1]-1, cfg.input_img_shape[0]-1])[None, None, :].to(input_img_object_points) * 2. - 1

                input_img_object_points_feat = F.grid_sample(feature_map, normalized_input_img_object_points[:, :, None, :], padding_mode=cfg.feature_map_padding_mode, align_corners=True)[:, :, :, 0]
                # input_img_object_points_feat: (cfg.num_input_views, C, -1)

            input_img_object_points_feat = input_img_object_points_feat.permute(0,2,1)  # (cfg.num_input_views, -1, C)
            if self.multiview_feat_aggregation == 'average':  
                object_points_feat = input_img_object_points_feat.mean(dim=0, keepdim=True)
            else:
                raise ValueError("No other mesh feature aggregation type defined!")

        """ Positional and Semantic encoding """
        batch_size = 1
        # positional encoding
        pe_hand_mesh_feat = self.positional_encoding(world_hand_mesh, hand_mesh_feat, world_hand_joints)
        # pe_hand_mesh_feat: (1, 778, C+3)
        # semantic encoding; assume batch 1; [None, :, :]
        pe_hand_mesh_feat = torch.cat([pe_hand_mesh_feat, self.hand_latent[None, :, :]], dim=-1)
        # reshape
        pe_hand_mesh_feat = pe_hand_mesh_feat.view(batch_size * pe_hand_mesh_feat.shape[1], -1) # (batch_size * 778, C+3)

        if world_object_points.sum() != 0:
            # positional encoding
            pe_object_points_feat = self.positional_encoding(world_object_points, object_points_feat, world_hand_joints)
            # pe_object_points_feat: (1, N, C+3)        
            # semantic encoding; assume batch 1; [None, :, :]
            pe_object_points_feat = torch.cat([pe_object_points_feat, self.object_latent[None, :, :].repeat(1, pe_object_points_feat.shape[1], 1)], dim=-1)
            # reshape
            pe_object_points_feat = pe_object_points_feat.view(batch_size * pe_object_points_feat.shape[1], -1) # (batch_size * N, C+3)


            # concat hand and object features
            # pe_3d_feat = torch.cat([pe_object_points_feat, pe_hand_mesh_feat], dim=0)  # (batch_size * (N + 778), C+3)
            # world_3d_voxel_coord = torch.cat([world_object_points_voxel_coord, world_hand_mesh_voxel_coord], dim=1)  # (1,  + 778, 3)

            pe_3d_feat = torch.cat([pe_hand_mesh_feat, pe_object_points_feat], dim=0)  # (batch_size * (N + 778), C+3)
            world_3d_voxel_coord = torch.cat([world_hand_mesh_voxel_coord, world_object_points_voxel_coord, ], dim=1)  # (1,  + 778, 3)
            
        else: # no object points
            # concat hand and object features
            pe_3d_feat = pe_hand_mesh_feat # (batch_size * 778, C+3)
            world_3d_voxel_coord = world_hand_mesh_voxel_coord # (1, 778, 3)
        
        # if you need faster running, skip this. This is for deterministic evaluation.
        if mode != 'train' and cfg.deterministic_eval:
            val, inv, cnt, idx =  deterministic_unique(world_3d_voxel_coord, dim=1)
            pe_3d_feat = pe_3d_feat[idx]
            world_3d_voxel_coord = world_3d_voxel_coord[:, idx]

        # positional encoding by distance between each voxel and mesh vertex location or object point location
        # concat 
        """ Concat object and hand features and run 3D sparse CNN """
        # encode feature volume with 3D sparse CNN
        feature_volume_list = self.encode_sparse_voxels(pe_3d_feat.to(torch.float32), world_3d_voxel_coord, world_mesh_voxel_out_sh, batch_size)

        return feature_volume_list

    def positional_encoding(self, points, points_feat, world_hand_joints):
        """
        points: (1, N, 3)
        points_feat: (1, N, C)
        world_hand_joints: (1, 21, 3)
        world2manojoints: (1,15,4,4), transformation matrices that ransform a point in the world to the each local mano joint space
        """

        # pe v1: normalize the hand mesh translation respect to the wrist joint
        norm_points = points - world_hand_joints[:, 0:1, :]
        pe_points_feat = torch.cat([points_feat, norm_points], dim=-1)  # (1, N, C+3)
        
        return pe_points_feat

    def encode_sparse_voxels(self, mesh_feat, coord, out_sh, batch_size):
        """ 
        mesh_feat: (batch_size * 778, feat_dim + 3)
        coord: (batch_size, 778, 3)
        out_sh: (batch_size, 3), dhw 
        batch_size: scalar
        """

        sp_coord, sp_out_sh = self.prepare_sp_input(coord, out_sh)
        # sp_coord: (batch_size * 778, 4), sp_out_sh: [d_dim, h_dim, w_dim], list

        xyzc = spconv.SparseConvTensor(mesh_feat, sp_coord, sp_out_sh, batch_size)
        # xyzc.dense(): (batch_size, feat_dim + 1, d_dim, h_dim, w_dim)

        feature_volume_list = self.sparse_3d_conv(xyzc)
        # list of feature_volumes; (batch_size, new_feat_dim, new_d_dim, new_h_dim, new_w_dim)

        return feature_volume_list

    def prepare_sp_input(self, coord, out_sh):
        """ 
        coord: (batch_size, 778, 3), dhw 
        out_sh: (batch_size, 3), dhw 
        """

        # get the input coorindate of mesh for the sparse convolution
        sh = coord.shape  # batch_size, point_num, 3
        idx = [torch.full([sh[1]], i) for i in range(sh[0])]
        idx = torch.cat(idx).to(coord)
        coord = coord.view(-1, 3)  # (batch_size * 778, 3)
        # (batch_size * 778, 4), batch_idx, z, y, x
        sp_coord = torch.cat([idx[:, None], coord], dim=1)

        # get maximum shape of the batch. so the batch consists of the same boundary... for efficient learning (?)
        sp_out_sh, _ = torch.max(out_sh, dim=0)  # (3, )
        sp_out_sh = sp_out_sh.tolist()

        return sp_coord, sp_out_sh

# https://github.com/traveller59/spconv/blob/master/docs/USAGE.md


class SparseConvNet(nn.Module):
    def __init__(self, input_dim):
        super(SparseConvNet, self).__init__()
        hidden_dim_first = 64
        hidden_dim_second = 64

        self.conv0 = double_conv(input_dim, hidden_dim_first)  
        self.down0 = stride_conv(hidden_dim_first, hidden_dim_first)

        self.conv1 = double_conv(hidden_dim_first, hidden_dim_first)
        self.down1 = stride_conv(hidden_dim_first, hidden_dim_first)

        self.conv2 = triple_conv(hidden_dim_first, hidden_dim_first)
        self.down2 = stride_conv(hidden_dim_first, hidden_dim_second)

        self.conv3 = triple_conv(hidden_dim_second, hidden_dim_second)
        self.down3 = stride_conv(hidden_dim_second, hidden_dim_second)

        self.conv4 = triple_conv(hidden_dim_second, hidden_dim_second)

    def forward(self, x):
        x = self.conv0(x)
        x = self.down0(x)

        x = self.conv1(x)
        out1 = x.dense()
        x = self.down1(x)

        x = self.conv2(x)
        out2 = x.dense()
        x = self.down2(x)

        x = self.conv3(x)
        out3 = x.dense()
        x = self.down3(x)

        x = self.conv4(x)
        out4 = x.dense()
        return [out1, out2, out3, out4]


def single_conv(in_channels, out_channels):
    return spconv.SparseSequential(spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(out_channels), nn.ReLU())


def double_conv(in_channels, out_channels):
    return spconv.SparseSequential(spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(out_channels), nn.ReLU(),
                                   spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(out_channels), nn.ReLU(), )


def triple_conv(in_channels, out_channels):
    return spconv.SparseSequential(spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=2), nn.BatchNorm1d(out_channels), nn.ReLU(),
                                   spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=2), nn.BatchNorm1d(
                                       out_channels), nn.ReLU(),
                                   spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=2), nn.BatchNorm1d(out_channels), nn.ReLU())


def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(spconv.SparseConv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, indice_key=indice_key), nn.BatchNorm1d(out_channels), nn.ReLU())
