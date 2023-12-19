import torch
from typing import Callable, Tuple, Optional

from nets.layers import LinearWithRepeat
from utils.harmonic_embedder import HarmonicEmbedding
from config import cfg

def create_embeddings_for_implicit_function(
    xyz_world: torch.Tensor,
    xyz_embedding_function: Optional[Callable],
) -> torch.Tensor:

    # (1, 1024, 1, 64, 3)
    bs, spatial_size1, spatial_size2, pts_per_ray, _ = xyz_world.shape

    ray_points_for_embed = xyz_world

    embeds = xyz_embedding_function(ray_points_for_embed).reshape(
        bs,
        1,
        spatial_size1 * spatial_size2,
        pts_per_ray,
        -1,
    )  # flatten spatial, add n_src dim

    return embeds



class OriginalNeRF(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.n_harmonic_functions_xyz: int = 10
        self.n_harmonic_functions_dir: int = 4
        self.n_hidden_neurons_xyz: int = 256
        self.n_hidden_neurons_dir: int = 128
        self.latent_dim: int = 0
        self.input_xyz: bool = True
        self.xyz_ray_dir_in_camera_coords: bool = False
        self.color_dim: int = 3

        self.harmonic_embedding_xyz = HarmonicEmbedding(
            self.n_harmonic_functions_xyz, append_input=True
        )
        self.harmonic_embedding_dir = HarmonicEmbedding(
            self.n_harmonic_functions_dir, append_input=True
        )

        mlp_input_dim = self.get_xyz_embedding_dim()
        mlp_skip_dim = mlp_input_dim
        mlp_n_hidden_neurons_xyz: int = self.n_hidden_neurons_xyz
        mlp_n_layers_xyz: int = 8
        append_xyz: Tuple[int, ...] = (5,)

        """ CUSTOM IMPLEMENTATION FOR HAND INFO ENCODING """
        if 'handjoint' in cfg.nerf_mode:
            mlp_input_dim = self.get_xyz_embedding_dim() + cfg.handpose_feat_dim

        if 'handmesh' in cfg.nerf_mode:
            # Where to put hand mesh feat
            if cfg.handmesh_feat_use_as_z:
                mlp_skip_dim = cfg.handmesh_feat_dim
                append_xyz: Tuple[int, ...] = (1, 3, 5)
            else:
                mlp_skip_dim = mlp_input_dim = self.get_xyz_embedding_dim() + cfg.handmesh_feat_dim

            # How to integrate hand mesh feat
            if cfg.handmesh_feat_aggregation == '' or cfg.handmesh_feat_aggregation == 'add' or cfg.handmesh_feat_aggregation == 'average':
                linear =  torch.nn.Linear(cfg.handmesh_feat_dim_from_spconv, cfg.handmesh_feat_dim)
                _xavier_init(linear)
                handmesh_linear_list = [torch.nn.Sequential(linear, torch.nn.Softplus())]
            elif cfg.handmesh_feat_aggregation == 'concat':
                linear = torch.nn.Linear(cfg.handmesh_feat_dim_from_spconv*2, cfg.handmesh_feat_dim)
                _xavier_init(linear)
                handmesh_linear_list = [torch.nn.Sequential(linear, torch.nn.Softplus())]
            elif cfg.handmesh_feat_aggregation == 'learn_weighted_sum':
                linear1 = torch.nn.Linear(2, 1, bias=False)
                _xavier_init(linear1)
                handmesh_linear_list = [torch.nn.Sequential(linear1, torch.nn.Softplus())]
                linear2 =  torch.nn.Linear(cfg.handmesh_feat_dim_from_spconv, cfg.handmesh_feat_dim)
                _xavier_init(linear2)
                handmesh_linear_list.append(torch.nn.Sequential(linear2, torch.nn.Softplus()))
            elif cfg.handmesh_feat_aggregation == 'learn_weighted_sum_multi_voxel_feat':
                linear1 = torch.nn.Linear(8, 1, bias=False)
                _xavier_init(linear1)
                handmesh_linear_list = [torch.nn.Sequential(linear1, torch.nn.Softplus())]
                linear2 =  torch.nn.Linear(cfg.handmesh_feat_dim_from_spconv, cfg.handmesh_feat_dim)
                _xavier_init(linear2)
                handmesh_linear_list.append(torch.nn.Sequential(linear2, torch.nn.Softplus()))
            else:
                raise ValueError("[Model] Undefined handmesh feature aggregation!")

            self.handmesh_layers = torch.nn.ModuleList(handmesh_linear_list)

        # if handmeshpxielnerf, concatenate pixelfeature to the initial xyz embedding input, instead of doing sum in mlps
        if 'pixel' in cfg.nerf_mode and 'handmesh' in cfg.nerf_mode:
            mlp_skip_dim = mlp_input_dim = (mlp_input_dim + cfg.img_feat_dim)
        else:
            mlp_skip_dim = mlp_input_dim
            
        self.xyz_encoder = MLPWithInputSkips(
            mlp_n_layers_xyz,
            mlp_input_dim,
            mlp_n_hidden_neurons_xyz,
            mlp_skip_dim,
            mlp_n_hidden_neurons_xyz,
            input_skips=append_xyz,
        )

        self.intermediate_linear = torch.nn.Linear(
            self.n_hidden_neurons_xyz, self.n_hidden_neurons_xyz
        )
        _xavier_init(self.intermediate_linear)

        self.density_layer = torch.nn.Linear(self.n_hidden_neurons_xyz, 1)
        _xavier_init(self.density_layer)

        # Zero the bias of the density layer to avoid
        # a completely transparent initialization.
        # fixme: Sometimes this is not enough
        self.density_layer.bias.data[:] = 0.0

        embedding_dim_dir = self.harmonic_embedding_dir.get_output_dim()
        self.color_layer = torch.nn.Sequential(
            LinearWithRepeat(
                self.n_hidden_neurons_xyz + embedding_dim_dir, self.n_hidden_neurons_dir
            ),
            torch.nn.ReLU(True),
            torch.nn.Linear(self.n_hidden_neurons_dir, self.color_dim),
            torch.nn.Sigmoid(),
        )
        
        if 'semantic' in cfg.nerf_mode:
            # semantic segmentation estimation
            self.num_classes = 3  # background, hand, object
            self.semantic_layer = torch.nn.Linear(self.n_hidden_neurons_xyz, self.num_classes) 
            _xavier_init(self.semantic_layer)

            # Zero the bias of the density layer to avoid
            # a completely transparent initialization.
            # fixme: Sometimes this is not enough
            self.semantic_layer.bias.data[:] = 0.0

    def _get_colors(self, features: torch.Tensor, rays_directions: torch.Tensor):
        """
        This function takes per-point `features` predicted by `self.xyz_encoder`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.
        """
        # Normalize the ray_directions to unit l2 norm.
        rays_directions_normed = torch.nn.functional.normalize(
            rays_directions, dim=-1)
        # Obtain the harmonic embedding of the normalized ray directions.
        # pyre-fixme[29]: `Union[torch.Tensor, torch.nn.Module]` is not a function.
        rays_embedding = self.harmonic_embedding_dir(rays_directions_normed)

        # pyre-fixme[29]: `Union[torch.Tensor, torch.nn.Module]` is not a function.
        return self.color_layer((self.intermediate_linear(features), rays_embedding))

    def get_xyz_embedding_dim(self):
        return (
            self.harmonic_embedding_xyz.get_output_dim() * int(self.input_xyz)
            + self.latent_dim
        )

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
        rays_semantics: (1, num_rays * num_points_per_ray, 3)
        rays_colors: (1, num_rays * num_points_per_ray, 3)
        """
        """ Prepare latent features """
        embeds = create_embeddings_for_implicit_function(rays_points_world, self.harmonic_embedding_xyz)
        # embeds: (1, 1, 1024, 64, 63)
        # pytorch3d; embeds.shape = [minibatch x n_src x n_rays x n_pts x self.n_harmonic_functions*6+3]

        # Use hand pose features!
        if ray_points_handpose_features is not None:
            
            embeds = torch.cat([embeds, ray_points_handpose_features[:, None]], dim=-1)
            # embeds: (1, 1, 1024, 64, 63+cfg.handpose_feat_dim)

        # Use hand mesh features
        if ray_points_handmesh_features is not None:
            for li, layer in enumerate(self.handmesh_layers):
                if li == 0 and 'learn_weighted_sum' in cfg.handmesh_feat_aggregation:
                    # ray_points_handmesh_features: (1, num_rays, num_points_per_ray, 2, cfg.handmesh_feat_dim_from_spconv)
                    # learn the weighted sum parameter
                    ray_points_handmesh_features = layer(ray_points_handmesh_features.transpose(3,4))[..., 0]
                else:
                    ray_points_handmesh_features = layer(ray_points_handmesh_features)

            ray_points_handmesh_features = ray_points_handmesh_features[:, None]

            if cfg.handmesh_feat_use_as_z:
                pass 
            else:
                embeds = torch.cat([embeds, ray_points_handmesh_features], dim=-1)
                # embeds: (1, 1, 1024, 64, ?+cfg.handmesh_feat_dim)
                ray_points_handmesh_features = None

        # Use img features!
        imgfeat = rays_points_world_img_features
        if rays_points_world_img_features is not None:
            imgfeat = rays_points_world_img_features.transpose(1, 2)
            # imgfeat: (cfg.num_input_views, 1, num_rays, num_points_per_ray, C)
            if 'handmesh' in cfg.nerf_mode:  
                # if there're both 'pixel' and 'hand' in 'nerf_mode',
                # concatenate pixelfeature to the initial xyz embedding input, instead of doing sum in mlps
                embeds = torch.cat([embeds, imgfeat.mean(dim=0, keepdim=True)], dim=-1)
                imgfeat = None
            else:
                # if 'hand' not in cfg.nerf_mode, and there's img feature, then do sum in mlps as pixelNeRF(2021CVPR)
                # if you concatenate, it will not trained well
                pass
        
        """ Feed to MLP layers """
        features = self.xyz_encoder(embeds, z=ray_points_handmesh_features, imgfeat=imgfeat) 
        # pytorch3d; features.shape = [minibatch x ... x self.n_hidden_neurons_xyz]
        features = features.reshape(*rays_points_world.shape[:-1], -1)
        # features: (1, 1024, 1, 64, 256)

        raw_densities = self.density_layer(features)
        # raw_densities.shape: (1, 1024, 1, 64, 1)
        
        raw_densities = raw_densities.reshape(
            raw_densities.shape[0], raw_densities.shape[1]*raw_densities.shape[2]*raw_densities.shape[3], -1)
        # raw_densities: (1, 1, -1)

        if 'semantic' in cfg.nerf_mode:
            rays_semantics = self.semantic_layer(features)
            # raw_semantics.shape: (1, 1024, 1, 64, self.num_classes)
            rays_semantics = rays_semantics.reshape(
                rays_semantics.shape[0], rays_semantics.shape[1]*rays_semantics.shape[2]*rays_semantics.shape[3], -1)
            # rays_semantics: (1, self.num_classes, -1)
        else:
            rays_semantics = None

        # reconstructing
        if ray_directions is None:
            rays_colors = None
        
        else:
            rays_colors = self._get_colors(features, ray_directions)
            # rays_colors.shape: (1, 1024, 1, 64, 3)

            rays_colors = rays_colors.reshape(
                rays_colors.shape[0], rays_colors.shape[1]*rays_colors.shape[2]*rays_colors.shape[3], -1)
            # rays_colors: (1, 3, -1)

        return raw_densities, rays_semantics, rays_colors


class MLPWithInputSkips(torch.nn.Module):
    # Copyright (c) Meta Platforms, Inc. and affiliates.
    # All rights reserved.
    #
    # This source code is licensed under the BSD-style license found in the
    # LICENSE file in  https://github.com/facebookresearch/pytorch3d/blob/main/LICENSE

    """
    Implements the multi-layer perceptron architecture of the Neural Radiance Field.

    As such, `MLPWithInputSkips` is a multi layer perceptron consisting
    of a sequence of linear layers with ReLU activations.__post_init__

    Additionally, for a set of predefined layers `input_skips`, the forward pass
    appends a skip tensor `z` to the output of the preceding layer.

    Note that this follows the architecture described in the Supplementary
    Material (Fig. 7) of [1], for which keep the defaults for:
        - `last_layer_bias_init` to None
        - `last_activation` to "relu"
        - `use_xavier_init` to `true`

    If you want to use this as a part of the color prediction in TensoRF model set:
        - `last_layer_bias_init` to 0
        - `last_activation` to "sigmoid"
        - `use_xavier_init` to `False`

    References:
        [1] Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik
            and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng:
            NeRF: Representing Scenes as Neural Radiance Fields for View
            Synthesis, ECCV2020

    Members:
        n_layers: The number of linear layers of the MLP.
        input_dim: The number of channels of the input tensor.
        output_dim: The number of channels of the output.
        skip_dim: The number of channels of the tensor `z` appended when
            evaluating the skip layers.
        hidden_dim: The number of hidden units of the MLP.
        input_skips: The list of layer indices at which we append the skip
            tensor `z`.
        last_layer_bias_init: If set then all the biases in the last layer
            are initialized to that value.
        last_activation: Which activation to use in the last layer. Options are:
            "relu", "softplus", "sigmoid" and "identity". Default is "relu".
        use_xavier_init: If True uses xavier init for all linear layer weights.
            Otherwise the default PyTorch initialization is used. Default True.
    """

    
    last_layer_bias_init: Optional[float] = None
    last_activation: str = 'relu'
    use_xavier_init: bool = True

    def __init__(self,  n_layers, input_dim, output_dim, skip_dim, hidden_dim, input_skips):
        super().__init__()

        self.n_layers: int = 8 if n_layers is None else n_layers
        self.input_dim: int = 39 if input_dim is None else input_dim
        self.output_dim: int = 256 if output_dim is None else output_dim
        self.skip_dim: int = 39 if skip_dim is None else skip_dim
        self.hidden_dim: int = 256 if hidden_dim is None else hidden_dim
        self.input_skips: Tuple[int, ...] = (5,) if input_skips is None else input_skips

        """ CUSTOM IMPLEMENTATION FOR PIXELNERF """
        self.n_imgfeat_layers = 3 # 3  # after this, combine multi-view image features
        self.imgfeat_dim = cfg.img_feat_dim
        assert self.n_layers > (self.n_imgfeat_layers+1), "Wrong MLP architecture... Increaes number of original layers"
        imgfeat_layers = []
        for imgfeat_layeri in range(self.n_imgfeat_layers):
            dimin = self.imgfeat_dim if imgfeat_layeri == 0 else self.hidden_dim
            dimout = self.hidden_dim

            if imgfeat_layeri > 0 and imgfeat_layeri in self.input_skips:
                dimin = self.hidden_dim + self.skip_dim
            if (imgfeat_layeri+1) in self.input_skips:
                dimout = self.hidden_dim + self.skip_dim

            linear = torch.nn.Linear(dimin, dimout)
            if self.use_xavier_init:
                _xavier_init(linear)
            if imgfeat_layeri == self.n_imgfeat_layers - 1 and self.last_layer_bias_init is not None:
                torch.nn.init.constant_(linear.bias, self.last_layer_bias_init)
            imgfeat_layers.append(
                # torch.nn.Sequential(linear, torch.nn.ReLU(True))
                torch.nn.Sequential(linear, torch.nn.Softplus())
            )
        self.imgfeat_mlp = torch.nn.ModuleList(imgfeat_layers)
        

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

            if layeri > 0 and layeri in self.input_skips:
                dimin = self.hidden_dim + self.skip_dim

            linear = torch.nn.Linear(dimin, dimout)
            if self.use_xavier_init:
                _xavier_init(linear)
            if layeri == self.n_layers - 1 and self.last_layer_bias_init is not None:
                torch.nn.init.constant_(linear.bias, self.last_layer_bias_init)
            layers.append(
                # torch.nn.Sequential(linear, torch.nn.ReLU(True))
                torch.nn.Sequential(linear, torch.nn.Softplus())
                if not layeri + 1 < self.n_layers
                else torch.nn.Sequential(linear, last_activation)
            )
        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(self.input_skips)


    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None, imgfeat: Optional[torch.Tensor] = None):

        """
        Args:
            x: The input tensor of shape `(..., input_dim)`.
            z: The input skip tensor of shape `(..., skip_dim)` which is appended
                to layers whose indices are specified by `input_skips`.
        Returns:
            y: The output tensor of shape `(..., output_dim)`.
        """
        y = x
        if z is None:
            # if the skip tensor is None, we use `x` instead.
            z = x
        skipi = 0
        # pyre-fixme[6]: For 1st param expected `Iterable[Variable[_T]]` but got
        #  `Union[Tensor, Module]`.

        for li, layer in enumerate(self.mlp):
            # pyre-fixme[58]: `in` is not supported for right operand type
            #  `Union[torch._tensor.Tensor, torch.nn.modules.module.Module]`.
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)
                skipi += 1

            """ CUSTOM IMPLEMENTATION FOR PIXELNERF """
            if imgfeat is not None:
                if 0 < li <= self.n_imgfeat_layers:
                    imgfeat = self.imgfeat_mlp[li-1](imgfeat)
                    y = y.expand(imgfeat.shape[0], -1, -1, -1, -1) + imgfeat

                elif li == (self.n_imgfeat_layers + 1):
                    # average multi-view features
                    y = torch.mean(y, axis=0, keepdim=True)

            y = layer(y)
        return y


def _xavier_init(linear) -> None:
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    torch.nn.init.xavier_uniform_(linear.weight.data)
