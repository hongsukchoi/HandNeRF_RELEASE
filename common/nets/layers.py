import math
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init, Parameter


class LinearWithRepeat(torch.nn.Module):
    # Copyright (c) Meta Platforms, Inc. and affiliates.
    # All rights reserved.
    #
    # This source code is licensed under the BSD-style license found in the
    # LICENSE file in  https://github.com/facebookresearch/pytorch3d/blob/main/LICENSE

    """
    if x has shape (..., k, n1)
    and y has shape (..., n2)
    then
    LinearWithRepeat(n1 + n2, out_features).forward((x,y))
    is equivalent to
    Linear(n1 + n2, out_features).forward(
        torch.cat([x, y.unsqueeze(-2).expand(..., k, n2)], dim=-1)
    )

    Or visually:
    Given the following, for each ray,

                feature   ->

    ray         xxxxxxxx
    position    xxxxxxxx
      |         xxxxxxxx
      v         xxxxxxxx


    and
                            yyyyyyyy

    where the y's do not depend on the position
    but only on the ray,
    we want to evaluate a Linear layer on both
    types of data at every position.

    It's as if we constructed

                xxxxxxxxyyyyyyyy
                xxxxxxxxyyyyyyyy
                xxxxxxxxyyyyyyyy
                xxxxxxxxyyyyyyyy

    and sent that through the Linear.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """
        Copied from torch.nn.Linear.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Copied from torch.nn.Linear.
        """
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)


def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def make_deconv_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.ConvTranspose2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)
                                  