import torch.nn as nn
import spconv.pytorch as spconv
import torch.nn.functional as F
import torch

from config import cfg
from nets.layers import make_deconv_layers, make_conv_layers

class ObjectSegmenter(nn.Module):
    def __init__(self):
        super(ObjectSegmenter, self).__init__()
        self.conv = make_conv_layers([cfg.img_feat_dim,256,256])
        self.deconv = make_deconv_layers([256,256])

        self.fg = make_conv_layers([256,1], kernel=1, stride=1, padding=0, bnrelu_final=False) 

    def forward(self, img_feat):
        """
        img_feat: (1, cfg.img_feat_dim, cfg.input_img_shape[1], cf.ginput_img_shape[0])

        // Returns //
        object_segmentation:  (1, 1, ? x cfg.input_img_shape[1], ? xcf.ginput_img_shape[0])
        """
        img_feat = self.conv(img_feat)
        img_feat = self.deconv(img_feat)
        object_segmentation = self.fg(img_feat)
    
        return torch.sigmoid(object_segmentation)