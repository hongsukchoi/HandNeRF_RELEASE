import torch.nn as nn
import spconv.pytorch as spconv
import torch.nn.functional as F
import torch


from nets.resnet import ResNetBackbone
from config import cfg


class FeatureEncoder(nn.Module):
    def __init__(self):
        super(FeatureEncoder, self).__init__()
        self.feature_extractor = ResNetBackbone(cfg.resnet_type)
        self.feature_extractor.init_weights()

    def forward(self, input_img):
        """
        input_img: (1, cfg.num_nput_views, cfg.input_img_shape[0], cfg.input_img_shape[1], 3) numpy matrix

        // Returns //
        feature_map: (batch_size, feat_dim, h, w)
        """
        img = input_img[0].permute(0,3,1,2) # (cfg.num_nput_views, 3, cfg.input_img_shape[0], cfg.input_img_shape[1])

        # encode img feature map
        feature_map = self.feature_extractor(img)  # (1, feat_dim, cfg.input_img_shape[0] / 2, cfg.input_img_shape[1] / 2)

        return feature_map