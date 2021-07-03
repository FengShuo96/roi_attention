import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import force_fp32

from mmdet.models.builder import ROI_EXTRACTORS
from .base_roi_extractor import BaseRoIExtractor


@ROI_EXTRACTORS.register_module()
class SingleRoIExtractorAttention(BaseRoIExtractor):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56,
                 attention_hidden_channels = 256,
                 conv_out_channels = 256,
                 attention_pool_size = 2):
        super(SingleRoIExtractorAttention, self).__init__(roi_layer, out_channels,
                                                 featmap_strides)
        self.finest_scale = finest_scale
        self.attention_hidden_channels = attention_hidden_channels
        self.q_conv = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1)
        self.k_conv = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1)
        self.v_conv = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1)
        self.y_conv = nn.Conv2d(attention_hidden_channels, conv_out_channels, kernel_size=1)
        self.attention_pool_size = attention_pool_size

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls
        
        
    def attention(self, feature, roi_feats):
    
    
        BS, C, H, W = feature.shape
        BS_num_rois, C_roi, roi_h, roi_w = roi_feats.shape
        num_rois = BS_num_rois // BS
        Q = self.q_conv(roi_feats)  # (BS*num_rois, attention_hidden_channels, H, W)
        _x = F.max_pool2d(feature, self.attention_pool_size, self.attention_pool_size)
        _H, _W = H // self.attention_pool_size, W // self.attention_pool_size
        K = self.k_conv(_x)  # (BS*num_rois, attention_hidden_channels, _H, _W)
        V = self.v_conv(_x)  # (BS*num_rois, attention_hidden_channels, _H, _W)

        Q = Q.permute(0, 2, 3, 1)  # (BS*num_rois, H, W, attention_hidden_channels)
        Q = Q.reshape(BS, num_rois, roi_h, roi_w, self.attention_hidden_channels)
        Q = Q.reshape(BS, num_rois*roi_h*roi_w, self.attention_hidden_channels)  # (BS, num_rois*H*W, attention_hidden_channels)

        K = K.permute(0, 2, 3, 1)  # (BS*num_rois, _H, _W, attention_hidden_channels)
        K = K.reshape(BS, _H*_W, self.attention_hidden_channels)  # (BS, num_rois*_H*_W, attention_hidden_channels)

        V = V.permute(0, 2, 3, 1)  # (BS*num_rois, _H, _W, attention_hidden_channels)
        V = V.reshape(BS, _H*_W, self.attention_hidden_channels)  # (BS, num_rois*_H*_W, attention_hidden_channels)

        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        WEIGHTS = torch.bmm(Q, K.permute(0, 2, 1))  # (BS, num_rois*H*W, _H*_W)
        WEIGHTS = torch.softmax(WEIGHTS, dim=2)

        Y = torch.bmm(WEIGHTS, V)  # (BS, num_rois*H*W, attention_hidden_channels)
        Y = Y.reshape(BS, num_rois, roi_h, roi_w, self.attention_hidden_channels)
        Y = Y.reshape(BS*num_rois, roi_h, roi_w, self.attention_hidden_channels)
        Y = Y.permute(0, 3, 1, 2)
        Y = Y.contiguous()
        y = self.y_conv(Y)
        y = y.contiguous()

        roi_enhanced = roi_feats + y
        
        return roi_enhanced

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function."""
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)
        # TODO: remove this when parrots supports
        if torch.__version__ == 'parrots':
            roi_feats.requires_grad = True

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(rois, num_levels)
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)
        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any():
                rois_ = rois[inds, :]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t
            else:
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.
        #roi_feats shape: (bs*num_rois, channels, 7 ,7) 
        roi_enhanced = self.attention(feats[2],roi_feats)
        return roi_enhanced
