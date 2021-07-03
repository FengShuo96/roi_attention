import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .convfc_bbox_head import ConvFCBBoxHead


@HEADS.register_module()
class RoIAttentionPixelToPixelConvFCBBoxHead2(ConvFCBBoxHead):

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 attention_hidden_channels=256,
                 attention_pool_size=2,
                 *args,
                 **kwargs):
        super().__init__(num_shared_convs, num_shared_fcs, num_cls_convs, num_cls_fcs, num_reg_convs, num_reg_fcs,
                         conv_out_channels, fc_out_channels, conv_cfg, norm_cfg, *args, **kwargs)
        self.attention_hidden_channels = attention_hidden_channels
        self.attention_pool_size = attention_pool_size
        self.multihead_attn = nn.modules.activation.MultiheadAttention(attention_hidden_channels, num_heads=1, dropout=0.1, need_weights=False)

    def init_weights(self):
        super(RoIAttentionPixelToPixelConvFCBBoxHead2, self).init_weights()
        # conv layers are already initialized by ConvModule

    def forward(self, x):
        """

        :param x: shape (BS, num_rois, C, H, W)
        :return:
        """
        BS, num_rois, C, H, W = x.shape
        x = x.permute(1, 3, 4, 0, 2)
        x = x.reshape(-1, BS, C)
        attn_output = self.multihead_attn(x, x, x)  # (num_rois*H*W, BS, attention_hidden_channels)
        x_enhanced = x + attn_output
        x_enhanced = x_enhanced.reshape(num_rois, H, W, BS, self.attention_hidden_channels)
        x_enhanced = x_enhanced.permute(3, 0, 4, 1, 2)
        x_enhanced = x_enhanced.reshape(-1, self.attention_hidden_channels, H, W)
        return super(RoIAttentionPixelToPixelConvFCBBoxHead2, self).forward(x_enhanced)


@HEADS.register_module()
class RoIAttentionPixelToPixelShared2FCBBoxHead2(RoIAttentionPixelToPixelConvFCBBoxHead2):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(RoIAttentionPixelToPixelShared2FCBBoxHead2, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class RoIAttentionPixelToPixelShared4Conv1FCBBoxHead2(RoIAttentionPixelToPixelConvFCBBoxHead2):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(RoIAttentionPixelToPixelShared4Conv1FCBBoxHead2, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class RoIAttentionPixelToObjectConvFCBBoxHead2(ConvFCBBoxHead):

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 attention_hidden_channels=256,
                 *args,
                 **kwargs):
        super().__init__(num_shared_convs, num_shared_fcs, num_cls_convs, num_cls_fcs, num_reg_convs, num_reg_fcs,
                         conv_out_channels, fc_out_channels, conv_cfg, norm_cfg, *args, **kwargs)
        self.attention_hidden_channels = attention_hidden_channels
        self.score_conv = nn.Conv2d(conv_out_channels, 1, kernel_size=1)
        self.q_conv = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1)
        self.k_fc = nn.Linear(conv_out_channels, attention_hidden_channels)
        self.v_fc = nn.Linear(conv_out_channels, attention_hidden_channels)
        self.y_conv = nn.Conv2d(attention_hidden_channels, conv_out_channels, kernel_size=1)
        self.multihead_attn = nn.modules.activation.MultiheadAttention(attention_hidden_channels, num_heads=1, dropout=0.1,
                                                                   need_weights=False)

    def init_weights(self):
        super(RoIAttentionPixelToObjectConvFCBBoxHead2, self).init_weights()
        # conv layers are already initialized by ConvModule
        for module_list in [self.k_fc, self.v_fc]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """

        :param x: shape (BS, num_rois, C, H, W)
        :return:
        """
        BS, num_rois, C, H, W = x.shape
        x = x.reshape(-1, C, H, W)
        Q = self.q_conv(x)  # (BS*num_rois, attention_hidden_channels, H, W)
        scores = self.score_conv(x)  # (BS*num_rois, 1, H, W)
        scores = scores.reshape(-1, 1, H*W)
        scores = torch.softmax(scores, dim=2)
        scores = scores.reshape(-1, 1, H, W)

        # from .get_param import get_count, increment_count, save
        # import os
        # _filename = os.path.join("/tmp/debug/", "weight-{}.pickle".format(get_count()))
        # increment_count()
        # save(_filename, scores.detach().cpu().numpy())

        f_rois = x * scores  # (BS*num_rois, C, H, W)
        f_rois = f_rois.sum(dim=[2, 3])  # (BS * num_rois, C)
        K = self.k_fc(f_rois)  # (BS * num_rois, attention_hidden_channels)
        V = self.v_fc(f_rois)  # (BS * num_rois, attention_hidden_channels)
        K = K.reshape(BS, num_rois, -1)
        V = V.reshape(BS, num_rois, -1)

        Q = Q.permute(0, 2, 3, 1)  # (BS*num_rois, H, W, attention_hidden_channels)
        Q = Q.reshape(BS, num_rois, H, W, -1)  # (BS, num_rois, H, W, attention_hidden_channels)
        Q = Q.reshape(BS, num_rois*H*W, -1)  # (BS, num_rois*H*W, attention_hidden_channels) or (BS, num_points, attention_hidden_channels)

        K = K.permute(0, 2, 1)
        weights = torch.bmm(Q, K)  # (BS, num_points, num_rois)
        weights = torch.softmax(weights, dim=2)
        Y = torch.bmm(weights, V)  # (BS, num_points, attention_hidden_channels)
        Y = Y.reshape(BS, num_rois, H, W, -1)
        Y = Y.reshape(-1, H, W, self.attention_hidden_channels)  # (BS*num_rois, H, W, attention_hidden_channels)
        Y = Y.permute(0, 3, 1, 2)  # (BS*num_rois, attention_hidden_channels, H, W)
        y = self.y_conv(Y)

        x_enhanced = x + y
        return super(RoIAttentionPixelToObjectConvFCBBoxHead2, self).forward(x_enhanced)


@HEADS.register_module()
class RoIAttentionPixelToObjectShared2FCBBoxHead2(RoIAttentionPixelToObjectConvFCBBoxHead2):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(RoIAttentionPixelToObjectShared2FCBBoxHead2, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class RoIAttentionPixelToObjectShared4Conv1FCBBoxHead2(RoIAttentionPixelToObjectConvFCBBoxHead2):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(RoIAttentionPixelToObjectShared4Conv1FCBBoxHead2, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class RoIAttentionObjectToObjectConvFCBBoxHead2(ConvFCBBoxHead):

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 attention_hidden_channels=1024,
                 *args,
                 **kwargs):
        super().__init__(num_shared_convs, num_shared_fcs, num_cls_convs, num_cls_fcs, num_reg_convs, num_reg_fcs,
                         conv_out_channels, fc_out_channels, conv_cfg, norm_cfg, *args, **kwargs)
        self.attention_hidden_channels = attention_hidden_channels
        _in_channels = self.in_channels * self.roi_feat_area
        self.multihead_attn = nn.modules.activation.MultiheadAttention(attention_hidden_channels, num_heads=1, dropout=0.1, need_weights=False)

    def init_weights(self):
        super(RoIAttentionObjectToObjectConvFCBBoxHead2, self).init_weights()
        # conv layers are already initialized by ConvModule

    def forward(self, x):
        """

        :param x: shape (BS, num_rois, C, H, W)
        :return:
        """
        BS, num_rois, C, H, W = x.shape
        x = x.reshape(BS, num_rois, -1)  # (BS, num_rois, C*H*W)
        x = x.permute(1, 0, 2)  # (num_rois, BS, C*H*W)
        x_enhanced = self.attention_module(x)
        x_enhanced = x_enhanced.permute(1, 0, 2)
        x_enhanced = x_enhanced.reshape(BS, num_rois, C, H, W)
        return super(RoIAttentionObjectToObjectConvFCBBoxHead2, self).forward(x_enhanced)


@HEADS.register_module()
class RoIAttentionObjectToObjectShared2FCBBoxHead2(RoIAttentionObjectToObjectConvFCBBoxHead2):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(RoIAttentionObjectToObjectShared2FCBBoxHead2, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class RoIAttentionObjectToObjectShared4Conv1FCBBoxHead2(RoIAttentionObjectToObjectConvFCBBoxHead2):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(RoIAttentionObjectToObjectShared4Conv1FCBBoxHead2, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
