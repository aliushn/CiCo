import torch
import torch.nn as nn
from mmcv.ops import DeformConv2d
from ..visualization_temporal import display_feature_map


class SpatioTemporalBlock(nn.Module):
    '''
    Spatio-temporal block aims to simultaneously explore spatial and temporal information from features.
    '''
    def __init__(self,
                 in_channels,
                 deformable_groups=1,
                 cascaded=False):
        super(SpatioTemporalBlock, self).__init__()
        self.cascaded = cascaded

        # Spatial branch
        self.spatio_kernel_size = (3, 3)
        self.spatio_padding = ((self.spatio_kernel_size[0] - 1) // 2, (self.spatio_kernel_size[1] - 1) // 2)
        # offset_channels = self.spatio_kernel_size[0] * self.spatio_kernel_size[1] * 2
        # self.spatio_offset = nn.Conv2d(in_channels,
        #                                deformable_groups * offset_channels,
        #                                kernel_size=self.spatio_kernel_size,
        #                                padding=self.spatio_padding,
        #                                bias=False)
        #
        # self.spatio_dcn = DeformConv2d(in_channels,
        #                                in_channels,
        #                                kernel_size=self.spatio_kernel_size,
        #                                padding=self.spatio_padding,
        #                                deform_groups=deformable_groups)
        # self.spatio_bn = nn.BatchNorm2d(in_channels)
        # self.spatio_relu = nn.ReLU(inplace=True)

        self.spatio_embed = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=self.spatio_kernel_size, padding=self.spatio_padding),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Temporal branch
        self.temporal_kernel_size = (3, 1, 1)
        self.temporal_padding = ((self.temporal_kernel_size[0] - 1) // 2, (self.temporal_kernel_size[1] - 1) // 2, (self.temporal_kernel_size[2] - 1) // 2)
        self.temporal_embed = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=self.temporal_kernel_size, padding=self.temporal_padding),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.fusion_layer = nn.Sequential(
            nn.Conv3d(2*in_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Init deformable convolution
        # self.init_weights()

    def init_weights(self):
        torch.nn.init.normal_(self.spatio_dcn.weight, std=0.01)

    def forward(self, x):
        bs, C_in, T, H, W = x.size()

        # Spatial branch
        # From 5D to 4D: [b, C, T, H, W] --> [bT, C, H, W]
        x_fold = x.permute(0, 2, 1, 3, 4).contiguous().reshape(bs*T, -1, H, W)
        # x_spatio_fold = self.spatio_dcn(x_fold, self.spatio_offset(x_fold.detach()))
        # x_spatio_fold = self.spatio_relu(self.spatio_bn(x_spatio_fold))
        x_spatio_fold = self.spatio_embed(x_fold)
        x_spatio = x_spatio_fold.reshape(bs, T, -1, H, W).permute(0, 2, 1, 3, 4).contiguous()

        # Temporal branch
        x_temporal = self.temporal_embed(x)

        # Fuse
        x_fusion = torch.cat([x_spatio, x_temporal], dim=1)
        x_out = self.fusion_layer(x_fusion)

        return x_out
        # display_feature_map(x_spatio, type='spatio')
        # display_feature_map(x_temporal, type='temporal')



