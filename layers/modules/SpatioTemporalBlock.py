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
                 kernel_size=(3, 3, 3),
                 deformable_groups=4,
                 cascaded=False):
        super(SpatioTemporalBlock, self).__init__()
        self.cascaded = cascaded
        self.kernel_size = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2, (self.kernel_size[2] - 1) // 2)

        # Spatial branch
        offset_channels = self.kernel_size[1] * self.kernel_size[2] * 2
        self.spatio_offset = nn.Conv3d(in_channels,
                                       deformable_groups * offset_channels,
                                       kernel_size=(1, self.kernel_size[1], self.kernel_size[2]),
                                       padding=(0, self.padding[1], self.padding[2]),
                                       bias=False)

        self.spatio_adaption = DeformConv2d(in_channels,
                                            in_channels,
                                            kernel_size=(self.kernel_size[1], self.kernel_size[2]),
                                            padding=(self.padding[1], self.padding[2]),
                                            deform_groups=deformable_groups)

        # Temporal barnch
        self.temporal = nn.Conv3d(in_channels, in_channels, kernel_size=self.kernel_size, padding=self.padding)

        # Init deformable convolution
        self.init_weights()

    def init_weights(self):
        torch.nn.init.normal_(self.spatio_adaption.weight, std=0.01)

    def forward(self, x):
        bs, C_in, T, H, W = x.size()

        # Spatial branch
        # offset: [b, C, T, H, W] --> [b, kh*kw, T, H, W] --> [b*T, kh*kw, H, W]
        offset = self.spatio_offset(x.detach())
        offset_fold = offset.permute(0,2,1,3,4).contiguous().reshape(bs*T, -1, H, W)
        x_fold = x.permute(0,2,1,3,4).contiguous().reshape(bs*T, -1, H, W)
        x_spatio_fold = self.spatio_adaption(x_fold, offset_fold)
        x_spatio = x_spatio_fold.reshape(bs,T,-1,H,W).permute(0,2,1,3,4).contiguous()

        # Temporal branch
        if self.cascaded:
            x_temporal = self.temporal(x_spatio)
            return x_temporal
        else:
            x_temporal = self.temporal(x)
            return x_spatio+x_temporal
        # display_feature_map(x_spatio, type='spatio')
        # display_feature_map(x_temporal, type='temporal')



