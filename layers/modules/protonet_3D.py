import torch.nn as nn
import torch.nn.functional as F
from .make_net import make_net
from .dynamic_mask_head import DynamicMaskHead
from ..utils import aligned_bilinear
from ..visualization_temporal import display_cubic_weights


class ProtoNet3D(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        self.cfg = cfg

        self.proto_src = cfg.MODEL.MASK_HEADS.PROTO_SRC
        if self.proto_src is not None and len(self.proto_src) > 1:
            self.mask_refine = nn.ModuleList([nn.Sequential(*[
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ]) for _ in range(len(self.proto_src))])

        # The include_last_relu=false here is because we might want to change it to another function
        self.clip_frames = cfg.SOLVER.NUM_CLIP_FRAMES
        self.use_3D = True if cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE_ON_PROTONET else False
        self.proto_net, proto_channels = make_net(in_channels, cfg.MODEL.MASK_HEADS.PROTO_NET, use_3D=self.use_3D,
                                                  include_bn=True, include_last_relu=True)
        # the last two Conv layers for predicting prototypes
        self.mask_dim = cfg.MODEL.MASK_HEADS.MASK_DIM
        if cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE_ON_PROTONET and cfg.STR.ST_CONSISTENCY.MASK_WITH_PROTOS:
            self.proto_conv = nn.Sequential(*[
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels, self.mask_dim, kernel_size=3, padding=(0, 1, 1)),
            ])

        else:
            proto_arch = [(proto_channels, 3, 1), (self.mask_dim, 1, 0)]
            self.proto_conv, _ = make_net(proto_channels, proto_arch, use_3D=self.use_3D, include_bn=True,
                                          include_last_relu=False)
        if cfg.MODEL.MASK_HEADS.USE_DYNAMIC_MASK:
            self.DynamicMaskHead = DynamicMaskHead()

    def forward(self, x, fpn_outs, img_meta=None):
        if self.proto_src is None:
            proto_x = x
        else:
            if len(self.proto_src) == 1:
                proto_x = fpn_outs[self.proto_src[0]]
            else:
                for src_i, src in enumerate(self.proto_src):
                    if src_i == 0:
                        proto_x = self.mask_refine[src_i](fpn_outs[src])
                    else:
                        proto_x_p = self.mask_refine[src_i](fpn_outs[src])
                        target_h, target_w = proto_x.size()[2:]
                        h, w = proto_x_p.size()[2:]
                        assert target_h % h == 0 and target_w % w == 0
                        factor = target_h // h
                        proto_x_p = aligned_bilinear(proto_x_p, factor)
                        proto_x = proto_x + proto_x_p

        bs, C_in, H_in, W_in = proto_x.size()
        bs = bs // self.clip_frames
        if self.use_3D:
            # Unfold inputs from 4D to 5D: [bs*T, C, H, W]=>[bs, C, T, H, W]
            proto_x = proto_x.reshape(-1, self.clip_frames, C_in, H_in, W_in).permute(0, 2, 1, 3, 4).contiguous()
        proto_out_features = self.proto_net(proto_x)
        # Activation function is Relu
        prototypes = F.relu(self.proto_conv(proto_out_features))
        # display_cubic_weights(prototypes, 0, type=2, name='prototypes', img_meta=img_meta)

        H_out, W_out = prototypes.size()[-2:]
        if self.use_3D:
            # Fold 5D t0 4D: [bs, C, T, H, W] => [bs*T, C, H, W]
            prototypes = prototypes.permute(0, 2, 1, 3, 4).contiguous().reshape(-1, self.mask_dim, H_out, W_out)
        # Move the features last so the multiplication is easy
        prototypes = prototypes.reshape(bs, -1, self.mask_dim, H_out, W_out).permute(0, 3, 4, 1, 2).contiguous()

        display_weights = False
        if display_weights:
            display_cubic_weights(proto_out_features, 0, type=2, name='proto', img_meta=img_meta)
            for jdx in range(0,9,3):
                display_cubic_weights(self.proto_net[jdx].weight, jdx, type=1, name='protonet')
            for jdx in range(0,6,3):
                display_cubic_weights(self.proto_conv[jdx].weight, jdx, type=1, name='protoconv')

        return prototypes