import torch.nn as nn
import torch.nn.functional as F
from .make_net import make_net
from .dynamic_mask_head import DynamicMaskHead
from ..utils import aligned_bilinear


class ProtoNet(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        self.proto_src = cfg.MODEL.MASK_HEADS.PROTO_SRC
        if self.proto_src is not None and len(self.proto_src) > 1:
            self.mask_refine = nn.ModuleList([nn.Sequential(*[
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ]) for _ in range(len(self.proto_src))])

        # The include_last_relu=false here is because we might want to change it to another function
        self.proto_net, proto_channles = make_net(in_channels, cfg.MODEL.MASK_HEADS.PROTO_NET, include_bn=True,
                                                  include_last_relu=True)
        # the last two Conv layers for predicting prototypes
        proto_arch = [(proto_channles, 3, 1), (cfg.MODEL.MASK_HEADS.MASK_DIM, 1, 0)]
        self.proto_conv, _ = make_net(proto_channles, proto_arch, include_bn=True, include_last_relu=False)
        if cfg.MODEL.MASK_HEADS.USE_DYNAMIC_MASK:
            self.DynamicMaskHead = DynamicMaskHead()
        self.clip_frames = cfg.SOLVER.NUM_CLIP_FRAMES if cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE else 1

    def forward(self, x, fpn_outs):
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

        proto_out_features = self.proto_net(proto_x)
        # Activation function is RELU
        prototypes = F.relu(self.proto_conv(proto_out_features))
        # Move the features last so the multiplication is easy
        # Unfold outputs from 4D to 5D: [bs*T, C, H, W]=>[bs, H, W, T, C]
        _, C, H, W = prototypes.size()
        prototypes = prototypes.reshape(-1, self.clip_frames, C, H, W).permute(0,3,4,1,2).contiguous()

        return prototypes