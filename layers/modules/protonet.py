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
            if cfg.MODEL.MASK_HEADS.USE_BN:
                self.mask_refine = nn.ModuleList([nn.Sequential(*[
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True)
                ]) for _ in range(len(self.proto_src))])
            else:
                self.mask_refine = nn.ModuleList([nn.Sequential(*[
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                ]) for _ in range(len(self.proto_src))])

        # The include_last_relu=false here is because we might want to change it to another function
        protonet_cfg = list(cfg.MODEL.MASK_HEADS.PROTO_NET)
        self.clip_frames = cfg.SOLVER.NUM_CLIP_FRAMES
        if cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE and cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE_ON_PROTONET:
            self.cubic_frames = cfg.SOLVER.NUM_CLIP_FRAMES
            for i, cfg_layer in enumerate(protonet_cfg):
                if isinstance(cfg_layer, tuple):
                    cfg_layer = list(cfg_layer)
                if cfg_layer[0] is not None:
                    cfg_layer[0] = cfg_layer[0]*self.cubic_frames
                    protonet_cfg[i] = tuple(cfg_layer)
        else:
            self.cubic_frames = 1

        norm_type = 'batch_norm' if cfg.MODEL.MASK_HEADS.USE_BN else None
        self.proto_net, proto_channels = make_net(in_channels*self.cubic_frames, protonet_cfg, norm_type=norm_type,
                                                  include_last_relu=True)
        self.mask_dim = cfg.MODEL.MASK_HEADS.MASK_DIM
        # the last two Conv layers for predicting prototypes
        proto_arch = [(proto_channels*self.cubic_frames, 3, 1), (self.mask_dim*self.cubic_frames, 1, 0)]
        self.proto_conv, _ = make_net(proto_channels, proto_arch, norm_type=norm_type, include_last_relu=False)
        if cfg.MODEL.MASK_HEADS.USE_DYNAMIC_MASK:
            self.DynamicMaskHead = DynamicMaskHead(cfg)

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

        C_in, H_in, W_in = proto_x.size()[-3:]
        proto_x = proto_x.reshape(-1, self.cubic_frames*C_in, H_in, W_in)
        proto_out_features = self.proto_net(proto_x)
        # Activation function is RELU
        prototypes = F.relu(self.proto_conv(proto_out_features))
        # Move the features last so the multiplication is easy
        # Unfold outputs from 4D to 5D: [bs*T, C, H, W]=>[bs, H, W, T, C]
        bs, C_out, H_out, W_out = prototypes.size()
        # Support single frame for the frame-level methods
        clip_frames = min(self.clip_frames, (bs*C_out) // self.mask_dim)
        prototypes = prototypes.reshape(-1, clip_frames, self.mask_dim, H_out, W_out).permute(0, 3, 4, 1, 2).contiguous()

        return prototypes