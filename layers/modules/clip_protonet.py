import torch.nn as nn
import torch.nn.functional as F
from .make_net import make_net
from ..utils import aligned_bilinear


class ClipProtoNet(nn.Module):
    """
    The feature maps of FPN layers {P3, P4, P5} are passed through a 2D conv layer
    and upsampled to the same resolution as P3 layer, the sum of which will be fed
    to the clip-level protonet to generate clip-level feature maps.
    Args:
        - in_channels: The input feature size.
        - mask_dim:    The number of prototypes
    """
    def __init__(self, cfg, in_channels, mask_dim):
        super().__init__()
        self.cfg = cfg
        self.mask_dim = mask_dim

        # build mask refine convs
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
        norm_type = 'batch_norm' if cfg.MODEL.MASK_HEADS.USE_BN else None
        self.clip_frames = cfg.SOLVER.NUM_CLIP_FRAMES
        self.proto_net, proto_channels = make_net(in_channels, cfg.MODEL.MASK_HEADS.PROTO_NET,
                                                  use_3D=self.cfg.CiCo.CPN.CUBIC_MODE,
                                                  norm_type=norm_type, include_last_relu=True)
        # Last two Conv layers for predicting prototypes
        proto_arch = [(proto_channels, 3, 1), (self.mask_dim, 1, 0)]
        self.proto_conv, _ = make_net(proto_channels, proto_arch, norm_type=norm_type,
                                      include_last_relu=False)

    def forward(self, x, fpn_outs, img_meta=None):
        # to upsample {P3, P4, P5} FPN layers to the same resolution as P3 layer
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
        if self.cfg.CiCo.CPN.CUBIC_MODE:
            # Unfold inputs from 4D to 5D: [bs*T, C, H, W] => [bs, C, T, H, W]
            proto_x_fold = proto_x.reshape(-1, self.clip_frames, C_in, H_in, W_in).permute(0, 2, 1, 3, 4).contiguous()
            proto_out_fold = self.proto_net(proto_x_fold)
            C_out, _, H_out, W_out = proto_out_fold.size()[-4:]
            proto_out = proto_out_fold.transpose(1, 2).reshape(-1, C_out, H_out, W_out)
        else:
            proto_out = self.proto_net(proto_x)

        # Activation function is Relu
        prototypes = F.relu(self.proto_conv(proto_out))
        H_out, W_out = prototypes.size()[-2:]
        # Move the features last so the multiplication is easy
        prototypes = prototypes.reshape(bs, -1, self.mask_dim, H_out, W_out).permute(0, 3, 4, 1, 2).contiguous()

        return prototypes