
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from datasets.config import cfg

from .make_net import make_net
from .Featurealign import FeatureAlign
from utils import timer
from itertools import product
from math import sqrt
from mmcv.ops import DeformConv2d
from ..utils import display_conf_outs


class PredictionModule(nn.Module):
    """
    The (c) prediction module adapted from DSSD:
    https://arxiv.org/pdf/1701.06659.pdf

    Note that this is slightly different to the module in the paper
    because the Bottleneck block actually has a 3x3 convolution in
    the middle instead of a 1x1 convolution. Though, I really can't
    be arsed to implement it myself, and, who knows, this might be
    better.
    Args:
        - in_channels:   The input feature size.
        - out_channels:  The output feature size (must be a multiple of 4).
        - aspect_ratios: A list of lists of priorbox aspect ratios (one list per scale).
        - scales:        A list of priorbox scales relative to this layer's convsize.
                         For instance: If this layer has convouts of size 30x30 for
                                       an image of size 600x600, the 'default' (scale
                                       of 1) for this layer would produce bounding
                                       boxes with an area of 20x20px. If the scale is
                                       .5 on the other hand, this layer would consider
                                       bounding boxes with area 10x10px, etc.
        - parent:        If parent is a PredictionModule, this module will use all the layers
                         from parent instead of from this module.
    """

    def __init__(self, in_channels, out_channels=1024,
                 pred_aspect_ratios=None, pred_scales=None, parent=None, deform_groups=1):
        super().__init__()

        self.out_channels = out_channels
        self.num_classes = cfg.num_classes
        self.mask_dim = cfg.mask_dim
        self.num_priors = len(pred_aspect_ratios[0]) * len(pred_scales)
        self.track_dim = cfg.track_dim
        self.pred_aspect_ratios = pred_aspect_ratios
        self.pred_scales = pred_scales
        self.deform_groups = deform_groups
        self.parent = [parent]  # Don't include this in the state dict

        if cfg.use_sipmask:
            self.mask_dim = self.mask_dim * cfg.sipmask_head
        elif cfg.mask_proto_coeff_occlusion:
            self.mask_dim = self.mask_dim * 3
        elif cfg.use_dynamic_mask:
            self.mask_dim = cfg.mask_dim ** 2 * (cfg.dynamic_mask_head_layers - 1) \
                            + cfg.mask_dim * cfg.dynamic_mask_head_layers + 1
            if not cfg.disable_rel_coords:
                self.mask_dim += cfg.mask_dim * 2
        elif cfg.mask_proto_with_levels:
            self.mask_dim = self.mask_dim * 2

        kernel_size = cfg.pred_conv_kernels[0]
        padding = [(kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2]

        if parent is None:

            self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4, kernel_size=kernel_size, padding=padding)

            if cfg.train_class:
                if cfg.use_dcn_class:
                    self.conf_layer = FeatureAlign(self.out_channels,
                                                   self.num_priors * self.num_classes,
                                                   kernel_size=kernel_size,
                                                   deformable_groups=self.deform_groups,
                                                   use_pred_offset=cfg.use_pred_offset)
                else:
                    self.conf_layer = nn.Conv2d(self.out_channels, self.num_priors * self.num_classes,
                                                kernel_size=kernel_size, padding=padding)

            if cfg.train_track and not cfg.track_by_Gaussian:
                if cfg.use_dcn_track:
                    self.track_layer = FeatureAlign(self.out_channels,
                                                    self.num_priors * self.embed_dim,
                                                    kernel_size=kernel_size,
                                                    deformable_groups=self.deform_groups,
                                                    use_pred_offset=cfg.use_pred_offset)
                else:
                    self.track_layer = nn.Conv2d(out_channels, self.num_priors * self.track_dim,
                                                 kernel_size=kernel_size, padding=padding)

            if cfg.use_dcn_mask:
                self.mask_layer = FeatureAlign(self.out_channels,
                                               self.num_priors * self.mask_dim,
                                               kernel_size=kernel_size,
                                               deformable_groups=self.deform_groups,
                                               use_pred_offset=cfg.use_pred_offset)
            else:
                self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim,
                                            kernel_size=kernel_size, padding=padding)

            if cfg.train_centerness:
                self.centerness_layer = nn.Conv2d(out_channels, self.num_priors,
                                                  kernel_size=kernel_size, padding=padding)

            # What is this ugly lambda doing in the middle of all this clean prediction module code?
            def make_extra(num_layers):
                if num_layers == 0:
                    return lambda x: x
                else:
                    # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
                    return nn.Sequential(*sum([[
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.GroupNorm(32, out_channels),
                        nn.ReLU(inplace=True)
                    ] for _ in range(num_layers)], []))

            self.bbox_extra, self.conf_extra = [make_extra(x) for x in cfg.extra_layers[:2]]
            if cfg.train_track and not cfg.track_by_Gaussian:
                self.track_extra = make_extra(cfg.extra_layers[-1])

    def forward(self, x, idx):
        """
        Args:
            - x: The convOut from a layer in the backbone network
                 Size: [batch_size, in_channels, conv_h, conv_w])
        Returns a tuple (bbox_coords, class_confs, mask_output, prior_boxes) with sizes
            - bbox_coords: [batch_size, conv_h*conv_w*num_priors, 4]
            - class_confs: [batch_size, conv_h*conv_w*num_priors, num_classes]
            - mask_output: [batch_size, conv_h*conv_w*num_priors, mask_dim]
            - prior_boxes: [conv_h*conv_w*num_priors, 4]
        """
        # In case we want to use another module's layers
        src = self if self.parent[0] is None else self.parent[0]

        bs, _, conv_h, conv_w = x.size()

        bbox_x = src.bbox_extra(x)
        conf_x = src.conf_extra(x)
        if cfg.train_track and not cfg.track_by_Gaussian:
            track_x = src.track_extra(x)

        bbox = src.bbox_layer(bbox_x)
        bbox_output = bbox.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)

        if cfg.train_class:
            if cfg.use_dcn_class:
                conf = src.conf_layer(conf_x, bbox.detach())
            else:
                conf = src.conf_layer(conf_x)
            conf = conf.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)

        if cfg.train_track and not cfg.track_by_Gaussian:
            if cfg.use_dcn_track:
                track = src.track_layer(track_x, bbox.detach())
            else:
                track = src.track_layer(track_x)
            track = track.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.track_dim)
            track = F.normalize(track, dim=-1)

        if cfg.use_dcn_mask:
            mask = src.mask_layer(bbox_x, bbox.detach())
        else:
            mask = src.mask_layer(bbox_x)
        mask = mask.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)

        if cfg.train_centerness:
            centerness = src.centerness_layer(bbox_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 1)
            centerness = torch.sigmoid(centerness)

        # See box_utils.decode for an explanation of this
        if cfg.use_yolo_regressors:
            bbox_output[:, :, :2] = torch.sigmoid(bbox_output[:, :, :2]) - 0.5
            bbox_output[:, :, 0] /= conv_w
            bbox_output[:, :, 1] /= conv_h

        priors, prior_levels = self.make_priors(idx, conv_h, conv_w, x.device)
        preds = {'loc': bbox_output, 'conf': conf, 'mask_coeff': mask,
                 'priors': priors, 'prior_levels': prior_levels}

        if cfg.train_track and not cfg.track_by_Gaussian:
            preds['track'] = track

        if cfg.train_centerness:
            preds['centerness'] = centerness

        return preds

    def make_priors(self, idx, conv_h, conv_w, device):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
        with timer.env('makepriors'):
            prior_data = []
            prior_levels = []
            # Iteration order is important (it has to sync up with the convout)
            for j, i in product(range(conv_h), range(conv_w)):
                # +0.5 because priors are in center-size notation
                x = (i + 0.5) / conv_w
                y = (j + 0.5) / conv_h

                for ars in self.pred_aspect_ratios:
                    for scale in self.pred_scales:
                        for ar in ars:
                            # [1, 1/2, 2]
                            ar = sqrt(ar)
                            r = scale / self.pred_scales[0] * 3
                            w = r * ar / conv_w
                            h = r / ar / conv_h

                            prior_data += [x, y, w, h]
                            prior_levels += [idx]

            priors = torch.Tensor(prior_data, device=device).view(1, -1, 4).detach()
            priors.requires_grad = False

            prior_levels = torch.Tensor(prior_levels, device=device).view(1, -1).detach()
            prior_levels.requires_grad = False

        return priors, prior_levels

