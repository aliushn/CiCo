import torch
import torch.nn as nn
import torch.nn.functional as F

from .Featurealign import FeatureAlign
from itertools import product
from math import sqrt


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

    def __init__(self, cfg, in_channels, deform_groups=1):
        super().__init__()

        self.cfg = cfg
        self.in_channels = in_channels
        self.num_classes = cfg.DATASETS.NUM_CLASSES if cfg.MODEL.CLASS_HEADS.USE_FOCAL_LOSS else cfg.DATASETS.NUM_CLASSES + 1
        self.mask_dim = cfg.MODEL.MASK_HEADS.MASK_DIM
        self.track_dim = cfg.MODEL.TRACK_HEADS.TRACK_DIM
        self.pred_aspect_ratios = cfg.MODEL.BACKBONE.PRED_ASPECT_RATIOS
        self.pred_scales = cfg.MODEL.BACKBONE.PRED_SCALES
        self.num_priors = len(self.pred_aspect_ratios[0]) * len(self.pred_scales[0])
        self.deform_groups = deform_groups
        self.clip_frames = cfg.SOLVER.NUM_CLIP_FRAMES if cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE else 1

        if cfg.MODEL.MASK_HEADS.USE_SIPMASK:
            self.mask_dim = self.mask_dim * cfg.MODEL.MASK_HEADS.SIPMASK_HEAD
        elif cfg.MODEL.MASK_HEADS.PROTO_COEFF_OCCLUSION:
            self.mask_dim = self.mask_dim * 3
        elif cfg.MODEL.MASK_HEADS.USE_DYNAMIC_MASK:
            self.mask_dim = self.mask_dim ** 2 * (cfg.MODEL.MASK_HEADS.DYNAMIC_MASK_HEAD_LAYERS - 1) \
                            + self.mask_dim * cfg.MODEL.MASK_HEADS.DYNAMIC_MASK_HEAD_LAYERS + 1
            if not cfg.MODEL.MASK_HEADS.DISABLE_REL_COORDS:
                self.mask_dim += self.mask_dim * 2
        else:
            self.mask_dim = self.mask_dim

        kernel_size = (3, 3)
        padding = [(kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2]
        if cfg.MODEL.PREDICTION_HEADS.CIRCUMSCRIBED_BOXES and cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
            # only needs a circumscribed box of multiple frames for instance segmentation
            self.boxes_dim = self.num_priors
        else:
            self.boxes_dim = self.num_priors*self.clip_frames
        self.bbox_layer = nn.Conv2d(self.in_channels, self.boxes_dim*4,
                                    kernel_size=kernel_size, padding=padding)
        if cfg.MODEL.BOX_HEADS.TRAIN_CENTERNESS:
            self.centerness_layer = nn.Conv2d(self.in_channels, self.boxes_dim,
                                              kernel_size=kernel_size, padding=padding)

        if cfg.MODEL.CLASS_HEADS.TRAIN_CLASS:
            if cfg.STMASK.FC.FCB_USE_DCN_CLASS:
                self.conf_layer = FeatureAlign(self.in_channels,
                                               self.num_priors * self.num_classes,
                                               kernel_size=kernel_size,
                                               deformable_groups=self.deform_groups,
                                               use_pred_offset=cfg.STMASK.FC.FCB_USE_PRED_OFFSET)
            else:
                self.conf_layer = nn.Conv2d(self.in_channels, self.num_priors*self.num_classes,
                                            kernel_size=kernel_size, padding=padding)

        if cfg.MODEL.TRACK_HEADS.TRAIN_TRACK and not cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
            if cfg.STMASK.FC.FCB_USE_DCN_TRACK:
                self.track_layer = FeatureAlign(self.in_channels,
                                                self.num_priors*self.track_dim,
                                                kernel_size=kernel_size,
                                                deformable_groups=self.deform_groups,
                                                use_pred_offset=cfg.STMASK.FC.FCB_USE_PRED_OFFSET)
            else:
                self.track_layer = nn.Conv2d(self.in_channels, self.num_priors*self.track_dim,
                                             kernel_size=kernel_size, padding=padding)

        if cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
            if cfg.STMASK.FC.FCB_USE_DCN_MASK:
                self.mask_layer = FeatureAlign(self.out_channels,
                                               self.num_priors * self.mask_dim,
                                               kernel_size=kernel_size,
                                               deformable_groups=self.deform_groups,
                                               use_pred_offset=cfg.STMASK.FC.FCB_USE_PRED_OFFSET)
            else:
                self.mask_layer = nn.Conv2d(self.in_channels, self.num_priors * self.mask_dim,
                                            kernel_size=kernel_size, padding=padding)

        # What is this ugly lambda doing in the middle of all this clean prediction module code?
        def make_extra(num_layers, in_channels):
            if num_layers == 0:
                return lambda x: x
            else:
                # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
                return nn.Sequential(*sum([[
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    # nn.BatchNorm2d(in_channels),
                    # nn.GroupNorm(32, in_channels),
                    nn.ReLU(inplace=True)
                ] for _ in range(num_layers)], []))

        self.bbox_extra = make_extra(cfg.MODEL.BOX_HEADS.TOWER_LAYERS, self.in_channels)
        self.conf_extra = make_extra(cfg.MODEL.CLASS_HEADS.TOWER_LAYERS, self.in_channels)
        if cfg.MODEL.TRACK_HEADS.TRAIN_TRACK and not cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
            self.track_extra = make_extra(cfg.MODEL.TRACK_HEADS.TOWER_LAYERS, self.in_channels)

    def forward(self, fpn_outs, img_meta=None):
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

        preds = {'priors': [], 'prior_levels': []}
        if self.cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
            preds['mask_coeff'] = []
        if self.cfg.MODEL.TRACK_HEADS.TRAIN_TRACK and not self.cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
            preds['track'] = []
        if self.cfg.MODEL.BOX_HEADS.TRAIN_BOXES:
            preds['loc'] = []
        if self.cfg.MODEL.BOX_HEADS.TRAIN_CENTERNESS:
            preds['centerness'] = []
        if self.cfg.MODEL.CLASS_HEADS.TRAIN_CLASS:
            preds['conf'] = []

        for idx in range(len(fpn_outs)):
            x = fpn_outs[idx]
            bs, _, conv_h, conv_w = x.size()
            priors, prior_levels = self.make_priors(idx, conv_h, conv_w, x.device)
            preds['priors'] += [priors]
            preds['prior_levels'] += [prior_levels]
    
            bbox_x = self.bbox_extra(x)
            bbox = self.bbox_layer(bbox_x).permute(0, 2, 3, 1).contiguous()
            preds['loc'] += [bbox.reshape(bs, conv_h*conv_w*self.num_priors, -1)]

            # Centerness for Boxes
            if self.cfg.MODEL.BOX_HEADS.TRAIN_CENTERNESS:
                centerness = self.centerness_layer(bbox_x).permute(0, 2, 3, 1).contiguous()
                preds['centerness'] += [centerness.reshape(bs, conv_h*conv_w*self.num_priors, -1).sigmoid()]

            # Classification
            if self.cfg.MODEL.CLASS_HEADS.TRAIN_CLASS:
                conf_x = self.conf_extra(x)
                conf = self.conf_layer(conf_x, bbox.detach()) if self.cfg.STMASK.FC.FCB_USE_DCN_CLASS else self.conf_layer(conf_x)
                preds['conf'] += [conf.permute(0, 2, 3, 1).contiguous().reshape(bs, -1, self.num_classes)]

            # Mask coefficients
            if self.cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
                mask = self.mask_layer(bbox_x, bbox.detach()) if self.cfg.STMASK.FC.FCB_USE_DCN_MASK else self.mask_layer(bbox_x)
                # Activation function is Tanh
                preds['mask_coeff'] += [torch.tanh(mask.permute(0, 2, 3, 1).contiguous().reshape(bs, -1, self.mask_dim))]

            # Tracking
            if self.cfg.MODEL.TRACK_HEADS.TRAIN_TRACK and not self.cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
                track_x = self.track_extra(x)
                track = self.track_layer(track_x, bbox.detach()) if self.cfg.STMASK.FC.FCB_USE_DCN_TRACK else self.track_layer(track_x)
                track = track.permute(0, 2, 3, 1).contiguous().reshape(bs, -1, self.track_dim)
                preds['track'] += [F.normalize(track, dim=-1)]

        for k, v in preds.items():
            preds[k] = torch.cat(v, 1)

        return preds

    def make_priors(self, idx, conv_h, conv_w, device):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
        prior_data = []
        prior_levels = []
        # Iteration order is important (it has to sync up with the convout)
        for j, i in product(range(conv_h), range(conv_w)):
            # +0.5 because priors are in center-size notation
            x = (i + 0.5) / conv_w
            y = (j + 0.5) / conv_h

            for scale in self.pred_scales[idx]:
                for ar in self.pred_aspect_ratios[idx]:
                    # [1, 1/2, 2]
                    ar = sqrt(ar)
                    r = scale / self.pred_scales[idx][0] * 3
                    w = r * ar / conv_w
                    h = r / ar / conv_h

                    prior_data += [x, y, w, h]
                    prior_levels += [idx]

        priors = torch.Tensor(prior_data, device=device).reshape(1, -1, 4).detach()
        priors.requires_grad = False

        prior_levels = torch.Tensor(prior_levels, device=device).reshape(1, -1).detach()
        prior_levels.requires_grad = False

        return priors, prior_levels

