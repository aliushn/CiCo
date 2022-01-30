import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from math import sqrt
# from ..visualization_temporal import display_cubic_weights, display_pixle_similarity


class ClipPredictionModule(nn.Module):
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

    def __init__(self, cfg, in_channels, mask_dim=32, deform_groups=1):
        super().__init__()

        self.cfg = cfg
        self.in_channels = in_channels
        self.num_classes = cfg.DATASETS.NUM_CLASSES if cfg.MODEL.CLASS_HEADS.USE_FOCAL_LOSS else cfg.DATASETS.NUM_CLASSES + 1
        self.mask_dim = mask_dim
        self.track_dim = cfg.MODEL.TRACK_HEADS.TRACK_DIM
        self.pred_aspect_ratios = cfg.MODEL.BACKBONE.PRED_ASPECT_RATIOS
        self.pred_scales = cfg.MODEL.BACKBONE.PRED_SCALES
        self.num_priors = len(self.pred_aspect_ratios[0]) * len(self.pred_scales[0])
        self.deform_groups = deform_groups
        self.clip_frames = cfg.SOLVER.NUM_CLIP_FRAMES

        # generate anchors
        # self.img_h = ((max(cfg.INPUT.MIN_SIZE_TRAIN)-1)//32+1) * 32
        # self.img_w = cfg.INPUT.MAX_SIZE_TRAIN
        # layers_idx = range(min(cfg.MODEL.BACKBONE.SELECTED_LAYERS), min(cfg.MODEL.BACKBONE.SELECTED_LAYERS)
        #                    + len(self.pred_scales))
        # self.priors, self.prior_levels = [], []
        # for idx, ldx in enumerate(layers_idx):
        #     priors, prior_levels = self.make_priors(idx, self.img_h//(2**(ldx+2)), self.img_w//(2**(ldx+2)))
        #     self.priors += [priors]
        #     self.prior_levels += [prior_levels]
        # self.priors = torch.cat(self.priors, dim=1).cpu()
        # self.prior_levels = torch.cat(self.prior_levels, dim=1).cpu()

        if cfg.CiCo.CPH.CUBIC_MODE:
            self.conv_type, conv = 'conv3d', nn.Conv3d
            if self.cfg.CiCo.CPH.MATCHER_CENTER:
                kernel_size, padding, stride = (self.clip_frames, 3, 3), (0, 1, 1), (1, 1, 1)
            else:
                kernel_size, padding = cfg.CiCo.CPH.LAYER_KERNEL_SIZE, cfg.CiCo.CPH.LAYER_PADDING
                stride = cfg.CiCo.CPH.LAYER_STRIDE
        else:
            self.conv_type, conv = 'conv2d', nn.Conv2d
            kernel_size, padding, stride = (3, 3), (1, 1), (1, 1)

        if cfg.MODEL.MASK_HEADS.TRAIN_MASKS and cfg.CiCo.CPH.CIRCUMSCRIBED_BOXES:
            self.bbox_layer = conv(self.in_channels, self.num_priors*4,
                                   kernel_size=kernel_size, padding=padding, stride=stride)
        else:
            self.bbox_layer = conv(self.in_channels, self.num_priors*4*self.clip_frames,
                                   kernel_size=kernel_size, padding=padding, stride=stride)

        if cfg.MODEL.BOX_HEADS.TRAIN_CENTERNESS:
            self.centerness_layer = conv(self.in_channels, self.num_priors,
                                         kernel_size=kernel_size, padding=padding, stride=stride)

        if cfg.MODEL.CLASS_HEADS.TRAIN_CLASS:
            self.conf_layer = conv(self.in_channels, self.num_priors*self.num_classes,
                                   kernel_size=kernel_size, padding=padding, stride=stride)

        if cfg.MODEL.TRACK_HEADS.TRAIN_TRACK and not cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
            self.track_layer = conv(self.in_channels, self.num_priors*self.track_dim,
                                    kernel_size=kernel_size, padding=padding, stride=stride)

        if cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
            self.mask_layer = conv(self.in_channels, self.num_priors*self.mask_dim,
                                   kernel_size=kernel_size, padding=padding, stride=stride)

        # What is this ugly lambda doing in the middle of all this clean prediction module code?
        self.conv_tower_type, conv_tower = ('conv3d', nn.Conv3d) if cfg.CiCo.CPH.TOWER_CUBIC_MODE else ('conv2d', nn.Conv2d)
        def make_extra(num_layers, in_channels):
            if num_layers == 0:
                return lambda x: x
            else:
                # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
                return nn.Sequential(*sum([[
                    conv_tower(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                ] for _ in range(num_layers)], []))

        self.bbox_extra = make_extra(cfg.MODEL.BOX_HEADS.TOWER_LAYERS, self.in_channels)
        self.conf_extra = make_extra(cfg.MODEL.CLASS_HEADS.TOWER_LAYERS, self.in_channels)
        if cfg.MODEL.TRACK_HEADS.TRAIN_TRACK and not cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
            self.track_extra = make_extra(cfg.MODEL.TRACK_HEADS.TOWER_LAYERS, self.in_channels)

    def forward(self, fpn_outs):
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
            _, c, conv_h, conv_w = fpn_outs[idx].size()
            n_proposals = conv_h * conv_w * self.num_priors
            priors, prior_levels = self.make_priors(idx, conv_h, conv_w, fpn_outs[idx].device)
            preds['priors'] += [priors]
            preds['prior_levels'] += [prior_levels]

            x = fpn_outs[idx]
            x = self.unfold_tensor(x) if self.conv_tower_type == 'conv3d' else x

            # bounding boxes regression
            bbox_x = self.bbox_extra(x)
            bbox_x = self.unfold_tensor(bbox_x) if self.conv_type != self.conv_tower_type else bbox_x
            bbox = self.permute_channels(self.bbox_layer(bbox_x))
            bs = bbox.size(0) * bbox.size(1) if bbox.dim() == 5 else bbox.size(0)
            preds['loc'] += [bbox.reshape(bs, n_proposals, -1)]

            # Centerness for Boxes
            if self.cfg.MODEL.BOX_HEADS.TRAIN_CENTERNESS:
                centerness = self.permute_channels(self.centerness_layer(bbox_x).sigmoid())
                preds['centerness'] += [centerness.reshape(bs, n_proposals, -1)]

            # Classification
            if self.cfg.MODEL.CLASS_HEADS.TRAIN_CLASS:
                conf_x = self.conf_extra(x)
                conf_x = self.unfold_tensor(conf_x) if self.conv_type != self.conv_tower_type else conf_x
                conf = self.permute_channels(self.conf_layer(conf_x))
                preds['conf'] += [conf.reshape(bs, n_proposals, -1)]

            # Mask coefficients
            if self.cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
                mask = self.permute_channels(self.mask_layer(bbox_x).tanh())
                # Activation function is Tanh
                preds['mask_coeff'] += [mask.reshape(bs, n_proposals, -1)]

            # Tracking
            if self.cfg.MODEL.TRACK_HEADS.TRAIN_TRACK and not self.cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
                track_x = self.track_extra(x)
                track_x = self.unfold_tensor(track_x) if self.conv_type != self.conv_tower_type else track_x
                track = self.permute_channels(self.track_layer(track_x))
                preds['track'] += [F.normalize(track.reshape(bs, n_proposals, -1), dim=-1)]

        for k, v in preds.items():
            preds[k] = torch.cat(v, 1)

        return preds

    def unfold_tensor(self, x):
        _, c, h, w = x.size()
        return x.reshape(-1, self.clip_frames, c, h, w).transpose(1, 2)

    def permute_channels(self, x):
        return x.permute(0, 2, 3, 1).contiguous() if x.dim() == 4 else x.permute(0, 2, 3, 4, 1).contiguous()

    def make_priors(self, idx, conv_h, conv_w, device=None):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
        # with timer.env('make priors'):
        prior_data = []
        prior_levels = []
        # Iteration order is important (it has to sync up with the convout)
        for j, i in product(range(conv_h), range(conv_w)):
            # +0.5 because priors are in center-size notation
            x = (i + 0.5) / conv_w
            y = (j + 0.5) / conv_h

            for scale in self.pred_scales[idx]:
                for ar in self.pred_aspect_ratios[idx]:
                    # [1, 0.5, 2]
                    ar = sqrt(ar)
                    r = scale / self.pred_scales[idx][0] * 3
                    w = r * ar / conv_w
                    h = r / ar / conv_h

                    prior_data += [x, y, w, h]
                    prior_levels += [idx]

        if device is None:
            priors = torch.Tensor(prior_data).reshape(1, -1, 4)
            prior_levels = torch.Tensor(prior_levels).reshape(1, -1)
        else:
            priors = torch.Tensor(prior_data, device=device).reshape(1, -1, 4).detach()
            priors.requires_grad = False
            prior_levels = torch.Tensor(prior_levels, device=device).reshape(1, -1).detach()
            prior_levels.requires_grad = False

        return priors, prior_levels


