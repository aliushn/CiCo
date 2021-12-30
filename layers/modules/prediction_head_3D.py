import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from utils import timer
from itertools import product
from math import sqrt
from mmcv.ops import DeformConv2d
from ..visualization_temporal import display_cubic_weights, display_pixle_similarity
from layers.utils import correlate_operator


class PredictionModule_3D(nn.Module):
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
        self.clip_frames = cfg.SOLVER.NUM_CLIP_FRAMES
        self.conf_feat_dim = cfg.MODEL.CLASS_HEADS.INTERCLIPS_CLAFEAT_DIM

        # generate anchors
        self.img_h = ((max(cfg.INPUT.MIN_SIZE_TRAIN)-1)//32+1) * 32
        self.img_w = cfg.INPUT.MAX_SIZE_TRAIN
        layers_idx = range(min(cfg.MODEL.BACKBONE.SELECTED_LAYERS), min(cfg.MODEL.BACKBONE.SELECTED_LAYERS)+len(self.pred_scales))
        self.priors_list, self.prior_levels_list = [], []
        for idx, ldx in enumerate(layers_idx):
            priors, prior_levels = self.make_priors(idx, self.img_h//(2**(ldx+2)), self.img_w//(2**(ldx+2)))
            self.priors_list += [priors]
            self.prior_levels_list += [prior_levels]
        self.priors_list = torch.cat(self.priors_list, dim=1).cpu()
        self.prior_levels_list = torch.cat(self.prior_levels_list, dim=1).cpu()

        if cfg.MODEL.MASK_HEADS.USE_SIPMASK:
            self.mask_dim = self.mask_dim * cfg.MODEL.MASK_HEADS.SIPMASK_HEAD
        elif cfg.MODEL.MASK_HEADS.PROTO_COEFF_OCCLUSION:
            self.mask_dim = self.mask_dim * 2
        elif cfg.MODEL.MASK_HEADS.USE_DYNAMIC_MASK:
            self.mask_dim = self.mask_dim ** 2 * (cfg.MODEL.MASK_HEADS.DYNAMIC_MASK_HEAD_LAYERS - 1) \
                            + self.mask_dim * cfg.MODEL.MASK_HEADS.DYNAMIC_MASK_HEAD_LAYERS + 1
            if not cfg.MODEL.MASK_HEADS.DISABLE_REL_COORDS:
                self.mask_dim += cfg.MODEL.MASK_HEADS.MASK_DIM
        else:
            self.mask_dim = self.mask_dim

        padding, stride = (0, 1, 1), (1, 1, 1)
        if self.cfg.CiCo.FRAME2CLIP_EXPAND_PROPOSALS:
            kernel_size = (1, 3, 3)
        elif self.cfg.CiCo.MATCHER_CENTER:
            kernel_size = (self.clip_frames, 3, 3)
        else:
            kernel_size, padding = cfg.CiCo.CPH_LAYER_KERNEL_SIZE, cfg.CiCo.CPH_LAYER_PADDING
            stride = cfg.CiCo.CPH_LAYER_STRIDE

        if cfg.MODEL.MASK_HEADS.TRAIN_MASKS and cfg.MODEL.PREDICTION_HEADS.CIRCUMSCRIBED_BOXES:
            self.bbox_layer = nn.Conv3d(self.in_channels, self.num_priors*4,
                                        kernel_size=kernel_size, padding=padding, stride=stride)
        else:
            self.bbox_layer = nn.Conv3d(self.in_channels, self.num_priors*4*self.clip_frames,
                                        kernel_size=kernel_size, padding=padding, stride=stride)

        if cfg.MODEL.BOX_HEADS.TRAIN_CENTERNESS:
            self.centerness_layer = nn.Conv3d(self.in_channels, self.num_priors,
                                              kernel_size=kernel_size, padding=padding, stride=stride)

        if cfg.MODEL.CLASS_HEADS.TRAIN_CLASS:
            if cfg.MODEL.CLASS_HEADS.TRAIN_INTERCLIPS_CLASS:
                self.conf_layer = nn.Conv3d(self.in_channels, self.num_priors,
                                            kernel_size=kernel_size, padding=padding, stride=stride)
                self.conf_feature_layer = nn.Conv3d(self.in_channels, self.num_priors*self.conf_feat_dim,
                                                    kernel_size=kernel_size, padding=padding, stride=stride)
            else:
                self.conf_layer = nn.Conv3d(self.in_channels, self.num_priors*self.num_classes,
                                            kernel_size=kernel_size, padding=padding, stride=stride)

        if cfg.MODEL.TRACK_HEADS.TRAIN_TRACK and not cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
            self.track_layer = nn.Conv3d(self.in_channels, self.num_priors*self.track_dim,
                                         kernel_size=kernel_size, padding=padding, stride=stride)

        if cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
            self.mask_layer = nn.Conv3d(self.in_channels, self.num_priors*self.mask_dim,
                                        kernel_size=kernel_size, padding=padding, stride=stride)

        # What is this ugly lambda doing in the middle of all this clean prediction module code?
        def make_extra(num_layers, in_channels):
            if num_layers == 0:
                return lambda x: x
            else:
                # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
                if cfg.CiCo.CPH_TOWER133:
                    kernel_size, padding = (1, 3, 3), (0, 1, 1)
                else:
                    kernel_size, padding = 3, 1
                return nn.Sequential(*sum([[
                    nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, padding=padding),
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

        preds = dict()
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
            if self.cfg.MODEL.CLASS_HEADS.TRAIN_INTERCLIPS_CLASS:
                preds['conf_feat'] = []

        for idx in range(len(fpn_outs)):
            if self.cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE:
                _, c, h, w = fpn_outs[idx].size()
                fpn_outs_clip = fpn_outs[idx].reshape(-1, self.clip_frames, c, h, w)
                x = fpn_outs_clip.permute(0, 2, 1, 3, 4).contiguous()
            else:
                x = fpn_outs[idx]

            bs, _, T, conv_h, conv_w = x.size()
            n_proposals = conv_h*conv_w*self.num_priors
            bbox_x = self.bbox_extra(x)
            bbox = self.bbox_layer(bbox_x).permute(0, 2, 3, 4, 1).contiguous()
            T_out = bbox.size(1)
            preds['loc'] += [bbox.reshape(bs*T_out, n_proposals, -1)]

            # Centerness for Boxes
            if self.cfg.MODEL.BOX_HEADS.TRAIN_CENTERNESS:
                centerness = self.centerness_layer(bbox_x).permute(0, 2, 3, 4, 1).contiguous()
                preds['centerness'] += [torch.sigmoid(centerness.reshape(bs*T_out, n_proposals, -1))]

            # Classification
            if self.cfg.MODEL.CLASS_HEADS.TRAIN_CLASS:
                conf_x = self.conf_extra(x)
                conf = self.conf_layer(conf_x).permute(0, 2, 3, 4, 1).contiguous()
                preds['conf'] += [conf.reshape(bs*T_out, n_proposals, -1)]
                if self.cfg.MODEL.CLASS_HEADS.TRAIN_INTERCLIPS_CLASS:
                    conf_feature = self.conf_feature_layer(conf_x).permute(0, 2, 3, 4, 1).contiguous()
                    preds['conf_feat'] += [conf_feature.reshape(bs*T_out, n_proposals, -1)]
                # display_pixle_similarity(conf, bbox_x, img_meta[0], idx=idx)

            # Mask coefficients
            if self.cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
                mask = self.mask_layer(bbox_x).permute(0, 2, 3, 4, 1).contiguous()
                # Activation function is Tanh
                preds['mask_coeff'] += [mask.tanh().reshape(bs*T_out, n_proposals, -1)]

            # Tracking
            if self.cfg.MODEL.TRACK_HEADS.TRAIN_TRACK and not self.cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
                track_x = self.track_extra(x)
                track = self.track_layer(track_x).permute(0, 2, 3, 4, 1).contiguous()
                preds['track'] += [F.normalize(track.reshape(bs*T_out, n_proposals, -1), dim=-1)]

        for k, v in preds.items():
            preds[k] = torch.cat(v, 1)

        preds['priors'] = self.priors_list.to(x.device).detach()
        preds['prior_levels'] = self.prior_levels_list.to(x.device).detach()
        return preds

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


