import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from .SpatioTemporalBlock import SpatioTemporalBlock
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
        self.clip_frames = cfg.SOLVER.NUM_CLIP_FRAMES if cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE else 1
        self.conf_feat_dim = cfg.MODEL.CLASS_HEADS.INTERCLIPS_CLAFEAT_DIM
        self.expand_proposals_clip = cfg.STR.ST_CONSISTENCY.EXPAND_PROPOSALS_CLIP

        if cfg.MODEL.MASK_HEADS.USE_SIPMASK:
            self.mask_dim = self.mask_dim * cfg.MODEL.MASK_HEADS.SIPMASK_HEAD
        elif cfg.MODEL.MASK_HEADS.PROTO_COEFF_OCCLUSION:
            self.mask_dim = self.mask_dim * 2
        elif cfg.MODEL.MASK_HEADS.USE_DYNAMIC_MASK:
            self.mask_dim = self.mask_dim ** 2 * (cfg.MODEL.MASK_HEADS.DYNAMIC_MASK_HEAD_LAYERS - 1) \
                            + self.mask_dim * cfg.MODEL.MASK_HEADS.DYNAMIC_MASK_HEAD_LAYERS + 1
            if not cfg.MODEL.MASK_HEADS.DISABLE_REL_COORDS:
                self.mask_dim += self.mask_dim * 2
        else:
            self.mask_dim = self.mask_dim

        if self.expand_proposals_clip:
            kernel_size, padding, stride = (1, 3, 3), (0, 1, 1), (1, 1, 1)
        else:
            kernel_size, padding, stride = 3, (0, 1, 1), (2, 1, 1)

        offset_channels = 18
        if cfg.MODEL.PREDICTION_HEADS.USE_SPATIO_DCN:
            self.bbox_spatio_offset = nn.Conv2d(self.in_channels, offset_channels, 1, bias=False)
            self.bbox_spatio_layer = DeformConv2d(self.in_channels,
                                                  self.num_priors*4*self.clip_frames,
                                                  kernel_size=(3, 3),
                                                  padding=(1, 1))
        if cfg.MODEL.MASK_HEADS.TRAIN_MASKS and cfg.MODEL.PREDICTION_HEADS.CIRCUMSCRIBED_BOXES:
            self.bbox_layer = nn.Conv3d(self.in_channels, self.num_priors*4,
                                        kernel_size=3, padding=(0, 1, 1))
        else:
            self.bbox_layer = nn.Conv3d(self.in_channels, self.num_priors*4*self.clip_frames,
                                        kernel_size=kernel_size, padding=padding, stride=stride)

        if cfg.MODEL.BOX_HEADS.TRAIN_CENTERNESS:
            if cfg.MODEL.PREDICTION_HEADS.USE_SPATIO_DCN:
                self.centerness_spatio_offset = nn.Conv2d(self.in_channels, offset_channels, 1, bias=False)
                self.centerness_spatio_layer = DeformConv2d(self.in_channels,
                                                            self.num_priors,
                                                            kernel_size=(3, 3),
                                                            padding=(1, 1))

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
            if cfg.MODEL.MASK_HEADS.USE_SPATIO_DCN:
                self.mask_spatio_offset = nn.Conv2d(self.num_priors*4, offset_channels, 1, bias=False)
                self.mask_spatio_layer = DeformConv2d(self.in_channels,
                                                      self.num_priors*self.mask_dim,
                                                      kernel_size=(3, 3),
                                                      padding=(1, 1))
            else:
                self.mask_layer = nn.Conv3d(self.in_channels, self.num_priors*self.mask_dim,
                                            kernel_size=kernel_size, padding=padding, stride=stride)

        # What is this ugly lambda doing in the middle of all this clean prediction module code?
        def make_extra(num_layers, in_channels):
            if num_layers == 0:
                return lambda x: x
            else:
                if cfg.MODEL.PREDICTION_HEADS.CUBIC_SPATIOTEMPORAL_BLOCK:
                    return nn.Sequential(*sum([[
                        SpatioTemporalBlock(in_channels)
                    ] for _ in range(num_layers)], []))
                else:
                    # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
                    if cfg.STR.ST_CONSISTENCY.CPH_WITH_TOWER133:
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
            if self.cfg.MODEL.CLASS_HEADS.TRAIN_INTERCLIPS_CLASS:
                preds['conf_feat'] = []

        x_list, bbox_x_list, conf_x_list, track_x_list = [], [], [], []
        for idx in range(len(fpn_outs)):
            if self.cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE:
                _, c, h, w = fpn_outs[idx].size()
                fpn_outs_clip = fpn_outs[idx].reshape(-1, self.clip_frames, c, h, w)
                if self.cfg.MODEL.PREDICTION_HEADS.CUBIC_3D_MODE:
                    x = fpn_outs_clip.permute(0, 2, 1, 3, 4).contiguous()
                else:
                    if self.init_cubic_mode == 'tsm':
                        x = []
                        fold = int(0.5/(self.clip_frames-1)*c)
                        left, right = 0, fold
                        for frame_idx in range(-(self.clip_frames//2), self.clip_frames//2+1):
                            x.append(fpn_outs_clip[:, frame_idx, left:right])
                            left = right
                            right = right + int(0.5*c) if frame_idx+1 == 0 else right+fold

                    else:
                        # stack frames in channels
                        x = [self.fpn_reduced_channels(fpn_outs_clip[:, 0])] \
                            if self.init_cubic_mode == 'reduced' else [fpn_outs_clip[:, 0]]
                        for frame_idx in range(self.clip_frames-1):
                            if self.cfg.MODEL.PREDICTION_HEADS.CUBIC_CORRELATION_MODE:
                                x += [correlate_operator(fpn_outs_clip[:, frame_idx].contiguous(),
                                                         fpn_outs_clip[:, frame_idx+1].contiguous(),
                                                         patch_size=self.correlation_patch_size,
                                                         kernel_size=1)]

                            cur_fpn_x = self.fpn_reduced_channels(fpn_outs_clip[:, frame_idx+1]) \
                                if self.init_cubic_mode == 'reduced' else fpn_outs_clip[:, frame_idx+1]
                            x.append(cur_fpn_x)
                    x = torch.cat(x, dim=1)
            else:
                x = fpn_outs[idx]
            # flip temporal information
            # x = 0.5*x + 0.5*torch.flip(x, [2])
            x_list += [x]

            bbox_x_list.append(self.bbox_extra(x))
            if self.cfg.MODEL.CLASS_HEADS.TRAIN_CLASS:
                conf_x_list.append(self.conf_extra(x))
            # Tracking
            if self.cfg.MODEL.TRACK_HEADS.TRAIN_TRACK and not self.cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
                track_x_list.append(self.track_extra(x))

            # generate anchors
            bs, _, T, conv_h, conv_w = x.size()
            priors, prior_levels = self.make_priors(idx, conv_h, conv_w, x.device)
            preds['priors'] += [priors]
            preds['prior_levels'] += [prior_levels]

        for idx in range(len(fpn_outs)):
            bs, _, T, conv_h, conv_w = x_list[idx].size()
            n_proposals = conv_h*conv_w*self.num_priors
            bbox_x_4D = bbox_x_list[idx].permute(0, 2, 1, 3, 4).contiguous().reshape(bs*T, -1, conv_h, conv_w)
            if self.cfg.MODEL.PREDICTION_HEADS.USE_SPATIO_DCN:
                bbox = self.bbox_spatio_layer(bbox_x_4D, self.bbox_spatio_offset(bbox_x_4D))
                bbox = bbox.reshape(bs, T, -1, conv_h, conv_w).permute(0, 2, 1, 3, 4).contiguous()
                bbox = self.bbox_layer(bbox)
            else:
                bbox = self.bbox_layer(bbox_x_list[idx])
            bbox_4D = bbox.permute(0, 2, 1, 3, 4).contiguous().reshape(bs*bbox.size(2), -1, conv_h, conv_w).detach()
            T_out = bbox.size(2)
            preds['loc'] += [bbox.permute(0, 2, 3, 4, 1).contiguous().reshape(bs*T_out, n_proposals, -1)]

            # Centerness for Boxes
            if self.cfg.MODEL.BOX_HEADS.TRAIN_CENTERNESS:
                if self.cfg.MODEL.PREDICTION_HEADS.USE_SPATIO_DCN:
                    centerness = self.centerness_spatio_layer(bbox_x_4D, self.centerness_spatio_offset(bbox_x_4D))
                    centerness = centerness.reshape(bs, T, -1, conv_h, conv_w).permute(0, 2, 1, 3, 4).contiguous()
                    centerness = self.centerness_layer(centerness).permute(0, 2, 3, 4, 1).contiguous()
                else:
                    centerness = self.centerness_layer(bbox_x_list[idx]).permute(0, 2, 3, 4, 1).contiguous()
                preds['centerness'] += [torch.sigmoid(centerness.reshape(bs*T_out, n_proposals, -1))]

            # Classification
            if self.cfg.MODEL.CLASS_HEADS.TRAIN_CLASS:
                conf = self.conf_layer(conf_x_list[idx]).permute(0, 2, 3, 4, 1).contiguous()
                preds['conf'] += [conf.reshape(bs*T_out, n_proposals, -1)]
                if self.cfg.MODEL.CLASS_HEADS.TRAIN_INTERCLIPS_CLASS:
                    conf_feature = self.conf_feature_layer(conf_x_list[idx]).permute(0, 2, 3, 4, 1).contiguous()
                    preds['conf_feat'] += [conf_feature.reshape(bs*T_out, n_proposals, -1)]
                # display_pixle_similarity(conf, bbox_x, img_meta[0], idx=idx)

            # Mask coefficients
            if self.cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
                if self.cfg.MODEL.MASK_HEADS.USE_SPATIO_DCN:
                    mask = self.mask_spatio_layer(bbox_x_4D, self.mask_spatio_offset(bbox_4D))
                    mask = mask.reshape(bs, T, -1, conv_h, conv_w).permute(0, 2, 1, 3, 4).contiguous()
                    mask = self.mask_layer(mask).permute(0, 2, 3, 4, 1).contiguous()
                else:
                    mask = self.mask_layer(bbox_x_list[idx]).permute(0, 2, 3, 4, 1).contiguous()
                # Activation function is Tanh
                preds['mask_coeff'] += [mask.tanh().reshape(bs*T_out, n_proposals, -1)]

            # Tracking
            if self.cfg.MODEL.TRACK_HEADS.TRAIN_TRACK and not self.cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
                track = self.track_layer(track_x_list[idx]).permute(0, 2, 3, 4, 1).contiguous()
                preds['track'] += [F.normalize(track.reshape(bs*T_out, -1, self.track_dim), dim=-1)]

            # Visualization
            # display_cubic_weights(bbox_x_list[idx], idx, type=2, name='box_x', img_meta=img_meta[0])
            # display_cubic_weights(conf_x_list[idx], idx, type=2, name='conf_x', img_meta=img_meta[0])
            # display_cubic_weights(track_x_list[idx], idx, type=2, name='track_x', img_meta=img_meta[0])
            display_weights = False
            if display_weights:
                if idx == 0:
                    for jdx in range(0,8,2):
                        display_cubic_weights(self.bbox_extra[jdx].weight, jdx, type=1, name='box_extra')
                        display_cubic_weights(self.conf_extra[jdx].weight, jdx, type=1, name='conf_extra')
                        display_cubic_weights(self.track_extra[jdx].weight, jdx, type=1, name='track_extra')
                    display_cubic_weights(self.bbox_layer.weight, 0, type=1, name='box')
                    display_cubic_weights(self.mask_layer.weight, 0, type=1, name='mask')
                    display_cubic_weights(self.conf_layer.weight, 0, type=1, name='conf')
                    display_cubic_weights(self.track_layer.weight, 0, type=1, name='track')

        for k, v in preds.items():
            preds[k] = torch.cat(v, 1)
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

                for scale in self.pred_scales[idx]:
                    for ar in self.pred_aspect_ratios[idx]:
                        # [1, 0.5, 2]
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


