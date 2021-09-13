import torch
import torch.nn as nn

from .Featurealign import FeatureAlign
from utils import timer
from itertools import product
import torch.nn.functional as F


class PredictionModule_FC(nn.Module):
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

    def __init__(self, cfg, in_channels, deform_groups=1, pred_scales=None, parent=None):
        super().__init__()

        self.cfg = cfg
        self.in_channels = in_channels
        self.num_classes = cfg.DATASETS.NUM_CLASSES if cfg.MODEL.CLASS_HEADS.USE_FOCAL_LOSS else cfg.DATASETS.NUM_CLASSES + 1
        self.mask_dim = cfg.MODEL.MASK_HEADS.MASK_DIM
        self.track_dim = cfg.MODEL.TRACK_HEADS.TRACK_DIM
        self.num_priors = len(pred_scales)
        self.deform_groups = deform_groups
        self.pred_scales = pred_scales
        self.pred_conv_kernels = cfg.STMASK.FC.FCA_CONV_KERNELS
        self.parent = [parent]  # Don't include this in the state dict

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

        self.clip_frames = cfg.SOLVER.NUM_CLIP_FRAMES if cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE else 1

        if parent is None:

            # init single or multi kernel prediction modules
            self.bbox_layer, self.mask_layer = nn.ModuleList([]), nn.ModuleList([])

            if cfg.MODEL.TRACK_HEADS.TRAIN_TRACK and not cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
                self.track_layer = nn.ModuleList([])

            if cfg.MODEL.BOX_HEADS.TRAIN_CENTERNESS:
                self.centerness_layer = nn.ModuleList([])

            if cfg.MODEL.CLASS_HEADS.TRAIN_CLASS:
                self.conf_layer = nn.ModuleList([])
            else:
                self.stuff_layer = nn.ModuleList([])

            for kernel_size in self.pred_conv_kernels:
                padding = [(kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2]
                
                self.bbox_layer.append(nn.Conv2d(self.in_channels, self.num_priors*4*self.clip_frames,
                                                 kernel_size=kernel_size, padding=padding))

                if cfg.MODEL.BOX_HEADS.TRAIN_CENTERNESS:
                    # self.DIoU_layer.append(nn.Conv2d(self.out_channels, self.num_priors, **cfg.head_layer_params[k]))
                    self.centerness_layer.append(nn.Conv2d(self.in_channels, self.num_priors*self.clip_frames,
                                                           kernel_size=kernel_size, padding=padding))

                if cfg.MODEL.CLASS_HEADS.TRAIN_CLASS:
                    if cfg.STMASK.FC.FCB_USE_DCN_CLASS:
                        self.conf_layer.append(FeatureAlign(self.in_channels,
                                                            self.num_priors * self.num_classes,
                                                            kernel_size=kernel_size,
                                                            deformable_groups=self.deform_groups,
                                                            use_pred_offset=cfg.STMASK.FC.FCB_USE_PRED_OFFSET))
                    else:
                        self.conf_layer.append(nn.Conv2d(self.in_channels, self.num_priors * self.num_classes,
                                                         kernel_size=kernel_size, padding=padding))
                else:
                    self.stuff_layer.append(nn.Conv2d(self.in_channels, self.num_priors,
                                                      kernel_size=kernel_size, padding=padding))

                if cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
                    if cfg.STMASK.FC.FCB_USE_DCN_MASK:
                        self.mask_layer.append(FeatureAlign(self.in_channels,
                                                            self.num_priors * self.mask_dim,
                                                            kernel_size=kernel_size,
                                                            deformable_groups=self.deform_groups,
                                                            use_pred_offset=cfg.STMASK.FC.FCB_USE_PRED_OFFSET))
                    else:
                        self.mask_layer.append(nn.Conv2d(self.in_channels, self.num_priors*self.mask_dim,
                                                         kernel_size=kernel_size, padding=padding))

                if cfg.MODEL.TRACK_HEADS.TRAIN_TRACK and not cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
                    if cfg.STMASK.FC.FCB_USE_DCN_TRACK:
                        self.track_layer.append(FeatureAlign(self.in_channels,
                                                             self.num_priors * self.track_dim,
                                                             kernel_size=kernel_size,
                                                             deformable_groups=self.deform_groups,
                                                             use_pred_offset=cfg.STMASK.FC.FCB_USE_PRED_OFFSET))
                    else:
                        self.track_layer.append(nn.Conv2d(self.in_channels,
                                                          self.num_priors * self.track_dim,
                                                          kernel_size=kernel_size, padding=padding))

            # What is this ugly lambda doing in the middle of all this clean prediction module code?
            def make_extra(num_layers, in_channels):
                if num_layers == 0:
                    return lambda x: x
                else:
                    # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
                    return nn.Sequential(*sum([[
                        nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                        # nn.GroupNorm(32, in_channels),
                        # nn.BatchNorm2d(in_channels),
                        nn.ReLU(inplace=True)
                    ] for _ in range(num_layers)], []))

            self.bbox_extra = make_extra(cfg.MODEL.BOX_HEADS.TOWER_LAYERS, self.in_channels)
            self.conf_extra = make_extra(cfg.MODEL.CLASS_HEADS.TOWER_LAYERS, self.in_channels)
            if cfg.MODEL.TRACK_HEADS.TRAIN_TRACK and not cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
                self.track_extra = make_extra(cfg.MODEL.TRACK_HEADS.TOWER_LAYERS, self.in_channels)

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

        batch_size, _, conv_h, conv_w = x.size()

        if self.cfg.MODEL.CLASS_HEADS.TRAIN_CLASS:
            conf_x = src.conf_extra(x)
            conf, stuff = [], []
        bbox_x = src.bbox_extra(x)
        if self.cfg.MODEL.TRACK_HEADS.TRAIN_TRACK and not self.cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
            track_x = src.track_extra(x)
            track = []

        bbox, centerness_data, mask, = [], [], []
        for k in range(len(self.pred_conv_kernels)):
            if self.cfg.MODEL.BOX_HEADS.TRAIN_CENTERNESS:
                centerness_cur = src.centerness_layer[k](bbox_x)
                centerness_data.append(centerness_cur.permute(0, 2, 3, 1).contiguous())

            bbox_cur = src.bbox_layer[k](bbox_x)
            bbox.append(bbox_cur.permute(0, 2, 3, 1).contiguous())

            if self.cfg.MODEL.CLASS_HEADS.TRAIN_CLASS:
                conf_cur = src.conf_layer[k](conf_x, bbox_cur.detach()) if self.cfg.STMASK.FC.FCB_USE_DCN_CLASS else src.conf_layer[k](conf_x)
                conf.append(conf_cur.permute(0, 2, 3, 1).contiguous())
            else:
                stuff_cur = src.stuff_layer[k](conf_x)
                stuff.append(stuff_cur.permute(0, 2, 3, 1).contiguous())

            if self.cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
                mask_cur = src.mask_layer[k](bbox_x, bbox_cur.detach()) if self.cfg.STMASK.FC.FCB_USE_DCN_MASK else src.mask_layer[k](bbox_x)
                mask.append(mask_cur.permute(0, 2, 3, 1).contiguous())

            if self.cfg.MODEL.TRACK_HEADS.TRAIN_TRACK and not self.cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
                track_cur = src.track_layer[k](track_x, bbox_cur.detach()) if self.cfg.STMASK.FC.FCB_USE_DCN_TRACK else src.track_layer[k](track_x)
                track.append(track_cur.permute(0, 2, 3, 1).contiguous())

        priors, prior_levels = self.make_priors(idx, x.size(2), x.size(3), x.device)  #[1, h*w*num_priors*num_ratios, 4]
        preds = {'priors': priors, 'prior_levels': prior_levels}

        # cat for all anchors
        bbox = torch.cat(bbox, dim=-1).view(x.size(0), -1, 4*self.clip_frames)
        preds['loc'] = bbox
        if self.cfg.MODEL.BOX_HEADS.TRAIN_CENTERNESS:
            preds['centerness'] = torch.cat(centerness_data, dim=1).view(x.size(0), -1, self.clip_frames).sigmoid()
        if self.cfg.MODEL.CLASS_HEADS.TRAIN_CLASS:
            preds['conf'] = torch.cat(conf, dim=-1).view(x.size(0), -1, src.num_classes)
        else:
            preds['stuff'] = torch.cat(stuff, dim=-1).view(x.size(0), -1, 1)
        if self.cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
            preds['mask_coeff'] = torch.tanh(torch.cat(mask, dim=-1).view(x.size(0), -1, src.mask_dim))
        if self.cfg.MODEL.TRACK_HEADS.TRAIN_TRACK and not self.cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
            track = torch.cat(track, dim=-1).view(x.size(0), -1, src.track_dim)
            preds['track'] = F.normalize(track, dim=-1)

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

                for ar in self.pred_conv_kernels:
                    # Replece original aspect ratios [1, 0.5, 2] with kernel size [3, 3], [3, 5], [5, 3]
                    for scale in self.pred_scales:
                        arh, arw = ar
                        ratio = scale / self.pred_scales[-1]
                        w = ratio * arw / conv_w
                        h = ratio * arh / conv_h
                        prior_data += [x, y, w, h]
                        prior_levels += [idx]

            priors = torch.clamp(torch.Tensor(prior_data, device=device), min=0, max=1).view(1, -1, 4).detach()
            priors.requires_grad = False

            prior_levels = torch.clamp(torch.Tensor(prior_levels, device=device), min=0, max=1).view(1, -1).detach()
            prior_levels.requires_grad = False

        return priors, prior_levels
