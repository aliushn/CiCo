import torch
import torch.nn as nn

from datasets.config import cfg
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

    def __init__(self, in_channels, deform_groups=1,
                 pred_aspect_ratios=None, pred_scales=None, parent=None):
        super().__init__()

        self.in_channels = in_channels
        if cfg.use_focal_loss:
            self.num_classes = cfg.num_classes
        else:
            self.num_classes = cfg.num_classes + 1
        self.mask_dim = cfg.mask_dim
        self.track_dim = cfg.track_dim
        self.num_priors = len(pred_scales)
        self.deform_groups = deform_groups
        self.pred_aspect_ratios = pred_aspect_ratios
        self.pred_scales = pred_scales
        self.parent = [parent]  # Don't include this in the state dict

        if cfg.use_sipmask:
            self.mask_dim = self.mask_dim * cfg.sipmask_head
        elif cfg.mask_proto_coeff_occlusion:
            self.mask_dim = self.mask_dim * 3
        elif cfg.use_dynamic_mask:
            self.mask_dim = cfg.mask_dim**2 * (cfg.dynamic_mask_head_layers-1) \
                            + cfg.mask_dim * cfg.dynamic_mask_head_layers + 1
            if not cfg.disable_rel_coords:
                self.mask_dim += cfg.mask_dim
        elif cfg.mask_proto_with_levels:
            self.mask_dim = self.mask_dim * 2

        if cfg.train_track and cfg.clip_prediction_mdoule:
            self.clip_frames = cfg.train_dataset.clip_frames
        else:
            self.clip_frames = 1

        if parent is None:

            # init single or multi kernel prediction modules
            self.bbox_layer, self.mask_layer = nn.ModuleList([]), nn.ModuleList([])

            if not cfg.track_by_Gaussian:
                self.track_layer = nn.ModuleList([])

            if cfg.train_centerness:
                self.centerness_layer = nn.ModuleList([])

            if cfg.train_class:
                self.conf_layer = nn.ModuleList([])
            else:
                self.stuff_layer = nn.ModuleList([])

            for k in range(len(cfg.pred_conv_kernels)):
                kernel_size = cfg.pred_conv_kernels[k]
                padding = [(kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2]

                if cfg.train_centerness:
                    # self.DIoU_layer.append(nn.Conv2d(self.out_channels, self.num_priors, **cfg.head_layer_params[k]))
                    self.centerness_layer.append(nn.Conv2d(self.in_channels, self.num_priors*self.clip_frames,
                                                           kernel_size=kernel_size, padding=padding))

                if cfg.train_boxes:
                    self.bbox_layer.append(nn.Conv2d(self.in_channels, self.num_priors*4*self.clip_frames,
                                                     kernel_size=kernel_size, padding=padding))

                if cfg.train_class:
                    if cfg.use_dcn_class:
                        self.conf_layer.append(FeatureAlign(self.in_channels,
                                                            self.num_priors * self.num_classes,
                                                            kernel_size=kernel_size,
                                                            deformable_groups=self.deform_groups,
                                                            use_pred_offset=cfg.use_pred_offset,
                                                            use_random_offset=cfg.use_random_offset))
                    else:
                        self.conf_layer.append(nn.Conv2d(self.in_channels, self.num_priors * self.num_classes,
                                                         kernel_size=kernel_size, padding=padding))
                else:
                    self.stuff_layer.append(nn.Conv2d(self.in_channels, self.num_priors,
                                                      kernel_size=kernel_size, padding=padding))

                if cfg.train_masks:
                    if cfg.use_dcn_mask:
                        self.mask_layer.append(FeatureAlign(self.in_channels,
                                                            self.num_priors * self.mask_dim,
                                                            kernel_size=kernel_size,
                                                            deformable_groups=self.deform_groups,
                                                            use_pred_offset=cfg.use_pred_offset,
                                                            use_random_offset=cfg.use_random_offset
                                                            ))
                    else:
                        self.mask_layer.append(nn.Conv2d(self.in_channels, self.num_priors*self.mask_dim,
                                                         kernel_size=kernel_size, padding=padding))

                if cfg.train_track and not cfg.track_by_Gaussian:
                    if cfg.use_dcn_track:
                        self.track_layer.append(FeatureAlign(self.in_channels,
                                                             self.num_priors * self.track_dim,
                                                             kernel_size=kernel_size,
                                                             deformable_groups=self.deform_groups,
                                                             use_pred_offset=cfg.use_pred_offset,
                                                             use_random_offset=cfg.use_random_offset
                                                             ))
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
                        nn.BatchNorm2d(in_channels),
                        nn.ReLU(inplace=True)
                    ] for _ in range(num_layers)], []))

            if cfg.train_class:
                self.conf_extra = make_extra(cfg.extra_layers[0], self.in_channels)
            else:
                self.stuff_extra = make_extra(cfg.extra_layers[0], self.in_channels)
            self.bbox_extra = make_extra(cfg.extra_layers[1], self.in_channels)
            if cfg.train_track and not cfg.track_by_Gaussian:
                self.track_extra = make_extra(cfg.extra_layers[-1], self.in_channels)

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

        if cfg.train_class:
            conf_x = src.conf_extra(x)
            conf = []
        else:
            stuff_x = src.stuff_extra(x)
            stuff = []
        bbox_x = src.bbox_extra(x)
        if cfg.train_track and not cfg.track_by_Gaussian:
            track_x = src.track_extra(x)
            track = []

        bbox, centerness_data, mask, = [], [], []
        for k in range(len(cfg.pred_conv_kernels)):
            if cfg.train_centerness:
                centerness_cur = src.centerness_layer[k](bbox_x)
                centerness_data.append(centerness_cur.permute(0, 2, 3, 1).contiguous())

            bbox_cur = src.bbox_layer[k](bbox_x)
            bbox.append(bbox_cur.permute(0, 2, 3, 1).contiguous())

            if cfg.train_class:
                if cfg.use_dcn_class:
                    conf_cur = src.conf_layer[k](conf_x, bbox_cur.detach())
                else:
                    conf_cur = src.conf_layer[k](conf_x)
                conf.append(conf_cur.permute(0, 2, 3, 1).contiguous())
            else:
                stuff_cur = src.stuff_layer[k](stuff_x)
                stuff.append(stuff_cur.permute(0, 2, 3, 1).contiguous())

            if cfg.train_masks:
                if cfg.use_dcn_mask:
                    mask_cur = src.mask_layer[k](bbox_x, bbox_cur.detach())
                else:
                    mask_cur = src.mask_layer[k](bbox_x)
                mask.append(mask_cur.permute(0, 2, 3, 1).contiguous())

            if cfg.train_track and not cfg.track_by_Gaussian:
                if cfg.use_dcn_track:
                    track_cur = src.track_layer[k](track_x, bbox_cur.detach())
                else:
                    track_cur = src.track_layer[k](track_x)
                track.append(track_cur.permute(0, 2, 3, 1).contiguous())

        priors, prior_levels = self.make_priors(idx, x.size(2), x.size(3), x.device)  #[1, h*w*num_priors*num_ratios, 4]
        preds = {'priors': priors, 'prior_levels': prior_levels}

        # cat for all anchors
        if cfg.train_boxes:
            bbox = torch.cat(bbox, dim=-1).view(x.size(0), -1, 4*self.clip_frames)
            if cfg.use_yolo_regressors:
                bbox[:, :, :2] = torch.sigmoid(bbox[:, :, :2]) - 0.5
                bbox[:, :, 0] /= conv_w
                bbox[:, :, 1] /= conv_h
            preds['loc'] = bbox
        if cfg.train_centerness:
            centerness_data = torch.cat(centerness_data, dim=1).view(x.size(0), -1, self.clip_frames)
            preds['centerness'] = torch.sigmoid(centerness_data)
        if cfg.train_class:
            preds['conf'] = torch.cat(conf, dim=-1).view(x.size(0), -1, src.num_classes)
        else:
            preds['stuff'] = torch.cat(stuff, dim=-1).view(x.size(0), -1, 1)

        if cfg.train_masks:
            preds['mask_coeff'] = torch.cat(mask, dim=-1).view(x.size(0), -1, src.mask_dim)
        if cfg.train_track and not cfg.track_by_Gaussian:
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

                for ars in self.pred_aspect_ratios:
                    for ar in ars:
                        for scale in self.pred_scales:
                            # [h, w]: [3, 3], [3, 5], [5, 3]
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
