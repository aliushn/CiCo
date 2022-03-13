import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from math import sqrt
# from ..visualization_temporal import display_cubic_weights, display_pixle_similarity


class ClipPredictionModule(nn.Module):
    """
    The clip-level prediction heads include four branches and each branch consists of
    a prediction tower (four convolutional layers) and a final layer. To keep spatial
    location consistency, we adopt a shared prediction tower for mask coefficients branch
    and bounding box regression branch.

    Note that this is slightly different to the module in the paper because the Bottleneck
    block actually has a 3x3 convolution in the middle instead of a 1x1 convolution. Though,
    I really can't be arsed to implement it myself, and, who knows, this might be better.
    Args:
        - in_channels: The input feature size.
        -    mask_dim: The number of mask parameters, including the weights and bias of
                       dynamic filters in FCN
    """

    def __init__(self, cfg, in_channels, mask_dim=32, deform_groups=1):
        super().__init__()

        self.cfg = cfg
        self.in_channels = in_channels
        self.num_classes = cfg.DATASETS.NUM_CLASSES if cfg.MODEL.CLASS_HEADS.USE_FOCAL_LOSS else cfg.DATASETS.NUM_CLASSES + 1
        self.mask_dim = mask_dim
        self.track_dim = cfg.MODEL.TRACK_HEADS.TRACK_DIM
        # aspect_ratios: A list of lists of priorbox aspect ratios (one list per scale).
        self.pred_aspect_ratios = cfg.MODEL.BACKBONE.PRED_ASPECT_RATIOS
        # A list of prior box scales relative to this layer's convsize.
        # For instance: If this layer has convouts of size 30x30 for
        #               an image of size 600x600, the 'default' (scale
        #               of 1) for this layer would produce bounding
        #               boxes with an area of 20x20px. If the scale is
        #               .5 on the other hand, this layer would consider
        #               bounding boxes with area 10x10px, etc.
        self.pred_scales = cfg.MODEL.BACKBONE.PRED_SCALES
        self.num_priors = len(self.pred_aspect_ratios[0]) * len(self.pred_scales[0])
        self.clip_frames = cfg.SOLVER.NUM_CLIP_FRAMES

        CPH = cfg.CiCo.CPH
        # There are two options for CPH: we set the shape of inputs as [bs, c, T, H, W].
        # if T == 1, CPH with 2D convs equals to the original frame-level PH. For  T > 1,
        # if CPH.MATCHER_MULTIPLE is False, the final layer adopts 2D convs and the shape
        # of output is unchanged, otherwise the shape of outputs is determined by the filter
        # shape of the final layer. For example, when T=5, if kernel size, padding and stride
        # are 3*3*3, 1*1*1, and 2*1*1 respectively, the outputs has the shape [bs, c, 3, H, W].
        # In general, you can try to reduce the temporal times of outputs for longer clips (T>7).
        if CPH.CUBIC_MODE and CPH.MATCHER_MULTIPLE:
            self.conv = nn.Conv3d
            kernel_size, padding = cfg.CiCo.CPH.LAYER_KERNEL_SIZE, cfg.CiCo.CPH.LAYER_PADDING
            stride = cfg.CiCo.CPH.LAYER_STRIDE
        else:
            self.conv = nn.Conv2d
            kernel_size, padding, stride = 3, 1, 1

        # ---------------------- Build final layer for each branch ---------------------
        # if CPH.CIRCUMSCRIBED_BOXES is True, directly predict the circumscribed boxes of instances
        # over a video clip, otherwise still predict the individual bounding boxes of instance in all frames
        # of the video clip
        num_bbox = self.num_priors*4 if CPH.CIRCUMSCRIBED_BOXES else self.num_priors*4*self.clip_frames
        self.bbox_layer = self.conv(self.in_channels, num_bbox,
                                    kernel_size=kernel_size, padding=padding, stride=stride)

        if cfg.MODEL.BOX_HEADS.TRAIN_CENTERNESS:
            # predict the centerness of bounding boxes, same as FCOS
            self.centerness_layer = self.conv(self.in_channels, self.num_priors,
                                              kernel_size=kernel_size, padding=padding, stride=stride)

        if cfg.MODEL.CLASS_HEADS.TRAIN_CLASS:
            self.conf_layer = self.conv(self.in_channels, self.num_priors*self.num_classes,
                                        kernel_size=kernel_size, padding=padding, stride=stride)

        if cfg.MODEL.TRACK_HEADS.TRAIN_TRACK and not cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
            # predict box-level embeddings for tracking
            self.track_layer = self.conv(self.in_channels, self.num_priors*self.track_dim,
                                         kernel_size=kernel_size, padding=padding, stride=stride)

        if cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
            self.mask_layer = self.conv(self.in_channels, self.num_priors*self.mask_dim,
                                        kernel_size=kernel_size, padding=padding, stride=stride)

        # ---------------------- Build prediction tower for each branch ---------------------
        self.conv_tower_box = nn.Conv3d if CPH.CUBIC_MODE and CPH.CUBIC_BOX_HEAD else nn.Conv2d
        self.conv_tower_conf = nn.Conv3d if CPH.CUBIC_MODE and CPH.CUBIC_CLASS_HEAD else nn.Conv2d
        self.conv_tower_track = nn.Conv3d if CPH.CUBIC_MODE and CPH.CUBIC_TRACK_HEAD else nn.Conv2d
        self.bbox_extra = self.make_prediction_tower(cfg.MODEL.BOX_HEADS.TOWER_LAYERS, self.in_channels,
                                                     self.conv_tower_box)
        self.conf_extra = self.make_prediction_tower(cfg.MODEL.CLASS_HEADS.TOWER_LAYERS, self.in_channels,
                                                     self.conv_tower_conf)
        if cfg.MODEL.TRACK_HEADS.TRAIN_TRACK and not cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
            self.track_extra = self.make_prediction_tower(cfg.MODEL.TRACK_HEADS.TOWER_LAYERS, self.in_channels,
                                                          self.conv_tower_track)

    # Build prediction tower
    def make_prediction_tower(self, num_layers, in_channels, conv_tower):
        assert num_layers > 0, print('Number of layers in prediction tower of CPH should > 0!')
        # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
        return nn.Sequential(*sum([[conv_tower(in_channels, in_channels, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True)]
                                   for _ in range(num_layers)], []))

    def forward(self, fpn_outs):
        """
        Args:
            - fpn_outs: A list of the multi-scale features after FPN, including {P3, P4, P5, P6, P7}.
                        Size: [batch_size*T, in_channels, conv_h, conv_w]
        Returns a dict including bbox_coords, class_confs, mask_output, prior_boxes, with sizes
            - bbox_coords: [batch_size, conv_h*conv_w*num_priors*T, 4T]
            - centerness: [batch_size, conv_h*conv_w*num_priors*T, 1]
            - class_confs: [batch_size, conv_h*conv_w*num_priors*T, num_classes]
            - mask_output: [batch_size, conv_h*conv_w*num_priors*T, mask_dim]
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

        for idx, x in enumerate(fpn_outs):
            # Anchor generation for each FPN features, which depends on the height and width of feature maps.
            # For VIS tasks, the input is fixed 360*640 shape, so we can move it to __init__ to avoid
            # generating it each iteration. But CoCo dataset usually adopts multi-scale training: the shorter
            # edge varies from 800 to 1333, we just move them here.
            _, c, conv_h, conv_w = x.size()
            n_proposals = conv_h * conv_w * self.num_priors
            priors, prior_levels = self.make_priors(idx, conv_h, conv_w, x.device)
            preds['priors'] += [priors]
            preds['prior_levels'] += [prior_levels]

            # [batch_size*T, in_channels, conv_h, conv_w] ==> [batch_size, in_channels, T, conv_h, conv_w]
            x_fold = self.fold_tensor(x)
            # Bounding boxes regression
            bbox_x = self.bbox_extra(x_fold) if self.conv_tower_box == nn.Conv3d else self.bbox_extra(x)
            bbox_x = self.unfold_tensor(bbox_x) if self.conv != self.conv_tower_box else bbox_x
            bbox = self.permute_channels(self.bbox_layer(bbox_x))
            bs = bbox.size(0) * bbox.size(1) if bbox.dim() == 5 else bbox.size(0)
            preds['loc'] += [bbox.reshape(bs, n_proposals, -1)]

            # Centerness for Boxes
            if self.cfg.MODEL.BOX_HEADS.TRAIN_CENTERNESS:
                centerness = self.permute_channels(self.centerness_layer(bbox_x).sigmoid())
                preds['centerness'] += [centerness.reshape(bs, n_proposals, -1)]

            # Classification
            if self.cfg.MODEL.CLASS_HEADS.TRAIN_CLASS:
                conf_x = self.conf_extra(x_fold) if self.conv_tower_conf == nn.Conv3d else self.conf_extra(x)
                conf_x = self.unfold_tensor(conf_x) if self.conv != self.conv_tower_conf else conf_x
                conf = self.permute_channels(self.conf_layer(conf_x))
                preds['conf'] += [conf.reshape(bs, n_proposals, -1)]

            # Mask coefficients: activation function is Tanh
            if self.cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
                mask = self.permute_channels(self.mask_layer(bbox_x).tanh())
                preds['mask_coeff'] += [mask.reshape(bs, n_proposals, -1)]

            # Tracking: need to normalization
            if self.cfg.MODEL.TRACK_HEADS.TRAIN_TRACK and not self.cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
                track_x = self.track_extra(x_fold) if self.conv_tower_track == nn.Conv3d else self.track_extra(x)
                track_x = self.unfold_tensor(track_x) if self.conv != self.conv_tower_track else track_x
                track = self.permute_channels(self.track_layer(track_x))
                preds['track'] += [F.normalize(track.reshape(bs, n_proposals, -1), dim=-1)]

        for k, v in preds.items():
            preds[k] = torch.cat(v, 1)

        return preds

    def fold_tensor(self, x):
        # [bs*T, c, h, w] => [bs, c, T, h, w]
        _, c, h, w = x.size()
        return x.reshape(-1, self.clip_frames, c, h, w).transpose(1, 2)

    def unfold_tensor(self, x):
        # [bs, c, T, h, w] => [bs*T, c, h, w]
        _, c, _, h, w = x.size()
        return x.transpose(1, 2).reshape(-1, c, h, w)

    def permute_channels(self, x):
        # move channels dim to the last dim [bs, c, h, w] => [bs, h, w, c]
        return x.permute(0, 2, 3, 1).contiguous() if x.dim() == 4 else x.permute(0, 2, 3, 4, 1).contiguous()

    def make_priors(self, idx, conv_h, conv_w, device=None):
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
                    ar = sqrt(ar)       # [1, 0.5, 2]
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


