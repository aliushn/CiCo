import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from datasets.config import cfg
from layers.modules import PredictionModule_FC, make_net, FPN, FastMaskIoUNet, TemporalNet
from layers.functions import Detect, Detect_TF, Track, generate_candidate, Track_TF, Track_TF_Clip
from backbone import construct_backbone
from utils import timer

# This is required for Pytorch 1.0.1 on Windows to initialize Cuda on some driver versions.
# See the bug report here: https://github.com/pytorch/pytorch/issues/17108
torch.cuda.current_device()
prior_cache = defaultdict(lambda: None)


class STMask(nn.Module):
    """
    The code comes from Yolact.
    You can set the arguments by chainging them in the backbone config object in config.py.

    Parameters (in cfg.backbone):
        - selected_layers: The indices of the conv layers to use for prediction.
        - pred_scales:     A list with len(selected_layers) containing tuples of scales (see PredictionModule)
        - pred_aspect_ratios: A list of lists of aspect ratios with len(selected_layers) (see PredictionModule)
    """

    def __init__(self):
        super().__init__()

        self.backbone = construct_backbone(cfg.backbone)

        # Compute mask_dim here and add it back to the config. Make sure Yolact's constructor is called early!
        if cfg.mask_proto_use_grid:
            self.grid = torch.Tensor(np.load(cfg.mask_proto_grid_file))
            self.num_grids = self.grid.size(0)
        else:
            self.num_grids = 0

        self.proto_src = cfg.mask_proto_src
        self.interpolation_mode = cfg.fpn.interpolation_mode

        if self.proto_src is None:
            in_channels = 3
        elif cfg.fpn is not None:
            in_channels = cfg.fpn.num_features
        else:
            in_channels = self.backbone.channels[self.proto_src]
        in_channels += self.num_grids

        # The include_last_relu=false here is because we might want to change it to another function
        self.proto_net, proto_channles = make_net(in_channels, cfg.mask_proto_net, include_last_relu=True)
        proto_arch = [(proto_channles, 3, {'padding': 1})] + [(cfg.mask_dim, 1, {})]
        self.proto_conv, cfg.mask_dim = make_net(proto_channles, proto_arch, include_last_relu=False)

        if cfg.mask_proto_bias:
            cfg.mask_dim += 1

        if cfg.train_track:
            track_arch = [(proto_channles, 3, {'padding': 1})] + [(cfg.track_n, 1, {})]
            self.track_conv, _ = make_net(proto_channles, track_arch, include_last_relu=False)
        if cfg.use_semantic_segmentation_loss:
            sem_seg_head = [(proto_channles, 3, {'padding': 1})] + [(cfg.num_classes, 1, {})]
            self.semantic_seg_conv, _ = make_net(proto_channles, sem_seg_head, include_last_relu=False)

        self.selected_layers = cfg.backbone.selected_layers
        self.pred_scales = cfg.backbone.pred_scales
        self.pred_aspect_ratios = cfg.backbone.pred_aspect_ratios
        self.num_priors = len(self.pred_scales[0])
        src_channels = self.backbone.channels

        if cfg.fpn is not None:
            # Some hacky rewiring to accomodate the FPN
            self.fpn = FPN([src_channels[i] for i in self.selected_layers])

            if cfg.backbone_C2_as_features:
                self.selected_layers = list(range(1, len(self.selected_layers) + cfg.fpn.num_downsample))
                src_channels = [cfg.fpn.num_features] * (len(self.selected_layers) + 1)
            else:
                self.selected_layers = list(range(len(self.selected_layers) + cfg.fpn.num_downsample))
                src_channels = [cfg.fpn.num_features] * len(self.selected_layers)

        # prediction layers for loc, conf, mask
        self.prediction_layers = nn.ModuleList()
        cfg.num_heads = len(self.selected_layers)  # yolact++
        for idx, layer_idx in enumerate(self.selected_layers):
            # If we're sharing prediction module weights, have every module's parent be the first one
            parent, parent_t = None, None
            if cfg.share_prediction_module and idx > 0:
                parent = self.prediction_layers[0]

            pred = PredictionModule_FC(src_channels[layer_idx], src_channels[layer_idx],
                                       deform_groups=1,
                                       pred_aspect_ratios=self.pred_aspect_ratios[idx],
                                       pred_scales=self.pred_scales[idx],
                                       parent=parent)

            self.prediction_layers.append(pred)

        if cfg.train_track and cfg.use_temporal_info:
            # temporal fusion
            self.correlation_selected_layer = cfg.correlation_selected_layer
            # evaluation for frame-level tracking
            self.Detect_TF = Detect_TF(cfg.num_classes, bkg_label=0, top_k=cfg.nms_top_k,
                                       conf_thresh=cfg.nms_conf_thresh, nms_thresh=cfg.nms_thresh)
            if cfg.eval_frames_of_clip > 1:
                self.Track_TF_Clip = Track_TF_Clip()
            else:
                self.Track_TF = Track_TF()

            # track to segment
            if cfg.temporal_fusion_module:
                corr_channels = 2*in_channels + cfg.correlation_patch_size**2
                self.TemporalNet = TemporalNet(corr_channels, cfg.mask_dim)
            elif cfg.use_FEELVOS:
                # using FEELVOS, VOS strategy, to track instances from previous to current frame
                VOS_in_channels = in_channels
                self.VOS_head, _ = make_net(VOS_in_channels, cfg.VOS_head, include_last_relu=False)
                self.VOS_attention = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        else:
            self.detect = Detect(cfg.num_classes, bkg_label=0, top_k=cfg.nms_top_k, conf_thresh=cfg.nms_conf_thresh,
                                 nms_thresh=cfg.nms_thresh)

            if cfg.train_track:
                self.Track = Track()

        # Extra parameters for the extra losses
        if cfg.use_class_existence_loss:
            # This comes from the smallest layer selected
            # Also note that cfg.num_classes includes background
            self.class_existence_fc = nn.Linear(src_channels[-1], cfg.num_classes - 1)

        self.candidate_clip = []
        self.img_meta_clip = []
        self.imgs_clip = []

    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)
        model_dict = self.state_dict()
        for key in list(state_dict.keys()):
            if key.startswith('module.'):
                state_dict[key[7:]] = state_dict[key]
                del state_dict[key]

        # For backward compatability, remove these (the new variable is called layers)
        for key in list(state_dict.keys()):
            if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
                del state_dict[key]

            # Also for backward compatibility with v1.0 weights, do this check
            if key.startswith('fpn.downsample_layers.'):
                if cfg.fpn is not None and int(key.split('.')[2]) >= cfg.fpn.num_downsample:
                    del state_dict[key]

        diff_layers1 = [k for k, v in state_dict.items() if k not in model_dict.keys()]
        print()
        print('layers in pre-trained model but not in current model:', diff_layers1)

        diff_layers2 = [k for k, v in model_dict.items() if k not in state_dict.keys()]
        print('layers in current model but not in pre-trained model:', diff_layers2)

        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)

    def init_weights_coco(self, backbone_path, local_rank=1):
        """ Initialize weights for training. """
        state_dict = torch.load(backbone_path, map_location=torch.device('cpu'))
        model_dict = self.state_dict()

        # only remain same modules and layers between pre-trained model and our model
        for key in list(state_dict.keys()):
            if key not in model_dict.keys():
                del state_dict[key]
            elif model_dict[key].shape != state_dict[key].shape:
                del state_dict[key]

        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        if local_rank == 0:
            print('parameters without load weights from pre-trained models')
            print([k for k, v in model_dict.items() if k not in state_dict])
        model_dict.update(state_dict)

        # Initialize the rest of the conv layers with xavier
        for k, v in model_dict.items():
            if k not in state_dict:
                if local_rank == 0:
                    print('init weights by Xavier:', k)
                if 'weight' in k:
                    nn.init.xavier_uniform_(model_dict[k])
                elif 'bias' in k:
                    if cfg.use_sigmoid_focal_loss and 'conf_layer' in k:
                        data0 = -torch.tensor(cfg.focal_loss_init_pi / (1 - cfg.focal_loss_init_pi)).log()
                        data1 = -torch.tensor((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi).log()
                        model_dict[k] = torch.cat([data0.repeat(self.num_priors), data1.repeat((cfg.num_classes-1)*self.num_priors)])
                    else:
                        model_dict[k].zero_()

        self.load_state_dict(model_dict)

    def init_weights(self, backbone_path, local_rank=1):
        """ Initialize weights for training. """
        # Initialize the backbone with the pretrained weights.
        self.backbone.init_backbone(backbone_path)

        conv_constants = getattr(nn.Conv2d(1, 1, 1), '__constants__')

        # Quick lambda to test if one list contains the other
        def all_in(x, y):
            for _x in x:
                if _x not in y:
                    return False
            return True

        # Initialize the rest of the conv layers with xavier
        for name, module in self.named_modules():
            # See issue #127 for why we need such a complicated condition if the module is a WeakScriptModuleProxy
            # Broke in 1.3 (see issue #175), WeakScriptModuleProxy was turned into just ScriptModule.
            # Broke in 1.4 (see issue #292), where RecursiveScriptModule is the new star of the show.
            # Note that this might break with future pytorch updates, so let me know if it does
            is_script_conv = False
            if 'Script' in type(module).__name__:
                # 1.4 workaround: now there's an original_name member so just use that
                if hasattr(module, 'original_name'):
                    is_script_conv = 'Conv' in module.original_name
                # 1.3 workaround: check if this has the same constants as a conv module
                else:
                    is_script_conv = (
                            all_in(module.__dict__['_constants_set'], conv_constants)
                            and all_in(conv_constants, module.__dict__['_constants_set']))

            is_conv_layer = isinstance(module, nn.Conv2d) or is_script_conv

            if is_conv_layer and module not in self.backbone.backbone_modules:
                nn.init.xavier_uniform_(module.weight.data)

                if module.bias is not None:
                    if cfg.use_focal_loss and 'conf_layer' in name:
                        if not cfg.use_sigmoid_focal_loss:
                            # Initialize the last layer as in the focal loss paper.
                            # Because we use softmax and not sigmoid, I had to derive an alternate expression
                            # on a notecard. Define pi to be the probability of outputting a foreground detection.
                            # Then let z = sum(exp(x)) - exp(x_0). Finally let c be the number of foreground classes.
                            # Chugging through the math, this gives us
                            #   x_0 = log(z * (1 - pi) / pi)    where 0 is the background class
                            #   x_i = log(z / c)                for all i > 0
                            # For simplicity (and because we have a degree of freedom here), set z = 1. Then we have
                            #   x_0 =  log((1 - pi) / pi)       note: don't split up the log for numerical stability
                            #   x_i = -log(c)                   for all i > 0
                            module.bias.data[0] = np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
                            module.bias.data[1:] = -np.log(module.bias.size(0) - 1)
                        else:
                            module.bias.data[0] = -np.log(cfg.focal_loss_init_pi / (1 - cfg.focal_loss_init_pi))
                            module.bias.data[1:] = -np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
                    else:
                        module.bias.data.zero_()

    def train(self, mode=True):
        super().train(mode)

    def forward_single(self, x):
        """ The input should be of size [batch_size, 3, img_h, img_w] """

        with timer.env('backbone'):
            bb_outs = self.backbone(x)

        if cfg.fpn is not None:
            with timer.env('fpn'):
                # Use backbone.selected_layers because we overwrote self.selected_layers
                outs = [bb_outs[i] for i in cfg.backbone.selected_layers]
                fpn_outs = self.fpn(outs)

        if cfg.eval_mask_branch:
            with timer.env('proto '):
                if self.proto_src is None:
                    proto_x = x
                else:
                    # h, w = bb_outs[self.proto_src].size()[2:]
                    # p3_upsample = F.interpolate(fpn_outs[self.proto_src], size=(h, w), mode=self.interpolation_mode,
                    #                             align_corners=False)
                    # proto_x = p3_upsample # + bb_outs[self.proto_src]
                    proto_x = fpn_outs[self.proto_src]

                proto_out_ori = self.proto_net(proto_x)
                proto_dict = self.proto_conv(proto_out_ori)
                if cfg.mask_proto_prototype_activation is not None:
                    proto_dict = cfg.mask_proto_prototype_activation(proto_dict)

                # Move the features last so the multiplication is easy
                proto_dict = proto_dict.permute(0, 2, 3, 1).contiguous()

                if cfg.mask_proto_bias:
                    bias_shape = [x for x in proto_dict.size()]
                    bias_shape[1] = 1
                    proto_dict = torch.cat([proto_dict, torch.ones(*bias_shape)], 1)

        with timer.env('pred_heads'):
            pred_outs = {'mask_coeff': [], 'priors': []}

            if cfg.train_boxes:
                pred_outs['loc'] = []

            if cfg.train_centerness:
                pred_outs['centerness'] = []

            if cfg.train_class:
                pred_outs['conf'] = []
            else:
                pred_outs['stuff'] = []

            for idx, pred_layer in zip(self.selected_layers, self.prediction_layers):
                pred_x = fpn_outs[idx]

                # A hack for the way dataparallel works
                if cfg.share_prediction_module and pred_layer is not self.prediction_layers[0]:
                    pred_layer.parent = [self.prediction_layers[0]]

                p = pred_layer(pred_x)

                for k, v in p.items():
                    pred_outs[k].append(v)  # [batch_size, h*w*anchors, dim]

                if cfg.backbone_C2_as_features:
                    idx -= 1

            for k, v in pred_outs.items():
                pred_outs[k] = torch.cat(v, 1)

        if cfg.train_track:
            with timer.env('track'):
                pred_outs['track'] = self.track_conv(proto_out_ori).permute(0, 2, 3, 1).contiguous()

        if cfg.use_semantic_segmentation_loss or not cfg.train_class:
            with timer.env('sem_seg'):
                pred_outs['sem_seg'] = self.semantic_seg_conv(proto_out_ori).permute(0, 2, 3, 1).contiguous()

        pred_outs['proto'] = proto_dict

        return fpn_outs, pred_outs

    def track_clip(self, candidate_clip, img_meta_clip, first_clip=False, last_clip=False):
        # n_frame_clip = 2T+1
        n_frame_clip = cfg.eval_frames_of_clip
        T = n_frame_clip // 2
        n_clip, pred_outs_all = 0, []
        index = torch.zeros(len(candidate_clip))
        while len(candidate_clip) >= n_frame_clip:
            pred_outs_cur = self.Track_TF_Clip(self, candidate_clip[:n_frame_clip], img_meta_clip[:n_frame_clip])
            if first_clip and n_clip == 0:
                pred_outs_all += pred_outs_cur
                index[:n_frame_clip] = 1
            else:
                pred_outs_all += pred_outs_cur[T:]
                min_idx = (n_clip+1)*(n_frame_clip-T)-1
                max_idx = min_idx + (n_frame_clip-T)
                index[min_idx:max_idx] = 1

            n_clip += 1
            candidate_clip = candidate_clip[T+1:]
            img_meta_clip = img_meta_clip[T+1:]

        if last_clip:
            if len(candidate_clip) > T:
                pred_outs_cur = self.Track_TF_Clip(self, candidate_clip, img_meta_clip)
                pred_outs_all += pred_outs_cur[T:]
                index[T-len(candidate_clip):] = 1

        return index, pred_outs_all

    def forward(self, x, img_meta=None):
        if self.training:
            batch_size, c, h, w = x.size()
            fpn_outs, pred_outs = self.forward_single(x)

            if cfg.train_track and cfg.use_temporal_info:
                # calculate correlation map
                pred_outs['fpn_feat'] = fpn_outs[self.correlation_selected_layer]

            # For the extra loss functions
            if cfg.use_class_existence_loss:
                pred_outs['classes'] = self.class_existence_fc(fpn_outs[-1].mean(dim=(2, 3)))

            # for nn.DataParallel
            pred_outs['priors'] = pred_outs['priors'].repeat(batch_size, 1, 1)

            return pred_outs
        else:
            bs = x.size(0)
            fpn_outs, pred_outs = self.forward_single(x)

            if cfg.train_class:
                pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)
            else:
                pred_outs['stuff'] = torch.sigmoid(pred_outs['stuff'])
                pred_outs['sem_seg'] = F.softmax(pred_outs['sem_seg'], dim=-1)

            # track instances frame-by-frame
            if cfg.train_track and cfg.use_temporal_info:

                # we only use the bbox features in the P3 layer
                pred_outs['fpn_feat'] = fpn_outs[1]
                candidate = generate_candidate(pred_outs)
                candidate_after_NMS = self.Detect_TF(candidate)
                pred_outs_after_track = []
                if cfg.eval_frames_of_clip == 1:
                    # two-frames
                    for i in range(bs):
                        pred_outs_after_track.append(self.Track_TF(self, candidate_after_NMS[i], img_meta[i], img=x))

                    return pred_outs_after_track, x, img_meta

                else:
                    # two-clips
                    n_frame_eval_clip = cfg.eval_frames_of_clip
                    T = n_frame_eval_clip // 2
                    self.imgs_clip += x
                    self.candidate_clip += candidate_after_NMS
                    self.img_meta_clip += img_meta
                    n_frames_cur_clip = len(self.candidate_clip)
                    out_imgs, out_img_metas = [], []
                    if n_frames_cur_clip >= n_frame_eval_clip:
                        is_first_idx = [i for i in range(1, n_frames_cur_clip) if self.img_meta_clip[i]['is_first']]

                        if len(is_first_idx) == 0:
                            # All frames of the clip come from a video
                            index, pred_outs_after_track = self.track_clip(self.candidate_clip, self.img_meta_clip,
                                                                           first_clip=self.img_meta_clip[0]['is_first'])

                        elif len(is_first_idx) == 1:
                            # The frames of the clip consist of two videos, we need to process them one-by-one
                            index1, pred_outs1 = self.track_clip(self.candidate_clip[:is_first_idx[0]],
                                                                 self.img_meta_clip[:is_first_idx[0]],
                                                                 first_clip=False, last_clip=True)
                            index2, pred_outs2 = self.track_clip(self.candidate_clip[is_first_idx[0]:],
                                                                 self.img_meta_clip[is_first_idx[0]:],
                                                                 first_clip=True, last_clip=False)
                            pred_outs_after_track = pred_outs1 + pred_outs2
                            index = torch.cat((index1, index2))

                        else:
                            print('Only support frames that less than 21, please try smaller batch size')

                        # to remove frames that has been processed
                        # the last T frames will be leaved to guarantee overlapped frames in two adjacent clips
                        # for tracking between clips

                        keep = index > 0
                        if keep.sum() > 0:
                            keep_idx = (torch.arange(n_frames_cur_clip)[keep]).tolist()
                            out_imgs = self.imgs_clip[min(keep_idx):max(keep_idx)+1]
                            out_img_metas = self.img_meta_clip[min(keep_idx):max(keep_idx)+1]
                            self.candidate_clip = self.candidate_clip[max(keep_idx)+1-T:]
                            self.img_meta_clip = self.img_meta_clip[max(keep_idx)+1-T:]
                            self.imgs_clip = self.imgs_clip[max(keep_idx)+1-T:]
                        else:
                            self.candidate_clip = self.candidate_clip[T:]
                            self.img_meta_clip = self.img_meta_clip[T:]
                            self.imgs_clip = self.imgs_clip[T:]

                    return pred_outs_after_track, out_imgs, out_img_metas

            else:
                # detect instances by NMS for each single frame
                pred_outs_after_NMS = self.detect(pred_outs, self)

                if cfg.train_track:
                    pred_outs_after_NMS = self.Track(pred_outs_after_NMS, img_meta)

                return pred_outs_after_NMS

