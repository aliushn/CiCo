import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from datasets.config import cfg
from layers.modules import PredictionModule_FC, PredictionModule, make_net, FPN, BiFPN, TemporalNet, DynamicMaskHead
from layers.functions import Detect, Detect_TF, Track, generate_candidate, Track_TF, Track_TF_Clip
from layers.utils import aligned_bilinear, display_conf_outs
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
        src_channels = self.backbone.channels
        if cfg.fpn is not None:
            # Some hacky rewiring to accomodate the FPN
            self.fpn = FPN([src_channels[i] for i in cfg.backbone.selected_layers])
            self.selected_layers = list(range(len(cfg.backbone.selected_layers) + cfg.fpn.num_downsample))
            src_channels = [cfg.fpn.num_features] * len(self.selected_layers)
            if cfg.use_bifpn:
                self.bifpn = nn.Sequential(*[BiFPN(cfg.fpn.num_features)
                                             for _ in range(cfg.num_bifpn)])

        # ------------ build ProtoNet ------------
        self.proto_src = cfg.mask_proto_src
        if self.proto_src is None:
            in_channels = 3
        elif cfg.fpn is not None:
            in_channels = cfg.fpn.num_features
        else:
            in_channels = self.backbone.channels[self.proto_src[0]]

        if self.proto_src is not None and len(self.proto_src) > 1:
            self.mask_refine = nn.ModuleList([nn.Sequential(*[
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ]) for _ in range(len(self.proto_src))])

        # The include_last_relu=false here is because we might want to change it to another function
        self.proto_net, proto_channles = make_net(in_channels, cfg.mask_proto_net, include_bn=True, include_last_relu=True)
        self.proto_dim = cfg.mask_dim * (len(self.selected_layers) + 1) if cfg.mask_proto_with_levels else cfg.mask_dim
        # the last two Conv layers for predicting prototypes
        proto_arch = [(proto_channles, 3, {'padding': 1})] + [(self.proto_dim, 1, {})]
        self.proto_conv, _ = make_net(proto_channles, proto_arch, include_bn=True, include_last_relu=False)
        if cfg.use_dynamic_mask:
            self.DynamicMaskHead = DynamicMaskHead()

        # ------- Build multi-scales prediction head  ------------
        self.pred_scales = cfg.backbone.pred_scales
        self.pred_aspect_ratios = cfg.backbone.pred_aspect_ratios
        self.num_priors = len(self.pred_scales[0])
        # prediction layers for loc, conf, mask
        self.prediction_layers = nn.ModuleList()
        for idx, layer_idx in enumerate(self.selected_layers):
            # If we're sharing prediction module weights, have every module's parent be the first one
            parent, parent_t = None, None
            if cfg.share_prediction_module and idx > 0:
                parent = self.prediction_layers[0]

            if cfg.use_feature_calibration:
                pred = PredictionModule_FC(src_channels[layer_idx], src_channels[layer_idx],
                                           deform_groups=1,
                                           pred_aspect_ratios=self.pred_aspect_ratios[idx],
                                           pred_scales=self.pred_scales[idx],
                                           parent=parent)
            else:
                pred = PredictionModule(src_channels[layer_idx], src_channels[layer_idx],
                                        deform_groups=1,
                                        pred_aspect_ratios=self.pred_aspect_ratios[idx],
                                        pred_scales=self.pred_scales[idx],
                                        parent=parent)

            self.prediction_layers.append(pred)

        # ---------------- Build track head ----------------
        if cfg.train_track:
            track_arch = [(proto_channles, 3, {'padding': 1})] * 2 + [(cfg.track_dim, 1, {})]
            self.track_conv, _ = make_net(proto_channles, track_arch, include_bn=True, include_last_relu=False)

            if cfg.use_temporal_info:
                # temporal fusion between multi frames for tracking
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

            else:
                # track instance frame-by-frame
                self.detect = Detect(cfg.num_classes, bkg_label=0, top_k=cfg.nms_top_k, conf_thresh=cfg.nms_conf_thresh,
                                     nms_thresh=cfg.nms_thresh)
                self.Track = Track()

        else:
            self.detect = Detect(cfg.num_classes, bkg_label=0, top_k=cfg.nms_top_k, conf_thresh=cfg.nms_conf_thresh,
                                 nms_thresh=cfg.nms_thresh)

        if cfg.use_semantic_segmentation_loss:
            sem_seg_head = [(proto_channles, 3, {'padding': 1})]*2 + [(cfg.num_classes, 1, {})]
            self.semantic_seg_conv, _ = make_net(proto_channles, sem_seg_head, include_bn=True,
                                                 include_last_relu=False)

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
        self.load_state_dict(state_dict, strict=True)

    def init_weights_coco(self, backbone_path, local_rank=1):
        """ Initialize weights for training. """
        state_dict = torch.load(backbone_path, map_location=torch.device('cpu'))
        model_dict = self.state_dict()

        # only remain same modules and layers between pre-trained model and our model
        for key in list(state_dict.keys()):
            new_key = key[7:] if key.startswith('module') else key
            if new_key not in model_dict.keys():
                del state_dict[key]
            elif model_dict[new_key].shape != state_dict[key].shape:
                del state_dict[key]
            else:
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        if local_rank == 0:
            for k in model_dict.keys():
                if k not in state_dict:
                    print('parameters without load weights from pre-trained models:', k)

        model_dict.update(state_dict)
        self.load_state_dict(model_dict, strict=True)

        # Initialize the rest of the conv layers with xavier
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) and name+'.weight' not in state_dict:
                if local_rank == 0:
                    print('init weights by Xavier:', name)

                nn.init.xavier_uniform_(module.weight.data)
                if module.bias is not None:
                    if 'conf_layer' in name:
                        module.bias.data[0:] = - np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
                    else:
                        module.bias.data.zero_()

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
                if local_rank == 0:
                    print('init weights by Xavier:', name)

                nn.init.xavier_uniform_(module.weight.data)

                if module.bias is not None:
                    if 'conf_layer' in name:
                        module.bias.data[0:] = - np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
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
                if cfg.use_bifpn:
                    fpn_outs = self.bifpn(fpn_outs)

        if cfg.eval_mask_branch:
            with timer.env('proto '):
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

                proto_out_ori = self.proto_net(proto_x)
                proto_dict = self.proto_conv(proto_out_ori)
                if cfg.mask_proto_prototype_activation is not None:
                    proto_dict = cfg.mask_proto_prototype_activation(proto_dict)

                # Move the features last so the multiplication is easy
                proto_dict = proto_dict.permute(0, 2, 3, 1).contiguous()

        with timer.env('pred_heads'):
            pred_outs = {'mask_coeff': [], 'priors': [], 'prior_levels': []}
            if cfg.train_track and not cfg.track_by_Gaussian:
                pred_outs['track'] = []

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

                p = pred_layer(pred_x, idx)

                for k, v in p.items():
                    pred_outs[k].append(v)  # [batch_size, h*w*anchors, dim

            for k, v in pred_outs.items():
                pred_outs[k] = torch.cat(v, 1)

        if cfg.train_track and cfg.track_by_Gaussian:
            with timer.env('track_by_Gaussian'):
                pred_outs['track'] = self.track_conv(proto_x).permute(0, 2, 3, 1).contiguous()

        if cfg.use_semantic_segmentation_loss or not cfg.train_class:
            with timer.env('sem_seg'):
                pred_outs['sem_seg'] = self.semantic_seg_conv(proto_x).permute(0, 2, 3, 1).contiguous()

        pred_outs['proto'] = proto_dict

        return fpn_outs, pred_outs

    def track_clip(self, candidate_clip, img_meta_clip, img_clip=None, first_clip=False, last_clip=False):
        # n_frame_clip = 2T+1
        n_frame_clip = cfg.eval_frames_of_clip
        T = n_frame_clip // 2
        n_clip, pred_outs_all = 0, []
        index = torch.zeros(len(candidate_clip))
        while len(candidate_clip) >= n_frame_clip:
            pred_outs_cur = self.Track_TF_Clip(self, candidate_clip[:n_frame_clip], img_meta_clip[:n_frame_clip],
                                               img_clip[:n_frame_clip])
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
                pred_outs_cur = self.Track_TF_Clip(self, candidate_clip, img_meta_clip, img_clip)
                pred_outs_all += pred_outs_cur[T:]
                index[T-len(candidate_clip):] = 1

        return index, pred_outs_all

    def forward(self, x, img_meta=None):
        batch_size, c, h, w = x.size()
        fpn_outs, pred_outs = self.forward_single(x)
        # for nn.DataParallel
        pred_outs['priors'] = pred_outs['priors'].repeat(batch_size, 1, 1)
        pred_outs['prior_levels'] = pred_outs['prior_levels'].repeat(batch_size, 1)

        if cfg.train_track and cfg.use_temporal_info:
            # calculate correlation map
            pred_outs['fpn_feat'] = fpn_outs[self.correlation_selected_layer]

        if self.training:

            return pred_outs
        else:

            if cfg.train_class:
                pred_outs['conf'] = pred_outs['conf'].sigmoid()
            else:
                pred_outs['stuff'] = pred_outs['stuff'].sigmoid()
                pred_outs['sem_seg'] = pred_outs['sem_seg'].sigmoid()

            # track instances frame-by-frame
            if cfg.train_track and cfg.use_temporal_info:

                candidate = generate_candidate(pred_outs)
                candidate_after_NMS = self.Detect_TF(self, candidate)

                if cfg.eval_frames_of_clip == 1:
                    # two-frames
                    pred_outs_after_NMS = self.Track_TF(self, candidate_after_NMS, img_meta, img=x)

                    return pred_outs_after_NMS, x, img_meta

                else:
                    # two-clips
                    n_frame_eval_clip = cfg.eval_frames_of_clip
                    T = n_frame_eval_clip // 2
                    self.imgs_clip += x
                    self.candidate_clip += candidate_after_NMS
                    self.img_meta_clip += img_meta
                    n_frames_cur_clip = len(self.candidate_clip)
                    pred_outs_after_track, out_imgs, out_img_metas = [], [], []
                    if n_frames_cur_clip >= n_frame_eval_clip:
                        is_first_idx = [i for i in range(1, n_frames_cur_clip) if self.img_meta_clip[i]['is_first']]

                        if len(is_first_idx) == 0:
                            # All frames of the clip come from a video
                            index, pred_outs_after_track = self.track_clip(self.candidate_clip, self.img_meta_clip,
                                                                           img_clip=self.imgs_clip,
                                                                           first_clip=self.img_meta_clip[0]['is_first'])

                        elif len(is_first_idx) == 1:
                            # The frames of the clip consist of two videos, we need to process them one-by-one
                            index1, pred_outs1 = self.track_clip(self.candidate_clip[:is_first_idx[0]],
                                                                 self.img_meta_clip[:is_first_idx[0]],
                                                                 img_clip=self.imgs_clip,
                                                                 first_clip=False, last_clip=True)
                            index2, pred_outs2 = self.track_clip(self.candidate_clip[is_first_idx[0]:],
                                                                 self.img_meta_clip[is_first_idx[0]:],
                                                                 img_clip=self.imgs_clip,
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

                # detect instances by NMS for each single frame on COCO datasets
                pred_outs_after_NMS = self.detect(self, pred_outs)

                if cfg.train_track:
                    pred_outs_after_NMS = self.Track(pred_outs_after_NMS, img_meta)

                return pred_outs_after_NMS, x, img_meta

