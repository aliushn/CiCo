import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from layers.modules import PredictionModule_FC, PredictionModule, make_net, FPN, BiFPN, TemporalNet, DynamicMaskHead
from layers.functions import Detect, Track, Track_TF, Track_TF_Clip
from layers.utils import aligned_bilinear, correlate_operator
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

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.backbone = construct_backbone(cfg.MODEL.BACKBONE)
        src_channels = self.backbone.channels
        self.clip_frames = 1

        # Some hacky rewiring to accomodate the FPN
        self.fpn = FPN([src_channels[i] for i in cfg.MODEL.BACKBONE.SELECTED_LAYERS])
        self.selected_layers = list(range(len(cfg.MODEL.BACKBONE.SELECTED_LAYERS)+cfg.MODEL.FPN.NUM_DOWNSAMPLE))
        self.fpn_num_features = cfg.MODEL.FPN.NUM_FEATURES
        self.fpn_num_downsample = cfg.MODEL.FPN.NUM_DOWNSAMPLE
        src_channels = [self.fpn_num_features] * len(self.selected_layers)

        if cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE:
            self.clip_frames = cfg.SOLVER.NUM_CLIP_FRAMES
            if cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE_WITH_INITIALIZATION == 'reduced':
                self.fpn_num_features //= self.clip_frames

            if cfg.MODEL.PREDICTION_HEADS.CUBIC_CORRELATION_MODE:
                self.correlation_patch_size = cfg.MODEL.PREDICTION_HEADS.CORRELATION_PATCH_SIZE
                src_channels = [self.clip_frames*self.fpn_num_features+self.correlation_patch_size**2] * len(self.selected_layers)
            else:
                src_channels = [self.clip_frames*self.fpn_num_features] * len(self.selected_layers)

        if cfg.MODEL.FPN.USE_BIFPN:
            self.bifpn = nn.Sequential(*[BiFPN(self.fpn_num_features) for _ in range(cfg.num_bifpn)])

        # ------------ build ProtoNet ------------
        self.train_masks = cfg.MODEL.MASK_HEADS.TRAIN_MASKS
        in_channels = cfg.MODEL.FPN.NUM_FEATURES
        if self.train_masks:
            self.proto_src = cfg.MODEL.MASK_HEADS.PROTO_SRC
            if self.proto_src is not None and len(self.proto_src) > 1:
                self.mask_refine = nn.ModuleList([nn.Sequential(*[
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True)
                ]) for _ in range(len(self.proto_src))])

            # The include_last_relu=false here is because we might want to change it to another function
            self.proto_net, proto_channles = make_net(in_channels, cfg.MODEL.MASK_HEADS.PROTO_NET, include_bn=True,
                                                      include_last_relu=True)
            # the last two Conv layers for predicting prototypes
            proto_arch = [(proto_channles, 3, 1), (cfg.MODEL.MASK_HEADS.MASK_DIM, 1, 0)]
            self.proto_conv, _ = make_net(proto_channles, proto_arch, include_bn=True, include_last_relu=False)
            if cfg.MODEL.MASK_HEADS.USE_DYNAMIC_MASK:
                self.DynamicMaskHead = DynamicMaskHead()

        # ------- Build multi-scales prediction head  ------------
        self.pred_scales = cfg.MODEL.BACKBONE.PRED_SCALES
        self.pred_aspect_ratios = cfg.MODEL.BACKBONE.PRED_ASPECT_RATIOS
        self.num_priors = len(self.pred_scales[0]) * len(self.pred_aspect_ratios[0][0])
        # reduced channels
        if cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE_WITH_INITIALIZATION == 'reduced':
            self.fpn_reduced_channels = nn.Sequential(*[
                nn.Conv2d(in_channels, in_channels//self.clip_frames, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)
            ])
        # prediction layers for loc, conf, mask, track
        self.prediction_layers = nn.ModuleList()
        for idx, layer_idx in enumerate(self.selected_layers):
            # If we're sharing prediction module weights, have every module's parent be the first one
            parent = None
            if cfg.MODEL.PREDICTION_HEADS.SHARE_PREDICTION_MODULE and idx > 0:
                parent = self.prediction_layers[0]

            if cfg.STMASK.FC.USE_FCA:
                pred = PredictionModule_FC(cfg, src_channels[idx],
                                           deform_groups=1,
                                           pred_scales=self.pred_scales[idx],
                                           parent=parent)
            else:
                pred = PredictionModule(cfg, src_channels[idx],
                                        deform_groups=1,
                                        pred_aspect_ratios=self.pred_aspect_ratios[idx],
                                        pred_scales=self.pred_scales[idx],
                                        parent=parent)

            self.prediction_layers.append(pred)

        # ---------------- Build detection ----------------
        self.Detect = Detect(cfg.DATASETS.NUM_CLASSES, cfg.TEST.DETECTIONS_PER_IMG, cfg.TEST.NMS_CONF_THRESH,
                             nms_thresh=cfg.TEST.NMS_IoU_THRESH, train_masks=self.train_masks,
                             nms_with_miou=cfg.TEST.NMS_WITH_MIoU, use_DIoU=cfg.MODEL.PREDICTION_HEADS.USE_DIoU)

        # ---------------- Build track head ----------------
        if cfg.MODEL.TRACK_HEADS.TRAIN_TRACK:
            if cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
                track_arch = [(in_channels, 3, {'padding': 1})] * 2 + [(cfg.track_dim, 1, {})]
                self.track_conv, _ = make_net(in_channels, track_arch, include_bn=True, include_last_relu=False)

            if not cfg.STMASK.T2S_HEADS.TEMPORAL_FUSION_MODULE:
                # Track objects frame-by-frame
                self.Track = Track(cfg.MODEL.TRACK_HEADS.MATCH_COEFF, self.clip_frames, train_masks=self.train_masks,
                                   track_by_Gaussian=cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN)
            else:
                if cfg.TEST.NUM_CLIP_FRAMES > 1:
                    self.Track = Track_TF_Clip(self, cfg.MODEL.TRACK_HEADS.MATCH_COEFF)
                else:
                    self.Track = Track_TF(self, cfg.MODEL.TRACK_HEADS.MATCH_COEFF,
                                          cfg.STMASK.T2S_HEADS.CORRELATION_PATCH_SIZE,
                                          train_maskshift=cfg.STMASK.T2S_HEADS.TRAIN_MASKSHIFT,
                                          conf_thresh=cfg.TEST.NMS_CONF_THRESH, train_masks=self.train_masks,
                                          track_by_Gaussian=cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN)

                # A Temporal Fusion built on two adjacent frames for tracking
                self.TF_correlation_selected_layer = cfg.STMASK.T2S_HEADS.CORRELATION_SELECTED_LAYER
                corr_channels = 2*in_channels + cfg.STMASK.T2S_HEADS.CORRELATION_PATCH_SIZE**2
                self.TemporalNet = TemporalNet(corr_channels, cfg.MODEL.MASK_HEADS.MASK_DIM,
                                               maskshift_loss=cfg.STMASK.T2S_HEADS.TRAIN_MASKSHIFT,
                                               use_sipmask=cfg.MODEL.MASK_HEADS.USE_SIPMASK,
                                               sipmask_head=cfg.MODEL.MASK_HEADS.SIPMASK_HEAD)

        if cfg.MODEL.MASK_HEADS.TRAIN_MASKS and cfg.MODEL.MASK_HEADS.USE_SEMANTIC_SEGMENTATION_LOSS:
            sem_seg_head = [(in_channels, 3, 1)]*2 + [(cfg.DATASETS.NUM_CLASSES, 1, 0)]
            self.semantic_seg_conv, _ = make_net(in_channels, sem_seg_head, include_bn=True,
                                                 include_last_relu=False)

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
                if int(key.split('.')[2]) >= self.fpn_num_downsample:
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
        print('Initialize weights from:', backbone_path)
        state_dict = torch.load(backbone_path, map_location=torch.device('cpu'))
        # In case of save models from distributed training
        for key in list(state_dict.keys()):
            if key.startswith('module'):
                state_dict[key[7:]] = state_dict.pop(key)

        model_dict = self.state_dict()

        # Initialize the rest of the conv layers with xavier
        print('init all weights by Xavier:')
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight.data)
                if module.bias is not None:
                    if self.cfg.MODEL.CLASS_HEADS.USE_FOCAL_LOSS and 'conf_layer' in name:
                        # The initial bias toward forground objects, as specified in the focal loss paper
                        focal_loss_init_pi = 0.01
                        module.bias.data[0:] = - np.log((1 - focal_loss_init_pi) / focal_loss_init_pi)
                    else:
                        module.bias.data.zero_()

                if name+'.weight' not in state_dict.keys():
                    print('parameters in current model but not in pre-trained model:', name)

        # If use correlation to encode temporal information, we only inflated weights from 2D to 3D in some channels
        if self.cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE and self.cfg.MODEL.PREDICTION_HEADS.CUBIC_CORRELATION_MODE:
            idx = []
            idx += range(self.fpn_num_features)
            for c in range(1, self.clip_frames):
                left = c*(self.fpn_num_features+self.correlation_patch_size**2)
                idx += range(left, left+self.fpn_num_features)

        # only update same modules and layers between pre-trained model and our model
        for key in list(state_dict.keys()):
            if key in model_dict.keys():
                if model_dict[key].shape == state_dict[key].shape:
                    model_dict[key] = state_dict[key]
                elif self.cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE and key.startswith('prediction_layers'):
                    if 'conf_layer' in key:
                        continue

                    # Inflated or reduced weights from 2D (single frame) to 3D (multi-frames)
                    if self.cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE_WITH_INITIALIZATION == 'reduced':
                        print('load parameters with reduced channels operation from pre-trained models:', key)
                        if key.split('.')[-2] in {'bbox_layer', 'centerness_layer'}:
                            if state_dict[key].dim() == 4:
                                _, c_in, kh, kw = state_dict[key].shape
                                init_weights = state_dict[key].reshape(self.num_priors, -1, c_in, kh, kw).repeat(1, self.clip_frames, 1, 1, 1).reshape(-1, c_in, kh, kw)
                            else:
                                init_weights = state_dict[key].reshape(self.num_priors, -1).repeat(1, self.clip_frames).reshape(-1)
                        else:
                            init_weights = state_dict[key]

                    elif self.cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE_WITH_INITIALIZATION == 'inflated':
                        print('load parameters with inflated operation from pre-trained models:', key)
                        if key.split('.')[-2] in {'bbox_layer', 'centerness_layer'}:
                            if state_dict[key].dim() == 4:
                                _, c_in, kh, kw = state_dict[key].shape
                                init_weights = state_dict[key].reshape(self.num_priors, -1, c_in, kh, kw).repeat(1, self.clip_frames, self.clip_frames, 1, 1).reshape(-1, self.clip_frames*c_in, kh, kw)
                            else:
                                init_weights = state_dict[key].reshape(self.num_priors, -1).repeat(1, self.clip_frames).reshape(-1)
                        elif key.split('.')[-2] in {'mask_layer'}:
                            init_weights = state_dict[key].repeat(1, self.clip_frames, 1, 1)
                        else:
                            init_weights = state_dict[key].repeat(self.clip_frames, self.clip_frames, 1, 1) \
                                        if state_dict[key].dim() == 4 else state_dict[key].repeat(self.clip_frames)
                        init_weights = (init_weights / float(self.clip_frames))
                    else:
                        RuntimeError('cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE_WITH_INITIALIZATION should be'
                                     ' reduced or inflated')

                    if not self.cfg.MODEL.PREDICTION_HEADS.CUBIC_CORRELATION_MODE:
                        model_dict[key] = init_weights.to(model_dict[key].device)
                    else:
                        if 'extra' in key:
                            if state_dict[key].dim() == 4:
                                model_dict[key][idx][:, idx] = init_weights.to(model_dict[key].device)
                            else:
                                model_dict[key][idx] = init_weights.to(model_dict[key].device)
                        else:
                            if state_dict[key].dim() == 4:
                                model_dict[key][:, idx] = init_weights.to(model_dict[key].device)
                            else:
                                model_dict[key] = init_weights.to(model_dict[key].device)
                else:
                    print('Size is different in pre-trained model and in current model:', key)
            else:
                print('parameters in pre-trained model but not in current model:', key)

        self.load_state_dict(model_dict, strict=True)

    def init_weights(self, backbone_path, local_rank=1):
        """ Initialize weights for training. """
        # Initialize the backbone with the pretrained weights.
        print('Initialize weights from:', backbone_path)
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
                    if self.cfg.MODEL.CLASS_HEADS.USE_FOCAL_LOSS and 'conf_layer' in name:
                        focal_loss_init_pi = 0.01
                        module.bias.data[0:] = - np.log((1 - focal_loss_init_pi) / focal_loss_init_pi)
                    else:
                        module.bias.data.zero_()

    def train(self, mode=True):
        super().train(mode)

    def forward_single(self, x):
        """ The input should be of size [batch_size, 3, img_h, img_w] """

        with timer.env('backbone'):
            bb_outs = self.backbone(x)

        with timer.env('fpn'):
            # Use backbone.selected_layers because we overwrote self.selected_layers
            outs = [bb_outs[i] for i in self.cfg.MODEL.BACKBONE.SELECTED_LAYERS]
            fpn_outs = self.fpn(outs)
            if self.cfg.MODEL.FPN.USE_BIFPN:
                fpn_outs = self.bifpn(fpn_outs)

        if self.cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
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

                proto_out_features = self.proto_net(proto_x)
                # Activation function is RELU
                # Move the features last so the multiplication is easy
                prototypes = F.relu(self.proto_conv(proto_out_features)).permute(0, 2, 3, 1).contiguous()

        with timer.env('pred_heads'):
            pred_outs = {'priors': [], 'prior_levels': []}
            if self.cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
                pred_outs['mask_coeff'] = []
            if self.cfg.MODEL.TRACK_HEADS.TRAIN_TRACK and not self.cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
                pred_outs['track'] = []
            if self.cfg.MODEL.BOX_HEADS.TRAIN_BOXES:
                pred_outs['loc'] = []
            if self.cfg.MODEL.BOX_HEADS.TRAIN_CENTERNESS:
                pred_outs['centerness'] = []
            if self.cfg.MODEL.CLASS_HEADS.TRAIN_CLASS:
                pred_outs['conf'] = []
            else:
                pred_outs['stuff'] = []

            for idx, pred_layer in zip(self.selected_layers, self.prediction_layers):
                # A hack for the way dataparallel works
                if self.cfg.MODEL.PREDICTION_HEADS.SHARE_PREDICTION_MODULE and idx != 0:
                    pred_layer.parent = [self.prediction_layers[0]]

                if self.cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE:
                    _, c, h, w = fpn_outs[idx].size()
                    fpn_outs_clip = fpn_outs[idx].reshape(-1, self.clip_frames, c, h, w)
                    if self.cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE_WITH_INITIALIZATION == 'reduced':
                        pred_x = [self.fpn_reduced_channels(fpn_outs_clip[:, 0])]
                    else:
                        pred_x = [fpn_outs_clip[:, 0]]
                    for frame_idx in range(self.clip_frames-1):
                        if self.cfg.MODEL.PREDICTION_HEADS.CUBIC_CORRELATION_MODE:
                            corr = correlate_operator(fpn_outs_clip[:, frame_idx].contiguous(),
                                                      fpn_outs_clip[:, frame_idx+1].contiguous(),
                                                      patch_size=self.correlation_patch_size,
                                                      kernel_size=1)
                            # if idx == 0:
                            #     x_clip = x.reshape(-1, self.clip_frames, 3, x.size(-2), x.size(-1))
                            #     display_correlation_map(corr, imgs=x_clip, img_meta=img_meta[-1], idx=idx)
                            pred_x.append(corr)
                        cur_fpn_x = self.fpn_reduced_channels(fpn_outs_clip[:, frame_idx+1]) \
                            if self.cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE_WITH_INITIALIZATION == 'reduced' else fpn_outs_clip[:, frame_idx+1]
                        pred_x.append(cur_fpn_x)

                    pred_x = torch.cat(pred_x, dim=1)
                else:
                    pred_x = fpn_outs[idx]

                p = pred_layer(pred_x, idx)

                for k, v in p.items():
                    pred_outs[k].append(v)  # [batch_size, h*w*anchors, dim

            for k, v in pred_outs.items():
                pred_outs[k] = torch.cat(v, 1)

        if self.cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
            pred_outs['proto'] = prototypes

        if self.cfg.MODEL.TRACK_HEADS.TRAIN_TRACK and self.cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
            with timer.env('track_by_Gaussian'):
                pred_outs['track'] = self.track_conv(fpn_outs[0]).permute(0, 2, 3, 1).contiguous()

        if self.training:
            if self.cfg.MODEL.MASK_HEADS.USE_SEMANTIC_SEGMENTATION_LOSS or not self.cfg.MODEL.CLASS_HEADS.TRAIN_CLASS:
                with timer.env('sem_seg'):
                    pred_outs['sem_seg'] = self.semantic_seg_conv(fpn_outs[0]).permute(0, 2, 3, 1).contiguous()

        return fpn_outs, pred_outs

    def forward(self, x, img_meta=None):
        batch_size, c, h, w = x.size()
        fpn_outs, pred_outs = self.forward_single(x)
        # Expand data for nn.DataParallel
        pred_outs['priors'] = pred_outs['priors'].repeat(batch_size//self.clip_frames, 1, 1)
        pred_outs['prior_levels'] = pred_outs['prior_levels'].repeat(batch_size//self.clip_frames, 1)

        if self.cfg.STMASK.T2S_HEADS.TEMPORAL_FUSION_MODULE:
            # calculate correlation map
            pred_outs['fpn_feat'] = fpn_outs[self.TF_correlation_selected_layer]

        if self.training:

            return pred_outs
        else:

            pred_outs['conf'] = pred_outs['conf'].sigmoid() if self.cfg.MODEL.CLASS_HEADS.USE_FOCAL_LOSS \
                else pred_outs['conf'].softmax(dim=-1)
            # Detection
            pred_outs_after_NMS = self.Detect(self, pred_outs)

            # track instances frame-by-frame
            if self.cfg.MODEL.TRACK_HEADS.TRAIN_TRACK:
                pred_outs_after_track = self.Track(pred_outs_after_NMS, img_meta, x)
                return pred_outs_after_track

            else:
                return pred_outs_after_NMS


