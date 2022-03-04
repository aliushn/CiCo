import torch
import torch.nn as nn
import numpy as np
from utils import timer
from collections import defaultdict
from layers.backbones import construct_backbone, SwinTransformer
from layers.modules import PredictionModule_FC, ClipPredictionModule, ClipProtoNet, make_net, FPN, T2S_Head, DynamicMaskHead
from layers.functions import Detect, Track_TF

# This is required for Pytorch 1.0.1 on Windows to initialize Cuda on some driver versions.
# See the bug report here: https://github.com/pytorch/pytorch/issues/17108
torch.cuda.current_device()
prior_cache = defaultdict(lambda: None)


class CoreNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # ------------ build Backbone ------------
        if self.cfg.MODEL.BACKBONE.SWINT.engine:
            self.backbone = SwinTransformer(depths=cfg.MODEL.BACKBONE.SWINT.depths)
            bb_embed_dim = self.cfg.MODEL.BACKBONE.SWINT.embed_dim
            self.fpn = FPN(cfg.MODEL.FPN, [2**i*bb_embed_dim for i in cfg.MODEL.BACKBONE.SELECTED_LAYERS])
        else:
            self.backbone = construct_backbone(cfg.MODEL.BACKBONE)
            self.fpn = FPN(cfg.MODEL.FPN, [self.backbone.channels[i] for i in cfg.MODEL.BACKBONE.SELECTED_LAYERS])

        self.selected_layers = list(range(len(cfg.MODEL.BACKBONE.SELECTED_LAYERS)+cfg.MODEL.FPN.NUM_DOWNSAMPLE))
        self.fpn_num_features = cfg.MODEL.FPN.NUM_FEATURES
        self.fpn_num_downsample = cfg.MODEL.FPN.NUM_DOWNSAMPLE
        self.clip_frames = cfg.SOLVER.NUM_CLIP_FRAMES

        # ------------ build ProtoNet ------------
        self.mask_dim = cfg.MODEL.MASK_HEADS.MASK_DIM
        if cfg.MODEL.MASK_HEADS.USE_SIPMASK:
            self.mask_dim = self.mask_dim * cfg.MODEL.MASK_HEADS.SIPMASK_HEAD
        if cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
            self.ProtoNet = ClipProtoNet(cfg, self.fpn_num_features, self.mask_dim)
            if cfg.MODEL.MASK_HEADS.USE_DYNAMIC_MASK:
                self.DynamicMaskHead = DynamicMaskHead(cfg)
                self.mask_coeff_dim = self.DynamicMaskHead.num_gen_params
            else:
                self.mask_coeff_dim = self.mask_dim

        # ------- Build multi-scales prediction head  ------------
        self.num_priors = len(cfg.MODEL.BACKBONE.PRED_SCALES[0]) * len(cfg.MODEL.BACKBONE.PRED_ASPECT_RATIOS[0])
        # prediction layers for loc, conf, mask, track
        if cfg.STMASK.FC.USE_FCA:
            self.prediction_layers = PredictionModule_FC(cfg, self.fpn_num_features, self.mask_coeff_dim, deform_groups=1)
        else:
            self.prediction_layers = ClipPredictionModule(cfg, self.fpn_num_features, self.mask_coeff_dim, deform_groups=1)

        # ---------------- Build detection ----------------
        self.Detect = Detect(cfg)

        # ---------------- Build track head ----------------
        if cfg.MODEL.TRACK_HEADS.TRAIN_TRACK:
            if cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
                self.use_3D = True if cfg.MODEL.CiCo.CPN.CUBIC_MODE else False
                track_arch = [(self.fpn_num_features, 3, 1)] * 2 + [(cfg.MODEL.TRACK_HEADS.TRACK_DIM, 1, 0)]
                self.track_conv, _ = make_net(self.fpn_num_features, track_arch, use_3D=self.use_3D,
                                              norm_type='batch_norm', include_last_relu=False)

            if cfg.STMASK.T2S_HEADS.TEMPORAL_FUSION_MODULE:
                # A Temporal Fusion built on two adjacent frames for tracking in STMask, clik here for more details
                # https://arxiv.org/abs/2104.05606
                self.TF_correlation_selected_layer = cfg.STMASK.T2S_HEADS.CORRELATION_SELECTED_LAYER
                corr_channels = 2*self.fpn_num_features + cfg.STMASK.T2S_HEADS.CORRELATION_PATCH_SIZE**2
                self.T2S_Head = T2S_Head(cfg.STMASK.T2S_HEADS, corr_channels, self.mask_dim)

        # Tracking for inference
        if cfg.DATASETS.TYPE == 'vis':
            self.Track = Track_TF(self, cfg)

        if cfg.MODEL.MASK_HEADS.TRAIN_MASKS and cfg.MODEL.MASK_HEADS.USE_SEMANTIC_SEGMENTATION_LOSS:
            sem_seg_head = [(self.fpn_num_features, 3, 1)]*2 + [(cfg.DATASETS.NUM_CLASSES, 1, 0)]
            self.semantic_seg_conv, _ = make_net(self.fpn_num_features, sem_seg_head, norm_type='batch_norm',
                                                 include_last_relu=False)

    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)
        model_dict = self.state_dict()
        for key in list(state_dict.keys()):
            if key.startswith('module.'):
                state_dict[key[7:]] = state_dict.pop(key)

            new_key = None
            if key.split('.')[0] in {'mask_refine', 'proto_net', 'proto_conv'}:
                new_key = 'ProtoNet.'+key
            elif key.split('.')[0] == 'prediction_layers' and key.split('.')[1] == '0':
                new_key = 'prediction_layers.' + '.'.join(key.split('.')[2:])
            elif key.startswith('TemporalNet'):
                new_key = 'T2S_Head.TemporalFusionNet.' + '.'.join(key.split('.')[1:])
            if new_key is not None:
                state_dict[new_key] = state_dict.pop(key)

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
        """ Initialize weights for training, pre-training on CoCo. """
        print('Initialize weights from:', backbone_path)
        state_dict = torch.load(backbone_path, map_location=torch.device('cpu'))
        for key in list(state_dict.keys()):
            # In case of save models from distributed training
            if key.startswith('module'):
                state_dict[key[7:]] = state_dict.pop(key)
            # In case of save models without 'ProtoNet.'
            if key.split('.')[0] in {'mask_refine', 'proto_net', 'proto_conv'}:
                new_key = 'ProtoNet.'+key
                state_dict[new_key] = state_dict.pop(key)
            # In case of save models with shared prediction heads like Yolact
            if '.'.join(key.split('.')[:2]) in {'prediction_layers.0'}:
                new_key = 'prediction_layers.' + '.'.join(key.split('.')[2:])
                state_dict[new_key] = state_dict.pop(key)

        # First initialize the rest of the conv layers with xavier
        print('Init all weights by Xavier:')
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
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
        print('Finish init all weights by Xavier.')
        print()

        model_dict = self.state_dict()
        # only update same modules and layers between pre-trained model and our model
        for key in list(state_dict.keys()):
            if key not in model_dict.keys():
                print('Layers in pre-trained model but not in current model:', key)
            else:
                if model_dict[key].shape == state_dict[key].shape:
                    model_dict[key] = state_dict[key]
                else:
                    if not self.cfg.CiCo.ENGINE:
                        print('Size is different in pre-trained model and current model:', key)
                    else:
                        # Init weights from 2D to 3D
                        if 'conf_layer' in key:
                            continue

                        device = model_dict[key].device
                        if state_dict[key].dim() == 1:
                            if model_dict[key].size(0) % state_dict[key].size(0) == 0:
                                scale = model_dict[key].size(0)//state_dict[key].size(0)
                                model_dict[key] = state_dict[key].repeat(scale).to(device)
                            else:
                                print('Size is different in pre-trained model and current model:', key)
                        elif state_dict[key].dim() == 4 and model_dict[key].dim() == 5:
                            c_out_p, c_in_p, kh_p, kw_p = state_dict[key].size()
                            c_out, c_in, t, kh, kw = model_dict[key].size()
                            if [kh_p, kw_p] == [kh, kw] and c_out % c_out_p == 0:
                                print('load 3D parameters from pre-trained models:', key)
                                scale = c_out // c_out_p
                                model_dict[key] = state_dict[key].unsqueeze(2).repeat(scale,1,t,1,1).to(device)/t
                            else:
                                print('Size is different in pre-trained model and current model:', key)

        self.load_state_dict(model_dict, strict=True)

    def init_weights(self, backbone_path, local_rank=1):
        """ Initialize weights for training, pre-training on ImageNet-1K"""
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
            if is_conv_layer and not name.startswith('backbone'):  # module not in self.backbone.backbone_modules:
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

        if mode:
            if self.cfg.freeze_bn:
                self.freeze_bn()

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for name, module in self.named_modules():
            if name.startswith('backbone') and isinstance(module, nn.BatchNorm2d):
                print('Freeze BN of backbone: ', name)
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

    def forward_single(self, x, img_meta=None):
        """ The input should be of size [batch_size, 3, img_h, img_w] """

        with timer.env('Backbone+FPN'):
            bb_outs = self.backbone(x)

            # Use backbone.selected_layers because we overwrote self.selected_layers
            outs = [bb_outs[i] for i in self.cfg.MODEL.BACKBONE.SELECTED_LAYERS]
            fpn_outs = self.fpn(outs)

        with timer.env('Pred_heads'):
            pred_outs = self.prediction_layers(fpn_outs)

        with timer.env('Protonet'):
            if self.cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
                pred_outs['proto'] = self.ProtoNet(x, fpn_outs, img_meta=img_meta)

        if self.cfg.MODEL.TRACK_HEADS.TRAIN_TRACK and self.cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
            with timer.env('track_by_Gaussian'):
                track_in = fpn_outs[0]
                _, c, h, w = track_in.size()
                if self.use_3D:
                    track_in = fpn_outs[0].reshape(-1, self.clip_frames, c, h, w).transpose(1, 2)
                track_out = self.track_conv(track_in)
                pred_outs['track'] = track_out.permute(0, 2, 3, 1).contiguous() if not self.use_3D else \
                    track_out.permute(0, 2, 3, 4, 1).contiguous()

        if self.training:
            if self.cfg.MODEL.MASK_HEADS.USE_SEMANTIC_SEGMENTATION_LOSS or not self.cfg.MODEL.CLASS_HEADS.TRAIN_CLASS:
                with timer.env('sem_seg'):
                    pred_outs['sem_seg'] = self.semantic_seg_conv(fpn_outs[0]).permute(0, 2, 3, 1).contiguous()

        return fpn_outs, pred_outs

    def forward(self, x, img_meta=None, use_cross_class_nms=False):
        batch_size, c, h, w = x.size()
        fpn_outs, pred_outs = self.forward_single(x, img_meta)
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
            with timer.env('Detect'):
                pred_outs_after_NMS = self.Detect(self, pred_outs, use_cross_class_nms)

            # track instances frame-by-frame
            if self.cfg.DATASETS.TYPE == 'vis':
                with timer.env('Track_TF'):
                    pred_outs_after_track = self.Track(pred_outs_after_NMS, img_meta, x)
                return pred_outs_after_track
            else:
                return pred_outs_after_NMS


