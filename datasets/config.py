from backbone import ResNetBackbone, VGGBackbone, ResNetBackboneGN, DarkNetBackbone
from math import sqrt
import torch

# for making bounding boxes pretty
COLORS = ((244,  67,  54),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          (  0, 150, 136),
          ( 76, 175,  80),
          (139, 195,  74),
          (205, 220,  57),
          (255, 235,  59),
          (255, 193,   7),
          (255, 152,   0),
          (255,  87,  34),
          (121,  85,  72),
          (158, 158, 158),
          ( 96, 125, 139))

# These are in BGR and are for YouTubeVOS
MEANS = (123.675, 116.28, 103.53)
STD = (58.395, 57.12, 57.375)

COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock',
                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

COCO_LABEL_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                  9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}

YouTube_VOS_CLASSES = ('person', 'giant_panda', 'lizard', 'parrot', 'skateboard',
                       'sedan', 'ape', 'dog', 'snake', 'monkey',
                       'hand', 'rabbit', 'duck', 'cat', 'cow',
                       'fish', 'train', 'horse', 'turtle', 'bear',
                       'motorbike', 'giraffe', 'leopard', 'fox', 'deer',
                       'owl', 'surfboard', 'airplane', 'truck', 'zebra',
                       'tiger', 'elephant', 'snowboard', 'boat', 'shark',
                       'mouse', 'frog', 'eagle', 'earless seal', 'tennis_racket')

YouTube_VOS_LABEL_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                         9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16,
                         17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24,
                         25: 25, 26: 26, 27: 27, 28: 28, 29: 29, 30: 30, 31: 31, 32: 32,
                         33: 33, 34: 34, 35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 40: 40}

YouTube_VOS2021_CLASSES = ('airplane', 'bear', 'bird', 'boat', 'car',
                           'cat', 'cow', 'deer', 'dog', 'duck',
                           'earless_seal', 'elephant', 'fish', 'flying_disc', 'fox',
                           'frog', 'giant_panda', 'giraffe', 'horse', 'leopard',
                           'lizard', 'monkey', 'motorbike', 'mouse', 'parrot',
                           'person', 'rabbit', 'shark', 'skateboard', 'snake',
                           'snowboard', 'squirrel', 'surfboard', 'tennis_racket', 'tiger',
                           'train', 'truck', 'turtle', 'whale', 'zebra')

OVIS_CLASSES = ('person', 'bird', 'cat', 'dog', 'horse',
                'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'poultry', 'giant panda', 'lizard', 'parrot',
                'monkey', 'rabbit', 'tiger', 'fish', 'turtle',
                'bicycle', 'motorcycle', 'airplane', 'boat', 'vehicle')

OVIS_LABEL_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                  9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16,
                  17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24,
                  25: 25}


# ----------------------- CONFIG CLASS ----------------------- #


class Config(object):
    """
References[]    Holds the configuration for anything you want it to.
    To get the currently active config, call get_cfg().

    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """
        ret = Config(vars(self))

        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)

    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)

# ----------------------- DATASETS ----------------------- #

dataset_base_coco = Config({
    'name': 'Base Dataset',

    # Training images and annotations
    'train_images': '../datasets/coco/train2017/',
    'train_info': 'path_to_annotation_file',

    # Validation images and annotations.
    'valid_images': '../datasets/coco/val2017/',
    'valid_info': '../datasets/coco/annotations/',

    # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
    'has_gt': True,

    # A list of names for each of you classes.
    'class_names': COCO_CLASSES,

    # COCO class ids aren't sequential, so this is a bandage fix. If your ids aren't sequential,
    # provide a map from category_id -> index in class_names + 1 (the +1 is there because it's 1-indexed).
    # If not specified, this just assumes category ids start at 1 and increase sequentially.
    'label_map': None
})

coco2014_dataset = dataset_base_coco.copy({
    'name': 'COCO 2014',

    'train_info': './data/coco/annotations/instances_train2014.json',
    'valid_info': './data/coco/annotations/instances_val2014.json',

    'label_map': COCO_LABEL_MAP
})

coco2017_dataset = dataset_base_coco.copy({
    'name': 'COCO 2017',

    'train_info': '../datasets/coco/annotations/instances_train2017.json',
    'valid_info': '../datasets/coco/annotations/instances_val2017.json',

    'label_map': COCO_LABEL_MAP
})

coco2017_testdev_dataset = dataset_base_coco.copy({
    'name': 'COCO 2017 Test-Dev',

    'valid_info': '../datasets/coco/annotations/image_info_test-dev2017.json',
    'valid_images': '../datasets/coco/test2017/',
    'has_gt': False,

    'label_map': COCO_LABEL_MAP
})

# ----------------------- DATASETS ----------------------- #

dataset_base_vis = Config({
    'type': 'YTVOSDataset',

    # images and annotations path
    'ann_file': 'path_to_annotation_file',
    'img_prefix': 'path_to_images_file',
    'img_scales': [(640, 384), (800, 480), (960, 576)],
    'MS_train': True,
    'preserve_aspect_ratio': True,
    'tranform': None,
    'size_divisor': 32,
    'with_mask': True,
    'with_crowd': True,
    'with_label': True,
    'with_track': True,
    'clip_frames': 1,
    'has_gt': True,


})

train_YouTube_VOS_dataset = dataset_base_vis.copy({
    'img_prefix': '../datasets/YouTube_VOS2019/train/JPEGImages',
    'ann_file': '../datasets/YouTube_VOS2019/annotations_instances/train_sub.json',
    'has_gt': True,
})

valid_sub_YouTube_VOS_dataset = dataset_base_vis.copy({
    'img_prefix': '../datasets/YouTube_VOS2019/train/JPEGImages',
    'ann_file': '../datasets/YouTube_VOS2019/annotations_instances/valid_sub.json',
    'has_gt': True,
})

valid_YouTube_VOS_dataset = dataset_base_vis.copy({

    'img_prefix': '../datasets/YouTube_VOS2019/valid/JPEGImages',
    'ann_file': '../datasets/YouTube_VOS2019/annotations_instances/valid.json',
    'has_gt': False,
})

test_YouTube_VOS_dataset = dataset_base_vis.copy({

    'img_prefix': '../datasets/YouTube_VOS2019/test/JPEGImages',
    'ann_file': '../datasets/YouTube_VOS2019/annotations_instances/test.json',
    'has_gt': False,
})

train_YouTube_VOS2021_dataset = dataset_base_vis.copy({
    'img_prefix': '../datasets/YouTube_VOS2021/train/JPEGImages',
    'ann_file': '../datasets/YouTube_VOS2021/train/instances.json',
    'has_gt': True,
})

valid_sub_YouTube_VOS2021_dataset = dataset_base_vis.copy({
    'img_prefix': '../datasets/YouTube_VOS2021/train/JPEGImages',
    'ann_file': '../datasets/YouTube_VOS2021/annotations_instances/valid_sub.json',
    'has_gt': True,
})

valid_YouTube_VOS2021_dataset = dataset_base_vis.copy({

    'img_prefix': '../datasets/YouTube_VOS2021/valid/JPEGImages',
    'ann_file': '../datasets/YouTube_VOS2021/valid/instances.json',
    'has_gt': False,
})

test_YouTube_VOS2021_dataset = dataset_base_vis.copy({
    'img_prefix': '../datasets/YouTube_VOS2021/test/JPEGImages',
    'ann_file': '../datasets/YouTube_VOS2021/test/instances.json',
    'has_gt': False,
})


train_OVIS_dataset = dataset_base_vis.copy({
    'img_prefix': '../datasets/OVIS/train',
    'ann_file': '../datasets/OVIS/annotations_train.json',
})

valid_sub_OVIS_dataset = dataset_base_vis.copy({
    'img_prefix': '../datasets/OVIS/train',
    'ann_file': '../datasets/OVIS/annotations_valid_sub.json',
    'has_gt': False,
})

valid_OVIS_dataset = dataset_base_vis.copy({

    'img_prefix': '../datasets/OVIS/valid',
    'ann_file': '../datasets/OVIS/annotations_valid.json',
    'has_gt': False,
})

test_OVIS_dataset = dataset_base_vis.copy({

    'img_prefix': '../datasets/OVIS/test',
    'ann_file': '../datasets/OVIS/annotations_test.json',
    'has_gt': False,
})


# ----------------------- TRANSFORMS ----------------------- #

resnet_transform = Config({
    'channel_order': 'RGB',
    'normalize': True,
    'subtract_means': False,
    'to_float': False,
})

vgg_transform = Config({
    # Note that though vgg is traditionally BGR,
    # the channel order of vgg_reducedfc.pth is RGB.
    'channel_order': 'RGB',
    'normalize': False,
    'subtract_means': True,
    'to_float': False,
})

darknet_transform = Config({
    'channel_order': 'RGB',
    'normalize': False,
    'subtract_means': False,
    'to_float': True,
})

# ----------------------- BACKBONES ----------------------- #

backbone_base = Config({
    'name': 'Base Backbone',
    'path': 'path/to/pretrained/weights',
    'type': object,
    'args': tuple(),
    'transform': resnet_transform,

    'selected_layers': list(),
    'pred_scales': list(),
    'pred_aspect_ratios': list(),

    'use_pixel_scales': False,
    'preapply_sqrt': True,
    'use_square_anchors': False,
})

resnet101_backbone = backbone_base.copy({
    'name': 'ResNet101',
    'path': 'yolact_base_54_800000.pth',
    # 'path': 'resnet101_reducedfc.pth',
    'type': ResNetBackbone,
    'args': ([3, 4, 23, 3],),
    'transform': resnet_transform,

    'selected_layers': list(range(2, 8)),
    'pred_scales': [[1]] * 6,
    'pred_aspect_ratios': [[[0.66685089, 1.7073535, 0.87508774, 1.16524493, 0.49059086]]] * 6,
})

resnet101_gn_backbone = backbone_base.copy({
    'name': 'ResNet101_GN',
    'path': 'R-101-GN.pkl',
    'type': ResNetBackboneGN,
    'args': ([3, 4, 23, 3],),
    'transform': resnet_transform,

    'selected_layers': list(range(2, 8)),
    'pred_scales': [[1]] * 6,
    'pred_aspect_ratios': [[[0.66685089, 1.7073535, 0.87508774, 1.16524493, 0.49059086]]] * 6,
})


resnet101_dcn_inter3_backbone = resnet101_backbone.copy({
    'name': 'ResNet101_DCN_Interval3',
    # 'path': 'STMask_plus_base_kl_YTVIS2019.pth',
    'path': 'yolact_plus_base_54_800000.pth',
    'args': ([3, 4, 23, 3], [0, 4, 23, 3], 3),
})

resnet101_gn_dcn_inter3_backbone = resnet101_gn_backbone.copy({
    'name': 'ResNet101_GN_DCN_Interval3',
    'args': ([3, 4, 23, 3], [0, 4, 23, 3], 3),
})

resnet50_backbone = resnet101_backbone.copy({
    'name': 'ResNet50',
    'path': 'yolact_resnet50_54_800000.pth',
    'type': ResNetBackbone,
    'args': ([3, 4, 6, 3],),
    'transform': resnet_transform,
})

resnet50_dcn_inter2_backbone = resnet50_backbone.copy({
    'name': 'ResNet50_DCN_Interval3',
    'path': 'yolact_plus_resnet50_54.pth',
    'args': ([3, 4, 6, 3], [0, 4, 6, 3], 2),
})

resnet152_backbone = backbone_base.copy({
    'name': 'ResNet152',
    'path': 'resnet152-b121ed2d.pth',
    'type': ResNetBackbone,
    'args': ([3, 8, 36, 3],),
    'transform': resnet_transform,
})

resnet152_dcn_inter3_backbone = resnet152_backbone.copy({
    'name': 'ResNet152_DCN_Interval3',
    'args': ([3, 8, 36, 3], [0, 8, 36, 3], 3),
})

darknet53_backbone = backbone_base.copy({
    'name': 'DarkNet53',
    'path': 'darknet53.pth',
    'type': DarkNetBackbone,
    'args': ([1, 2, 8, 8, 4],),
    'transform': darknet_transform,

    'selected_layers': list(range(3, 9)),
    'pred_scales': [[3.5, 4.95], [3.6, 4.90], [3.3, 4.02], [2.7, 3.10], [2.1, 2.37], [1.8, 1.92]],
    'pred_aspect_ratios': [[[1, sqrt(2), 1 / sqrt(2), sqrt(3), 1 / sqrt(3)][:n], [1]] for n in [3, 5, 5, 5, 3, 3]],
})

vgg16_arch = [[64, 64],
              ['M', 128, 128],
              ['M', 256, 256, 256],
              [('M', {'kernel_size': 2, 'stride': 2, 'ceil_mode': True}), 512, 512, 512],
              ['M', 512, 512, 512],
              [('M', {'kernel_size': 3, 'stride': 1, 'padding': 1}),
               (1024, {'kernel_size': 3, 'padding': 6, 'dilation': 6}),
               (1024, {'kernel_size': 1})]]

vgg16_backbone = backbone_base.copy({
    'name': 'VGG16',
    'path': 'vgg16_reducedfc.pth',
    'type': VGGBackbone,
    'args': (vgg16_arch, [(256, 2), (128, 2), (128, 1), (128, 1)], [3]),
    'transform': vgg_transform,

    'selected_layers': [3] + list(range(5, 10)),
    'pred_scales': [[5, 4]] * 6,
    'pred_aspect_ratios': [[[1], [1, sqrt(2), 1 / sqrt(2), sqrt(3), 1 / sqrt(3)][:n]] for n in [3, 5, 5, 5, 3, 3]],
})

# ----------------------- ACTIVATION FUNCTIONS ----------------------- #

activation_func = Config({
    'tanh': torch.tanh,
    'sigmoid': torch.sigmoid,
    'softmax': lambda x: torch.nn.functional.softmax(x, dim=-1),
    'relu': lambda x: torch.nn.functional.relu(x, inplace=False),
    'none': lambda x: x,
})

# ----------------------- FPN DEFAULTS ----------------------- #

fpn_base = Config({
    # The number of features to have in each FPN layer
    'num_features': 256,

    # The upsampling mode used
    'interpolation_mode': 'bilinear',

    # The number of extra layers to be produced by downsampling starting at P5
    'num_downsample': 1,

    # Whether to down sample with a 3x3 stride 2 conv layer instead of just a stride 2 selection
    'use_conv_downsample': False,

    # Whether to pad the pred layers with 1 on each side (I forgot to add this at the start)
    # This is just here for backwards compatibility
    'pad': True,

    # Whether to add relu to the downsampled layers.
    'relu_downsample_layers': False,

    # Whether to add relu to the regular layers
    'relu_pred_layers': True,
})

# ----------------------- CONFIG DEFAULTS ----------------------- #


base_config = Config({
    'COLORS': COLORS,

    'max_epoch': 12,

    # The maximum number of detections for evaluation
    'max_num_detections': 100,

    # dw' = momentum * dw - lr * (grad + decay * w)
    'lr': 1e-3,
    'momentum': 0.9,
    'decay': 1e-4,

    # For each lr step, what to multiply the lr with
    'gamma': 0.1,
    'lr_steps': (6, 9),

    # Initial learning rate to linearly warmup from (if until > 0)
    'lr_warmup_init': 1e-4,

    # If > 0 then increase the lr linearly from warmup_init to lr each iter for until iters
    'lr_warmup_until': 500,

    # The terms to scale the respective loss by
    'conf_alpha': 1,
    'stuff_alpha': 1,
    'bbox_alpha': 1.5,
    'track_alpha': 5,
    'mask_alpha': 0.4 / 256 * 140 * 140,  # Some funky equation. Don't worry about it.

    # Eval.py sets this if you just want to run YOLACT as a detector
    'eval_mask_branch': True,

    # Top_k examples to consider for NMS
    'nms_top_k': 200,
    # Examples with confidence less than this are not considered by NMS
    'nms_conf_thresh': 0.3,
    # Boxes with IoU overlap greater than this threshold will be culled during NMS
    'nms_thresh': 0.5,
    # used in detection of eval, lower than conf_thresh will be ignored
    'eval_conf_thresh': 0.3,

    # See mask_type for details.
    'mask_size': 16,
    'masks_to_train': 100,
    'mask_proto_src': None,
    'mask_proto_net': [(256, 3, {}), (256, 3, {})],
    'mask_proto_prototype_activation': activation_func.relu,
    'mask_proto_mask_activation': activation_func.sigmoid,
    'mask_proto_coeff_activation': activation_func.tanh,
    'mask_proto_crop': False,
    'mask_proto_remove_empty_masks': False,
    'mask_proto_reweight_coeff': 1,
    'mask_proto_coeff_diversity_loss': False,
    'mask_proto_coeff_diversity_alpha': 1,
    'mask_proto_normalize_emulate_roi_pooling': False,
    'mask_proto_double_loss': False,
    'mask_proto_double_loss_alpha': 1,
    'mask_proto_crop_with_pred_box': False,

    # SSD data augmentation parameters
    # Randomize hue, vibrance, etc.
    'augment_photometric_distort': False,
    # Have a chance to scale down the image and pad (to emulate smaller detections)
    'augment_expand': False,
    # Potentialy sample a random crop from the image and put it in a random place
    'augment_random_sample_crop': False,
    # Mirror the image with a probability of 1/2
    'augment_random_mirror': False,
    # Flip the image vertically with a probability of 1/2
    'augment_random_flip': True,
    # With uniform probability, rotate the image [0,90,180,270] degrees
    'augment_random_rot90': False,

    # Set this to a config object if you want an FPN (inherit from fpn_base). See fpn_base for details.
    'fpn': None,

    # Use the same weights for each network head
    'share_prediction_module': False,

    # Whether to use sigmoid focal loss instead of softmax, all else being the same.
    'focal_loss_alpha': 0.25,
    'focal_loss_gamma': 2,

    # The initial bias toward forground objects, as specified in the focal loss paper
    'focal_loss_init_pi': 0.01,

    # Adds a 1x1 convolution directly to the biggest selected layer that predicts a semantic segmentations for each of the 80 classes.
    # This branch is only evaluated during training time and is just there for multitask learning.
    'use_semantic_segmentation_loss': False,
    'semantic_segmentation_alpha': 1,

    # Match gt boxes using the Box2Pix change metric instead of the standard IoU metric.
    # Note that the threshold you set for iou_threshold should be negative with this setting on.
    'use_change_matching': False,

    # What params should the final head layers have (the ones that predict box, confidence, and mask coeffs)
    'pred_conv_kernels':[[3,3], [3,5], [5,3]],

    # Add extra layers between the backbone and the network heads
    # The order is (conf, bbox, track, mask)
    'extra_layers': (0, 0, 0, 0),

    # During training, to match detections with gt, first compute the maximum gt IoU for each prior.
    # Then, any of those priors whose maximum overlap is over the positive threshold, mark as positive.
    # For any priors whose maximum is less than the negative iou threshold, mark them as negative.
    # The rest are neutral and not used in calculating the loss.
    'positive_iou_threshold': 0.5,
    'negative_iou_threshold': 0.4,

    # If less than 1, anchors treated as a negative that have a crowd iou over this threshold with
    # the crowd boxes will be treated as a neutral.
    'crowd_iou_threshold': 1,

    # This is filled in at runtime by Yolact's __init__, so don't touch it
    'mask_dim': None,

    # Whether or not to do post processing on the cpu at test time
    'force_cpu_nms': True,

    # Whether to use mask coefficient cosine similarity nms instead of bbox iou nms
    'use_coeff_nms': False,

    # Whether or not to tie the mask loss / box loss to 0
    'train_masks': True,
    'train_boxes': True,
    'train_track': True,
    'train_class': True,

    # Whether or not to preserve aspect ratio when resizing the image.
    # If True, uses the faster r-cnn resizing scheme.
    # If False, all images arte resized to max_size x max_size
    'preserve_aspect_ratio': True,

    # Whether or not to use the predicted coordinate scheme from Yolo v2
    'use_yolo_regressors': False,

    # For training, bboxes are considered "positive" if their anchors have a 0.5 IoU overlap
    # or greater with a ground truth box. If this is true, instead of using the anchor boxes
    # for this IoU computation, the matching function will use the predicted bbox coordinates.
    # Don't turn this on if you're not using yolo regressors!
    'use_prediction_matching': False,

    # Use command-line arguments to set this.
    'no_jit': False,

    'backbone': None,
    'name': 'base_config',

    # display output results in each epoch
    'train_output_visualization': True,

    # the format of mask in output .json file including 'polygon' and 'rle'.
    'mask_output_json': 'polygon',
})


# ----------------------- STMask CONFIGS ----------------------- #
STMask_base_config = base_config.copy({
    'name': 'STMask_base_config',
    'use_prediction_matching': False,

    # Dataset stuff
    'train_dataset': train_YouTube_VOS_dataset,
    'valid_sub_dataset': valid_sub_YouTube_VOS_dataset,
    'valid_dataset': valid_YouTube_VOS_dataset,
    'test_dataset': test_YouTube_VOS_dataset,
    'num_classes': 40,  # This should include the background class
    'classes': YouTube_VOS_CLASSES,

    # Training params
    'lr_steps': (8, 10),
    'max_epoch': 12,

    # loss
    'conf_alpha': 5,
    'stuff_alpha': 1,
    'bbox_alpha': 1,
    'BIoU_alpha': 2,
    'track_alpha': 5,
    'mask_proto_coeff_diversity_alpha': 1,
    'center_alpha': 1,

    # backbone and FCA settings
    'backbone': resnet101_backbone.copy({
        'selected_layers': list(range(1, 4)),

        'pred_aspect_ratios': [[[ 1, 1/2,  2 ]]] * 5,
        'pred_scales': [[i*2**(j/1.0) for j in range(1)] for i in [24, 48, 96, 192, 384]],
    }),

    # FPN Settings
    'fpn': fpn_base.copy({
        'num_features': 256,
        'use_conv_downsample': True,
        'num_downsample': 2,
    }),
    'use_bifpn': False,
    'num_bifpn': 3,

    # FCA and prediction module settings
    'share_prediction_module': True,
    'extra_layers': (4, 4, 4),   # class, box, track
    'pred_conv_kernels': [[3,3], [3,3], [3,3]],

    # Mask Settings
    'mask_alpha': 6.125,
    'mask_proto_src': [0, 1, 2],
    'mask_proto_crop': True,
    'mask_dim': 32,
    'mask_proto_with_levels': False,   # generate proto with per level
    'mask_proto_net': [(256, 3, {'padding': 1})] * 3 + [(None, -2, {})],
    'mask_proto_coeff_diversity_loss': False,
    'mask_proto_crop_with_pred_box': False,
    'mask_proto_coeff_occlusion': False,
    'mask_dice_coefficient': False,
    'mask_loss_with_ori_size': False,

    # Dynamic Mask Settings
    'use_dynamic_mask': False,
    'dynamic_mask_head_layers': 3,
    'disable_rel_coords': False,

    # SipMask uses multi heads for obtaining better mask segmentation
    'use_sipmask': False,
    'sipmask_head': 4,
    'use_semantic_segmentation_loss': False,
    'semantic_segmentation_alpha': 1,

    # Proto_net settings
    'display_protos': False,

    # train boxes
    'train_boxes': True,
    'train_centerness': True,
    'use_boxiou_loss': True,
    'use_repulsion_loss': False,

    # train class
    'train_class': True,
    'use_DIoU': False,
    'use_focal_loss': True,
    'focal_loss_alpha': 0.25,
    'focal_loss_gamma': 2,

    # Track settings
    'train_track': True,
    'track_by_Gaussian': False,
    'match_coeff': [0, 1, 1, 0],   # scores, mask_iou, box_iou, label
    'track_dim': 64,
    'track_crop_with_pred_mask': False,
    'track_crop_with_pred_box': False,

    # temporal settings
    'use_temporal_info': True,
    # temporal fusion module settings
    'temporal_fusion_module': True,
    'correlation_patch_size': 11,
    'correlation_selected_layer': 1,
    'boxshift_with_pred_box': False,
    'maskshift_alpha': 6.125,
    'maskshift_loss': False,
    'forward_flow': True,
    'backward_flow': False,

    # FCB settings
    # FCB_ada set 'use_pred_offset' as True, FCB_ali set 'use_pred_offset' as False
    'use_feature_calibration': False,
    'use_pred_offset': False,
    'use_random_offset': False,
    'use_dcn_class': False,
    'use_dcn_track': False,
    'use_dcn_mask': False,

    # loss settings
    'positive_iou_threshold': 0.5,
    'negative_iou_threshold': 0.4,
    'crowd_iou_threshold': 0.7,

    # eval
    'eval_frames_of_clip': 1,
    'nms_conf_thresh': 0.25,
    'nms_thresh': 0.5,
    'eval_conf_thresh': 0.25,
    'candidate_conf_thresh': 0.25,
    'nms_as_miou': False,
    'remove_false_inst': True,
    'add_missed_masks': False,
    'use_train_sub': False,
    'use_valid_sub': True,
    'use_test': False,
    'only_calc_metrics': False,
    'only_count_classes': False,
    'use_DIoU_in_comp_scores': False,
    'display_corr': False,
    'eval_single_im': False,
})


# ----------------------- STMask-plus CONFIGS ----------------------- #
STMask_resnet152_ori_config = STMask_base_config.copy({
    'name': 'STMask_resnet152_base',

    'backbone': resnet152_backbone.copy({
        'path': 'STMask_resnet152_coco_ori_19.pth',
        'selected_layers': list(range(1, 4)),
        'pred_aspect_ratios': STMask_base_config.backbone.pred_aspect_ratios,   # FCA
        'pred_scales': STMask_base_config.backbone.pred_scales,
    }),

})

# only turn on feature calibration for anchors (FCA) and temporal fusion module (TF)
STMask_plus_base_config = STMask_base_config.copy({
    'name': 'STMask_plus_base',

    'backbone': resnet101_dcn_inter3_backbone.copy({
        'selected_layers': list(range(1, 4)),
        'pred_aspect_ratios': [[[ [3, 3], [3, 5],  [5, 3] ]]] * 5,   # FCA
        'pred_scales': STMask_base_config.backbone.pred_scales,
    }),

    'use_feature_calibration': True,
    'pred_conv_kernels': [[3,3], [3,5], [5,3]],

})

STMask_plus_base_ada_config = STMask_plus_base_config.copy({
    'name': 'STMask_plus_base_ada',

    # FCB settings
    'use_feature_calibration': True,
    'use_pred_offset': True,
    'use_dcn_class': True,
    'use_dcn_track': False,
    'use_dcn_mask': False,

})

STMask_plus_base_ali_config = STMask_plus_base_config.copy({
    'name': 'STMask_plus_base_ali',

    # FCB settings
    'use_feature_calibration': True,
    'use_pred_offset': False,
    'use_dcn_class': True,
    'use_dcn_track': False,
    'use_dcn_mask': False,

})

# ----------------------- STMask-resnet50 CONFIGS ----------------------- #
STMask_resnet50_config = STMask_base_config.copy({
    'name': 'STMask_resnet50',
    'backbone': resnet50_backbone.copy({
        'selected_layers': list(range(1, 4)),
        'pred_scales': STMask_base_config.backbone.pred_scales,
        'pred_aspect_ratios': STMask_base_config.backbone.pred_aspect_ratios,
    }),
})

STMask_plus_resnet50_config = STMask_plus_base_config.copy({
    'name': 'STMask_plus_resnet50',
    'backbone': resnet50_dcn_inter2_backbone.copy({
        'selected_layers': list(range(1, 4)),

        'pred_scales': STMask_plus_base_config.backbone.pred_scales,
        'pred_aspect_ratios': STMask_plus_base_config.backbone.pred_aspect_ratios,
    }),

})

STMask_plus_resnet50_ada_config = STMask_plus_resnet50_config.copy({
    'name': 'STMask_plus_resnet50_ada',

    # FCB settings
    'use_feature_calibration': True,
    'use_pred_offset': True,
    'use_dcn_class': True,
    'use_dcn_track': False,
    'use_dcn_mask': False,

})

STMask_plus_resnet50_ali_config = STMask_plus_resnet50_config.copy({
    'name': 'STMask_plus_resnet50_ali',

    # FCB settings
    'use_feature_calibration': True,
    'use_pred_offset': False,
    'use_dcn_class': True,
    'use_dcn_track': False,
    'use_dcn_mask': False,

})


STMask_darknet53_config = STMask_base_config.copy({
    'name': 'STMask_darknet53',
    'backbone': darknet53_backbone.copy({
        'selected_layers': list(range(2, 5)),
        'pred_scales': STMask_base_config.backbone.pred_scales,
        'pred_aspect_ratios': STMask_base_config.backbone.pred_aspect_ratios,
        'use_pixel_scales': True,
        'preapply_sqrt': False,
        'use_square_anchors': True,  # This is for backward compatability with a bug
    }),
})

# ------------------------ OVIS datasets -----------------------
STMask_base_OVIS_config = STMask_base_config.copy({
    'name': 'STMask_base_OVIS',

    # Dataset stuff
    'train_dataset': train_OVIS_dataset,
    'valid_sub_dataset': valid_sub_OVIS_dataset,
    'valid_dataset': valid_OVIS_dataset,
    'test_dataset': test_OVIS_dataset,
    'num_classes': 25,  # This should include the background class
    'classes': OVIS_CLASSES,
})

STMask_plus_base_OVIS_config = STMask_plus_base_config.copy({
    'name': 'STMask_plus_base_OVIS',

    # Dataset stuff
    'train_dataset': train_OVIS_dataset,
    'valid_sub_dataset': valid_sub_OVIS_dataset,
    'valid_dataset': valid_OVIS_dataset,
    'test_dataset': test_OVIS_dataset,
    'num_classes': 25,  # This should include the background class
    'classes': OVIS_CLASSES,
})

STMask_plus_base_ada_OVIS_config = STMask_plus_base_ada_config.copy({
    'name': 'STMask_plus_base_ada_OVIS',

    # Dataset stuff
    'train_dataset': train_OVIS_dataset,
    'valid_sub_dataset': valid_sub_OVIS_dataset,
    'valid_dataset': valid_OVIS_dataset,
    'test_dataset': test_OVIS_dataset,
    'num_classes': 25,  # This should include the background class
    'classes': OVIS_CLASSES,
})

STMask_plus_base_ali_OVIS_config = STMask_plus_base_ali_config.copy({
    'name': 'STMask_plus_base_ada_OVIS',

    # Dataset stuff
    'train_dataset': train_OVIS_dataset,
    'valid_sub_dataset': valid_sub_OVIS_dataset,
    'valid_dataset': valid_OVIS_dataset,
    'test_dataset': test_OVIS_dataset,
    'num_classes': 25,  # This should include the background class
    'classes': OVIS_CLASSES,
})

STMask_plus_resnet50_OVIS_config = STMask_plus_resnet50_config.copy({
    'name': 'STMask_plus_resnet50_OVIS',

    # Dataset stuff
    'train_dataset': train_OVIS_dataset,
    'valid_sub_dataset': valid_sub_OVIS_dataset,
    'valid_dataset': valid_OVIS_dataset,
    'test_dataset': test_OVIS_dataset,
    'num_classes': 25,  # This should include the background class
    'classes': OVIS_CLASSES,
})

STMask_plus_resnet50_ada_OVIS_config = STMask_plus_resnet50_ada_config.copy({
    'name': 'STMask_plus_resnet50_ada_OVIS',

    # Dataset stuff
    'train_dataset': train_OVIS_dataset,
    'valid_sub_dataset': valid_sub_OVIS_dataset,
    'valid_dataset': valid_OVIS_dataset,
    'test_dataset': test_OVIS_dataset,
    'num_classes': 25,  # This should include the background class
    'classes': OVIS_CLASSES,
})

STMask_plus_resnet50_ali_OVIS_config = STMask_plus_resnet50_ali_config.copy({
    'name': 'STMask_plus_resnet50_ada_OVIS',

    # Dataset stuff
    'train_dataset': train_OVIS_dataset,
    'valid_sub_dataset': valid_sub_OVIS_dataset,
    'valid_dataset': valid_OVIS_dataset,
    'test_dataset': test_OVIS_dataset,
    'num_classes': 25,  # This should include the background class
    'classes': OVIS_CLASSES,
})

STMask_resnet152_OVIS_ori_config = STMask_resnet152_ori_config.copy({
    'name': 'STMask_resnet152_OVIS_ori',

    # Dataset stuff
    'train_dataset': train_OVIS_dataset,
    'valid_sub_dataset': valid_sub_OVIS_dataset,
    'valid_dataset': valid_OVIS_dataset,
    'test_dataset': test_OVIS_dataset,
    'num_classes': 25,  # This should include the background class
    'classes': OVIS_CLASSES,

})


STMask_plus_resnet152_OVIS_config = STMask_base_OVIS_config.copy({
    'name': 'STMask_plus_resnet152_OVIS',
    'backbone': resnet152_dcn_inter3_backbone.copy({
        'path': 'STMask_plus_resnet152_coco_20.pth',
        'selected_layers': list(range(1, 4)),
        'pred_scales': STMask_plus_base_config.backbone.pred_scales,
        'pred_aspect_ratios': STMask_plus_base_config.backbone.pred_aspect_ratios,
    }),
    'use_feature_calibration': True,
    'pred_conv_kernels': [[3,3], [3,5], [5,3]],

})

# --------------------------- YTVIS2021 datasets ----------------------
STMask_plus_base_YTVIS2021_config = STMask_plus_base_config.copy({
    'name': 'STMask_plus_base_YTVIS2021',

    # Dataset stuff
    'train_dataset': train_YouTube_VOS2021_dataset,
    'valid_sub_dataset': valid_sub_YouTube_VOS2021_dataset,
    'valid_dataset': valid_YouTube_VOS2021_dataset,
    'test_dataset': test_YouTube_VOS2021_dataset,
    'num_classes': 40,  # This should include the background class
    'classes': YouTube_VOS2021_CLASSES,
})

STMask_plus_base_ada_YTVIS2021_config = STMask_plus_base_ada_config.copy({
    'name': 'STMask_plus_base_ada_YTVIS2021',

    # Dataset stuff
    'train_dataset': train_YouTube_VOS2021_dataset,
    'valid_sub_dataset': valid_sub_YouTube_VOS2021_dataset,
    'valid_dataset': valid_YouTube_VOS2021_dataset,
    'test_dataset': test_YouTube_VOS2021_dataset,
    'num_classes': 40,  # This should include the background class
    'classes': YouTube_VOS2021_CLASSES,
})

STMask_plus_resnet50_YTVIS2021_config = STMask_plus_resnet50_config.copy({
    'name': 'STMask_plus_resnet50_YTVIS2021',

    # Dataset stuff
    'train_dataset': train_YouTube_VOS2021_dataset,
    'valid_sub_dataset': valid_sub_YouTube_VOS2021_dataset,
    'valid_dataset': valid_YouTube_VOS2021_dataset,
    'test_dataset': test_YouTube_VOS2021_dataset,
    'num_classes': 40,  # This should include the background class
    'classes': YouTube_VOS2021_CLASSES,
})

STMask_plus_resnet50_ada_YTVIS2021_config = STMask_plus_resnet50_ada_config.copy({
    'name': 'STMask_plus_resnet50_ada_YTVIS2021',

    # Dataset stuff
    'train_dataset': train_YouTube_VOS2021_dataset,
    'valid_sub_dataset': valid_sub_YouTube_VOS2021_dataset,
    'valid_dataset': valid_YouTube_VOS2021_dataset,
    'test_dataset': test_YouTube_VOS2021_dataset,
    'num_classes': 40,  # This should include the background class
    'classes': YouTube_VOS2021_CLASSES,
})

# ----------------------- COCO_YOLACT++ CONFIGS ----------------------- #

STMask_base_coco_ori_config = STMask_base_config.copy({
    'name': 'STMask_base_coco_ori',

    # Dataset stuff
    'dataset': coco2017_dataset,
    'num_classes': len(coco2017_dataset.class_names),

    # Image Size
    'MS_train': True,
    'preserve_aspect_ratio': True,
    'min_size': 640,
    'max_size': 768,

    # Training params
    'lr_steps': (12, 16, 20),
    'max_epoch': 24,

    'backbone': STMask_base_config.backbone.copy({
        'path': 'resnet101_reducedfc.pth',
    }),

    'train_track': False,

})

STMask_base_coco_config = STMask_base_coco_ori_config.copy({
    'name': 'STMask_base_coco',

    'backbone': STMask_base_config.backbone.copy({
        'path': 'resnet101_reducedfc.pth',
        'pred_aspect_ratios': STMask_plus_base_config.backbone.pred_aspect_ratios,   # FCA
        'pred_scales': STMask_plus_base_config.backbone.pred_scales,
    }),

    'use_feature_calibration': True,
    'pred_conv_kernels': [[3,3], [3,5], [5,3]],

})

STMask_plus_base_coco_config = STMask_base_coco_config.copy({
    'name': 'STMask_plus_base_coco',

    'backbone': STMask_plus_base_config.backbone.copy({
        'path': 'resnet101_reducedfc.pth',
    }),

})

STMask_resnet50_coco_ori_config = STMask_base_coco_ori_config.copy({
    'name': 'STMask_resnet50_coco_ori',

    'backbone': STMask_resnet50_config.backbone.copy({
        'path': 'resnet50-19c8e357.pth',
        'selected_layers': list(range(1, 4)),

        'pred_aspect_ratios': STMask_base_config.backbone.pred_aspect_ratios,
        'pred_scales': STMask_base_config.backbone.pred_scales,

    }),
})

STMask_plus_resnet50_coco_config = STMask_resnet50_coco_ori_config.copy({
    'name': 'STMask_plus_resnet50_coco',

    'backbone': STMask_plus_resnet50_config.backbone.copy({
        'path': 'resnet50-19c8e357.pth',

    }),

    'use_feature_calibration': True,
    'pred_conv_kernels': [[3,3], [3,5], [5,3]],
})

STMask_resnet152_coco_ori_config = STMask_base_coco_ori_config.copy({
    'name': 'STMask_resnet152_coco_ori',
    'backbone': resnet152_backbone.copy({
        'selected_layers': list(range(1, 4)),
        'pred_scales': STMask_base_config.backbone.pred_scales,
        'pred_aspect_ratios': STMask_base_config.backbone.pred_aspect_ratios,
    }),

})

STMask_plus_resnet152_coco_config = STMask_plus_base_coco_config.copy({
    'name': 'STMask_plus_resnet152_coco',
    'backbone': resnet152_dcn_inter3_backbone.copy({
        'selected_layers': list(range(1, 4)),
        'pred_scales': STMask_base_config.backbone.pred_scales,
        'pred_aspect_ratios': STMask_base_config.backbone.pred_aspect_ratios,
    }),

    'use_feature_calibration': False,
    'pred_conv_kernels': [[3,3], [3,3], [3,3]],
})


# Default config
cfg = STMask_plus_base_config.copy()


def set_cfg(config_name: str):
    """ Sets the active config. Works even if cfg is already imported! """
    global cfg

    # Note this is not just an eval because I'm lazy, but also because it can
    # be used like ssd300_config.copy({'max_size': 400}) for extreme fine-tuning
    cfg.replace(eval(config_name))

    if cfg.name is None:
        cfg.name = config_name.split('_config')[0]


def set_dataset(dataset_name: str, type: str):
    """ Sets the dataset of the current config. """
    if type == 'train':
        cfg.train_dataset = eval(dataset_name)
    elif type == 'eval':
        cfg.valid_dataset = eval(dataset_name)