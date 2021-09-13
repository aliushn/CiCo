# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the maximum image side during training will be
# INPUT.MAX_SIZE_TRAIN, while for testing it will be
# INPUT.MAX_SIZE_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.RPN_ONLY = False
_C.MODEL.MASK_ON = False
_C.MODEL.RETINANET_ON = False
_C.MODEL.KEYPOINT_ON = False
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
_C.MODEL.CLS_AGNOSTIC_BBOX_REG = False

# If the WEIGHT starts with a catalog://, like :R-50, the code will look for
# the path in paths_catalog. Else, it will use it as the specified absolute
# path
_C.MODEL.WEIGHT = ""


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = (800,)  # (800,)
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing
_C.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1333
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [1., 1., 1.]
# Convert image to BGR format (for Caffe2 models), in range 0-255
_C.INPUT.TO_BGR255 = True
_C.INPUT.MULTISCALE_TRAIN = True
_C.INPUT.PRESERVE_ASPECT_RATIO = True
_C.INPUT.SIZE_DIVISOR = 32

# Image ColorJitter
_C.INPUT.BRIGHTNESS = 0.0
_C.INPUT.CONTRAST = 0.0
_C.INPUT.SATURATION = 0.0
_C.INPUT.HUE = 0.0

# Flips
_C.INPUT.HORIZONTAL_FLIP_PROB_TRAIN = 0.5
_C.INPUT.VERTICAL_FLIP_PROB_TRAIN = 0.0

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.TYPE = ''
_C.DATASETS.NUM_CLASSES = 0
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ''
# List of the dataset names for valid, as present in paths_catalog.py
_C.DATASETS.VALID_SUB = ''
_C.DATASETS.VALID = ''
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ''

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 0
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = True


# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

# The backbone conv body to use
# The string must match a function that is imported in modeling.model_builder
# (e.g., 'FPN.add_fpn_ResNet101_conv5_body' to specify a ResNet-101-FPN
# backbone)
_C.MODEL.BACKBONE.CONV_BODY = 'ResNet101'
_C.MODEL.BACKBONE.PATH = 'resnet101_reducedfc.pth'
_C.MODEL.BACKBONE.TYPE = 'ResNetBackbone'
_C.MODEL.BACKBONE.ARGS = ([3, 4, 6, 3]),
_C.MODEL.BACKBONE.SELECTED_LAYERS = list(range(2, 8))
_C.MODEL.BACKBONE.PRED_SCALES = [[32], [64], [128], [256], [512]]
_C.MODEL.BACKBONE.PRED_ASPECT_RATIOS = [[[0.66685089, 1.7073535, 0.87508774, 1.16524493, 0.49059086]]]

# Add StopGrad at a specified stage so the bottom layers are frozen
_C.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 2

# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.FPN = CN()
_C.MODEL.FPN.USE_GN = False
_C.MODEL.FPN.USE_RELU = False
_C.MODEL.FPN.NUM_FEATURES = 256
_C.MODEL.FPN.INTERPOLATION_MODE = 'bilinear'
# The number of extra layers to be produced by downsampling starting at P5
_C.MODEL.FPN.NUM_DOWNSAMPLE = 2
# Whether to down sample with a 3x3 stride 2 conv layer instead of just a stride 2 selection
_C.MODEL.FPN.USE_CONV_DOWNSAMPLE = False
# Whether to pad the pred layers with 1 on each side (I forgot to add this at the start)
# This is just here for backwards compatibility
_C.MODEL.FPN.PAD = True
# Whether to add relu to the downsampled layers.
_C.MODEL.FPN.RELU_DOWNSAMPLE_LAYERS = False
# Whether to add relu to the regular layers
_C.MODEL.FPN.RELU_PRED_LAYERS = True
_C.MODEL.FPN.USE_BIFPN = False


# ---------------------------------------------------------------------------- #
# Group Norm options
# ---------------------------------------------------------------------------- #
_C.MODEL.GROUP_NORM = CN()
# Number of dimensions per group in GroupNorm (-1 if using NUM_GROUPS)
_C.MODEL.GROUP_NORM.DIM_PER_GP = -1
# Number of groups in GroupNorm (-1 if using DIM_PER_GP)
_C.MODEL.GROUP_NORM.NUM_GROUPS = 32
# GroupNorm's small constant in the denominator
_C.MODEL.GROUP_NORM.EPSILON = 1e-5


# ---------------------------------------------------------------------------- #
# PREDICTION HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.PREDICTION_HEADS = CN()
_C.MODEL.PREDICTION_HEADS.USE_DIoU = True
_C.MODEL.PREDICTION_HEADS.SHARE_PREDICTION_MODULE = True
# Whether to use predicted boxes to define positive and negative samples
_C.MODEL.PREDICTION_HEADS.USE_PREDICTION_MATCHING = True
_C.MODEL.PREDICTION_HEADS.POSITIVE_IoU_THRESHOLD = 0.5
_C.MODEL.PREDICTION_HEADS.NEGATIVE_IoU_THRESHOLD = 0.4
_C.MODEL.PREDICTION_HEADS.CROWD_IoU_THRESHOLD = 0.7

# CUBIC PREDICTION HEADS Options
_C.MODEL.PREDICTION_HEADS.CUBIC_MODE = False
_C.MODEL.PREDICTION_HEADS.CUBIC_CORRELATION_MODE = False
_C.MODEL.PREDICTION_HEADS.CUBIC_3D_MODE = False
_C.MODEL.PREDICTION_HEADS.CORRELATION_PATCH_SIZE = 5
# INITIALIZATION: reduced or inflated
_C.MODEL.PREDICTION_HEADS.CUBIC_MODE_WITH_INITIALIZATION = ''
_C.MODEL.PREDICTION_HEADS.CIRCUMSCRIBED_BOXES = False


# ---------------------------------------------------------------------------- #
# DETECTION HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.BOX_HEADS = CN()
_C.MODEL.BOX_HEADS.TRAIN_BOXES = True
_C.MODEL.BOX_HEADS.TOWER_LAYERS = 2
_C.MODEL.BOX_HEADS.TRAIN_CENTERNESS = True
_C.MODEL.BOX_HEADS.USE_BOXIOU_LOSS = True
_C.MODEL.BOX_HEADS.USE_REPULSION_LOSS = False
_C.MODEL.BOX_HEADS.LOSS_ALPHA = 1
_C.MODEL.BOX_HEADS.BIoU_ALPHA = 1
_C.MODEL.BOX_HEADS.CENTERNESS_ALPHA = 1


# ---------------------------------------------------------------------------- #
# CLASSIFICATION HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.CLASS_HEADS = CN()
_C.MODEL.CLASS_HEADS.TRAIN_CLASS = True
_C.MODEL.CLASS_HEADS.TOWER_LAYERS = 2
_C.MODEL.CLASS_HEADS.USE_FOCAL_LOSS = True
_C.MODEL.CLASS_HEADS.FOCAL_LOSS_ALPHA = 0.25
_C.MODEL.CLASS_HEADS.FOCAL_LOSS_GAMMA = 2
_C.MODEL.CLASS_HEADS.LOSS_ALPHA = 5


# ---------------------------------------------------------------------------- #
# TRACKING HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.TRACK_HEADS = CN()
_C.MODEL.TRACK_HEADS.TRAIN_TRACK = True
_C.MODEL.TRACK_HEADS.TRACK_DIM = 64
_C.MODEL.TRACK_HEADS.TOWER_LAYERS = 2
_C.MODEL.TRACK_HEADS.LOSS_ALPHA = 5
_C.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN = False
_C.MODEL.TRACK_HEADS.CROP_WITH_PRED_MASK = False
# Weights of other clues: scores, mask_iou, box_iou, label, for tracking
_C.MODEL.TRACK_HEADS.MATCH_COEFF = [0, 1, 1, 0]


# ---------------------------------------------------------------------------- #
# MASK HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.MASK_HEADS = CN()
_C.MODEL.MASK_HEADS.TRAIN_MASKS = True
_C.MODEL.MASK_HEADS.MASK_DIM = 16
_C.MODEL.MASK_HEADS.TOWER_LAYERS = 2
# Which layers will be used to obtain prototypes
_C.MODEL.MASK_HEADS.PROTO_SRC = [0]
_C.MODEL.MASK_HEADS.PROTO_NET = ()
_C.MODEL.MASK_HEADS.PROTO_CROP = True
_C.MODEL.MASK_HEADS.PROTO_CROP_WITH_PRED_BOX = False
# generate coefficients with per level
_C.MODEL.MASK_HEADS.LOSS_ALPHA = 6.125
_C.MODEL.MASK_HEADS.LOSS_WITH_OIR_SIZE = False
_C.MODEL.MASK_HEADS.LOSS_WITH_DICE_COEFF = False
_C.MODEL.MASK_HEADS.SEMANTIC_SEGMENTATION_ALPHA = 1
_C.MODEL.MASK_HEADS.USE_SEMANTIC_SEGMENTATION_LOSS = False
# proto diversity loss
_C.MODEL.MASK_HEADS.PROTO_COEFF_DIVERSITY_LOSS = False
_C.MODEL.MASK_HEADS.PROTO_COEFF_DIVERSITY_ALPHA = 1
_C.MODEL.MASK_HEADS.PROTO_COEFF_OCCLUSION = False

# SipMask (ICCV2020) uses multi heads for obtaining better mask segmentation
_C.MODEL.MASK_HEADS.USE_SIPMASK = False
_C.MODEL.MASK_HEADS.SIPMASK_HEAD = 4

# Dynamic Mask Settings proposed by CondInst (ICCV 2020)
_C.MODEL.MASK_HEADS.USE_DYNAMIC_MASK = False
_C.MODEL.MASK_HEADS.DYNAMIC_MASK_HEAD_LAYERS = 3
_C.MODEL.MASK_HEADS.DISABLE_REL_COORDS = False


# ---------------------------------------------------------------------------- #
# STMask (CVPR2021) Options
# ---------------------------------------------------------------------------- #
_C.STMASK = CN()

# FCB settings
# FCB_ada set 'use_pred_offset' as True, FCB_ali set 'use_pred_offset' as False
_C.STMASK.FC = CN()
_C.STMASK.FC.USE_FCA = False
_C.STMASK.FC.FCA_CONV_KERNELS = [[3,3], [3,3], [3,3]]
_C.STMASK.FC.USE_FCB = False
_C.STMASK.FC.FCB_USE_PRED_OFFSET = False
_C.STMASK.FC.FCB_USE_DCN_CLASS = False
_C.STMASK.FC.FCB_USE_DCN_TRACK = False
_C.STMASK.FC.FCB_USE_DCN_MASK = False

_C.STMASK.T2S_HEADS = CN()
_C.STMASK.T2S_HEADS.TEMPORAL_FUSION_MODULE = False
_C.STMASK.T2S_HEADS.CORRELATION_PATCH_SIZE = 5
_C.STMASK.T2S_HEADS.CORRELATION_SELECTED_LAYER = 1
_C.STMASK.T2S_HEADS.TRAIN_BOXSHIFT = True
_C.STMASK.T2S_HEADS.BOXSHIFT_ALPHA = 1
_C.STMASK.T2S_HEADS.TRAIN_MASKSHIFT = False
_C.STMASK.T2S_HEADS.MASKSHIFT_ALPHA = 6.125
_C.STMASK.T2S_HEADS.SHIFT_WITH_PRED_BOX = False
_C.STMASK.T2S_HEADS.FORWARD_FLOW = True
_C.STMASK.T2S_HEADS.BACKWARD_FLOW = False


# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
_C.MODEL.RESNETS.NUM_GROUPS = 1

# Baseline width of each group
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True

# Residual transformation function
_C.MODEL.RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"
# ResNet's stem function (conv1 and pool1)
_C.MODEL.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"

# Apply dilation in stage "res5"
_C.MODEL.RESNETS.RES5_DILATION = 1

_C.MODEL.RESNETS.BACKBONE_OUT_CHANNELS = 256 * 4
_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

_C.MODEL.RESNETS.STAGE_WITH_DCN = (False, False, False, False)
_C.MODEL.RESNETS.WITH_MODULATED_DCN = False
_C.MODEL.RESNETS.DEFORMABLE_GROUPS = 1


# ---------------------------------------------------------------------------- #
# RetinaNet Options (Follow the Detectron version)
# ---------------------------------------------------------------------------- #
_C.MODEL.RETINANET = CN()

# This is the number of foreground classes and background.
_C.MODEL.RETINANET.NUM_CLASSES = 81

# Anchor aspect ratios to use
_C.MODEL.RETINANET.ANCHOR_SIZES = (32, 64, 128, 256, 512)
_C.MODEL.RETINANET.ASPECT_RATIOS = (0.5, 1.0, 2.0)
_C.MODEL.RETINANET.ANCHOR_STRIDES = (8, 16, 32, 64, 128)
_C.MODEL.RETINANET.STRADDLE_THRESH = 0

# Anchor scales per octave
_C.MODEL.RETINANET.OCTAVE = 2.0
_C.MODEL.RETINANET.SCALES_PER_OCTAVE = 3

# Use C5 or P5 to generate P6
_C.MODEL.RETINANET.USE_C5 = True

# Convolutions to use in the cls and bbox tower
# NOTE: this doesn't include the last conv for logits
_C.MODEL.RETINANET.NUM_CONVS = 4

# Weight for bbox_regression loss
_C.MODEL.RETINANET.BBOX_REG_WEIGHT = 4.0

# Smooth L1 loss beta for bbox regression
_C.MODEL.RETINANET.BBOX_REG_BETA = 0.11

# During inference, #locs to select based on cls score before NMS is performed
# per FPN level
_C.MODEL.RETINANET.PRE_NMS_TOP_N = 1000

# IoU overlap ratio for labeling an anchor as positive
# Anchors with >= iou overlap are labeled positive
_C.MODEL.RETINANET.FG_IOU_THRESHOLD = 0.5

# IoU overlap ratio for labeling an anchor as negative
# Anchors with < iou overlap are labeled negative
_C.MODEL.RETINANET.BG_IOU_THRESHOLD = 0.4

# Focal loss parameter: alpha
_C.MODEL.RETINANET.LOSS_ALPHA = 0.25

# Focal loss parameter: gamma
_C.MODEL.RETINANET.LOSS_GAMMA = 2.0

# Prior prob for the positives at the beginning of training. This is used to set
# the bias init for the logits layer
_C.MODEL.RETINANET.PRIOR_PROB = 0.01

# Inference cls score threshold, anchors with score > INFERENCE_TH are
# considered for inference
_C.MODEL.RETINANET.INFERENCE_TH = 0.05

# NMS threshold used in RetinaNet
_C.MODEL.RETINANET.NMS_TH = 0.4


# ---------------------------------------------------------------------------- #
# FBNet options
# ---------------------------------------------------------------------------- #
_C.MODEL.FBNET = CN()
_C.MODEL.FBNET.ARCH = "default"
# custom arch
_C.MODEL.FBNET.ARCH_DEF = ""
_C.MODEL.FBNET.BN_TYPE = "bn"
_C.MODEL.FBNET.SCALE_FACTOR = 1.0
# the output channels will be divisible by WIDTH_DIVISOR
_C.MODEL.FBNET.WIDTH_DIVISOR = 1
_C.MODEL.FBNET.DW_CONV_SKIP_BN = True
_C.MODEL.FBNET.DW_CONV_SKIP_RELU = True

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.DET_HEAD_LAST_SCALE = 1.0
_C.MODEL.FBNET.DET_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.DET_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.KPTS_HEAD_LAST_SCALE = 0.0
_C.MODEL.FBNET.KPTS_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.KPTS_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.MASK_HEAD_LAST_SCALE = 0.0
_C.MODEL.FBNET.MASK_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.MASK_HEAD_STRIDE = 0

# 0 to use all blocks defined in arch_def
_C.MODEL.FBNET.RPN_HEAD_BLOCKS = 0
_C.MODEL.FBNET.RPN_BN_TYPE = ""


# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER = CN()
_C.SOLVER.IMS_PER_BATCH = 16
_C.SOLVER.NUM_CLIP_FRAMES = 1
_C.SOLVER.NUM_WORKERS = 4
_C.SOLVER.LR = 0.001
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.LR_STEPS = (8, 10)
_C.SOLVER.MAX_EPOCH = 12
_C.SOLVER.LR_WARMUP_UNTIL = 500
_C.SOLVER.LR_WARMUP_INIT = 0.0001
_C.SOLVER.SAVE_INTERVAL = 5000

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.EXPECTED_RESULTS = []
_C.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, NUM_CLIP_FRAMES = 2, each GPU will
# see 4 images per batch
_C.TEST.IMS_PER_BATCH = 8
_C.TEST.NUM_CLIP_FRAMES = 1
# Number of detections per image
_C.TEST.DETECTIONS_PER_IMG = 100
_C.TEST.NMS_IoU_THRESH = 0.5
_C.TEST.NMS_CONF_THRESH = 0.1
_C.TEST.NMS_WITH_BIoU = True
_C.TEST.NMS_WITH_MIoU = False


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
_C.NAME = 'r50_base'
_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
