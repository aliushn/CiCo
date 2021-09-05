
_base_ = ['configs/_base_/models/r50_base.yaml']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet50',
        PATH='STMask_resnet50_coco_960_46_340000.pth'),

    FPN=dict(
        NUM_FEATURES=256,
        USE_CONV_DOWNSAMPLE=True,
        NUM_DOWNSAMPLE=2),

    PREDICTION_HEADS=dict(
        USE_DIoU=True,
        CUBIC_MODE=False,
        CUBIC_CORRELATION_MODE=False,
        USE_PREDICTION_MATCHING=True),

    BOX_HEADS=dict(
        TRAIN_BOXES=True,
        TRAIN_CENTERNESS=True),

    CLASS_HEADS=dict(
        TRAIN_CLASS=True,
        USE_FOCAL_LOSS=True,
        FOCAL_LOSS_ALPHA=0.25,
        FOCAL_LOSS_GAMMA=2),
    
    TRACK_HEADS=dict(
        TRAIN_TRACK=True,
        TRACK_DIM=64,
        LOSS_ALPHA=5,
        MATCH_COEFF=[0, 1, 1, 0]),
    
    MASK_HEADS=dict(
        TRAIN_MASKS=True,
        MASK_DIM=32,
        PROTO_SRC=[0, 1, 2],
        PROTO_CROP=True,
        PROTO_CROP_WITH_PRED_BOX=False,
        LOSS_WITH_OIR_SIZE=False,
        LOSS_WITH_DICE_COEFF=False)
)

INPUT = dict(
    MIN_SIZE_TRAIN=(360,),
    MAX_SIZE_TRAIN=640,
    MIN_SIZE_TEST=360,
    MAX_SIZE_TEST=640,
    MULTISCALE_TRAIN=False,
    PRESERVE_ASPECT_RATIO=True,
    SIZE_DIVISOR=32)
    
DATASETS = dict(
    TYPE='vis',
    NUM_CLASSES=40,
    TRAIN='train_YouTube_VOS2019_dataset',
    VALID_SUB='valid_sub_YouTube_VOS2019_dataset',
    VALID='valid_YouTube_VOS2019_dataset',
    TEST='test_YouTube_VOS2019_dataset')
    
SOLVER = dict(
    IMS_PER_BATCH=8,
    NUM_CLIP_FRAMES=2,
    LR=0.0001,
    LR_STEPS=(8, 10),
    MAX_EPOCH=12,
    SAVE_INTERVAL=5000)
    
TEST = dict(
    IMS_PER_BATCH=4,
    NUM_CLIP_FRAMES=1,
    DETECTIONS_PER_IMG=100,
    NMS_IoU_THRESH=0.5,
    NMS_CONF_THRESH=0.1,
    NMS_WITH_BIoU=True,
    NMS_WITH_MIoU=True)

OUTPUT_DIR = 'weights/YTVIS2019/'
NAME = 'r50_base_YTVIS2019_1X'


