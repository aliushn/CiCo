_base_ = ['configs/_base_/models/r50_base.yaml']

MODEL = dict(
    FPN=dict(
        NUM_FEATURES=256,
        USE_CONV_DOWNSAMPLE=True,
        NUM_DOWNSAMPLE=2),

    PREDICTION_HEADS=dict(
        USE_DIoU=True),

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
        MATCH_COEFF=[0, 0.5, 0.5, 0.3]),   # scores, mask_iou, box_iou, label

    MASK_HEADS=dict(
        TRAIN_MASKS=True,
        MASK_DIM=32,
        PROTO_SRC=[0, 1, 2],
        PROTO_NET=[(256, 3, 1), (256, 3, 1), (256, 3, 1), (None, -2, {})],
        PROTO_CROP=True,
        PROTO_CROP_WITH_PRED_BOX=False,
        LOSS_WITH_OIR_SIZE=False,
        LOSS_WITH_DICE_COEFF=False,
        USE_SEMANTIC_SEGMENTATION_LOSS=False)
)

INPUT = dict(
    MIN_SIZE_TRAIN=(400, ),
    MAX_SIZE_TRAIN=640,
    MIN_SIZE_TEST=640,
    MAX_SIZE_TEST=640,
    MULTISCALE_TRAIN=True,
    PRESERVE_ASPECT_RATIO=True,
    SIZE_DIVISOR=32)

DATASETS = dict(
    TYPE='coco',
    NUM_CLASSES=40,
    TRAIN='coco2017_train_dataset',
    VALID_SUB='coco2017_valid_dataset',
    VALID='coco2017_valid_dataset',
    TEST='coco2017_testdev_dataset')

SOLVER = dict(
    IMS_PER_BATCH=4,
    NUM_CLIP_FRAMES=1,
    LR=0.01,
    LR_STEPS=(16, 24, 32),
    MAX_EPOCH=36,
    SAVE_INTERVAL=5000)

TEST = dict(
    IMS_PER_BATCH=1,
    NUM_CLIP_FRAMES=1,
    DETECTIONS_PER_IMG=200,
    NMS_CONF_THRESH=0.05,
    NMS_WITH_BIoU=True,
    NMS_WITH_MIoU=False)

OUTPUT_DIR = 'weights/COCOVIS/'
NAME = 'r50_base_cocovis'
