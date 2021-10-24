_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/cocovis/base_cocovis.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet50',
        PATH='STMask_resnet50_coco_960_46_340000.pth'),

    PREDICTION_HEADS=dict(
        CUBIC_MODE=True,
        CUBIC_CORRELATION_MODE=False,
        CUBIC_MODE_ON_PROTONET=True,
        CUBIC_3D_MODE=True,
        CIRCUMSCRIBED_BOXES=False,
        # INITIALIZATION=reduced or inflated
        CUBIC_MODE_WITH_INITIALIZATION='inflated')
)

DATASETS = dict(
    TYPE='cocovis',
    NUM_CLASSES=40,
    TRAIN='coco2017_train_dataset',
    VALID_SUB='coco2017_valid_dataset',
    VALID='coco2017_valid_dataset',
    TEST='coco2017_testdev_dataset')

INPUT = dict(
    MIN_SIZE_TRAIN=(300, ),
    MAX_SIZE_TRAIN=360,
    MIN_SIZE_TEST=360,
    MAX_SIZE_TEST=360)

SOLVER = dict(
    IMS_PER_BATCH=4,
    NUM_CLIP_FRAMES=3,
    NUM_CLIPS=1,
    LR=0.001,
    LR_STEPS=(12, 18),
    MAX_EPOCH=20)

TEST = dict(
    IMS_PER_BATCH=1,
    NUM_CLIP_FRAMES=3)

OUTPUT_DIR = 'weights/COCOVIS/'
NAME = 'r50_base_cocovis_cubic_3D_c3_indbox_300_360_1X'
