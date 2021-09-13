_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/VID/base_VID.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet50',
        PATH='STMask_plus_resnet50_DET_960_22_50000.pth'),

    PREDICTION_HEADS=dict(
        CUBIC_MODE=True,
        CUBIC_CORRELATION_MODE=False,
        # INITIALIZATION=reduced or inflated
        CUBIC_MODE_WITH_INITIALIZATION='inflated')
)

DATASETS = dict(
    TYPE='vid',
    NUM_CLASSES=30,
    TRAIN='train_VID_dataset',
    VALID_SUB='valid_sub_VID_dataset',
    VALID='valid_VID_dataset',
    TEST='test_VID_dataset')

SOLVER = dict(
    IMS_PER_BATCH=8,
    NUM_CLIP_FRAMES=3,
    LR_STEPS=(8, 10),
    MAX_EPOCH=12)

TEST = dict(
    IMS_PER_BATCH=1,
    NUM_CLIP_FRAMES=3)

OUTPUT_DIR = 'weights/VID/'
NAME = 'r50_base_VID_cubic_c3_1X'
