_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/VIS/base_VIS.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet50',
        PATH='STMask_resnet50_coco_960_46_340000.pth'),

    PREDICTION_HEADS=dict(
        CUBIC_MODE=True,
        CUBIC_3D_MODE=True,
        CUBIC_CORRELATION_MODE=False,
        CUBIC_MODE_ON_PROTONET=True,
        CIRCUMSCRIBED_BOXES=False,
        # INITIALIZATION=reduced (even 2,4,6,8) or inflated
        CUBIC_MODE_WITH_INITIALIZATION='inflatied'),

    CLASS_HEADS=dict(
        TRAIN_CLASS=True,
        USE_FOCAL_LOSS=False)
)

DATASETS = dict(
    TYPE='vis',
    NUM_CLASSES=25,
    TRAIN='train_OVIS_dataset',
    VALID_SUB='valid_sub_OVIS_dataset',
    VALID='valid_OVIS_dataset',
    TEST='test_OVIS_dataset')

SOLVER = dict(
    IMS_PER_BATCH=8,
    NUM_CLIP_FRAMES=3,
    NUM_CLIPS=2,
    LR_STEPS=(12, 18),
    MAX_EPOCH=20,)

TEST = dict(
    IMS_PER_BATCH=1,
    NUM_CLIP_FRAMES=3)

OUTPUT_DIR = 'weights/OVIS/'
NAME = 'r50_base_OVIS_cubic_3D_c3_indbox_ohem_2X'

