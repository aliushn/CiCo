_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/VIS/base_VIS.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet50',
        PATH='STMask_resnet50_coco_960_46_340000.pth'),

    PREDICTION_HEADS=dict(
        CUBIC_MODE=True,
        CUBIC_3D_MODE=True,
        CUBIC_MODE_ON_PROTONET=False,
        CUBIC_CORRELATION_MODE=False,
        # INITIALIZATION=reduced or inflated
        CUBIC_MODE_WITH_INITIALIZATION='inflated'),

    CLASS_HEADS=dict(
        USE_FOCAL_LOSS=False)
)

DATASETS = dict(
    TYPE='vis',
    NUM_CLASSES=25,
    TRAIN='train_OVIS_dataset',
    VALID_SUB='valid_sub_OVIS_dataset',
    VALID='valid_OVIS_dataset',
    TEST='test_OVIS_dataset')

STR = dict(
    ST_CONSISTENCY=dict(
        CPH_WITH_TOWER133=True)
)

SOLVER = dict(
    IMS_PER_BATCH=18,
    NUM_CLIP_FRAMES=3,
    NUM_CLIPS=1,
    LR_STEPS=(8, 12),
    MAX_EPOCH=16)

TEST = dict(
    IMS_PER_BATCH=1,
    NUM_CLIP_FRAMES=3)

OUTPUT_DIR = 'weights/OVIS/'
NAME = 'r50_base_OVIS_cubic_3D_c3_indbox_CPH133_proto2d_ohem_1X'
