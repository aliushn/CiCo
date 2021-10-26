_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/VIS/base_VIS.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet50',
        PATH='STMask_resnet50_coco_960_46_340000.pth'),

    PREDICTION_HEADS=dict(
        CUBIC_MODE=True,
        CUBIC_3D_MODE=True,
        CUBIC_CORRELATION_MODE=False,
        CUBIC_MODE_ON_PROTONET=False,
        CIRCUMSCRIBED_BOXES=False,
        # INITIALIZATION=reduced or inflated
        CUBIC_MODE_WITH_INITIALIZATION='inflated')
)

STR = dict(
    ST_CONSISTENCY=dict(
        CPH_WITH_TOWER133=True)
)

DATASETS = dict(
    TYPE='vis',
    NUM_CLASSES=40,
    TRAIN='train_YouTube_VOS2021_dataset',
    VALID_SUB='valid_sub_YouTube_VOS2021_dataset',
    VALID='valid_YouTube_VOS2021_dataset',
    TEST='test_YouTube_VOS20121_dataset')

SOLVER = dict(
    IMS_PER_BATCH=16,
    NUM_CLIP_FRAMES=3,
    NUM_CLIPS=1)

TEST = dict(
    IMS_PER_BATCH=1,
    NUM_CLIP_FRAMES=3)

OUTPUT_DIR = 'weights/YTVIS2021/'
NAME = 'r50_base_YTVIS2021_cubic_3D_c3_indbox_CPH133_proto2d_1X'
