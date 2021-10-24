_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/VIS/base_VIS.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet50',
        PATH='r50_base_cocovis_cubic_3D_c3_indbox_300_400_1X_5_75000.pth'),

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
NAME = 'r50_base_YTVIS2021_cubic_3D_c3_indbox_covis_1X'
