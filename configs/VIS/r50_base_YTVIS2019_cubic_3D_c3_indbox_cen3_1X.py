_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/VIS/base_VIS.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet50',
        PATH='STMask_resnet50_coco_960_46_340000.pth'),

    PREDICTION_HEADS=dict(
        CUBIC_MODE=True,
        CUBIC_3D_MODE=True,
        CUBIC_MODE_ON_PROTONET=True,
        CUBIC_CORRELATION_MODE=False,
        CIRCUMSCRIBED_BOXES=False)
)

CiCo = dict(
    MATCHER_CENTER=True)

DATASETS = dict(
    TYPE='vis',
    NUM_CLASSES=40,
    TRAIN='train_YouTube_VOS2019_dataset',
    VALID_SUB='valid_sub_YouTube_VOS2019_dataset',
    VALID='valid_YouTube_VOS2019_dataset',
    TEST='test_YouTube_VOS2019_dataset')

SOLVER = dict(
    IMS_PER_BATCH=4,
    NUM_CLIP_FRAMES=3,
    NUM_CLIPS=1)

TEST = dict(
    IMS_PER_BATCH=1,
    NUM_CLIP_FRAMES=3,
    NMS_CONF_THRESH=0.1,)

OUTPUT_DIR = 'weights/YTVIS2019/'
NAME = 'r50_base_YTVIS2019_cubic_3D_c3_indbox_cen3_1X'