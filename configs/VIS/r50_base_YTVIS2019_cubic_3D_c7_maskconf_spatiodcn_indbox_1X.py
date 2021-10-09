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
        CIRCUMSCRIBED_BOXES=False,
        CUBIC_MODE_WITH_INITIALIZATION='inflated'
    ),

    CLASS_HEADS=dict(
        USE_SPATIO_DCN=True
    ),

    MASK_HEADS=dict(
        USE_SPATIO_DCN=True
    ),
)

DATASETS = dict(
    TYPE='vis',
    NUM_CLASSES=40,
    TRAIN='train_YouTube_VOS2019_dataset',
    VALID_SUB='valid_sub_YouTube_VOS2019_dataset',
    VALID='valid_YouTube_VOS2019_dataset',
    TEST='test_YouTube_VOS2019_dataset')

SOLVER = dict(
    IMS_PER_BATCH=4,
    NUM_CLIP_FRAMES=7)

TEST = dict(
    IMS_PER_BATCH=1,
    NUM_CLIP_FRAMES=7)

OUTPUT_DIR = 'weights/YTVIS2019/'
NAME = 'r50_base_YTVIS2019_cubic_3D_c7_maskconf_spatiodcn_indbox_1X'
