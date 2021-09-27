_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/VIS/base_VIS.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet50',
        PATH='STMask_resnet50_coco_960_46_340000.pth'),

    PREDICTION_HEADS=dict(
        CUBIC_MODE=True,
        CUBIC_MODE_ON_PROTONET=True,
        CUBIC_CORRELATION_MODE=False,
        CUBIC_SPATIOTEMPORAL_BLOCK=False,
        CUBIC_3D_MODE=True,
        # INITIALIZATION=reduced or inflated
        CUBIC_MODE_WITH_INITIALIZATION='inflated',
        CIRCUMSCRIBED_BOXES=True,
    ),

    MASK_HEADS=dict(
        PROTO_COEFF_OCCLUSION=True)

)

DATASETS = dict(
    TYPE='vis',
    NUM_CLASSES=40,
    TRAIN='train_YouTube_VOS2019_dataset',
    VALID_SUB='valid_sub_YouTube_VOS2019_dataset',
    VALID='valid_YouTube_VOS2019_dataset',
    TEST='test_YouTube_VOS2019_dataset')

SOLVER = dict(
    IMS_PER_BATCH=2,
    NUM_CLIP_FRAMES=3,
    LR_STEPS=(8, 10),
    MAX_EPOCH=12)

TEST = dict(
    IMS_PER_BATCH=1,
    NUM_CLIP_FRAMES=3)

OUTPUT_DIR = 'weights/YTVIS2019/'
NAME = 'r50_base_YTVIS2019_cubic_3D_c3_occluded_1X'
