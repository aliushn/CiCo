_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/_base_/models/swint_base.yaml',
          'configs/CiCo/base_vis.py']

MODEL = dict(
    BACKBONE=dict(
        SWINT=dict(
            path='swint_coco_yolact.pth')),

    FPN=dict(
        NUM_FEATURES=256),

    MASK_HEADS=dict(
        TRAIN_MASKS=True,
        MASK_DIM=32,
        PROTO_NET=[(192, 3, 1), (192, 3, 1), (192, 3, 1), (None, -2, {})]),

)

CiCo = dict(
    ENGINE=True,

    CPH=dict(
        CUBIC_MODE=True,
        LAYER_KERNEL_SIZE=(1, 3, 3),
        LAYER_STRIDE=(1, 1, 1),
        CUBIC_CLASS_HEAD=False,
        CUBIC_BOX_HEAD=True,
        CUBIC_TRACK_HEAD=True,
        MATCHER_MULTIPLE=False,
        CIRCUMSCRIBED_BOXES=False),

    CPN=dict(
        CUBIC_MODE=True)

)

DATASETS = dict(
    TYPE='vis',
    NUM_CLASSES=40,
    TRAIN='train_YouTube_VOS2021_dataset',
    VALID_SUB='valid_sub_YouTube_VOS2021_dataset',
    VALID='valid_YouTube_VOS2021_dataset',
    TEST='test_YouTube_VOS2021_dataset')

SOLVER = dict(
    IMS_PER_BATCH=4,
    NUM_CLIP_FRAMES=3,
    NUM_CLIPS=1)

TEST = dict(
    IMS_PER_BATCH=1,
    NUM_CLIP_FRAMES=3)

OUTPUT_DIR = 'outputs/YTVIS2021/'
