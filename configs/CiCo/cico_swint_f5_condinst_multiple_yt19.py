_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/_base_/models/swint_base.yaml',
          'configs/CiCo/base_vis.py']

MODEL = dict(
    BACKBONE=dict(
        SWINT=dict(
            path='swint_base_coco_dynamic_mask_head_300_576_21_420000.pth')
    ),

    MASK_HEADS=dict(
        TRAIN_MASKS=True,
        USE_DYNAMIC_MASK=True,
        LOSS_WITH_DICE_COEFF=False,
        MASK_DIM=8,
        PROTO_NET=[(192, 3, 1), (192, 3, 1), (192, 3, 1), (None, -2, {})])
)

CiCo = dict(
    ENGINE=True,

    CPH=dict(
        TOWER_CUBIC_MODE=False,
        MATCHER_CENTER=False,
        CUBIC_MODE=True,
        LAYER_KERNEL_SIZE=(3, 3, 3),
        LAYER_STRIDE=(1, 1, 1),
        CIRCUMSCRIBED_BOXES=False),

    CPN=dict(
        CUBIC_MODE=True)

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
    NUM_CLIP_FRAMES=5,
    NUM_CLIPS=1)

TEST = dict(
    IMS_PER_BATCH=1,
    NUM_CLIP_FRAMES=5)

OUTPUT_DIR = 'outputs/YTVIS2019/'
NAME = 'cico_swint_f5_dynamic_mask_head_multiple_yt19'
