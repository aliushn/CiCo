_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/CiCo/base_vis.py']

MODEL = dict(
    BACKBONE=dict(
        PATH='swint_base_coco_dynamic_mask_head_300_576_21_420000.pth')
    ),

    PREDICTION_HEADS=dict(
        CUBIC_MODE=True,
        CUBIC_3D_MODE=True,
        CUBIC_MODE_ON_PROTONET=True,
        CIRCUMSCRIBED_BOXES=False),

    MASK_HEADS=dict(
        TRAIN_MASKS=True,
        USE_DYNAMIC_MASK=True,
        LOSS_WITH_DICE_COEFF=False,
        MASK_DIM=8,
        PROTO_NET=[(192, 3, 1), (192, 3, 1), (192, 3, 1), (None, -2, {})])
)

CiCo = dict(
    MATCHER_CENTER=False,
    CPH_LAYER_KERNEL_SIZE=(1, 3, 3),
    CPH_LAYER_STRIDE=(1, 1, 1))

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
    NUM_CLIP_FRAMES=3)

OUTPUT_DIR = 'weights/YTVIS2019/'
NAME = 'cico_swint_f3_dynamic_mask_head_multiple_yt19'
