_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/CiCo/base_vis.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet50',
        PATH='resnet50_coco_46.pth'),

    # MASK_HEADS=dict(
    #     TRAIN_MASKS=True,
    #     USE_DYNAMIC_MASK=True,
    #     LOSS_WITH_DICE_COEFF=False,
    #     MASK_DIM=8)
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

OUTPUT_DIR = 'outputs/YTVIS2019/'
NAME = 'cico_r50_f3_int1_cir3_boxtrack_cphcmh_yt19'
