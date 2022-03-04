_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/CiCo/base_vis.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet50',
        PATH='resnet50_coco_46.pth'),

)

CiCo = dict(
    ENGINE=True,

    CPH=dict(
        TOWER_CUBIC_MODE=True,
        CUBIC_MODE=True,
        LAYER_KERNEL_SIZE=(2, 3, 3),
        LAYER_STRIDE=(2, 1, 1),
        MATCHER_CENTER=False,
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
    NUM_CLIP_FRAMES=6,
    NUM_CLIPS=1)

TEST = dict(
    IMS_PER_BATCH=1,
    NUM_CLIP_FRAMES=6,
    NMS_IoU_THRESH=0.4)

OUTPUT_DIR = 'outputs/YTVIS2021/'
NAME = 'cico_r50_f6_multiple_yt21'
