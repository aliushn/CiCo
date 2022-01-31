_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/CiCo/base_vis.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet50',
        PATH='resnet50_coco_46.pth'),

    CLASS_HEADS=dict(
        USE_FOCAL_LOSS=False)
)

CiCo = dict(
    ENGINE=True,

    CPH=dict(
        TOWER_CUBIC_MODE=True,
        CUBIC_MODE=True,
        LAYER_KERNEL_SIZE=(1, 3, 3),
        LAYER_STRIDE=(1, 1, 1),
        MATCHER_CENTER=False,
        CIRCUMSCRIBED_BOXES=False),

    CPN=dict(
        CUBIC_MODE=True)

)

DATASETS = dict(
    TYPE='vis',
    NUM_CLASSES=25,
    TRAIN='train_OVIS_dataset',
    VALID_SUB='valid_sub_OVIS_dataset',
    VALID='valid_OVIS_dataset',
    TEST='test_OVIS_dataset')

SOLVER = dict(
    IMS_PER_BATCH=4,
    NUM_CLIP_FRAMES=3,
    NUM_CLIPS=1)

TEST = dict(
    IMS_PER_BATCH=1,
    NUM_CLIP_FRAMES=3,
    NMS_CONF_THRESH=0.1)

OUTPUT_DIR = 'outputs/OVIS/'
NAME = 'cico_r50_f3c2_multiple_ohem_ovis'
