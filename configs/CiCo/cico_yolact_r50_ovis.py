_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/CiCo/base_vis.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet50',
        PATH='resnet50_coco_yolact.pth'),

    CLASS_HEADS=dict(
        USE_FOCAL_LOSS=False)
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
    NUM_CLASSES=25,
    TRAIN='train_OVIS_dataset',
    VALID_SUB='valid_sub_OVIS_dataset',
    VALID='valid_OVIS_dataset',
    TEST='test_OVIS_dataset')

OUTPUT_DIR = 'outputs/OVIS/'
