_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/VIS/base_VIS.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet50',
        PATH='STMask_resnet50_coco_960_46_340000.pth'),

    CLASS_HEADS=dict(
        USE_FOCAL_LOSS=False)
)

DATASETS = dict(
    TYPE='vis',
    NUM_CLASSES=25,
    TRAIN='train_OVIS_dataset',
    VALID_SUB='valid_sub_OVIS_dataset',
    VALID='valid_OVIS_dataset',
    TEST='test_OVIS_dataset')

OUTPUT_DIR = 'weights/OVIS/'
NAME = 'r50_base_OVIS_1X'


