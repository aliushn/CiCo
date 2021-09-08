_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/_base_/models/r101_base.yaml',
          'configs/VIS/base_VIS.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet101',
        PATH='STMask_resnet101_coco_960_52_220000.pth')
)

DATASETS = dict(
    TYPE='vis',
    NUM_CLASSES=40,
    TRAIN='train_YouTube_VOS2021_dataset',
    VALID_SUB='valid_sub_YouTube_VOS2021_dataset',
    VALID='valid_YouTube_VOS2021_dataset',
    TEST='test_YouTube_VOS2021_dataset')

SOLVER = dict(
    LR=0.0001)

OUTPUT_DIR = 'weights/YTVIS2021/'
NAME = 'r101_base_YTVIS2021_1X'


