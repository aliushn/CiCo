_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/VID/base_VID.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet50',
        PATH='STMask_plus_resnet50_DET_960_22_50000.pth'),
)

DATASETS = dict(
    TYPE='vid',
    NUM_CLASSES=30,
    TRAIN='train_VID_dataset',
    VALID_SUB='valid_sub_VID_dataset',
    VALID='valid_VID_dataset',
    TEST='test_VID_dataset')

OUTPUT_DIR = 'weights/VID/'
NAME = 'r50_base_VID_1X'


