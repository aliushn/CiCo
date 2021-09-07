_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/VID/r50_base_VID.py']

DATASETS = dict(
    TYPE='vid',
    NUM_CLASSES=30,
    TRAIN='train_VID_dataset',
    VALID_SUB='valid_sub_VID_dataset',
    VALID='valid_VID_dataset',
    TEST='test_VID_dataset')

OUTPUT_DIR = 'weights/VID/'
NAME = 'r50_base_VID_1X'


