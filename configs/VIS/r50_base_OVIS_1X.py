_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/VIS/r50_base_VIS.py']

DATASETS = dict(
    TYPE='vis',
    NUM_CLASSES=25,
    TRAIN='train_OVIS_dataset',
    VALID_SUB='valid_sub_OVIS_dataset',
    VALID='valid_OVIS_dataset',
    TEST='test_OVIS_dataset')

OUTPUT_DIR = 'weights/OVIS/'
NAME = 'r50_base_OVIS_1X'


