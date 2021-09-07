_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/VIS/r50_base_VIS.py']

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
NAME = 'r50_base_YTVIS2021_1X'


