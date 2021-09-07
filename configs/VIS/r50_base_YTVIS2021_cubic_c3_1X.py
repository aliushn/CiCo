_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/VIS/r50_base_VIS.py']

MODEL = dict(
    PREDICTION_HEADS=dict(
        CUBIC_MODE=True,
        CUBIC_CORRELATION_MODE=False,
        # INITIALIZATION=reduced or inflated
        CUBIC_MODE_WITH_INITIALIZATION='inflated')
)

DATASETS = dict(
    TYPE='vis',
    NUM_CLASSES=40,
    TRAIN='train_YouTube_VOS2021_dataset',
    VALID_SUB='valid_sub_YouTube_VOS2021_dataset',
    VALID='valid_YouTube_VOS2021_dataset',
    TEST='test_YouTube_VOS2021_dataset')

SOLVER = dict(
    IMS_PER_BATCH=8,
    NUM_CLIP_FRAMES=3,
    LR_STEPS=(8, 10),
    MAX_EPOCH=12)

TEST = dict(
    IMS_PER_BATCH=1,
    NUM_CLIP_FRAMES=3)

OUTPUT_DIR = 'weights/YTVIS2021/'
NAME = 'r50_base_YTVIS2021_cubic_c3_1X'
