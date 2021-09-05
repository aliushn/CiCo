_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/r50_base_YTVIS2019_1X.py']

MODEL = dict(
    PREDICTION_HEADS=dict(
        CUBIC_MODE=True,
        CUBIC_CORRELATION_MODE=False,
        # INITIALIZATION=reduced or inflated
        CUBIC_MODE_WITH_INITIALIZATION='inflated')
)

SOLVER = dict(
    IMS_PER_BATCH=4,
    NUM_CLIP_FRAMES=3,
    LR=0.0001,
    LR_STEPS=(8, 10),
    MAX_EPOCH=12)

TEST = dict(
    IMS_PER_BATCH=1,
    NUM_CLIP_FRAMES=3)

NAME = 'r50_base_YTVIS2019_cubic_1X_C3'
