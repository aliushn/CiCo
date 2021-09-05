_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/r50_base_YTVIS2019_1X.py']

STMASK = dict(
    FC=dict(
        USE_FCA=True,
        FCA_CONV_KERNELS=[[3,3], [3,5], [5,3]],
        USE_FCB=True,
        FCB_USE_PRED_OFFSET=True,
        FCB_USE_DCN_CLASS=True,
        FCB_USE_DCN_TRACK=False,
        FCB_USE_DCN_MASK=False),
    T2S_HEADS=dict(
        TEMPORAL_FUSION_MODULE=True,
        CORRELATION_PATCH_SIZE=5,
        CORRELATION_SELECTED_LAYER=1,
        TRAIN_BOXSHIFT=True,
        BOXSHIFT_ALPHA=1,
        TRAIN_MASKSHIFT=True,
        MASKSHIFT_ALPHA=6.125,
        SHIFT_WITH_PRED_BOX=False,
        FORWARD_FLOW=True,
        BACKWARD_FLOW=False)
)

INPUT = dict(
    MIN_SIZE_TRAIN=(360,),
    MAX_SIZE_TRAIN=640,
    MIN_SIZE_TEST=360,
    MAX_SIZE_TEST=640,
    MULTISCALE_TRAIN=False,
    PRESERVE_ASPECT_RATIO=True,
    SIZE_DIVISOR=32)

SOLVER = dict(
    IMS_PER_BATCH=4,
    NUM_CLIP_FRAMES=2,
    LR=0.0001,
    LR_STEPS=(8, 10),
    MAX_EPOCH=12)

TEST = dict(
    IMS_PER_BATCH=4,
    NUM_CLIP_FRAMES=1)

NAME = 'r50_base_YTVIS2019_stmask_1X_TF2'

