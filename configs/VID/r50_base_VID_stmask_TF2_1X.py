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

OUTPUT_DIR = 'weights/VID/'
NAME = 'r50_base_VID_stmask_TF2_1X'

