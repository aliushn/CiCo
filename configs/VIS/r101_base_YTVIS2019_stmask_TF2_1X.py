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
    TRAIN='train_YouTube_VOS2019_dataset',
    VALID_SUB='valid_sub_YouTube_VOS2019_dataset',
    VALID='valid_YouTube_VOS2019_dataset',
    TEST='test_YouTube_VOS2019_dataset')

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

OUTPUT_DIR = 'weights/YTVIS2019/'
NAME = 'r101_base_YTVIS2019_stmask_TF2_1X'

