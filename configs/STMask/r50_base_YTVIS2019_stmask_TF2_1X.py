_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/VIS/base_VIS.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet50',
        PATH='STMask_resnet50_coco_960_46_340000.pth'),
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
        FCA_CONV_KERNELS=[[3, 3], [3, 5], [5, 3]],
        USE_FCB=True,
        FCB_USE_PRED_OFFSET=True,
        FCB_USE_DCN_CLASS=True,
        FCB_USE_DCN_TRACK=False,
        FCB_USE_DCN_MASK=False),
    T2S_HEADS=dict(
        TEMPORAL_FUSION_MODULE=True,
        CORRELATION_PATCH_SIZE=11,
        CORRELATION_SELECTED_LAYER=1,
        TRAIN_BOXSHIFT=True,
        BOXSHIFT_ALPHA=1,
        TRAIN_MASKSHIFT=True,
        MASKSHIFT_ALPHA=6.125,
        FORWARD_FLOW=True,
        BACKWARD_FLOW=False)
)

TEST = dict(
    NMS_IoU_THRESH=0.5,
    NMS_CONF_THRESH=0.1)


OUTPUT_DIR = 'weights/YTVIS2019/'
NAME = 'r50_base_YTVIS2019_stmask_TF2_1X'

