
_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/_base_/models/r50_inter2_base.yaml',
          'configs/VID/base_VID.py', 'configs/VID/r50_base_VID_stmask_TF2_1X.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet50',
        PATH='STMask_plus_resnet50_DET_960_22_50000.pth')
)

NAME = 'r50_inter2_base_VID_stmask_TF2_1X'


