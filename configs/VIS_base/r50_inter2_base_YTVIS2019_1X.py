
_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/_base_/models/r50_inter2_base.yaml',
          'configs/VIS/base_VIS.py', 'configs/VIS/r50_base_YTVIS2019_1X.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet50',
        PATH='STMask_plus_resnet50_coco_960_53_260000.pth')
)

NAME = 'r50_inter2_base_YTVIS2019_1X'


