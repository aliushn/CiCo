_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/_base_/models/r101_base.yaml',
          'configs/VIS/base_VIS.py', 'configs/VIS/r50_base_YTVIS2019_cubic_3D_c3_indbox_1X.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet101',
        PATH='STMask_resnet101_coco_960_52_220000.pth')
)

OUTPUT_DIR = 'weights/YTVIS2019/'
NAME = 'r101_base_YTVIS2019_cubic_3D_c3_indbox_1X'


