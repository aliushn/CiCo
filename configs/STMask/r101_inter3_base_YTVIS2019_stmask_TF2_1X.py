_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/_base_/models/r101_inter3_base.yaml',
          'configs/VIS/base_VIS.py', 'configs/VIS/r101_base_YTVIS2019_stmask_TF2_1X.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet101',
        PATH='STMask_plus_resnet101_coco_960_40_200000.pth'),
)

OUTPUT_DIR = 'weights/YTVIS2019/'
NAME = 'r101_inter3_base_YTVIS2019_stmask_TF2_1X'

