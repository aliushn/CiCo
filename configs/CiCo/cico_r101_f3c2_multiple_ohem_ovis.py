_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/_base_/models/r101_base.yaml',
          'configs/CiCo/base_vis.py', 'configs/CiCo/cico_r50_f3c2_multiple_ohem_ovis.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet101',
        PATH='STMask_resnet101_coco_960_52_220000.pth'),
)

OUTPUT_DIR = 'outputs/OVIS/'
# NAME = 'cico_r101_f3c2_multiple_ohem_ovis'
NAME = 'r101_base_OVIS_cubic_3D_c3s2_indbox_multiple_ohem_1X'
