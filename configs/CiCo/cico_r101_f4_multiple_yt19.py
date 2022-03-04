_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/_base_/models/r101_base.yaml',
          'configs/CiCo/base_vis.py', 'configs/CiCo/cico_r50_f4_multiple_sparse_yt19.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet101',
        PATH='STMask_resnet101_coco_960_52_220000.pth'),
)

OUTPUT_DIR = 'outputs/YTVIS2019/'
NAME = 'r101_base_YTVIS2019_cubic_3D_c4_indbox_multiple_1X'
