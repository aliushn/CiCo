_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/_base_/models/r101_base.yaml',
          'configs/CiCo/base_vis.py', 'configs/CiCo/cico_r50_f3_multiple_yt21.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet101',
        PATH='STMask_resnet101_coco_960_52_220000.pth'),
)

TEST = dict(
    NMS_IoU_THRESH=0.4)

OUTPUT_DIR = 'outputs/YTVIS2021/'
# NAME = 'cico_r101_f3_multiple_yt21'
NAME = 'r101_base_YTVIS2021_cubic_3D_c3_indbox_multiple_1X'
