_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/_base_/models/r101_base.yaml',
          'configs/CiCo/base_vis.py', 'configs/CiCo/cico_r50_f3_dynamic_mask_head_multiple_yt19.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet101',
        PATH='r101_base_coco_dynamic_mask_head_300_640_35_230000.pth'),
)

OUTPUT_DIR = 'outputs/YTVIS2019/'
NAME = 'cico_r101_f3_dynamic_mask_head_multiple_yt19'
