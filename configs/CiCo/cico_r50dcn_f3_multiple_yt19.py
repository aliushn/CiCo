_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/_base_/models/r50_inter2_base.yaml',
          'configs/CiCo/base_vis.py', 'configs/CiCo/cico_r50_f3_multiple_yt19.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet50',
        PATH='STMask_plus_resnet50_coco_960_53_260000.pth'),
)

SOLVER = dict(
    IMS_PER_BATCH=24,
    NUM_CLIP_FRAMES=3,
    NUM_CLIPS=1,
    LR=0.001,
    LR_STEPS=(4, 9, 11))

TEST = dict(
    IMS_PER_BATCH=1,
    NUM_CLIP_FRAMES=3)

OUTPUT_DIR = 'outputs/YTVIS2019/'
NAME = 'cico_r50dcn_f3_multiple_yt19'
