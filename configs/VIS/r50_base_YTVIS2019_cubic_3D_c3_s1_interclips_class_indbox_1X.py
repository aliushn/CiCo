_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/VIS/base_VIS.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet50',
        PATH='STMask_resnet50_coco_960_46_340000.pth'),

    PREDICTION_HEADS=dict(
        CUBIC_MODE=True,
        CUBIC_CORRELATION_MODE=False,
        CUBIC_MODE_ON_PROTONET=True,
        CUBIC_3D_MODE=True,
        CIRCUMSCRIBED_BOXES=False,
        # INITIALIZATION=reduced or inflated
        CUBIC_MODE_WITH_INITIALIZATION='inflated'),

    CLASS_HEADS=dict(
        TRAIN_INTERCLIPS_CLASS=True)

)

DATASETS = dict(
    TYPE='vis',
    NUM_CLASSES=40,
    TRAIN='train_YouTube_VOS2019_dataset',
    VALID_SUB='valid_sub_YouTube_VOS2019_dataset',
    VALID='valid_YouTube_VOS2019_dataset',
    TEST='test_YouTube_VOS2019_dataset')

SOLVER = dict(
    IMS_PER_BATCH=8,
    NUM_CLIP_FRAMES=3,
    NUM_CLIPS=1)

TEST = dict(
    IMS_PER_BATCH=1,
    NUM_CLIP_FRAMES=3)

OUTPUT_DIR = 'weights/YTVIS2019/'
NAME = 'r50_base_YTVIS2019_cubic_3D_c3_s1_interclips_class_indbox_1X'
