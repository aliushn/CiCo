_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/VID/base_VID.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet50',
        PATH='STMask_resnet50_coco_960_46_340000.pth'),

    TRACK_HEADS=dict(
        TRAIN_TRACK=False)
)

DATASETS = dict(
    TYPE='det',
    NUM_CLASSES=30,
    TRAIN='train_DET_dataset')

SOLVER = dict(
    IMS_PER_BATCH=16,
    NUM_CLIP_FRAMES=1,
    LR=0.001,
    LR_STEPS=(12, 20),
    MAX_EPOCH=24,
    SAVE_INTERVAL=5000)

OUTPUT_DIR = 'weights/VID/'
NAME = 'r50_base_DET_2X'


