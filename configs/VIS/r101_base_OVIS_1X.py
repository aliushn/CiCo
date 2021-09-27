_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/_base_/models/r101_base.yaml',
          'configs/VIS/base_VIS.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet101',
        PATH='STMask_resnet101_coco_960_52_220000.pth')
)

DATASETS = dict(
    TYPE='vis',
    NUM_CLASSES=25,
    TRAIN='train_OVIS_dataset',
    VALID_SUB='valid_sub_OVIS_dataset',
    VALID='valid_OVIS_dataset',
    TEST='test_OVIS_dataset')

TEST = dict(
    DETECTIONS_PER_IMG=200,
    NMS_IoU_THRESH=0.4,
    NMS_CONF_THRESH=0.2,
    NMS_WITH_BIoU=True,
    NMS_WITH_MIoU=True)

OUTPUT_DIR = 'weights/OVIS/'
NAME = 'r101_base_OVIS_1X'


