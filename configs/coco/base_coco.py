_base_ = ['configs/_base_/models/r50_base.yaml']

MODEL = dict(
    PREDICTION_HEADS=dict(
      USE_DIoU=True),
  
    BOX_HEADS=dict(
      TRAIN_BOXES=True,
      TRAIN_CENTERNESS=True),
  
    CLASS_HEADS=dict(
      TRAIN_CLASS=True,
      USE_FOCAL_LOSS=True,
      FOCAL_LOSS_ALPHA=0.25,
      FOCAL_LOSS_GAMMA=2),
  
    TRACK_HEADS=dict(
      TRAIN_TRACK=False),
  
    MASK_HEADS=dict(
      TRAIN_MASKS=True,
      MASK_DIM=32,
      PROTO_SRC=[0, 1, 2],
      PROTO_NET=[(256, 3, 1), (256, 3, 1), (256, 3, 1), (None, -2, {})],
      PROTO_CROP=True,
      PROTO_CROP_WITH_PRED_BOX=False,
      LOSS_WITH_OIR_SIZE=False,
      LOSS_WITH_DICE_COEFF=False,
      USE_SEMANTIC_SEGMENTATION_LOSS=False)
  )
  
INPUT = dict(
    MIN_SIZE_TRAIN=(600, ),
    MAX_SIZE_TRAIN=960,
    MIN_SIZE_TEST=960,
    MAX_SIZE_TEST=960,
    MULTISCALE_TRAIN=True,
    PRESERVE_ASPECT_RATIO=True,
    SIZE_DIVISOR=32)
  
DATASETS = dict(
    TYPE='coco',
    NUM_CLASSES=80,
    TRAIN='coco2017_train_dataset',
    VALID_SUB='coco2017_valid_dataset',
    VALID='coco2017_valid_dataset',
    TEST='coco2017_testdev_dataset')
  
SOLVER = dict(
    IMS_PER_BATCH=6,
    NUM_CLIP_FRAMES=1,
    LR=0.001,
    LR_STEPS=(18, 26, 30, 34),
    MAX_EPOCH=36,
    SAVE_INTERVAL=10000)
  
TEST = dict(
    IMS_PER_BATCH=1,
    NUM_CLIP_FRAMES=1,
    DETECTIONS_PER_IMG=100,
    NMS_IoU_THRESH=0.5,
    NMS_CONF_THRESH=0.1,
    NMS_WITH_BIoU=True,
    NMS_WITH_MIoU=True)
  
OUTPUT_DIR = 'outputs/COCO/'
NAME = 'r50_base_coco'
