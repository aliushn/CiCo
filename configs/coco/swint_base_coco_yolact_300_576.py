_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/_base_/models/swint_base.yaml',
          'configs/coco/base_coco.py']

MODEL = dict(
    MASK_HEADS=dict(
        TRAIN_MASKS=True,
        PROTO_NET=[(192, 3, 1), (192, 3, 1), (192, 3, 1), (None, -2, {})])
)

INPUT = dict(
    MIN_SIZE_TRAIN=(300, 576),
    MAX_SIZE_TRAIN=800,
    MIN_SIZE_TEST=576,
    MAX_SIZE_TEST=800,
    MULTISCALE_TRAIN=True,
    PRESERVE_ASPECT_RATIO=True,
    SIZE_DIVISOR=32)

SOLVER = dict(
    IMS_PER_BATCH=4,
    NUM_CLIP_FRAMES=1,
    LR=0.001,
    LR_STEPS=(20, 40, 46, 50),
    MAX_EPOCH=54,
    SAVE_INTERVAL=500)

OUTPUT_DIR = 'weights/COCO/'
NAME = 'swint_base_coco_300_576'
