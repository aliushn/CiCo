_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/coco/base_coco.py']

MODEL = dict(
    MASK_HEADS=dict(
        USE_BN=True)
)

INPUT = dict(
    MIN_SIZE_TRAIN=(300, 576),
    MAX_SIZE_TRAIN=800,
    MIN_SIZE_TEST=576,
    MAX_SIZE_TEST=800,
    MULTISCALE_TRAIN=True,
    PRESERVE_ASPECT_RATIO=True,
    SIZE_DIVISOR=32)

OUTPUT_DIR = 'outputs/COCO/'
NAME = 'r50_base_coco_300_640'
