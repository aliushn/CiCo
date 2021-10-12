_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/coco/base_coco.py']

INPUT = dict(
    MIN_SIZE_TRAIN=(600, ),
    MAX_SIZE_TRAIN=960,
    MIN_SIZE_TEST=960,
    MAX_SIZE_TEST=960,
    MULTISCALE_TRAIN=True,
    PRESERVE_ASPECT_RATIO=True,
    SIZE_DIVISOR=32)

OUTPUT_DIR = 'weights/COCO/'
NAME = 'r50_base_coco_960'
