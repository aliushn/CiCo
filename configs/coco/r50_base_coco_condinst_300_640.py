_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/coco/base_coco.py']

MODEL = dict(
    MASK_HEADS=dict(
        TRAIN_MASKS=True,
        USE_BN=True,
        USE_DYNAMIC_MASK=True,
        LOSS_WITH_DICE_COEFF=False,
        MASK_DIM=8)
)

INPUT = dict(
    MIN_SIZE_TRAIN=(300, 640),
    MAX_SIZE_TRAIN=800,
    MIN_SIZE_TEST=640,
    MAX_SIZE_TEST=800,
    MULTISCALE_TRAIN=True,
    PRESERVE_ASPECT_RATIO=True,
    SIZE_DIVISOR=32)

OUTPUT_DIR = 'outputs/COCO/'
NAME = 'r50_base_coco_dynamic_mask_head_300_640'
