_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/coco/base_coco.py']

MODEL = dict(
    MASK_HEADS=dict(
        PROTO_DIVERGENCE_LOSS=True,
        PROTO_COEFF_DIVERSITY_LOSS=True,
        PROTO_COEFF_DIVERSITY_ALPHA=0.1,
    )
)

INPUT = dict(
    MIN_SIZE_TRAIN=(300, ),
    MAX_SIZE_TRAIN=640,
    MIN_SIZE_TEST=640,
    MAX_SIZE_TEST=640,
    MULTISCALE_TRAIN=True,
    PRESERVE_ASPECT_RATIO=True,
    SIZE_DIVISOR=32)

SOLVER = dict(
    IMS_PER_BATCH=4,
    NUM_CLIP_FRAMES=1,
    LR=0.005,
    LR_STEPS=(20, 40, 46, 50),
    MAX_EPOCH=54,
    SAVE_INTERVAL=10000)

OUTPUT_DIR = 'weights/COCO/'
NAME = 'r50_base_coco_protodiv_300_640'
