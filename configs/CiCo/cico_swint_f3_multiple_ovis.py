_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/_base_/models/swint_base.yaml',
          'configs/CiCo/base_vis.py']

MODEL = dict(
    BACKBONE=dict(
        SWINT=dict(
            path='swint_base_coco_300_576_40_320000.pth')
    ),

    FPN=dict(
        NUM_FEATURES=256),

    PREDICTION_HEADS=dict(
        CUBIC_MODE=True,
        CUBIC_3D_MODE=True,
        CUBIC_MODE_ON_PROTONET=True,
        CIRCUMSCRIBED_BOXES=False),

    # CLASS_HEADS=dict(
    #     USE_FOCAL_LOSS=False),

    MASK_HEADS=dict(
        TRAIN_MASKS=True,
        PROTO_NET=[(192, 3, 1), (192, 3, 1), (192, 3, 1), (None, -2, {})])
)

CiCo = dict(
    MATCHER_CENTER=False,
    CPH_LAYER_KERNEL_SIZE=(1, 3, 3),
    CPH_LAYER_STRIDE=(1, 1, 1))

DATASETS = dict(
    TYPE='vis',
    NUM_CLASSES=25,
    TRAIN='train_OVIS_dataset',
    VALID_SUB='valid_sub_OVIS_dataset',
    VALID='valid_OVIS_dataset',
    TEST='test_OVIS_dataset')

SOLVER = dict(
    IMS_PER_BATCH=4,
    NUM_CLIP_FRAMES=3,
    NUM_CLIPS=2)

TEST = dict(
    IMS_PER_BATCH=1,
    NUM_CLIP_FRAMES=3)

OUTPUT_DIR = 'weights/OVIS/'
NAME = 'cico_swint_f3_multiple_ovis'
