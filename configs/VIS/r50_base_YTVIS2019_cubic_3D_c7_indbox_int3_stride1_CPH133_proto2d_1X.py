_base_ = ['configs/_base_/models/r50_base.yaml', 'configs/VIS/base_VIS.py']

MODEL = dict(
    BACKBONE=dict(
        CONV_BODY='ResNet50',
        PATH='STMask_resnet50_coco_960_46_340000.pth'),

    PREDICTION_HEADS=dict(
        CUBIC_MODE=True,
        CUBIC_3D_MODE=True,
        CUBIC_MODE_ON_PROTONET=False,
        CUBIC_CORRELATION_MODE=False,
        CIRCUMSCRIBED_BOXES=False,
        # INITIALIZATION=reduced or inflated
        CUBIC_MODE_WITH_INITIALIZATION='inflated')
)


CiCo = dict(
    CPH_TOWER133=True,
    MATCHER_CENTER=False,
    CPH_LAYER_KERNEL_SIZE=(3, 3, 3),
    CPH_LAYER_PADDING=(0, 1, 1),
    CPH_LAYER_STRIDE=(1, 1, 1))

DATASETS = dict(
    TYPE='vis',
    NUM_CLASSES=40,
    TRAIN='train_YouTube_VOS2019_dataset',
    VALID_SUB='valid_sub_YouTube_VOS2019_dataset',
    VALID='valid_YouTube_VOS2019_dataset',
    TEST='test_YouTube_VOS2019_dataset')

SOLVER = dict(
    IMS_PER_BATCH=2,
    NUM_CLIP_FRAMES=7,
    NUM_CLIPS=1)

TEST = dict(
    IMS_PER_BATCH=1,
    NUM_CLIP_FRAMES=7)

OUTPUT_DIR = 'weights/YTVIS2019/'
NAME = 'r50_base_YTVIS2019_cubic_3D_c7_indbox_int3_stride1_CPH133_proto2d_1X'
