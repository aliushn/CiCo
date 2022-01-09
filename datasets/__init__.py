from .utils import random_scale, show_ann, get_dataset, ImageList_from_tensors

from .ytvos import YTVOSDataset
from .ytvos import detection_collate as detection_collate_vis
from .ytvos import prepare_data as prepare_data_vis
from .loader import GroupSampler, DistributedGroupSampler, build_dataloader
from .augmentations_vis import BaseTransform_vis
from .augmentations_vid import BaseTransform_vid
from .augmentations_coco import SSDAugmentation, BaseTransform_coco, FastBaseTransform

from .coco import COCODetection
from .coco import detection_collate as detection_collate_coco
from .coco import prepare_data as prepare_data_coco
from .cocovis import COCOVISDetection
from .cocovis import detection_collate as detection_collate_cocovis
from .cocovis import prepare_data as prepare_data_cocovis
from .VID import VIDDataset
from .VID import detection_collate as detection_collate_vid
from .VID import prepare_data as prepare_data_vid
from .CustomDataParallel import NetLoss, CustomDataParallel

# These are in BGR and are for YouTubeVOS
MEANS = (123.675, 116.28, 103.53)
STD = (58.395, 57.12, 57.375)

__all__ = [
    'MEANS', 'STD',
    'detection_collate_coco', 'detection_collate_vis', 'get_dataset',
    'YTVOSDataset', 'COCODetection',
    'GroupSampler', 'DistributedGroupSampler', 'build_dataloader',
    'random_scale', 'show_ann', 'prepare_data_vis', 'ImageList_from_tensors', 'prepare_data_coco',
    'SSDAugmentation', 'BaseTransform_coco', 'FastBaseTransform', 'BaseTransform_vis',
    'VIDDataset', 'detection_collate_vid', 'prepare_data_vid',
    'COCOVISDetection', 'detection_collate_cocovis', 'prepare_data_cocovis',
    'NetLoss', 'CustomDataParallel'
]


