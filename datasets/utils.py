from typing import List

import mmcv
import torch

import matplotlib.pyplot as plt
import numpy as np
from .augmentations_vis import BaseTransform_vis
from .augmentations_coco import BaseTransform_coco
import torch.nn.functional as F


def random_scale(img_scales, mode='range'):
    """Randomly select a scale from a list of scales or scale ranges.
    Args:
        img_scales (list[tuple]): Image scale or scale range.
        mode (str): "range" or "value".
    Returns:
        tuple: Sampled image scale.
    """
    num_scales = len(img_scales)
    if num_scales == 1:  # fixed scale is specified
        img_scale = img_scales[0]
    elif num_scales == 2:  # randomly sample a scale
        if mode == 'range':
            img_scale_long = [max(s) for s in img_scales]
            img_scale_short = [min(s) for s in img_scales]
            long_edge = np.random.randint(
                min(img_scale_long),
                max(img_scale_long) + 1)
            short_edge = np.random.randint(
                min(img_scale_short),
                max(img_scale_short) + 1)
            img_scale = (long_edge, short_edge)
        elif mode == 'range_keep_ratio':
            img_scale_long = [max(s) for s in img_scales]
            img_scale_short = [min(s) for s in img_scales]
            scale = np.random.rand(1) * (max(img_scale_long) / min(img_scale_long)-1) + 1
            img_scale = (int(min(img_scale_long) * scale), int(min(img_scale_short) * scale))
        elif mode == 'value':
            img_scale = img_scales[np.random.randint(num_scales)]
    else:
        if mode != 'value':
            raise ValueError(
                'Only "value" mode supports more than 2 image scales')
        img_scale = img_scales[np.random.randint(num_scales)]
    return img_scale


def show_ann(coco, img, ann_info):
    plt.imshow(mmcv.bgr2rgb(img))
    plt.axis('off')
    coco.showAnns(ann_info)
    plt.show()


# modify from detectron2.structures.image_list
def ImageList_from_tensors(
        tensors: List[torch.Tensor], size_divisibility: int = 0, pad_value: float = 0.0):
    """
    Args:
        tensors: a tuple or list of `torch.Tensor`, each of shape (Hi, Wi) or
            (C_1, ..., C_K, Hi, Wi) where K >= 1. The Tensors will be padded
            to the same shape with `pad_value`.
        size_divisibility (int): If `size_divisibility > 0`, add padding to ensure
            the common height and width is divisible by `size_divisibility`.
            This depends on the model and many models need a divisibility of 32.
        pad_value (float): value to pad

    Returns:
        an `ImageList`.
    """
    assert len(tensors) > 0
    assert isinstance(tensors, (tuple, list))
    for t in tensors:
        assert isinstance(t, torch.Tensor), type(t)
        assert t.shape[:-2] == tensors[0].shape[:-2], t.shape

    image_sizes_h = [im.shape[-2] for im in tensors]
    image_sizes_w = [im.shape[-1] for im in tensors]
    max_size_h, max_size_w = max(image_sizes_h), max(image_sizes_w)

    if size_divisibility > 1:
        # the last two dims are H,W, both subject to divisibility requirement
        max_size_h = (max_size_h + (size_divisibility - 1)) // size_divisibility * size_divisibility
        max_size_w = (max_size_w + (size_divisibility - 1)) // size_divisibility * size_divisibility

    if len(tensors) == 1:
        # This seems slightly (2%) faster.
        # TODO: check whether it's faster for multiple images as well
        padding_size = [0, max_size_w - image_sizes_w[0], 0, max_size_h - image_sizes_h[0]]
        batched_imgs = F.pad(tensors[0], padding_size, value=pad_value).unsqueeze_(0)
    else:
        # max_size can be a tensor in tracing mode, therefore convert to list
        batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + [max_size_h, max_size_w]
        batched_imgs = tensors[0].new_full(batch_shape, pad_value)
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)

    return batched_imgs.contiguous()


def get_dataset(data_type, dataset, backbone_transform, inference=False):
    if dataset.has_gt and not inference:
        flip, MS_train = True, dataset.MS_train
        resize_gt, pad_gt = True, True
    else:
        flip, MS_train = False, False
        resize_gt, pad_gt = False, False

    from .ytvos import YTVOSDataset
    from .coco import COCODetection

    if data_type == 'vis':
        dataset = YTVOSDataset(ann_file=dataset.ann_file,
                               img_prefix=dataset.img_prefix,
                               transform=BaseTransform_vis(
                                    img_scales=dataset.img_scales,
                                    Flip=flip,
                                    MS_train=MS_train,
                                    preserve_aspect_ratio=dataset.preserve_aspect_ratio,
                                    backbone_transform=backbone_transform,
                                    resize_gt=resize_gt,
                                    pad_gt=pad_gt),
                               has_gt=dataset.has_gt)
    elif data_type == 'coco':
        dataset = COCODetection(image_path=dataset.img_prefix,
                                info_file=dataset.ann_file,
                                transform=BaseTransform_coco(dataset.img_scales,
                                                             Flip=flip,
                                                             MS_train=MS_train,
                                                             preserve_aspect_ratio=dataset.preserve_aspect_ratio,
                                                             backbone_transform=backbone_transform,
                                                             resize_gt=resize_gt,
                                                             pad_gt=pad_gt))
    return dataset



