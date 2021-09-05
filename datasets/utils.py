from typing import List
import os
import mmcv
import torch
import matplotlib.pyplot as plt
import numpy as np
from .augmentations_vis import BaseTransform_vis
from .augmentations_coco import BaseTransform_coco
from .augmentations_vid import BaseTransform_vid
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


def GtList_from_tensor(height, width, masks=None, boxes=None, img_metas=None):
    '''
    :param pad_h:
    :param pad_w:
    :param masks: List[torch.Tensor]
    :param boxes: List[torch.Tensor]
    :param img_metas:
    :return:
    '''
    assert masks is not None or img_metas is not None
    n_list = len(masks) if masks is not None else len(boxes)
    for i in range(n_list):
        if masks is not None:
            if len(masks[i].shape) == 4:
                n_obj, n_frames, im_h, im_w = masks[i].shape
                n = n_obj*n_frames
            else:
                n, im_h, im_w = masks[i].shape
            expand_masks = np.zeros(
                (n, height, width), dtype=masks[i].dtype)
            expand_masks[:, :im_h, :im_w] = masks[i].reshape(-1, im_h, im_w)
            masks[i] = expand_masks.reshape(n_obj,n_frames,height,width) if len(masks[i].shape) == 4 else expand_masks

        if boxes is not None:
            if masks is not None:
                im_h, im_w = masks[i].shape[-2:]
            else:
                if isinstance(img_metas[i], list):
                    im_h, im_w = img_metas[i][0]['img_shape'][:2]
                else:
                    im_h, im_w = img_metas[i]['img_shape'][:2]
            # Scale bounding boxes (which are currently percent coordinates)
            boxes[i][..., [0, 2]] *= (im_w / width)
            boxes[i][..., [1, 3]] *= (im_h / height)

        if img_metas is not None:
            if isinstance(img_metas[i], list):
                for j in range(len(img_metas[i])):
                    img_metas[i][j]['img_shape'] = (height, width, 3)
            else:
                img_metas[i]['img_shape'] = (height, width, 3)

    return masks, boxes, img_metas


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
        assert t.shape[:-2] == tensors[0].shape[:-2], print('Mismatch size:', t.shape, tensors[0].shape)

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


def get_dataset(data_type, data_name, input, num_clip_frame, inference=False):
    from configs._base_.datasets import get_dataset_config
    from .ytvos import YTVOSDataset
    from .coco import COCODetection
    from .VID import VIDDataset

    dataset_config = get_dataset_config(data_name, data_type)
    if not inference:
        flip, MS_train = True, input.MULTISCALE_TRAIN
        resize_gt, pad_gt = True, True
        img_scales = [input.MIN_SIZE_TRAIN, input.MAX_SIZE_TRAIN]
    else:
        flip, MS_train = False, False
        resize_gt, pad_gt = False, False
        img_scales = [input.MIN_SIZE_TEST, input.MAX_SIZE_TEST]

    backbone_transform = {
        'channel_order': 'RGB',
        'normalize': True,
        'subtract_means': False,
        'to_float': False,
    }

    dataset_config['ann_file'] = os.path.join(dataset_config['data_dir'], dataset_config['ann_file'])
    dataset_config['img_prefix'] = os.path.join(dataset_config['data_dir'], dataset_config['img_prefix'])

    if data_type == 'vis':
        dataset = YTVOSDataset(ann_file=dataset_config['ann_file'],
                               img_prefix=dataset_config['img_prefix'],
                               has_gt=dataset_config['has_gt'],
                               clip_frames=num_clip_frame,
                               size_divisor=input.SIZE_DIVISOR,
                               transform=BaseTransform_vis(
                                    min_size=input.MIN_SIZE_TRAIN,
                                    max_size=input.MAX_SIZE_TRAIN,
                                    Flip=flip,
                                    MS_train=MS_train,
                                    backbone_transform=backbone_transform,
                                    resize_gt=resize_gt,
                                    pad_gt=pad_gt))
    elif data_type == 'vid':
        dataset_config['img_idx'] = os.path.join(dataset_config['data_dir'], dataset_config['img_idx'])
        dataset = VIDDataset(ann_file=dataset_config['ann_file'],
                             img_prefix=dataset_config['img_prefix'],
                             img_index=dataset_config['img_index'],
                             has_gt=dataset_config['has_gt'],
                             clip_frames=num_clip_frame,
                             size_divisor=input.SIZE_DIVISOR,
                             transform=BaseTransform_vid(
                                   img_scales=img_scales,
                                   Flip=flip,
                                   MS_train=MS_train,
                                   preserve_aspect_ratio=input.PRESERVE_ASPECT_RATIO,
                                   backbone_transform=backbone_transform,
                                   resize_gt=resize_gt,
                                   pad_gt=pad_gt))

    elif data_type == 'coco':
        dataset = COCODetection(image_path=dataset_config['img_prefix'],
                                info_file=dataset_config['ann_file'],
                                transform=BaseTransform_coco(img_scales,
                                                             Flip=flip,
                                                             MS_train=MS_train,
                                                             preserve_aspect_ratio=input.PRESERVE_ASPECT_RATIO,
                                                             backbone_transform=backbone_transform,
                                                             resize_gt=resize_gt,
                                                             pad_gt=pad_gt))
    else:
        RuntimeError("Dataset not available: {}".format(data_name))

    return dataset



