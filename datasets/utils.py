import copy
from collections import Sequence
from typing import Any, List, Tuple

import mmcv
from mmcv.runner import obj_from_dict
import torch

import matplotlib.pyplot as plt
import numpy as np
from .concat_dataset import ConcatDataset
from .repeat_dataset import RepeatDataset
import datasets
from torch.autograd import Variable
import torch.nn.functional as F
import random


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.
    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


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


def get_dataset(data_cfg):
    data_cfg = vars(data_cfg)
    if data_cfg['type'] == 'RepeatDataset':
        return RepeatDataset(
            get_dataset(data_cfg['dataset']), data_cfg['times'])

    if isinstance(data_cfg['ann_file'], (list, tuple)):
        ann_files = data_cfg['ann_file']
        num_dset = len(ann_files)
    else:
        ann_files = [data_cfg['ann_file']]
        num_dset = 1

    if 'proposal_file' in data_cfg.keys():
        if isinstance(data_cfg['proposal_file'], (list, tuple)):
            proposal_files = data_cfg['proposal_file']
        else:
            proposal_files = [data_cfg['proposal_file']]
    else:
        proposal_files = [None] * num_dset
    assert len(proposal_files) == num_dset

    if isinstance(data_cfg['img_prefix'], (list, tuple)):
        img_prefixes = data_cfg['img_prefix']
    else:
        img_prefixes = [data_cfg['img_prefix']] * num_dset
    assert len(img_prefixes) == num_dset

    dsets = []
    for i in range(num_dset):
        data_info = copy.deepcopy(data_cfg)
        data_info['ann_file'] = ann_files[i]
        data_info['proposal_file'] = proposal_files[i]
        data_info['img_prefix'] = img_prefixes[i]
        dset = obj_from_dict(data_info, datasets)
        dsets.append(dset)
    if len(dsets) > 1:
        dset = ConcatDataset(dsets)
    else:
        dset = dsets[0]
    return dset


def detection_collate_vis(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and (lists of annotations, masks)

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list<tensor>, list<tensor>, list<int>) annotations for a given image are stacked
                on 0 dim. The output gt is a tuple of annotations and masks.
    """
    batch_out = {}
    # batch_out['img'] = torch.cat([batch[i]['img'].data for i in range(batch_size)])
    # if 'ref_imgs' in batch[0].keys():
    #     batch_out['ref_imgs'] = torch.cat([batch[i]['ref_imgs'].data for i in range(batch_size)])

    for k in batch[0].keys():
        batch_out[k] = []

    for i in range(len(batch)):
        for k in batch_out.keys():
            if isinstance(batch[i][k], list):
                batch_out[k].append(batch[i][k])
            else:
                batch_out[k].append(batch[i][k].data)

    return batch_out


def detection_collate_coco(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and (lists of annotations, masks)

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list<tensor>, list<tensor>, list<int>) annotations for a given image are stacked
                on 0 dim. The output gt is a tuple of annotations and masks.
    """
    targets = []
    imgs = []
    masks = []
    num_crowds = []

    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1][0]))
        masks.append(torch.FloatTensor(sample[1][1]))
        num_crowds.append(sample[1][2])

    return imgs, targets, masks, num_crowds


def prepare_data_vis(data_batch, devices, train_mode=True):
    if train_mode:
        with torch.no_grad():
            images = torch.cat(data_batch['img']).cuda(devices)
            bs = images.size(0)
            bboxes_list = [sum(data_batch['bboxes'], [])[i].cuda(devices) for i in range(bs)]
            labels_list = [sum(data_batch['labels'], [])[i].cuda(devices) for i in range(bs)]
            masks_list = [sum(data_batch['masks'], [])[i].cuda(devices) for i in range(bs)]
            ids_list = [sum(data_batch['ids'], [])[i].cuda(devices) for i in range(bs)]
            images_meta_list = data_batch['img_meta']

            return images, bboxes_list, labels_list, masks_list, ids_list, images_meta_list

    else:
        # [0] is downsample image [1, 3, 384, 640], [1] is original image [1, 3, 736, 1280]
        images = torch.stack([img[0].data for img in data_batch['img']], dim=0)
        images_meta = [img_meta[0].data for img_meta in data_batch['img_meta']]

        images = gradinator(images.cuda(devices))
        images_meta = images_meta

        return images, images_meta


def prepare_data_coco(data_batch, devices, train_mode=True):
    if train_mode:
        with torch.no_grad():
            imgs = ImageList_from_tensors(data_batch[0], size_divisibility=32).cuda(devices)
            image_sizes_ori = torch.tensor([[im.shape[-2], im.shape[-1]] for im in data_batch[0]])
            h, w = imgs.size()[-2:]
            bboxes_list = []
            for target in data_batch[1]:
                target[:, 0:4:2] /= w
                target[:, 1:4:2] /= h
                bboxes_list.append(target[:, :4].cuda(devices))

            labels_list = [target[:, -1].data.long().cuda(devices) for target in data_batch[1]]
            masks_list = []
            for i, mask in enumerate(data_batch[2]):
                padding_size = [0, w - image_sizes_ori[i, 1], 0, h - image_sizes_ori[i, 0]]
                masks_list.append(F.pad(mask, padding_size, value=0).cuda(devices))
            num_crowds = data_batch[3]

            return imgs, bboxes_list, labels_list, masks_list, num_crowds

    else:
        # [0] is downsample image [1, 3, 384, 640], [1] is original image [1, 3, 736, 1280]
        images = torch.stack([img[0].data for img in data_batch['img']], dim=0)
        images_meta = [img_meta[0].data for img_meta in data_batch['img_meta']]

        images = gradinator(images.cuda(devices))
        images_meta = images_meta

        return images, images_meta


def gradinator(x):
    x.requires_grad = False
    return x


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

    image_sizes = torch.tensor([[im.shape[-2], im.shape[-1]] for im in tensors])
    max_size, _ = image_sizes.max(0)
    max_size_h, max_size_w = max_size[0], max_size[1]

    if size_divisibility > 1:
        # the last two dims are H,W, both subject to divisibility requirement
        max_size_h = (max_size_h + (size_divisibility - 1)) // size_divisibility * size_divisibility
        max_size_w = (max_size_w + (size_divisibility - 1)) // size_divisibility * size_divisibility

    if len(tensors) == 1:
        # This seems slightly (2%) faster.
        # TODO: check whether it's faster for multiple images as well
        image_size = image_sizes[0]
        padding_size = [0, max_size_w - image_size[1], 0, max_size_h - image_size[0]]
        batched_imgs = F.pad(tensors[0], padding_size, value=pad_value).unsqueeze_(0)
    else:
        # max_size can be a tensor in tracing mode, therefore convert to list
        batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + [max_size_h, max_size_w]
        batched_imgs = tensors[0].new_full(batch_shape, pad_value)
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)

    return batched_imgs.contiguous()


def enforce_size(img, targets, masks, num_crowds, new_w, new_h):
    """ Ensures that the image is the given size without distorting aspect ratio. """
    with torch.no_grad():
        _, h, w = img.size()

        if h == new_h and w == new_w:
            return img, targets, masks, num_crowds

        # Resize the image so that it fits within new_w, new_h
        w_prime = new_w
        h_prime = h * new_w / w

        if h_prime > new_h:
            w_prime *= new_h / h_prime
            h_prime = new_h

        w_prime = int(w_prime)
        h_prime = int(h_prime)

        # Do all the resizing
        img = F.interpolate(img.unsqueeze(0), (h_prime, w_prime), mode='bilinear', align_corners=False)
        img.squeeze_(0)

        # Act like each object is a color channel
        masks = F.interpolate(masks.unsqueeze(0), (h_prime, w_prime), mode='bilinear', align_corners=False)
        masks.squeeze_(0)

        # Scale bounding boxes (this will put them in the top left corner in the case of padding)
        targets[:, [0, 2]] *= (w_prime / new_w)
        targets[:, [1, 3]] *= (h_prime / new_h)

        # Finally, pad everything to be the new_w, new_h
        pad_dims = (0, new_w - w_prime, 0, new_h - h_prime)
        img = F.pad(img, pad_dims, mode='constant', value=0)
        masks = F.pad(masks, pad_dims, mode='constant', value=0)

        return img, targets, masks, num_crowds

