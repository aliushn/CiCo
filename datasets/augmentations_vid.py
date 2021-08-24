import cv2
import mmcv
import numpy as np
import torch
from numpy import random
from .config import MEANS, STD


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        if boxes is not None:
            n_f = len(image)
            height, width, _ = image[0].shape
            for i in range(n_f):
                boxes[i][:, 0] /= width
                boxes[i][:, 2] /= width
                boxes[i][:, 1] /= height
                boxes[i][:, 3] /= height

        return image, boxes, labels


class RandomFlip(object):
    def __call__(self, image, boxes, labels):
        if random.randint(2):
            for i in range(len(image)):
                image[i] = image[i][::-1, :]
                boxes = boxes.copy()
                boxes[i][:, 1::2] = 1 - boxes[i][:, 3::-2]
        return image, boxes, labels


class Resize(object):
    """
    The same resizing scheme as used in faster R-CNN
    https://arxiv.org/pdf/1506.01497.pdf

    We resize the image so that the shorter side is min_size.
    If the longer side is then over max_size, we instead resize
    the image so the long side is max_size.
    """

    def __init__(self, img_scales, MS_train=False, preserve_aspect_ratio=True, resize_gt=True):
        self.divisibility = 32
        self.resize_gt = resize_gt
        self.MS_train = MS_train
        self.img_scales = img_scales
        self.preserve_aspect_ratio = preserve_aspect_ratio

    def __call__(self, image, boxes, labels=None):
        n_f = len(image)
        img_h, img_w, _ = image[0].shape

        if self.MS_train:
            ms_idx = np.random.randint(len(self.img_scales))
            pad_h, pad_w = min(self.img_scales[ms_idx]), max(self.img_scales[ms_idx])
        else:
            pad_h, pad_w = min(self.img_scales[-1]), max(self.img_scales[-1])
        img_ratio, pad_ratio = img_h / img_w, pad_h / pad_w

        if self.preserve_aspect_ratio:
            # resize long edges
            if img_ratio < pad_ratio:
                width, height = pad_w, int(img_h * (pad_w / img_w))
            else:
                width, height = int(img_w * (pad_h / img_h)), pad_h
        else:
            width, height = pad_w, pad_h
        assert width <= pad_w and height <= pad_h

        # the input of cv2.resize() should be 3-dimention with (h, w, c)
        image = [cv2.resize(image[i], (width, height)) for i in range(n_f)]

        # for precent coords, bboxes do not need resize

        return image, boxes, labels


class Pad(object):
    """
    Pads the image to the input width and height, filling the
    background with mean and putting the image in the top-left.

    Note: this expects im_w <= width and im_h <= height
    """
    def __init__(self, img_scales, mean=MEANS, pad_gt=True):
        self.mean = mean
        self.img_scales = img_scales
        self.pad_gt = pad_gt

    def __call__(self, image, boxes=None, labels=None):
        n_f = len(image)
        im_h, im_w, depth = image[0].shape

        width, height = max(self.img_scales[-1]), min(self.img_scales[-1])
        for i in range(n_f):
            expand_image = np.zeros((height, width, depth), dtype=image[i].dtype)
            expand_image[:, :, :] = self.mean
            expand_image[:im_h, :im_w] = image[i]
            image[i] = expand_image

        if self.pad_gt:
            for i in range(n_f):
                # Scale bounding boxes (which are currently percent coordinates)
                boxes[i][:, [0, 2]] *= (im_w / width)
                boxes[i][:, [1, 3]] *= (im_h / height)

        return image, boxes, labels


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        image = [image[i].astype(np.float32) for i in range(len(image))]
        return image, boxes, labels


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img,boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


def do_nothing(img=None, boxes=None, labels=None):
    return img, boxes, labels


def enable_if(condition, obj):
    return obj if condition else do_nothing


class BaseTransform_vid(object):
    """ Transorm to be used when evaluating. """

    def __init__(self, img_scales, Flip=False, MS_train=False, preserve_aspect_ratio=True,
                 backbone_transform=None, resize_gt=True, pad_gt=True, mean=MEANS, std=STD):
        self.augment = Compose([
            ConvertFromInts(),
            ToPercentCoords(),
            enable_if(Flip, RandomFlip()),
            Resize(img_scales, MS_train=MS_train, preserve_aspect_ratio=preserve_aspect_ratio, resize_gt=resize_gt),
            Pad(img_scales, mean, pad_gt=pad_gt),
            BackboneTransform(backbone_transform, mean, std, 'BGR')
        ])

    def __call__(self, img, boxes=None, labels=None):
        return self.augment(img, boxes, labels)


class BackboneTransform(object):
    """
    Transforms a BRG image made of floats in the range [0, 255] to whatever
    input the current backbone network needs.

    transform is a transform config object (see config.py).
    in_channel_order is probably 'BGR' but you do you, kid.
    """
    def __init__(self, transform, mean, std, in_channel_order):
        self.mean = np.array(mean, dtype=np.float32)
        self.std  = np.array(std,  dtype=np.float32)
        self.transform = transform

        # Here I use "Algorithms and Coding" to convert string permutations to numbers
        self.channel_map = {c: idx for idx, c in enumerate(in_channel_order)}
        self.channel_permutation = [self.channel_map[c] for c in transform.channel_order]

    def __call__(self, imgs, boxes=None, labels=None):

        for i in range(len(imgs)):
            img = imgs[i].astype(np.float32)

            if self.transform.normalize:
                img = (img - self.mean) / self.std
            elif self.transform.subtract_means:
                img = (img - self.mean)
            elif self.transform.to_float:
                img = img / 255

            imgs[i] = img[:, :, self.channel_permutation].astype(np.float32)

        return imgs, boxes, labels
