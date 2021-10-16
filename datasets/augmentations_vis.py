import cv2
import mmcv
import numpy as np
import torch
from numpy import random
from .config import MEANS, STD

# __all__ = ['ImageTransform', 'BboxTransform', 'MaskTransform', 'Numpy2Tensor']


class ToPercentCoords(object):
    def __call__(self, image, masks=None, boxes=None, labels=None):
        if boxes is not None:
            n_f = len(image)
            height, width, _ = image[0].shape
            for i in range(n_f):
                boxes[i][:, 0] /= width
                boxes[i][:, 2] /= width
                boxes[i][:, 1] /= height
                boxes[i][:, 3] /= height

        return image, masks, boxes, labels


class RandomFlip(object):
    def __call__(self, image, masks, boxes, labels):
        if random.randint(2):
            for i in range(len(image)):
                image[i] = image[i][::-1, :]
                masks[i] = masks[i][:, ::-1, :]
                boxes = boxes.copy()
                boxes[i][:, 1::2] = 1 - boxes[i][:, 3::-2]
        return image, masks, boxes, labels


class Resize(object):
    """
    The same resizing scheme as used in faster R-CNN
    https://arxiv.org/pdf/1506.01497.pdf

    We resize the image so that the shorter side is min_size.
    If the longer side is then over max_size, we instead resize
    the image so the long side is max_size.
    """

    def __init__(self, min_size, max_size, MS_train=False, resize_gt=True):
        self.resize_gt = resize_gt
        self.MS_train = MS_train
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, masks, boxes, labels=None):
        n_f = len(image)
        img_h, img_w, _ = image[0].shape
        if isinstance(self.min_size, int):
            min_size = self.min_size
        else:
            min_size = self.min_size[np.random.randint(len(self.min_size))] if self.MS_train else self.min_size[-1]
        scale_img, scale = img_h / img_w, min_size / self.max_size
        if scale_img < scale:
            width, height = self.max_size, int(img_h * (self.max_size / img_w))
        else:
            width, height = int(img_w * (min_size / img_h)), min_size

        # the input of cv2.resize() should be 3-dimention with (h, w, c)
        image = [cv2.resize(image[i], (width, height)) for i in range(n_f)]

        if self.resize_gt:
            # Act like each object is a color channel
            for i in range(n_f):
                masks_resize = cv2.resize(masks[i].transpose((1, 2, 0)), (width, height))

                # OpenCV resizes a (w,h,1) array to (s,s), so fix that
                if len(masks_resize.shape) == 2:
                    masks[i] = np.expand_dims(masks_resize, 0)
                else:
                    masks[i] = masks_resize.transpose((2, 0, 1))

                # for precent coords, bboxes do not need resize

        return image, masks, boxes, labels


class Pad(object):
    """
    Pads the image to the input width and height, filling the
    background with mean and putting the image in the top-left.

    Note: this expects im_w <= width and im_h <= height
    """
    def __init__(self, min_size, max_size, mean=MEANS, pad_gt=True):
        self.min_size = min_size
        self.max_size = max_size
        self.mean = mean
        self.divisibility = 32
        self.pad_gt = pad_gt

    def __call__(self, image, masks, boxes=None, labels=None):
        n_f = len(image)
        im_h, im_w, depth = image[0].shape

        width = ((self.max_size-1)//self.divisibility+1)*self.divisibility
        height = ((self.min_size-1)//self.divisibility+1)*self.divisibility
        if im_h != height or width != im_w:
            for i in range(n_f):
                expand_image = np.zeros((height, width, depth), dtype=image[i].dtype)
                expand_image[:, :, :] = self.mean
                expand_image[:im_h, :im_w] = image[i]
                image[i] = expand_image

            if self.pad_gt:
                for i in range(n_f):
                    expand_masks = np.zeros(
                            (masks[i].shape[0], height, width), dtype=masks[i].dtype)
                    expand_masks[:, :im_h, :im_w] = masks[i]
                    masks[i] = expand_masks

                    # Scale bounding boxes (which are currently percent coordinates)
                    boxes[i][:, [0, 2]] *= (im_w / width)
                    boxes[i][:, [1, 3]] *= (im_h / height)

        return image, masks, boxes, labels


class ConvertFromInts(object):
    def __call__(self, image, masks=None, boxes=None, labels=None):
        image = [image[i].astype(np.float32) for i in range(len(image))]
        return image, masks, boxes, labels


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

    def __call__(self, img, masks=None, boxes=None, labels=None):
        for t in self.transforms:
            img, masks, boxes, labels = t(img, masks, boxes, labels)
        return img, masks, boxes, labels


def do_nothing(img=None, masks=None, boxes=None, labels=None):
    return img, masks, boxes, labels


def enable_if(condition, obj):
    return obj if condition else do_nothing


class BaseTransform_vis(object):
    """ Transorm to be used when evaluating. """

    def __init__(self, min_size, max_size, Flip=False, MS_train=False, backbone_transform=None,
                 resize_gt=True, pad_gt=True, mean=MEANS, std=STD):
        self.augment = Compose([
            ConvertFromInts(),
            ToPercentCoords(),
            enable_if(Flip, RandomFlip()),
            Resize(min_size, max_size, MS_train=MS_train, resize_gt=resize_gt),
            # Pad(min_size, max_size, mean, pad_gt=pad_gt),
            BackboneTransform(backbone_transform, mean, std, 'BGR')
        ])

    def __call__(self, img, masks=None, boxes=None, labels=None):
        return self.augment(img, masks, boxes, labels)


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
        self.channel_permutation = [self.channel_map[c] for c in transform['channel_order']]

    def __call__(self, imgs, masks=None, boxes=None, labels=None):

        for i in range(len(imgs)):
            img = imgs[i].astype(np.float32)

            if self.transform['normalize']:
                img = (img - self.mean) / self.std
            elif self.transform['subtract_means']:
                img = (img - self.mean)
            elif self.transform['to_float']:
                img = img / 255

            imgs[i] = img[:, :, self.channel_permutation].astype(np.float32)

        return imgs, masks, boxes, labels


class ImageTransform(object):
    """Preprocess an image.

    1. rescale the image to expected size
    2. normalize the image
    3. flip the image (if needed)
    4. pad the image (if needed)
    5. transpose to (c, h, w)
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 to_rgb=True,
                 size_divisor=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.size_divisor = size_divisor

    def __call__(self, img, scale, flip=False, keep_ratio=True):
        if keep_ratio:
            img, scale_factor = mmcv.imrescale(img, scale, return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img, scale, return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        img_shape = img.shape
        img = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
        if flip:
            img = mmcv.imflip(img)
        if self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor)
            pad_shape = img.shape
        else:
            pad_shape = img_shape
        img = img.transpose(2, 0, 1)
        return img, img_shape, pad_shape, scale_factor


def bbox_flip(bboxes, img_shape):
    """Flip bboxes horizontally.

    Args:
        bboxes(ndarray): shape (..., 4*k)
        img_shape(tuple): (height, width)
    """
    assert bboxes.shape[-1] % 4 == 0
    w = img_shape[1]
    flipped = bboxes.copy()
    flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
    flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
    return flipped


class BboxTransform(object):
    """Preprocess gt bboxes.

    1. rescale bboxes according to image size
    2. flip bboxes (if needed)
    3. pad the first dimension to `max_num_gts`
    """

    def __init__(self, max_num_gts=None):
        self.max_num_gts = max_num_gts

    def __call__(self, bboxes, img_shape, pad_shape, scale_factor, flip=False):
        gt_bboxes = bboxes * scale_factor
        if flip:
            gt_bboxes = bbox_flip(gt_bboxes, img_shape)
        # normalization [0, 1] [x1,y1,x2,y2]
        gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1])/pad_shape[1]
        gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0])/pad_shape[0]

        if self.max_num_gts is None:
            return gt_bboxes
        else:
            num_gts = gt_bboxes.shape[0]
            padded_bboxes = np.zeros((self.max_num_gts, 4), dtype=np.float32)
            padded_bboxes[:num_gts, :] = gt_bboxes
            return padded_bboxes


class MaskTransform(object):
    """Preprocess masks.

    1. resize masks to expected size and stack to a single array
    2. flip the masks (if needed)
    3. pad the masks (if needed)
    """

    def __call__(self, masks, pad_shape, scale, flip=False, keep_ratio=True):
        if keep_ratio:
            masks = [
                mmcv.imrescale(mask, scale, interpolation='nearest')
                for mask in masks
            ]
        else:
            masks = [
                mmcv.imresize(mask, scale, interpolation='nearest')
                for mask in masks
            ]

        if flip:
            masks = [mask[:, ::-1] for mask in masks]

        padded_masks = [
            mmcv.impad(mask, shape=pad_shape[:2], pad_val=0) for mask in masks
        ]
        padded_masks = np.stack(padded_masks, axis=0)
        return padded_masks


class Numpy2Tensor(object):

    def __init__(self):
        pass

    def __call__(self, *args):
        if len(args) == 1:
            return torch.from_numpy(args[0])
        else:
            return tuple([torch.from_numpy(np.array(array)) for array in args])
