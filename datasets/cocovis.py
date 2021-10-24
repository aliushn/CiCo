import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torch.nn.functional as F
import cv2
import numpy as np
from pycocotools import mask as maskUtils
from numpy import random


COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock',
                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

COCO_LABEL_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                  9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}


COCO_VIS_CLASSES = ('person', 'car', 'motorcycle', 'airplane',
                    'train', 'truck', 'boat', 'bird', 'cat',
                    'dog', 'horse', 'cow', 'elephant', 'bear',
                    'zebra', 'giraffe', 'snowboard', 'skateboard', 'surfboard',
                    'tennis racket', 'mouse')

COCO_VIS_LABEL_MAP = {1: 26,   3: 5,   4: 23,  5: 1,
                      7: 36,   8: 37,  9: 4,  16: 3,  17: 6,
                      18: 9,  19: 19, 21: 7,  22: 12, 23: 2,
                      24: 40, 25: 18, 36: 31, 41: 29, 42: 33,
                      43: 34, 74: 24}


class COCOVISAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """

    def __init__(self, coco):
        self.label_map = COCO_VIS_LABEL_MAP
        self.coco = coco

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res, res_mask = [], []
        for i, obj in enumerate(target):
            if 'bbox' in obj:
                bbox = obj['bbox']
                label_idx = obj['category_id']

                # only remain objects whose category also exist in YTVIS2021
                if label_idx in self.label_map:
                    label_idx = self.label_map[label_idx]
                    final_box = list(np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]) / scale)
                    final_box.append(label_idx)
                    res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
                    res_mask += [self.coco.annToMask(obj).reshape(-1)]
            else:
                print("No bbox found for object ", obj)

        if len(res_mask) > 0:
            res_mask = np.vstack(res_mask).reshape((-1, height, width))
        return res, res_mask


class COCO2VIS(object):
    def __init__(self, frames, scale=0.05):
        self.frames = frames
        self.scale = scale

    def __call__(self, img, boxes, masks):
        clip_imgs, clip_boxes, clip_masks = [], [], []
        for i in range(self.frames):
            # Translation
            if random.randint(2):
                height, width, depth = img.shape
                expand_img = np.zeros((height, width, depth), dtype=img.dtype)
                dx, dy = int(random.randn()*self.scale*width), int(random.randn()*self.scale*height)
                clip_imgs.append(self.translation(img, expand_img, dx, dy, width, height))
                expand_masks = np.zeros((height, width, masks.shape[0]), dtype=masks.dtype)
                expand_masks = self.translation(masks.transpose(1, 2, 0), expand_masks, dx, dy, width, height)
                clip_masks.append(expand_masks.transpose(2, 0, 1))
                dxy = np.expand_dims(np.array([dx/width, dy/height, dx/width, dy/height]), axis=0)
                clip_boxes.append(np.clip(boxes+dxy, 0, 1))

            # rotation
            # if random.randint(2):
            else:
                clip_imgs.append(img)
                clip_boxes.append(boxes)
                clip_masks.append(masks)

        clip_imgs = np.concatenate(clip_imgs, axis=-1)              # [height, width, T*depth]
        clip_masks = np.stack(clip_masks, axis=1)                   # [n_objs, T, height, width]
        clip_boxes = np.stack(clip_boxes, axis=1)                   # [n_objs, T, 4]
        return clip_imgs, clip_boxes, clip_masks

    def translation(self, img, expand_img, dx, dy, width, height):
        if dx >= 0 and dy >= 0:
            expand_img[dy:, dx:, :] = img[:height-dy, :width-dx, :]     # Move right and bottom
        elif dx < 0 and dy >= 0:
            expand_img[dy:, :width+dx, :] = img[:height-dy, -dx:, :]     # Move left and bottom
        elif dx >= 0 and dy < 0:
            expand_img[:height+dy, dx:, :] = img[-dy:, :width-dx, :]     # Move right and upper
        else:
            expand_img[:height+dy, :width+dx, :] = img[-dy:, -dx:, :]     # Move left and upper

        return expand_img


class COCOVISDetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        in the target (bbox) and transforms it.
        prep_crowds (bool): Whether or not to prepare crowds for the evaluation step.
    """

    def __init__(self, image_path, info_file, transform=None, frames=1,
                 dataset_name='MS COCO', has_gt=True):
        # Do this here because we have too many things named COCO
        from pycocotools.coco import COCO

        self.root = image_path
        self.coco = COCO(info_file)

        self.ids = list(self.coco.imgToAnns.keys())
        if len(self.ids) == 0 or not has_gt:
            self.ids = list(self.coco.imgs.keys())

        self.transform = transform
        self.target_transform = COCOVISAnnotationTransform(self.coco)

        self.name = dataset_name
        self.has_gt = has_gt

        self.frames = frames
        self.COCO2VIS = COCO2VIS(self.frames)

        self.class_names = COCO_CLASSES
        self.label_map = COCO_LABEL_MAP

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, (target, masks, num_crowds)).
                   target is the object returned by ``coco.loadAnns``.
        """
        return self.pull_item(index)

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, masks, height, width, crowd).
                   target is the object returned by ``coco.loadAnns``.
            Note that if no crowd annotations exist, crowd will be None
        """
        img_id = self.ids[index]

        if self.has_gt:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)

            # Target has {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
            target = [x for x in self.coco.loadAnns(ann_ids) if x['image_id'] == img_id]
        else:
            target = None

        # The split here is to have compatibility with both COCO2014 and 2017 annotations.
        # In 2014, images have the pattern COCO_{train/val}2014_%012d.jpg, while in 2017 it's %012d.jpg.
        # Our script downloads the images as %012d.jpg so convert accordingly.
        file_name = self.coco.loadImgs(img_id)[0]['file_name']

        if file_name.startswith('COCO'):
            file_name = file_name.split('_')[-1]

        path = osp.join(self.root, file_name)
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)

        img = cv2.imread(path)
        height, width, _ = img.shape
        if self.target_transform is not None and target is not None:
            if len(target) > 0:
                target, masks = self.target_transform(target, width, height)

        if target is not None and (len(target) == 0):
            # print('Warning: Augmentation output an example with no overlapped categories ground truth. Resampling...')
            return self.pull_item(random.randint(0, len(self.ids)-1))

        if self.transform is not None:
            if len(target) > 0:
                target = np.array(target)
                img, masks, boxes, labels = self.transform(img, masks, target[:, :4], {'labels': target[:, 4]})
                labels = labels['labels']
                # Along with temporal dim to expand a single static image to a clip with small translation
                img, boxes, masks = self.COCO2VIS(img, boxes, masks)
                valid = ((boxes[:, :, 2:] - boxes[:, :, :2]) >= 0.05).sum(axis=(1, 2)) == 2*boxes.shape[1]
                if sum(~valid) > 0:
                    if sum(valid) > 0:
                        boxes = np.stack([boxes[i] for i, v in enumerate(valid) if v], axis=0)
                        masks = np.stack([masks[i] for i, v in enumerate(valid) if v], axis=0)
                        labels = np.array([labels[i] for i, v in enumerate(valid) if v])
                    else:
                        boxes, masks, labels = np.array([]), np.array([]), np.array([])

            else:
                img, _, _, _ = self.transform(img, np.zeros((1, height, width), dtype=np.float),
                                              np.array([[0, 0, 1, 1]]), {'labels': np.array([0])})
                masks, boxes, labels = None, None, None

        if boxes is not None and boxes.shape[0] == 0:
            # print('Warning: Augmentation output an example with no ground truth. Resampling...')
            return self.pull_item(random.randint(0, len(self.ids)-1))

        img_meta = [dict(
            ori_shape=(height, width, 3),
            img_shape=img.shape,
            frame_id=index)] * self.frames

        return img, img_meta, (masks, boxes, labels, np.arange(boxes.shape[0]))

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def detection_collate_cocovis(batch):
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
    imgs = []
    masks = []
    boxes = []
    labels = []
    ids = []
    img_metas = []

    for sample in batch:
        imgs += [torch.from_numpy(sample[0]).permute(2, 0, 1)]
        img_metas += [sample[1]]
        if sample[2][0] is not None:
            masks += [sample[2][0]]
            boxes += [sample[2][1]]
            labels += [sample[2][2]]
            ids += [sample[2][3]]

    from .utils import ImageList_from_tensors, GtList_from_tensor
    imgs_batch = ImageList_from_tensors(imgs, size_divisibility=32)
    pad_h, pad_w = imgs_batch.size()[-2:]

    if len(masks) > 0:
        masks, bboxes, img_metas = GtList_from_tensor(pad_h, pad_w, masks, boxes, img_metas)
        return imgs_batch, img_metas, (bboxes, labels, masks, ids)
    else:
        _, _, img_metas = GtList_from_tensor(pad_h, pad_w, None, None, img_metas)
        return imgs_batch, img_metas


def prepare_data_cocovis(data_batch, devices):
    images, image_metas, (bboxes, labels, masks, obj_ids) = data_batch
    h, w = images.size()[-2:]
    d = len(devices)
    if images.size(0) % d != 0:
        # TODO: if read multi frames (n_f) as a clip, thus the condition should be images.size(0) / n_f % len(devices)
        idx = [i % images.size(0) for i in range(d)]
        remainder = d - images.size(0) % d
        images = torch.cat([images, images[idx[:remainder]]])
        bboxes += [bboxes[i] for i in idx[:remainder]]
        labels += [labels[i] for i in idx[:remainder]]
        masks += [masks[i] for i in idx[:remainder]]
        obj_ids += [obj_ids[i] for i in idx[:remainder]]
        image_metas += [image_metas[i] for i in idx[:remainder]]
    n = images.size(0) // len(devices)

    with torch.no_grad():
        images_list, masks_list, bboxes_list, labels_list, obj_ids_list, num_crowds_list = [], [], [], [], [], []
        image_metas_list = []
        for idx, device in enumerate(devices):
            images_list.append(gradinator(images[idx * n:(idx + 1) * n].reshape(-1, 3, h, w).to(device)))
            masks_list.append([gradinator(torch.from_numpy(masks[jdx]).to(device)) for jdx in range(idx * n, (idx + 1) * n)])
            bboxes_list.append([gradinator(torch.from_numpy(bboxes[jdx]).float().to(device)) for jdx in range(idx * n, (idx + 1) * n)])
            labels_list.append([gradinator(torch.from_numpy(labels[jdx]).to(device)) for jdx in range(idx * n, (idx + 1) * n)])
            obj_ids_list.append([gradinator(torch.from_numpy(obj_ids[jdx]).to(device)) for jdx in range(idx * n, (idx + 1) * n)])
            num_crowds_list.append([0] * n)
            image_metas_list.append(image_metas[idx * n:(idx + 1) * n])

        return images_list, bboxes_list, labels_list, masks_list, obj_ids_list, num_crowds_list, image_metas_list


def gradinator(x):
    x.requires_grad = False
    return x