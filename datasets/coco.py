import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torch.nn.functional as F
import cv2
import numpy as np
from pycocotools import mask as maskUtils
import random


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


class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """

    def __init__(self):
        self.label_map = COCO_LABEL_MAP

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
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                label_idx = obj['category_id']
                if label_idx >= 0:
                    label_idx = self.label_map[label_idx]
                final_box = list(np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]) / scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("No bbox found for object ", obj)

        return res


class COCODetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
        prep_crowds (bool): Whether or not to prepare crowds for the evaluation step.
    """

    def __init__(self, image_path, info_file, transform=None, target_transform=None,
                 dataset_name='MS COCO', has_gt=True):
        # Do this here because we have too many things named COCO
        from pycocotools.coco import COCO

        self.root = image_path
        self.coco = COCO(info_file)

        self.ids = list(self.coco.imgToAnns.keys())
        if len(self.ids) == 0 or not has_gt:
            self.ids = list(self.coco.imgs.keys())

        self.transform = transform
        self.target_transform = COCOAnnotationTransform()

        self.name = dataset_name
        self.has_gt = has_gt

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
            target = []

        # Separate out crowd annotations. These are annotations that signify a large crowd of
        # objects of said class, where there is no annotation for each individual object. Both
        # during testing and training, consider these crowds as neutral.
        crowd = [x for x in target if ('iscrowd' in x and x['iscrowd'])]
        target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
        num_crowds = len(crowd)

        for x in crowd:
            x['category_id'] = -1

        # This is so we ensure that all crowd annotations are at the end of the array
        target += crowd

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

        if len(target) > 0:
            # Pool all the masks for this image into one [num_objects,height,width] matrix
            masks = [self.coco.annToMask(obj).reshape(-1) for obj in target]
            masks = np.vstack(masks)
            masks = masks.reshape((-1, height, width))

        if self.target_transform is not None and len(target) > 0:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            if len(target) > 0:
                target = np.array(target)
                img, masks, boxes, labels = self.transform(img, masks, target[:, :4],
                                                           {'num_crowds': num_crowds, 'labels': target[:, 4]})

                # I stored num_crowds in labels so I didn't have to modify the entirety of augmentations
                num_crowds = labels['num_crowds']
                labels = labels['labels']

                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            else:
                img, _, _, _ = self.transform(img, np.zeros((1, height, width), dtype=np.float),
                                              np.array([[0, 0, 1, 1]]), {'num_crowds': 0, 'labels': np.array([0])})
                masks = None
                target = None

        if target is not None and (target.shape[0] == 0 or target.shape[0] == num_crowds):
            print('Warning: Augmentation output an example with no ground truth. Resampling...')
            return self.pull_item(random.randint(0, len(self.ids)-1))

        img_meta = dict(
            ori_shape=(height, width, 3),
            img_shape=img.shape,
            frame_id=index)

        return torch.from_numpy(img).permute(2, 0, 1).contiguous(), img_meta, (target, masks, num_crowds)

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


def detection_collate(batch):
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
    boxes, labels = [], []
    imgs = []
    masks = []
    num_crowds = []
    img_metas = []

    for sample in batch:
        imgs.append(sample[0])
        img_metas.append(sample[1])
        if sample[2][0] is not None:
            boxes.append(sample[2][0][:, :4])
            labels.append(sample[2][0][:, -1])
            masks.append(sample[2][1])
            num_crowds.append(sample[2][2])

    return imgs, img_metas, (boxes, labels, masks, num_crowds)


def prepare_data(data_batch, devices):
    images, img_metas, (boxes, labels, masks, num_crowds) = data_batch

    from .utils import ImageList_from_tensors, GtList_from_tensor
    images = ImageList_from_tensors(images, size_divisibility=32)
    if len(masks) > 0:
        pad_h, pad_w = images.size()[-2:]
        masks, boxes, img_metas = GtList_from_tensor(pad_h, pad_w, masks, boxes, img_metas)

    if images.size(0) % len(devices) != 0:
        n_total = (images.size(0) // len(devices) + 1) * len(devices)
        idx = [i % images.size(0) for i in range(images.size(0), n_total)]
        images = torch.cat([images, images[idx]], dim=0)
        boxes += [boxes[i] for i in idx]
        labels += [labels[i] for i in idx]
        masks += [masks[i] for i in idx]
        num_crowds += [num_crowds[i] for i in idx]
        img_metas += [img_metas[i] for i in idx]
    n = images.size(0) // len(devices)

    with torch.no_grad():
        images_list, masks_list, boxes_list, labels_list, obj_ids_list, num_crowds_list = [], [], [], [], [], []
        img_metas_list = []
        for idx, device in enumerate(devices):
            images_list.append(gradinator(images[idx*n:(idx+1)*n].to(device)))
            masks_list.append([gradinator(torch.tensor(masks[jdx]).to(device)) for jdx in range(idx*n, (idx+1)*n)])
            boxes_list.append([gradinator(torch.tensor(boxes[jdx]).to(device)) for jdx in range(idx*n, (idx+1)*n)])
            labels_list.append([gradinator(torch.tensor(labels[jdx]).to(device)) for jdx in range(idx*n, (idx+1)*n)])
            obj_ids_list.append([None] * n)
            num_crowds_list.append(num_crowds[idx*n:(idx+1)*n])
            img_metas_list.append(img_metas[idx*n:(idx+1)*n])

        return images_list, boxes_list, labels_list, masks_list, obj_ids_list, num_crowds_list, img_metas_list


def gradinator(x):
    x.requires_grad = False
    return x