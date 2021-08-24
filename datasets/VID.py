import os
import json

import torch
import torch.utils.data

from PIL import Image
import cv2
import sys
import numpy as np
import random

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from .BoxList import BoxList


class VIDDataset(torch.utils.data.Dataset):
    classes = ['__background__',  # always index 0
               'airplane', 'antelope', 'bear', 'bicycle', 'bird',
               'bus', 'car', 'cattle', 'dog', 'domestic_cat',
               'elephant', 'fox', 'giant_panda', 'hamster', 'horse',
               'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit',
               'red_panda', 'sheep', 'snake', 'squirrel', 'tiger',
               'train', 'turtle', 'watercraft', 'whale', 'zebra']
    classes_map = ['__background__',  # always index 0
                   'n02691156', 'n02419796', 'n02131653', 'n02834778', 'n01503061',
                   'n02924116', 'n02958343', 'n02402425', 'n02084071', 'n02121808',
                   'n02503517', 'n02118333', 'n02510455', 'n02342885', 'n02374451',
                   'n02129165', 'n01674464', 'n02484322', 'n03790512', 'n02324045',
                   'n02509815', 'n02411705', 'n01726692', 'n02355227', 'n02129604',
                   'n04468005', 'n01662784', 'n04530566', 'n02062744', 'n02391049']

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_index,
                 preserve_aspect_ratio=True,
                 transform=None,
                 size_divisor=None,
                 with_mask=False,
                 with_crowd=True,
                 with_label=True,
                 with_track=False,
                 clip_frames=1,
                 has_gt=False):

        self.det_vid = ann_file.split('/')[-1]
        self.image_set = img_index.split('/')[-1][:-4]
        self.transform = transform

        self.img_prefix = img_prefix
        self.ann_file = ann_file
        self.img_index = img_index

        self.has_gt = has_gt
        self.clip_frames = clip_frames

        self._img_dir = os.path.join(self.img_prefix, "%s.JPEG")
        self._anno_path = os.path.join(self.ann_file, "%s.xml")

        with open(self.img_index) as f:
            lines = [x.strip().split(" ") for x in f.readlines()]
        if len(lines[0]) == 2:
            self.image_set_index = [x[0] for x in lines]
            self.frame_id = [int(x[1]) for x in lines]
        else:

            self.image_set_index = ["%s/%06d" % (x[0], int(x[2])) for x in lines]
            self.pattern = [x[0] + "/%06d" for x in lines]
            self.frame_id = [int(x[1]) for x in lines]
            self.frame_seg_id = [int(x[2]) for x in lines]
            self.frame_seg_len = [int(x[3]) for x in lines]

        if self.has_gt:
            keep = self.filter_annotation()

            if len(lines[0]) == 2:
                self.image_set_index = [self.image_set_index[idx] for idx in range(len(keep)) if keep[idx]]
                self.frame_id = [self.frame_id[idx] for idx in range(len(keep)) if keep[idx]]
            else:
                self.image_set_index = [self.image_set_index[idx] for idx in range(len(keep)) if keep[idx]]
                self.pattern = [self.pattern[idx] for idx in range(len(keep)) if keep[idx]]
                self.frame_id = [self.frame_id[idx] for idx in range(len(keep)) if keep[idx]]
                self.frame_seg_id = [self.frame_seg_id[idx] for idx in range(len(keep)) if keep[idx]]
                self.frame_seg_len = [self.frame_seg_len[idx] for idx in range(len(keep)) if keep[idx]]

        self.classes_to_ind = dict(zip(self.classes_map, range(len(self.classes_map))))
        self.categories = dict(zip(range(len(self.classes)), self.classes))
        self.annos = self.load_annos(os.path.join(self.cache_dir, self.image_set + "_anno.json"))

    def __getitem__(self, idx):
        return self._get_train(idx)

    def _get_train(self, idx):
        if self.has_gt and self.clip_frames > 1:
            ref_frame_idx = self.sample_ref(idx)
            clip_frame_idx = ref_frame_idx + [idx]
            clip_frame_idx.sort()
        else:
            clip_frame_idx = [idx]

        imgs = []
        for idx in clip_frame_idx:
            filename = self.image_set_index[idx]
            imgs.append(Image.open(self._img_dir % filename).convert("RGB"))

        if self.has_gt:
            boxes, labels, obj_ids, boxes_occluded = [], [], [], []
            for idx in clip_frame_idx:
                anno = self.annos[idx]
                boxes.append(anno['boxes'])
                labels.append(anno['lables'])
                obj_ids.append(anno['obj_ids'])
                boxes_occluded.append(anno['occluded'])
        else:
            boxes, labels, obj_ids, boxes_occluded = None, None, None, None

        if self.transform is not None:
            imgs, boxes, labels = self.transform(imgs, boxes, labels)

        return imgs, boxes, labels, obj_ids, boxes_occluded

    def __len__(self):
        return len(self.image_set_index)

    def sample_ref(self, idx):
        # sample another frame in the same sequence as reference
        left = max(0, idx-self.clip_frames)
        right = min(len(self.image_set_index), idx+self.clip_frames+1)
        key_vid, key_fid = self.image_set_index[idx].split('/')[-2:]
        valid_samples = []
        for ref_idx in range(left, right):
            # check if the video id is same as key frame
            if ref_idx != idx:
                ref_vid, ref_fid = self.image_set_index[ref_idx].split('/')[-2:]
                if key_vid == ref_vid and key_fid != ref_fid:
                    valid_samples.append(ref_idx)
        if len(valid_samples) == 0:
            ref_frames_idx = [idx] * (self.clip_frames-1)
        else:
            ref_frames_idx = random.sample(valid_samples, self.clip_frames-1)
        return ref_frames_idx

    def filter_annotation(self):
        cache_file = os.path.join(self.cache_dir, self.image_set + "_keep.json")

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as fid:
                keep = json.load(fid)
            print("{}'s keep information loaded from {}".format(self.det_vid, cache_file))
            return keep

        keep = np.zeros((len(self)), dtype=np.int32)
        for idx in range(len(self)):
            if idx % 10000 == 0:
                print("Had filtered {} images".format(idx))

            filename = self.image_set_index[idx]

            tree = ET.parse(self._anno_path % filename).getroot()
            objs = tree.findall("object")
            keep[idx] = 0 if len(objs) == 0 else 1
        print("Had filtered {} images".format(len(self)))

        with open(cache_file, 'w', encoding='utf-8') as fid:
            json.dump(keep.tolist(), fid)
        print("Saving {}'s keep information into {}".format(self.det_vid, cache_file))

        return keep

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        obj_ids = []
        occluded = []

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        objs = target.findall("object")
        for obj in objs:
            if not obj.find("name").text in self.classes_to_ind:
                continue

            bbox = obj.find("bndbox")
            box = [
                np.maximum(float(bbox.find("xmin").text), 0),
                np.maximum(float(bbox.find("ymin").text), 0),
                np.minimum(float(bbox.find("xmax").text), im_info[1] - 1),
                np.minimum(float(bbox.find("ymax").text), im_info[0] - 1)
            ]
            boxes.append(box)
            gt_classes.append(self.classes_to_ind[obj.find("name").text.lower().strip()])
            obj_ids.append(int(obj.find("trackid").text))
            occluded.append(int(obj.find("occluded").text))

        res = {
            "boxes": boxes,
            "labels": gt_classes,
            'obj_ids': obj_ids,
            'occluded': occluded,
            "im_info": im_info,
        }
        return res

    def load_annos(self, cache_file):
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as fid:
                annos = json.load(fid)
            print("{}'s annotation information loaded from {}".format(self.det_vid, cache_file))
        else:
            annos = []
            for idx in range(len(self)):
                if idx % 10000 == 0:
                    print("Had processed {} images".format(idx))

                filename = self.image_set_index[idx]

                tree = ET.parse(self._anno_path % filename).getroot()
                anno = self._preprocess_annotation(tree)
                annos.append(anno)
            print("Had processed {} images".format(len(self)))

            with open(cache_file, 'w', encoding='utf-8') as fid:
                json.dump(annos, fid)
            print("Saving {}'s annotation information into {}".format(self.det_vid, cache_file))

        return annos

    def get_img_info(self, idx):
        im_info = self.annos[idx]["im_info"]
        return {"height": im_info[0], "width": im_info[1]}

    @property
    def cache_dir(self):
        """
        make a directory to store all caches
        :return: cache path
        """
        cache_dir = os.path.join(self.img_prefix, 'cache')
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        return cache_dir

    def get_visualization(self, idx):
        filename = self.image_set_index[idx]

        img = cv2.imread(self._img_dir % filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = self.get_groundtruth(idx)
        target = target.clip_to_image(remove_empty=True)

        return img, target, filename

    def get_groundtruth(self, idx):
        anno = self.annos[idx]
        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])

        return target

    @staticmethod
    def map_class_id_to_class_name(class_id):
        return VIDDataset.classes[class_id]


def detection_collate_vid(batch):
    transposed_batch = list(zip(*batch))
    images = [torch.from_numpy(img).permute(2, 0, 1) for img in transposed_batch[0]]
    targets = transposed_batch[1]
    img_ids = transposed_batch[2]
    return images, targets, img_ids
