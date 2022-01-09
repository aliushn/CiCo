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
                 clip_frames=1,
                 has_gt=False,
                 transform=None,
                 size_divisor=None,
                 with_crowd=True):

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

        self.vid_ids = list(set([img_set_index.split('/')[-2] for img_set_index in self.image_set_index]))
        self.vid_ids.sort()
        self.vid_infos = [[] for _ in range(len(self.vid_ids))]
        for img_set_index in self.image_set_index:
            idx = self.vid_ids.index(img_set_index.split('/')[-2])
            self.vid_infos[idx].append(img_set_index)

        # self.anns2json_video_bbox(os.path.join(self.cache_dir, self.image_set + "_anno_eval.json"))

    def __getitem__(self, idx):
        return self._get_train(idx)

    def _get_train(self, idx):
        if self.clip_frames > 1:
            ref_frame_idx = self.sample_ref(idx)
            clip_frame_idx = ref_frame_idx + [idx]
            clip_frame_idx.sort()
        else:
            clip_frame_idx = [idx]

        imgs = []
        for idx in clip_frame_idx:
            filename = self.image_set_index[idx]
            imgs.append(np.array(Image.open(self._img_dir % filename).convert("RGB")))
        ori_shape = imgs[0].shape

        if self.has_gt:
            boxes, labels = [], []
            if 'VID' in self.ann_file:
                obj_ids, boxes_occluded = [], []
            else:
                obj_ids, boxes_occluded = None, None
            for idx in clip_frame_idx:
                anno = self.annos[idx]
                boxes.append(np.stack(anno['boxes'], axis=0))
                labels.append(anno['labels'])
                if 'VID' in self.ann_file:
                    obj_ids.append(anno['obj_ids'])
                    boxes_occluded.append(anno['occluded'])
        else:
            boxes, labels, obj_ids, boxes_occluded = None, None, None, None

        if self.transform is not None:
            imgs, boxes, labels = self.transform(imgs, boxes, labels)

        img_meta = []
        for i, idx in enumerate(clip_frame_idx):
            vid, frame_id = self.image_set_index[idx].split('/')[-2:]
            img_h, img_w = imgs[i].shape[:2]
            img_meta.append(dict(
                ori_shape=ori_shape,
                img_shape=(img_h, img_w, 3),
                video_id=vid,
                frame_id=frame_id))
            if 'VID' in self.ann_file:
                img_meta[i]['is_first'] = (int(frame_id) == 0)

        # Stack annotations into [n_objs, n_frames, ...]
        imgs = np.concatenate(imgs, axis=-1)
        if self.has_gt:
            if 'VID' in self.ann_file:
                clip_obj_ids = list(set([id for ids in obj_ids for id in ids]))
                clip_boxes, clip_labels, clip_boxes_occluded = [], [], []
                for id in clip_obj_ids:
                    boxes_obj, labels_obj, boxes_occluded_obj = [], [], []
                    for ids, box, label, box_occluded in zip(obj_ids, boxes, labels, boxes_occluded):
                        if id in ids:
                            boxes_obj.append(box[ids.index(id)])
                            labels_obj += [label[ids.index(id)]]
                            boxes_occluded_obj += [box_occluded[ids.index(id)]]
                        else:
                            # -1 means object do not exist in the frame
                            boxes_obj.append(np.array([0.]*4))
                            boxes_occluded_obj += [1]
                    clip_boxes.append(np.stack(boxes_obj, axis=0))
                    clip_labels.append(np.argmax(np.bincount(np.array(labels_obj))))
                    clip_boxes_occluded.append(np.stack(boxes_occluded_obj, axis=0))
                clip_boxes = np.stack(clip_boxes, axis=0)
                clip_boxes_occluded = np.stack(clip_boxes_occluded, axis=0)
                clip_labels = np.array(clip_labels)
                clip_obj_ids = np.array(clip_obj_ids)
                return imgs, img_meta, (clip_boxes, clip_labels, clip_obj_ids, clip_boxes_occluded)
            else:
                return imgs, img_meta, (boxes[0], labels[0], obj_ids, boxes_occluded)
        else:
            return imgs, img_meta, (boxes, labels, obj_ids, boxes_occluded)

    def __len__(self):
        return len(self.image_set_index)

    def pull_clip_from_video(self, vid, clip_frame_idx):
        # prepare a sequence from a video
        vid_idx = self.vid_ids.index(vid)
        vid_info = self.vid_infos[vid_idx]
        imgs = []
        for idx in clip_frame_idx:
            filename = vid_info[idx]
            imgs.append(np.array(Image.open(self._img_dir % filename).convert("RGB")))

        height, width, depth = imgs[0].shape
        ori_shape = (height, width, depth)

        # load annotation of ref_frames
        if self.has_gt:
            boxes, labels = [], []
            if 'VID' in self.ann_file:
                obj_ids, boxes_occluded = [], []
            else:
                obj_ids, boxes_occluded = None, None
            for idx in clip_frame_idx:
                anno = self.annos[self.image_set_index.index(vid_info[idx])]
                boxes.append(np.stack(anno['boxes'], axis=0) if len(anno['boxes']) > 0 else anno['boxes'])
                labels.append(anno['labels'])
                if 'VID' in self.ann_file:
                    obj_ids.append(anno['obj_ids'])
                    boxes_occluded.append(anno['occluded'])

        else:
            boxes, labels, obj_ids, boxes_occluded = None, None, None, None

        # apply transforms
        if self.transform is not None:
            imgs, boxes, labels = self.transform(imgs, boxes, labels)

        img_meta = []
        for i, img in enumerate(imgs):
            img_h, img_w = img.shape[:2]
            img_meta.append(dict(
                ori_shape=ori_shape,
                img_shape=(img_h, img_w, 3),
                video_id=vid,
                frame_id=clip_frame_idx[i],
                is_first=(clip_frame_idx[i] == 0)))

        return imgs, img_meta, (boxes, labels, obj_ids, boxes_occluded)

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
            if self.clip_frames-1 > len(valid_samples):
                valid_samples += (valid_samples*self.clip_frames)[:self.clip_frames-1-len(valid_samples)]
            ref_frames_idx = random.sample(valid_samples, self.clip_frames-1)
        return ref_frames_idx

    def _interval_samples(self, lines):
        lines_interval10 = []
        for idx, line in enumerate(lines):
            if int(line[2]) % 10 == 0:
                lines_interval10.append(line)

        new_img_index_file = self.img_index[:-4] + '_every10frames.txt'
        with open(new_img_index_file, 'w') as fid:
            for line in lines_interval10:
                for x in line:
                    fid.write(x)
                    fid.write(' ')
                fid.write('\n')

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
            if 'VID' in self.ann_file:
                obj_ids.append(int(obj.find("trackid").text))
                occluded.append(int(obj.find("occluded").text))

        res = {
            "boxes": boxes,
            "labels": gt_classes,
            "im_info": im_info,
        }
        if 'VID' in self.ann_file:
            res['obj_ids'] = obj_ids
            res['occluded'] = occluded
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

    def anns2json_video_bbox(self, json_file):
        if not os.path.exists(json_file):
            print('Prepare video annotation as json format for metric evaluation')
            json_results = []
            for idx, vid_info in enumerate(self.vid_infos):
                if idx % 100 == 0:
                    print("Had processed {} videos".format(idx))
                # assume results is ordered
                vid_id = vid_info[0].split('/')[-2]
                vid_anns = []
                obj_ids_vid = []
                for frame_info in vid_info:
                    jdx = self.image_set_index.index(frame_info)
                    obj_ids_vid += self.annos[jdx]['obj_ids']
                    vid_anns.append(self.annos[jdx])

                obj_ids_vid = list(set(obj_ids_vid))
                vid_objs = dict()
                for obj_id in obj_ids_vid:
                    vid_objs[obj_id] = {'video_id': vid_id, 'bbox': [], 'category_id': [], 'occluded': []}
                    for ann in vid_anns:
                        if obj_id in ann['obj_ids']:
                            kdx = ann['obj_ids'].index(obj_id)
                            vid_objs[obj_id]['bbox'].append(ann['boxes'][kdx])
                            vid_objs[obj_id]['category_id'].append(ann['labels'][kdx])
                            vid_objs[obj_id]['occluded'].append(ann['occluded'][kdx])
                        else:
                            vid_objs[obj_id]['bbox'].append(None)

                for obj_id, obj in vid_objs.items():
                    # majority voting of those frames with top k highest scores for sequence catgory
                    obj['category_id'] = np.bincount(obj['category_id']).argmax().item()
                    json_results.append(obj)

            results = {'annotations': json_results, 'categories': self.classes[1:], 'videos': self.vid_ids}

            with open(json_file, 'w', encoding='utf-8') as fid:
                json.dump(results, fid)
            print('Done')


def detection_collate(batch):
    imgs = []
    boxes = []
    labels = []
    ids = []
    occluded_boxes = []
    img_metas = []

    for sample in batch:
        imgs += [torch.from_numpy(sample[0]).permute(2, 0, 1)]
        img_metas += [sample[1]]
        boxes += [sample[2][0]]
        labels += [sample[2][1]]
        if sample[2][2] is not None:
            ids += [sample[2][2]]
        if sample[2][3] is not None:
            occluded_boxes += [sample[2][3]]

    from .utils import ImageList_from_tensors, GtList_from_tensor
    imgs_batch = ImageList_from_tensors(imgs, size_divisibility=32)
    pad_h, pad_w = imgs_batch.size()[-2:]

    if len(boxes) > 0:
        _, boxes, img_metas = GtList_from_tensor(pad_h, pad_w, None, boxes, img_metas)
    else:
        _, _, img_metas = GtList_from_tensor(pad_h, pad_w, None, None, img_metas)

    return imgs_batch, img_metas, (boxes, labels, ids, occluded_boxes)


def prepare_data(data_batch, devices):
    images, image_metas, (bboxes, labels, obj_ids, occluded_boxes) = data_batch
    h, w = images.size()[-2:]
    d = len(devices)
    if images.size(0) % d != 0:
        # TODO: if read multi frames (n_f) as a clip, thus the condition should be images.size(0) / n_f % len(devices)
        idx = [i % images.size(0) for i in range(d)]
        remainder = d - images.size(0) % d
        images = torch.cat([images, images[idx[:remainder]]])
        bboxes += [bboxes[i] for i in idx[:remainder]]
        labels += [labels[i] for i in idx[:remainder]]
        if obj_ids is not None and len(obj_ids) > 0:
            obj_ids += [obj_ids[i] for i in idx[:remainder]]
        if occluded_boxes is not None and len(occluded_boxes) > 0:
            occluded_boxes += [occluded_boxes[i] for i in idx[:remainder]]
        image_metas += [image_metas[i] for i in idx[:remainder]]
    n = images.size(0) // len(devices)

    with torch.no_grad():
        images_list, masks_list, bboxes_list, labels_list, obj_ids_list, num_crowds_list = [], [], [], [], [], []
        image_metas_list = []
        for idx, device in enumerate(devices):
            images_list.append(gradinator(images[idx * n:(idx + 1) * n].reshape(-1, 3, h, w).to(device)))
            masks_list.append([None] * n)
            bboxes_list.append([gradinator(torch.from_numpy(bboxes[jdx]).float().to(device)) for jdx in range(idx * n, (idx + 1) * n)])
            labels_list.append([gradinator(torch.from_numpy(labels[jdx]).to(device)) for jdx in range(idx * n, (idx + 1) * n)])
            if obj_ids is not None and len(obj_ids) > 0:
                obj_ids_list.append([gradinator(torch.from_numpy(obj_ids[jdx]).to(device)) for jdx in range(idx * n, (idx + 1) * n)])
            else:
                obj_ids_list.append([None] * n)
            num_crowds_list.append([0] * n)
            image_metas_list.append(image_metas[idx * n:(idx + 1) * n])

        return images_list, bboxes_list, labels_list, masks_list, obj_ids_list, num_crowds_list, image_metas_list


def gradinator(x):
    x.requires_grad = False
    return x