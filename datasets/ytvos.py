import numpy as np
import os.path as osp
import random
import mmcv
import torch.utils.data as data
from .augmentations_vis import Numpy2Tensor
from cocoapi.PythonAPI.pycocotools.ytvos import YTVOS
import torch
import torch.nn.functional as F
from .utils import ImageList_from_tensors


class YTVOSDataset(data.Dataset):

    def __init__(self,
                 ann_file,
                 img_prefix,
                 preserve_aspect_ratio=True,
                 transform=None,
                 size_divisor=None,
                 with_mask=True,
                 with_crowd=False,
                 with_label=True,
                 with_track=False,
                 clip_frames=1,
                 has_gt=False):
        # prefix of images path
        self.img_prefix = img_prefix
        self.preserve_aspect_ratio = preserve_aspect_ratio

        # load annotations (and proposals)
        self.vid_infos = self.load_annotations(ann_file)
        img_ids = []
        for idx, vid_info in zip(self.vid_ids, self.vid_infos):
            for frame_id in range(len(vid_info['filenames'])):
                img_ids.append((idx, frame_id))
        self.img_ids = img_ids

        # has gt or not
        self.has_gt = has_gt
        # filter images with no annotation during training
        if self.has_gt:
            valid_inds = [i for i, (v, f) in enumerate(self.img_ids)
                          if len(self.get_ann_info(v, f)['bboxes'])]
            self.img_ids = [self.img_ids[i] for i in valid_inds]

        # if using mutil-scale training, random an integer from [min_size, max_size]
        self.clip_frames = clip_frames

        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor

        # with mask or not (reserved field, takes no effect)
        self.with_mask = with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = with_crowd
        # with label is False for RPN
        self.with_label = with_label
        self.with_track = with_track

        # set group flag for the sampler
        if self.has_gt:
            self._set_group_flag()

        self.transform = transform
        self.numpy2tensor = Numpy2Tensor()

    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        return self.pull_item(self.img_ids[idx])

    def load_annotations(self, ann_file):
        self.ytvos = YTVOS(ann_file)
        self.cat_ids = self.ytvos.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.vid_ids = self.ytvos.getVidIds()
        vid_infos = []
        for i in self.vid_ids:
            info = self.ytvos.loadVids([i])[0]
            info['filenames'] = info['file_names']
            vid_infos.append(info)
        return vid_infos

    def get_ann_info(self, vid_id, frame_id):
        ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
        ann_info = self.ytvos.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, frame_id)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            vid_id, _ = self.img_ids[i]
            vid_idx = self.vid_ids.index(vid_id)
            vid_info = self.vid_infos[vid_idx]
            if vid_info['width'] / vid_info['height'] > 1:
                self.flag[i] = 1

    def bbox_aug(self, bbox, img_size):
        center_off = self.aug_ref_bbox_param[0]
        size_perturb = self.aug_ref_bbox_param[1]

        n_bb = bbox.shape[0]
        # bbox center offset
        center_offs = (2*np.random.rand(n_bb, 2) - 1) * center_off
        # bbox resize ratios
        resize_ratios = (2*np.random.rand(n_bb, 2) - 1) * size_perturb + 1
        # bbox: x1, y1, x2, y2
        centers = (bbox[:,:2]+ bbox[:,2:])/2.
        sizes = bbox[:,2:] - bbox[:,:2]
        new_centers = centers + center_offs * sizes
        new_sizes = sizes * resize_ratios
        new_x1y1 = new_centers - new_sizes/2.
        new_x2y2 = new_centers + new_sizes/2.
        c_min = [0,0]
        c_max = [img_size[1], img_size[0]]
        new_x1y1 = np.clip(new_x1y1, c_min, c_max)
        new_x2y2 = np.clip(new_x2y2, c_min, c_max)
        bbox = np.hstack((new_x1y1, new_x2y2)).astype(np.float32)
        return bbox

    def sample_ref(self, idx):
        # sample another frame in the same sequence as reference
        vid, frame_id = idx
        valid_samples = []
        for i in range(-self.clip_frames+1, self.clip_frames):
            # check if the frame id is valid
            ref_idx = (vid, i+frame_id)
            if i != 0 and ref_idx in self.img_ids:
                valid_samples.append(i+frame_id)
        if len(valid_samples) == 0:
            ref_frames = [frame_id]
        else:
            ref_frames = random.sample(valid_samples, self.clip_frames-1)
        return ref_frames

    def pull_item(self, idx):
        # prepare a pair of image in a sequence
        vid,  frame_id = idx
        vid_idx = self.vid_ids.index(vid)
        vid_info = self.vid_infos[vid_idx]
        if self.has_gt and self.clip_frames > 1:
            ref_frame_ids = self.sample_ref(idx)
            clip_frame_ids = ref_frame_ids + [frame_id]
            clip_frame_ids.sort()
        else:
            clip_frame_ids = [frame_id]
        imgs = [mmcv.imread(osp.join(self.img_prefix, vid_info['filenames'][id])) for id in clip_frame_ids]
        height, width, depth = imgs[0].shape
        ori_shape = (height, width, depth)

        # load annotation of ref_frames
        if self.has_gt:
            masks, bboxes, labels, obj_ids, bboxes_ignore = [], [], [], [], []
            for id in clip_frame_ids:
                ann = self.get_ann_info(vid, id)
                bboxes.append(ann['bboxes'])
                labels.append(ann['labels'])
                # obj ids attribute does not exist in current annotation
                # need to add it
                obj_ids.append(np.array(ann['obj_ids']))
                if self.with_mask:
                    masks.append(np.stack(ann['masks'], axis=0))
                # compute matching of reference frame with current frame
                # 0 denote there is no matching
                # gt_pids = [ref_ids.index(i)+1 if i in ref_ids else 0 for i in gt_ids]
                if self.with_crowd:
                    bboxes_ignore.append(ann['bboxes_ignore'])

        else:
            masks, bboxes, labels, obj_ids = None, None, None, None

        # apply transforms
        imgs, masks, bboxes, labels = self.transform(imgs, masks, bboxes, labels)

        img_meta = []
        for i, img in enumerate(imgs):
            pad_h, pad_w = img.shape[:2]

            if self.preserve_aspect_ratio:
                # resize long edges
                if (height/float(width)) < (pad_h/float(pad_w)):
                    img_w, img_h = pad_w, int(height * (pad_w / width))
                else:
                    img_w, img_h = int(width * (pad_h / height)), pad_h
            else:
                img_w, img_h = pad_w, pad_h

            img_meta.append(dict(
                ori_shape=ori_shape,
                img_shape=(img_h, img_w, 3),
                pad_shape=(pad_h, pad_w, 3),
                video_id=vid,
                frame_id=clip_frame_ids[i],
                is_first=(clip_frame_ids[i] == 0)))

        # TODO: how to deal with crowded bounding boxes
        # if self.with_crowd:
        #     for i in range(len(clip_frame_ids)):
        #         bboxes_ignore[i] = self.transform(bboxes_ignore[i], img_shape, pad_shape, scale_factor, flip)

        return imgs, img_meta, (masks, bboxes, labels, obj_ids)

    def pull_clip_from_video(self, vid, clip_frame_ids):
        # prepare a sequence from a video
        vid_idx = self.vid_ids.index(vid)
        vid_info = self.vid_infos[vid_idx]
        imgs = [mmcv.imread(osp.join(self.img_prefix, vid_info['filenames'][id])) for id in clip_frame_ids]
        height, width, depth = imgs[0].shape
        ori_shape = (height, width, depth)

        # load annotation of ref_frames
        if self.has_gt:
            masks, bboxes, labels, obj_ids, bboxes_ignore = [], [], [], [], []
            for id in clip_frame_ids:
                ann = self.get_ann_info(vid, id)
                bboxes.append(ann['bboxes'])
                labels.append(ann['labels'])
                # obj ids attribute does not exist in current annotation
                # need to add it
                obj_ids.append(np.array(ann['obj_ids']))
                if self.with_mask:
                    masks.append(np.stack(ann['masks'], axis=0))
                # compute matching of reference frame with current frame
                # 0 denote there is no matching
                # gt_pids = [ref_ids.index(i)+1 if i in ref_ids else 0 for i in gt_ids]
                if self.with_crowd:
                    bboxes_ignore.append(ann['bboxes_ignore'])

        else:
            masks, bboxes, labels, obj_ids = None, None, None, None

        # apply transforms
        imgs, masks, bboxes, labels = self.transform(imgs, masks, bboxes, labels)

        img_meta = []
        for i, img in enumerate(imgs):
            pad_h, pad_w = img.shape[:2]

            if self.preserve_aspect_ratio:
                # resize long edges
                if (height/float(width)) < (pad_h/float(pad_w)):
                    img_w, img_h = pad_w, int(height * (pad_w / width))
                else:
                    img_w, img_h = int(width * (pad_h / height)), pad_h
            else:
                img_w, img_h = pad_w, pad_h

            img_meta.append(dict(
                ori_shape=ori_shape,
                img_shape=(img_h, img_w, 3),
                pad_shape=(pad_h, pad_w, 3),
                video_id=vid,
                frame_id=clip_frame_ids[i],
                is_first=(clip_frame_ids[i] == 0)))

        return imgs, img_meta, (masks, bboxes, labels, obj_ids)

    def _parse_ann_info(self, ann_info, frame_id, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_ids = []
        gt_bboxes_ignore = []

        with_occlusion = False
        if len(ann_info) > 0:
            if 'occlusion' in ann_info[0].keys():
                with_occlusion = True
                occlusion = []

        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []
        for i, ann in enumerate(ann_info):
            # each ann is a list of masks
            # ann:
            # bbox: list of bboxes
            # segmentation: list of segmentation
            # category_id
            # area: list of area
            bbox = ann['bboxes'][frame_id]
            area = ann['areas'][frame_id]
            segm = ann['segmentations'][frame_id]
            if bbox is None: continue
            x1, y1, w, h = bbox
            if area <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_ids.append(ann['id'])
                gt_labels.append(self.cat2label[ann['category_id']])
            if with_mask:
                gt_masks.append(self.ytvos.annToMask(ann, frame_id))
                mask_polys = [
                    p for p in segm if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)
            if with_occlusion:
                occlusion.append(ann['occlusion'][frame_id])
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, obj_ids=gt_ids, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        if with_occlusion:
            ann['occlusion'] = occlusion
        return ann


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
    imgs = []
    masks = []
    bboxes = []
    labels = []
    ids = []
    img_metas = []

    for sample in batch:
        imgs += [torch.from_numpy(img).permute(2, 0, 1) for img in sample[0]]
        img_metas += sample[1]
        if sample[2][0] is not None:
            masks += [torch.from_numpy(mask) for mask in sample[2][0]]
            bboxes += [torch.from_numpy(box) for box in sample[2][1]]
            labels += [torch.from_numpy(label) for label in sample[2][2]]
            ids += [torch.from_numpy(id) for id in sample[2][3]]

    imgs_batch = torch.stack(imgs, dim=0)

    if len(masks) > 0:
        return imgs_batch, img_metas, (bboxes, labels, masks, ids)
    else:
        return imgs_batch, img_metas


def prepare_data_vis(data_batch, devices):
    images, image_metas, (bboxes, labels, masks, obj_ids) = data_batch
    d = len(devices) * 2
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
            images_list.append(gradinator(images[idx * n:(idx + 1) * n].to(device)))
            masks_list.append([gradinator(masks[jdx].to(device)) for jdx in range(idx * n, (idx + 1) * n)])
            bboxes_list.append([gradinator(bboxes[jdx].to(device)) for jdx in range(idx * n, (idx + 1) * n)])
            labels_list.append([gradinator(labels[jdx].to(device)) for jdx in range(idx * n, (idx + 1) * n)])
            obj_ids_list.append([gradinator(obj_ids[jdx].to(device)) for jdx in range(idx * n, (idx + 1) * n)])
            num_crowds_list.append([0] * n)
            image_metas_list.append(image_metas[idx * n:(idx + 1) * n])

        return images_list, bboxes_list, labels_list, masks_list, obj_ids_list, num_crowds_list, image_metas_list


def gradinator(x):
    x.requires_grad = False
    return x
