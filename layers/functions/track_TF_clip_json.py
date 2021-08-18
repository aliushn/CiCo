import torch
from layers.utils import jaccard, mask_iou, compute_comp_scores, compute_kl_div
from .TF_clip_utils import Fold_candidates_by_order, UnFold_candidate_clip, select_distinct_track
from .track_TF_within_clip import Track_TF_within_clip, Backward_Track_TF_within_clip
from utils import timer
import pycocotools.mask as mask_util

from datasets import cfg, activation_func
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import os

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)


class Track_TF_Clip_json(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """

    def __init__(self, remove_duplicated=True, iou_threshold=0.6, overlap=9):
        self.remove_duplicated = remove_duplicated
        self.iou_threshold = iou_threshold
        self.prev_result = None
        self.max_id = 0
        # used for two clips, so the threshold should be small
        self.bbox_dummy_iou = 0.1
        self.overlap = overlap

    def __call__(self, candidate):
        with timer.env('Track'):
            clips = candidate['clips']
            results = []
            for clip_idx, clip in enumerate(clips):

                if len(clip['clip_res']) == 0:
                    print('Objects in the  clip is 0. The video Id is: ', candidate['video_id'],
                          'The clip Id is: ', clip['clip_idx'])
                    results.append([])

                else:
                    # first tracking within a clip and then tracking within clips
                    results.append(self.track_clips(clip))

        return results

    def track_clips(self, clip):
        clip_idx = clip['clip_idx']
        is_first = clip_idx == 0
        clip_res = clip['clip_res']
        for inst_idx in range(len(clip_res)):
            for seg_rle in clip_res[inst_idx]['segmentations']:
                if seg_rle is not None:
                    ori_h, ori_w = seg_rle['size']
                    continue

        h, w = int(ori_h//4), int(ori_w//4)

        results_clip = {'segm_binary': []}
        for k, v in clip_res[0].items():
            if k not in {'segmentations'}:
                results_clip[k] = []
        # decode all masks from RLE to binary mask
        for inst_idx in range(len(clip_res)):
            inst_segs_rle = clip_res[inst_idx]['segmentations']
            inst_segs = []
            for inst_seg_rle in inst_segs_rle:
                if inst_seg_rle is None:
                    inst_segs.append(torch.zeros(h, w).type(torch.uint8).cuda())
                else:
                    binary_mask = torch.from_numpy(mask_util.decode(inst_seg_rle)).cuda()
                    binary_mask = F.interpolate(binary_mask.view(1, 1, ori_h, ori_w).float(), (h, w),
                                                mode='bilinear', align_corners=False).view(h, w).gt(0.5)
                    inst_segs.append(binary_mask.type(torch.uint8))

            results_clip['segm_binary'].append(torch.stack(inst_segs, dim=0))       # [18, h, w]
            n_frames = results_clip['segm_binary'][0].size(0)
            for k, v in clip_res[inst_idx].items():
                if k not in {'segmentations'}:
                    results_clip[k].append(torch.tensor(v).view(1, -1).repeat(n_frames, 1).cuda())

        # [inst1, inst2, ..., inst_k] => [18, k, -1]  or [1, k]
        for k, v in results_clip.items():
            results_clip[k] = torch.stack(v, dim=1)

        # First NMS to remove duplicated masks from a single clip
        if self.remove_duplicated:
            _, sort_idx = results_clip['score'][0].view(-1).sort(descending=True)

            # Compute clip NMS with mask iou
            # Out of memory in my computer, so we compute them frame-by-frame
            cur_seg = results_clip['segm_binary']
            # miou = []
            # for idx in range(cur_seg.size(0)):
            #     miou.append(mask_iou(cur_seg[idx], cur_seg[idx]))
            # miou = torch.stack(miou, dim=0).mean(0)
            if clip_idx == 0:
                miou = mask_iou(cur_seg, cur_seg).mean(dim=0)
            else:
                miou = mask_iou(cur_seg[self.overlap-n_frames:], cur_seg[self.overlap-n_frames:]).mean(dim=0)
            miou.triu_(diagonal=1)
            iou_max, _ = torch.max(miou, dim=0)
            idx_out = sort_idx[iou_max <= self.iou_threshold]
            idx_out, _ = idx_out.sort()
            keep_score = results_clip['score'][0].view(-1)[idx_out] > 0.2
            idx_out = idx_out[keep_score]
            # print(clip_idx, idx_out)

            for k, v in results_clip.items():
                results_clip[k] = v[:, idx_out]

            temp_clip_res = clip_res
            clip_res = []
            for idx in idx_out:
                clip_res.append(temp_clip_res[idx])

        n_frames, n_dets = results_clip['segm_binary'].size()[:2]
        cur_clip_frame_ids = torch.arange(clip_idx * (n_frames - self.overlap),
                                          clip_idx * (n_frames - self.overlap) + n_frames)

        # compared bboxes in current frame with bboxes in previous frame to achieve tracking
        if is_first or (not is_first and self.prev_result is None):
            # the parameters of Gaussian in last frame as the keyframe
            results_clip['obj_ids'] = torch.arange(n_dets).view(-1)
            self.max_id = n_dets
            self.prev_result = dict()

        else:

            assert self.prev_result is not None

            mask_ious_over = []
            prev_box_ids = torch.tensor(list(self.prev_result.keys()))
            for obj_id, obj_prev in self.prev_result.items():
                prev_frame_ids = torch.tensor(list(obj_prev.keys()))
                # find idx of overlapped frames ids between current clip and last clip
                # >>> a, b = torch.arange(2,7), torch.arange(4,9)
                # >>> overlapped_idx = (a.unsqueeze(1) == b).nonzero(as_tuple=False)
                # >>> overlapped_idx
                # tensor([[2, 0],
                #         [3, 1],
                #         [4, 2]])

                frame_ids_overlapped_idx = (prev_frame_ids.unsqueeze(1) == cur_clip_frame_ids).nonzero(as_tuple=False)
                prev_overlapped_idx = frame_ids_overlapped_idx[:, 0].tolist()
                prev_overlapped_id = prev_frame_ids[prev_overlapped_idx].tolist()
                cur_overlapped_idx = frame_ids_overlapped_idx[:, 1].tolist()
                cur_masks = results_clip['segm_binary']

                if len(prev_overlapped_idx) > 0:

                    # compute clip-level BIoU and MIoU for objects in two clips to assign IDs
                    # last_bbox = self.last_clip_result['box'][last_overlapped_idx]  # [T_over, N_prev, 4]
                    # cur_bbox = results_clip['box'][cur_overlapped_idx]  # [T_over, N_cur, 4]
                    # bbox_ious_over = jaccard(cur_bbox, last_bbox)
                    # bbox_ious_over = bbox_ious_over.mean(dim=0)   # [N_cur,  N_prev]
                    mask_ious_over_single = []
                    prev_masks = self.prev_result[obj_id]
                    for prev_id, cur_idx in zip(prev_overlapped_id, cur_overlapped_idx):
                        prev_masks_frame = prev_masks[prev_id]['segm_binary']
                        cur_masks_frame = cur_masks[cur_idx]
                        mask_ious_over_single.append(mask_iou(cur_masks_frame, prev_masks_frame).mean(dim=-1))
                    mask_ious_over.append(torch.stack(mask_ious_over_single, dim=-1).mean(dim=-1))

                else:
                    mask_ious_over.append(torch.zeros(n_dets))

            mask_ious_over = torch.stack(mask_ious_over, dim=-1)  # [n_dets, n_prev]

            # match the same instances on two overlappled clips,
            # otherwise, it will be viewed as new objects to compute the similarity on all previous instances
            comp_scores_over = mask_ious_over
            match_likelihood_prev, match_ids_prev = torch.max(comp_scores_over, dim=1)
            new_obj_prev = match_likelihood_prev < 0.5
            match_ids_prev[~new_obj_prev] = prev_box_ids[match_ids_prev[~new_obj_prev]]
            match_ids_prev[new_obj_prev] = torch.arange(self.max_id, self.max_id+new_obj_prev.sum())
            self.max_id += new_obj_prev.sum()

            for obj_id in match_ids_prev.unique():
                keep = match_ids_prev == obj_id
                if keep.sum() > 1:
                    # two objects matched 1 prev instacne, we remain the one with higher conf
                    _, sort_idx = results_clip['score'][0].view(-1)[keep].sort(descending=True)
                    temp = torch.arange(n_dets)[keep]
                    for j in range(1, keep.sum()):
                        match_ids_prev[temp[sort_idx[j]]] = -1

            results_clip['obj_ids'] = match_ids_prev.view(-1)

        for obj_idx, obj_id in enumerate(results_clip['obj_ids'].tolist()):
            if obj_id == -1:
                continue

            # remove those frames that non overlapped with current clips
            if clip_idx > 0 and obj_id in self.prev_result:
                frame_ids = list(self.prev_result[obj_id].keys())
                for frame_id in frame_ids:
                    if frame_id < min(cur_clip_frame_ids):
                       self.prev_result[obj_id].pop(frame_id)

            if obj_id not in self.prev_result:
                self.prev_result[obj_id] = {}
            for frame_idx, frame_id in enumerate(cur_clip_frame_ids.tolist()):
                if frame_id not in self.prev_result[obj_id]:
                    self.prev_result[obj_id][frame_id] = dict()
                    for k, v in results_clip.items():
                        if k in {'score', 'category_id', 'segm_binary'}:
                            self.prev_result[obj_id][frame_id][k] = v[frame_idx, obj_idx].unsqueeze(0)
                else:
                    for k, v in results_clip.items():
                        if k in {'score', 'category_id', 'segm_binary'}:
                            self.prev_result[obj_id][frame_id][k] = torch.cat([self.prev_result[obj_id][frame_id][k],
                                                                               v[frame_idx, obj_idx].unsqueeze(0)], dim=0)

        for obj_idx, obj_id in enumerate(results_clip['obj_ids'].tolist()):
            clip_res[obj_idx]['obj_ids'] = obj_id

        if clip_idx != 0:
            # only remiain the newly frames in the clip to output the final .json file
            for inst_idx in range(len(clip_res)):
                clip_res[inst_idx]['segmentations'] = clip_res[inst_idx]['segmentations'][self.overlap - n_frames:]

        print(results_clip['obj_ids'].tolist())

        return clip_res

