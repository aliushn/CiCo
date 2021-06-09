import torch
from layers.utils import jaccard,  mask_iou, generate_track_gaussian, compute_comp_scores, generate_mask, compute_kl_div, compute_DIoU
from .TF_utils import CandidateShift
from .TF_clip_utils import Fold_candidates, UnFold_candidate_clip, Anchor_independented_info
from utils import timer

from datasets import cfg, activation_func
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import os

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)


class Track_TF_Clip(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """
    # TODO: Refactor this whole class away. It needs to go.

    def __init__(self):
        self.last_clip_result = None
        self.prev_track = None
        self.bbox_dummy_iou = 0.3

    def __call__(self, net, candidates, img_metas, img=None):
        """
        Args:
             loc_data: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]

        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.

            Note that the outputs are sorted only if cross_class_nms is False
        """

        with timer.env('Track'):
            # only support batch_size = 1 for video test
            is_first = img_metas[0]['is_first']
            if is_first:
                self.last_clip_result = None
                self.prev_track = None

            # total number of objects in the clip
            n_objs_clip = [candidate['box'].nelement() for candidate in candidates]

            if sum(n_objs_clip) == 0:
                results = candidates

            else:
                candidate_clip = Fold_candidates(candidates, img_metas)

                # add frames_ids for every objects in the clip
                base_info_candidates = Anchor_independented_info(candidates, img_metas)

                # first tracking within a clip and then tracking within clips
                results = self.track_clips(net, candidate_clip, base_info_candidates, is_first, img=img)

        return results

    def track_clips(self, net, candidate_clip, base_info_candidates, is_first, img=None):

        # to track masks within a clip first and to predict masks from short-term frames
        result_clip, distinct_track = self.track_within_clip(net, candidate_clip, base_info_candidates)
        T, n = result_clip['box'].size()[:2]

        # compared bboxes in current frame with bboxes in previous frame to achieve tracking
        if is_first or (not is_first and self.last_clip_result is None):
            # save bbox and features for later matching
            result_clip['box_ids'] = torch.arange(n).view(1, -1, 1).repeat(T, 1, 1)
            results = UnFold_candidate_clip(result_clip)
            # the parameters of Gaussian in last frame as the keyframe
            self.prev_track = distinct_track
            self.last_clip_result = result_clip
        else:

            assert self.last_clip_result is not None
            n_prev = self.prev_track['mu'].size(0)
            n_cur = distinct_track['mu'].size(0)

            # calculate KL divergence of Gaussian distribution between instances of the current clip
            # and all instances of previous instances
            kl_divergence = compute_kl_div(self.prev_track['mu'], self.prev_track['var'],
                                           distinct_track['mu'], distinct_track['var'])
            sim_dummy = torch.ones(n_cur, 1) * 1  # threshold for kl_divergence = 10
            # from [0, +infinite] to [0, 1]: sim = 1/ (exp(0.1*kl_div))
            sim = torch.div(1., torch.exp(0.1 * torch.cat([sim_dummy, kl_divergence], dim=-1)))

            bbox_ious = torch.ones(n_cur, n_prev) * self.bbox_dummy_iou
            mask_ious = torch.ones(n_cur, n_prev) * self.bbox_dummy_iou
            cur_scores = torch.mean(result_clip['score'], dim=0).view(-1, 1)
            # compute clip-level lables and scores
            label_delta = (distinct_track['class'].view(-1, 1) == self.prev_track['class'].view(1, -1)).float()

            # find idx of overlapped frames ids between current clip and last clip
            # >>> a, b = torch.arange(2,7), torch.arange(4,9)
            # >>> overlapped_idx = (a.unsqueeze(1) == b).nonzero(as_tuple=False)
            # >>> overlapped_idx
            # tensor([[2, 0],
            #         [3, 1],
            #         [4, 2]])
            last_clip_frame_ids = self.last_clip_result['frame_id'][:, 0]
            cur_clip_frame_ids = result_clip['frame_id'][:, 0]
            frame_ids_overlapped_idx = (last_clip_frame_ids.unsqueeze(1) == cur_clip_frame_ids).nonzero(as_tuple=False)
            last_overlapped_idx = frame_ids_overlapped_idx[:, 0]
            cur_overlapped_idx = frame_ids_overlapped_idx[:, 1]

            if last_overlapped_idx.nelement() > 0:

                # calculate KL divergence for Gaussian distribution of isntances
                last_clip_track_mu = self.last_clip_result['track_mu'][last_overlapped_idx]
                last_clip_track_var = self.last_clip_result['track_var'][last_overlapped_idx]
                cur_track_mu_over = result_clip['track_mu'][cur_overlapped_idx]
                cur_track_var_over = result_clip['track_var'][cur_overlapped_idx]
                kl_divergence_over = compute_kl_div(last_clip_track_mu, last_clip_track_var,
                                                    cur_track_mu_over, cur_track_var_over).mean(dim=0)
                # from [0, +infinite] to [0, 1]: sim = 1/ (exp(0.1*kl_div))
                sim_over = torch.div(1., torch.exp(0.1 * kl_divergence_over))

                # compute clip-level BIoU and MIoU for objects in two clips to assign IDs
                last_bbox = self.last_clip_result['box'][last_overlapped_idx]  # [T_over, N_prev, 4]
                cur_bbox = result_clip['box'][cur_overlapped_idx]  # [T_over, N_cur, 4]
                bbox_ious_over = jaccard(cur_bbox, last_bbox).mean(dim=0)  # [N_cur,  N_prev]
                last_masks_soft = self.last_clip_result['mask'][last_overlapped_idx]  # [T_over, N_prev, d]
                cur_masks_soft = result_clip['mask'][cur_overlapped_idx]  # [T_over, N_cur,  d]
                mask_ious_over = mask_iou(cur_masks_soft.gt(0.5), last_masks_soft.gt(0.5)).mean(dim=0)

                # update sim with cur_sim to computer precise tracking similarity
                last_box_ids = self.last_clip_result['box_ids'][0].view(-1)
                sim[:, last_box_ids + 1] = sim_over
                bbox_ious[:, last_box_ids] = bbox_ious_over
                mask_ious[:, last_box_ids] = mask_ious_over

            # compute comprehensive score for two clips
            comp_scores = compute_comp_scores(sim,
                                              cur_scores,
                                              bbox_ious,
                                              mask_ious,
                                              label_delta,
                                              add_bbox_dummy=True,
                                              bbox_dummy_iou=self.bbox_dummy_iou,
                                              match_coeff=cfg.match_coeff,
                                              use_FEELVOS=cfg.use_FEELVOS)
            match_likelihood, match_ids = torch.max(comp_scores, dim=1)

            # translate match_ids to det_obj_ids, assign new id to new objects
            # update tracking features/bboxes of exisiting object,
            # add tracking features/bboxes of new object
            det_obj_ids = torch.ones(n_cur, dtype=torch.int64) * (-1)
            best_match_scores = torch.ones(n_prev) * (-1)
            best_match_idx = torch.ones(n_prev) * (-1)
            for idx, match_id in enumerate(match_ids):
                if match_id == 0:
                    det_obj_ids[idx] = self.prev_track['mu'].size(0)
                    for k, v in self.prev_track.items():
                        self.prev_track[k] = torch.cat([v, distinct_track[k][idx][None]], dim=0)

                else:
                    # multiple candidate might match with previous object, here we choose the one with
                    # largest comprehensive score
                    obj_id = match_id - 1
                    match_score = cur_scores[idx]  # match_likelihood[idx]
                    if match_score > best_match_scores[obj_id]:
                        if best_match_idx[obj_id] != -1:
                            det_obj_ids[int(best_match_idx[obj_id])] = -1
                        det_obj_ids[idx] = obj_id
                        best_match_scores[obj_id] = match_score
                        best_match_idx[obj_id] = idx
                        for k, v in self.prev_track.items():
                            self.prev_track[k][obj_id] = distinct_track[k][idx]

            result_clip['box_ids'] = det_obj_ids.view(1, -1, 1).repeat(T, 1, 1)
            # det_obj_ids = torch.arange(n_cur)
            # result_clip['box_ids'] = det_obj_ids.view(1, -1, 1).repeat(T, 1, 1)
            # remove replicated masks
            keep_idx = (torch.arange(n_cur)[det_obj_ids >= 0]).tolist()
            for k, v in result_clip.items():
                result_clip[k] = v[:, keep_idx]

            # divide result_clip into results [result1, ..., result_T]
            results = UnFold_candidate_clip(result_clip)
            self.last_clip_result = result_clip

        return results

    def track_within_clip(self, net, candidate_clip, base_info_candidates):

        assert candidate_clip['box'].nelement() > 0

        # sort all objects in the clip by scores
        _, idx = candidate_clip['score'].view(-1).sort(0, descending=True)
        for k, v in candidate_clip.items():
            candidate_clip[k] = v[idx]

        # get bbox and class after NMS
        det_bbox = candidate_clip['box']
        det_scores, det_labels = candidate_clip['score'], candidate_clip['class']
        # get masks
        det_masks_soft = candidate_clip['mask']

        det_track_embed = F.normalize(candidate_clip['track'], dim=-1)
        det_track_mu, det_track_var = generate_track_gaussian(det_track_embed, det_masks_soft, det_bbox)
        candidate_clip['track_mu'] = det_track_mu
        candidate_clip['track_var'] = det_track_var
        n_dets = det_bbox.size(0)

        # calculate KL divergence for Gaussian distribution of isntances
        kl_divergence = compute_kl_div(det_track_mu, det_track_var)  # value in [[0, +infinite]]
        # from [0, +infinite] to [0, 1]: sim = 1/ (exp(0.1*kl_div))
        # sim = -1 * torch.div(1., torch.exp(0.1 * kl_divergence)).log()

        # Calculate BIoU and MIoU between detected masks and tracked masks for assign IDs
        bbox_ious = compute_DIoU(det_bbox, det_bbox)  # [n_dets, n_prev]
        mask_ious = mask_iou(det_masks_soft.gt(0.5).float(), det_masks_soft.gt(0.5).float())  # [n_dets, n_prev]

        # compute comprehensive score
        label_delta = (det_labels.view(1, -1) == det_labels.view(-1, 1)).float()
        comp_scores = compute_comp_scores(1. / kl_divergence,
                                          det_scores.view(-1, 1),
                                          bbox_ious,
                                          mask_ious,
                                          label_delta,
                                          add_bbox_dummy=False,
                                          bbox_dummy_iou=0.3,
                                          match_coeff=cfg.match_coeff,
                                          use_FEELVOS=cfg.use_FEELVOS)

        # sim + scores + 0.7*MIoU + 0.3*BIoU + labels
        score_threshold = 1 + cfg.match_coeff[0] * 0. \
                          + cfg.match_coeff[1] * 0.3 + cfg.match_coeff[2] * 0.3 + cfg.match_coeff[3] * 0.5

        # Zero out the lower triangle of the cosine similarity matrix and diagonal
        comp_scores = torch.triu(comp_scores, diagonal=1)

        # Now that everything in the diagonal and below is zeroed out, if we take the max
        # of the IoU matrix along the columns, each column will represent the maximum IoU
        # between this element and every element with a higher score than this element.
        comp_scores_max, comp_scores_max_idx = torch.max(comp_scores, dim=0)

        # Now just filter out the ones greater than the threshold, i.e., only keep boxes that
        # don't have a higher scoring box that would supress it in normal NMS.
        keep_unique = comp_scores_max <= score_threshold
        obj_ids_unique = torch.arange(keep_unique.sum())
        obj_ids = torch.ones(n_dets, device=comp_scores.device).type(torch.int64) * -1
        obj_ids[keep_unique] = obj_ids_unique
        for i in range(1, n_dets):
            if obj_ids[i] == -1:
                obj_ids[i] = obj_ids[comp_scores_max_idx[i]]

        # if an instance is missed in a frame of the clip because of occlusion, blur or other cases,
        # we hope to track it from other frames of the clip. Otherwise, if the instance does disappear in the frame,
        # we hope the masks predicted from other frames should be blank.
        # for the output, [T, N, 4] for boxes, [T, N, d] for mask_coeff, [T, N, h, w, d] for proto_data
        # [T] for frame_id and so on.
        result_clip = {}
        for k in candidate_clip.keys():
            result_clip[k] = []

        # select a distinct frame for each object to represent the object in the clip
        distinct_track = {'mu': [], 'var': [], 'class': []}

        all_frame_ids = torch.tensor([base_info_candidate['frame_id'] for base_info_candidate in base_info_candidates])
        for id in obj_ids_unique:
            # to do: the classes of an object in the clip should be same
            keep = obj_ids == id
            frame_ids_cur = candidate_clip['frame_id'][keep].view(-1)
            frame_ids_cur_unique = frame_ids_cur.unique()
            if len(frame_ids_cur_unique) < len(all_frame_ids):
                # using temporal information to embed motion between frames to predict masks
                # from an unmissed frame to the missed frame
                missed_frames = (frame_ids_cur_unique.unsqueeze(1) == all_frame_ids).sum(0) == 0

                # selected the frame which has max confident for the instance
                selected_idx = torch.argmax(candidate_clip['score'][keep])
                selected_candidate = {}
                for k, v in candidate_clip.items():
                    selected_candidate[k] = v[keep][selected_idx].unsqueeze(0)

                # update distinct_track for each object
                distinct_track['mu'].append(selected_candidate['track_mu'])
                distinct_track['var'].append(selected_candidate['track_var'])
                distinct_track['class'].append(selected_candidate['class'])

                candidates_single_obj = []
                for idx, base_info_candidate in enumerate(base_info_candidates):
                    if missed_frames[idx]:
                        # to predict masks from selected frame to the missed frame
                        # TODO: add img_meta and img for display_box_shift.py
                        candidate_shift_cur = CandidateShift(net, base_info_candidate, selected_candidate)
                        for k, v in selected_candidate.items():
                            if k not in candidate_shift_cur.keys():
                                candidate_shift_cur[k] = v
                        # we assume an object should have same category with the selected frame in the clip
                        candidate_shift_cur['class'] = selected_candidate['class']
                        candidates_single_obj.append(candidate_shift_cur)

                    else:
                        # to find the corresponding info of the current frame from candidate clip
                        candidate_cur = {}
                        obj_idx = (frame_ids_cur == all_frame_ids[idx]).view(-1)
                        for k, v in candidate_clip.items():
                            # for each frame, only the object with max score will be added
                            if obj_idx.sum() > 1:
                                candidate_cur[k] = v[keep][obj_idx][0].unsqueeze(0)  # [1, 4]
                            else:
                                candidate_cur[k] = v[keep][obj_idx]
                        candidate_cur['class'] = selected_candidate['class']
                        candidates_single_obj.append(candidate_cur)

                # merge those candidates including single object to a clip-level candidate
                candidate_clip_single_obj = Fold_candidates(candidates_single_obj)  # [T, 4]
                for k, v in candidate_clip_single_obj.items():
                    result_clip[k].append(v)                             # [T, 4]

            else:

                # update distinct_track for each object
                selected_idx = torch.argmax(candidate_clip['score'][keep])
                distinct_track['mu'].append(candidate_clip['track_mu'][keep][selected_idx].unsqueeze(0))
                distinct_track['var'].append(candidate_clip['track_var'][keep][selected_idx].unsqueeze(0))
                distinct_track['class'].append(candidate_clip['class'][keep][selected_idx].unsqueeze(0))

                sorted_frame_ids_cur, sorted_idx = frame_ids_cur.sort()
                if len(frame_ids_cur) > len(frame_ids_cur_unique):
                    unique_ids, unique_idx = [], []
                    for id, idx in zip(sorted_frame_ids_cur, sorted_idx):
                        if id not in unique_ids:
                            unique_ids.append(id)
                            unique_idx.append(idx)
                    sorted_idx = torch.stack(unique_idx, dim=0)

                for k, v in candidate_clip.items():
                    result_clip[k].append(v[keep][sorted_idx])                       # [T, 4]

        # integer multi objects candidates to a final clip candidate: N * [T, 4] -> [T, N, 4]
        for k, v in result_clip.items():
            result_clip[k] = torch.stack(v, dim=1)                       # [T, N, 4]

        for k, v in distinct_track.items():
            distinct_track[k] = torch.cat(v)

        return result_clip, distinct_track