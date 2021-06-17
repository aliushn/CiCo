import torch
from layers.utils import jaccard, mask_iou, compute_comp_scores, compute_kl_div
from .TF_clip_utils import Fold_candidates_by_order, UnFold_candidate_clip, select_distinct_track
from .track_TF_within_clip import Track_TF_within_clip, Backward_Track_TF_within_clip
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
                # first tracking within a clip and then tracking within clips
                results = self.track_clips(net, candidates, is_first, img_metas=img_metas, img=img)

        return results

    def track_clips(self, net, candidates, is_first, img_metas=None, img=None):

        # forward_tracking: assign instances IDs for each frame
        results_for = Track_TF_within_clip(net, candidates, img_metas, img)

        # backward shift for adding missed objects
        results = Backward_Track_TF_within_clip(net, results_for[::-1], img_metas[::-1], img[::-1])
        results_clip = Fold_candidates_by_order(results, img_metas)
        distinct_track = select_distinct_track(results_clip)
        n_frames, n_dets = results_clip['box'].size()[:2]

        # compared bboxes in current frame with bboxes in previous frame to achieve tracking
        if is_first or (not is_first and self.last_clip_result is None):
            # the parameters of Gaussian in last frame as the keyframe
            self.prev_track = dict()
            for k, v in distinct_track.items():
                self.prev_track[k] = v.clone()
            self.last_clip_result = dict()

        else:

            assert self.last_clip_result is not None
            n_prev = self.prev_track['track_mu'].size(0)
            n_cur = distinct_track['track_mu'].size(0)

            # calculate KL divergence of Gaussian distribution between instances of the current clip
            # and all instances of previous instances
            kl_divergence = compute_kl_div(self.prev_track['track_mu'], self.prev_track['track_var'],
                                           distinct_track['track_mu'], distinct_track['track_var'])
            sim_dummy = torch.ones(n_cur, 1) * 1  # threshold for kl_divergence = 10
            # from [0, +infinite] to [0, 1]: sim = 1/ (exp(0.1*kl_div))
            sim = torch.div(1., torch.exp(0.1 * torch.cat([sim_dummy, kl_divergence], dim=-1)))

            bbox_ious = torch.ones(n_cur, n_prev) * self.bbox_dummy_iou
            mask_ious = torch.ones(n_cur, n_prev) * self.bbox_dummy_iou
            cur_scores = torch.mean(results_clip['score'], dim=0).view(-1, 1)
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
            cur_clip_frame_ids = results_clip['frame_id'][:, 0]
            frame_ids_overlapped_idx = (last_clip_frame_ids.unsqueeze(1) == cur_clip_frame_ids).nonzero(as_tuple=False)
            last_overlapped_idx = frame_ids_overlapped_idx[:, 0]
            cur_overlapped_idx = frame_ids_overlapped_idx[:, 1]

            if last_overlapped_idx.nelement() > 0:

                # calculate KL divergence for Gaussian distribution of isntances
                last_clip_track_mu = self.last_clip_result['track_mu'][last_overlapped_idx]
                last_clip_track_var = self.last_clip_result['track_var'][last_overlapped_idx]
                cur_track_mu_over = results_clip['track_mu'][cur_overlapped_idx]
                cur_track_var_over = results_clip['track_var'][cur_overlapped_idx]
                kl_divergence_over = compute_kl_div(last_clip_track_mu, last_clip_track_var,
                                                    cur_track_mu_over, cur_track_var_over).mean(dim=0)
                # from [0, +infinite] to [0, 1]: sim = 1/ (exp(0.1*kl_div))
                sim_over = torch.div(1., torch.exp(0.1 * kl_divergence_over))

                # compute clip-level BIoU and MIoU for objects in two clips to assign IDs
                last_bbox = self.last_clip_result['box'][last_overlapped_idx]  # [T_over, N_prev, 4]
                cur_bbox = results_clip['box'][cur_overlapped_idx]  # [T_over, N_cur, 4]
                bbox_ious_over = jaccard(cur_bbox, last_bbox).mean(dim=0)  # [N_cur,  N_prev]
                last_masks_soft = self.last_clip_result['mask'][last_overlapped_idx]  # [T_over, N_prev, d]
                cur_masks_soft = results_clip['mask'][cur_overlapped_idx]  # [T_over, N_cur,  d]
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
                    det_obj_ids[idx] = self.prev_track['track_mu'].size(0)
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

            results_clip['box_ids'] = det_obj_ids.view(1, -1, 1).repeat(n_frames, 1, 1)
            # det_obj_ids = torch.arange(n_cur)
            # result_clip['box_ids'] = det_obj_ids.view(1, -1, 1).repeat(T, 1, 1)
            # remove replicated masks
            # keep_idx = (torch.arange(n_cur)[det_obj_ids >= 0]).tolist()
            # for k, v in result_clip.items():
            #     result_clip[k] = v[:, keep_idx]

            # divide result_clip into results [result1, ..., result_T]
            results = UnFold_candidate_clip(results_clip, remove_blank=False)

        for k, v in results_clip.items():
            self.last_clip_result[k] = v.clone()

        return results
