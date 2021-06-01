import torch
from layers.utils import jaccard,  mask_iou, generate_track_gaussian, compute_comp_scores, generate_mask, compute_kl_div
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
        self.prev_result_clip = None
        self.prev_max_inst_ids = 0
        self.prev_track = None

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
            candidate_clip = Fold_candidates(candidates, img_metas)

            # add frames_ids for every objects in the clip
            base_info_candidates = Anchor_independented_info(candidates, img_metas)

            # first tracking within a clip and then tracking within clips
            results = self.track_clips(net, candidate_clip, base_info_candidates, img_metas, img=img)

        return results

    def track_clips(self, net, candidate_clip, base_info_candidates, img_metas, img=None):
        # only support batch_size = 1 for video test
        is_first = img_metas[0]['is_first']
        if is_first:
            self.prev_result_clip = None
            self.prev_track = None
            self.prev_max_inst_ids = 0

        if candidate_clip['box'].nelement() == 0:
            return {'box': torch.Tensor(), 'box_ids': torch.Tensor(), 'mask_coeff': torch.Tensor(),
                    'class': torch.Tensor(), 'score': torch.Tensor()}

        # to track masks within a clip first and to predict masks from short-term frames
        result_clip = self.track_within_clip(net, candidate_clip, base_info_candidates)
        T, n = result_clip['box'].size()[:2]

        # compared bboxes in current frame with bboxes in previous frame to achieve tracking
        if is_first or (not is_first and self.prev_result_clip is None):
            # save bbox and features for later matching
            result_clip['box_ids'] = torch.arange(n).view(1, -1, 1).repeat(T, 1, 1)
            results = UnFold_candidate_clip(result_clip)
            self.prev_track = result_clip['track_mu'][-1]
            self.prev_result_clip = result_clip
            self.prev_max_inst_ids = result_clip['box'].size(1)
        else:

            assert self.prev_result_clip is not None
            # find idx of overlapped frames ids in two clips
            # >>> a, b = torch.arange(2,7), torch.arange(4,9)
            # >>> overlapped_idx = (a.unsqueeze(1) == b).nonzero(as_tuple=False)
            # >>> overlapped_idx
            # tensor([[2, 0],
            #         [3, 1],
            #         [4, 2]])
            prev_frame_ids = self.prev_result_clip['frame_id'][:, 0]
            cur_frame_ids = result_clip['frame_id'][:, 0]
            frame_ids_overlapped_idx = (prev_frame_ids.unsqueeze(1) == cur_frame_ids).nonzero(as_tuple=False)
            prev_overlapped_idx = frame_ids_overlapped_idx[:, 0]
            cur_overlapped_idx = frame_ids_overlapped_idx[:, 1]

            # compute clip-level BIoU and MIoU for objects in two clips to assign IDs
            n_prev = self.prev_result_clip['box'].size(1)
            n_cur = result_clip['box'].size(1)
            prev_bbox = self.prev_result_clip['box'][prev_overlapped_idx]          # [T_over, N_prev, 4]
            cur_bbox = result_clip['box'][cur_overlapped_idx]                      # [T_over, N_cur, 4]
            bbox_ious = jaccard(cur_bbox, prev_bbox).mean(dim=0)                   # [N_cur,  N_prev]
            prev_masks_soft = self.prev_result_clip['mask'][prev_overlapped_idx]   # [T_over, N_prev, d]
            cur_masks_soft = result_clip['mask'][cur_overlapped_idx]               # [T_over, N_cur,  d]
            mask_ious = mask_iou(cur_masks_soft.gt(0.5), prev_masks_soft.gt(0.5)).mean(dim=0)

            # calculate KL divergence for Gaussian distribution of isntances
            prev_track_mu = self.prev_result_clip['track_mu'][prev_overlapped_idx]
            prev_track_var = self.prev_result_clip['track_var'][prev_overlapped_idx]
            cur_track_mu = result_clip['track_mu'][cur_overlapped_idx]
            cur_track_var = result_clip['track_var'][cur_overlapped_idx]
            kl_divergence = compute_kl_div(prev_track_mu, prev_track_var,
                                           cur_track_mu, cur_track_var).mean(dim=0)  # value in [0, +infinite]
            sim_dummy = torch.ones(n_cur, 1, device=cur_bbox.device) * 1  # threshold for kl_divergence = 10
            # from [0, +infinite] to [0, 1]: sim = 1/ (exp(0.1*kl_div))
            # sim = torch.div(1., torch.exp(torch.cat([sim_dummy, kl_divergence], dim=-1) * 0.1))
            sim = torch.cat([sim_dummy, 1. / kl_divergence], dim=-1)

            # compute clip-level lables and scores
            prev_classes = self.prev_result_clip['class'][0]
            cur_classes = result_clip['class'][0]
            cur_scores = torch.mean(result_clip['score'][cur_overlapped_idx], dim=0)
            label_delta = (cur_classes.view(-1, 1) == prev_classes.view(1, -1)).float()

            # compute comprehensive score for two clips
            comp_scores = compute_comp_scores(sim,
                                              cur_scores.view(-1, 1),
                                              bbox_ious,
                                              mask_ious,
                                              label_delta,
                                              add_bbox_dummy=True,
                                              bbox_dummy_iou=0.3,
                                              match_coeff=cfg.match_coeff,
                                              use_FEELVOS=cfg.use_FEELVOS)

            match_likelihood, match_ids = torch.max(comp_scores, dim=1)
            # translate match_ids to det_obj_ids, assign new id to new objects
            # update tracking features/bboxes of exisiting object,
            # add tracking features/bboxes of new object
            prev_obj_ids = self.prev_result_clip['box_ids'][0].view(-1)
            det_obj_ids = torch.ones(n_cur, dtype=torch.int32) * (-1)
            best_match_scores = torch.ones(self.prev_max_inst_ids) * (-1)
            best_match_idx = torch.ones(self.prev_max_inst_ids) * (-1)
            for idx, match_id in enumerate(match_ids):
                if match_id == 0:
                    det_obj_ids[idx] = self.prev_max_inst_ids
                    self.prev_max_inst_ids += 1

                else:
                    # multiple candidate might match with previous object, here we choose the one with
                    # largest comprehensive score
                    obj_id = prev_obj_ids[match_id - 1]
                    match_score = cur_scores[idx]  # match_likelihood[idx]
                    if match_score > best_match_scores[obj_id]:
                        if best_match_idx[obj_id] != -1:
                            det_obj_ids[int(best_match_idx[obj_id])] = -1
                        det_obj_ids[idx] = obj_id
                        best_match_scores[obj_id] = match_score
                        best_match_idx[obj_id] = idx

            # divide result_clip into results [result1, ..., result_T]
            result_clip['box_ids'] = det_obj_ids.view(1, -1, 1).repeat(T, 1, 1)
            results = UnFold_candidate_clip(result_clip)
            self.prev_result_clip = result_clip

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
        det_masks_coeff = candidate_clip['mask_coeff']
        proto_data = candidate_clip['proto']
        # get masks
        det_masks_soft = generate_mask(proto_data, det_masks_coeff, det_bbox)
        candidate_clip['mask'] = det_masks_soft

        det_track_embed = F.normalize(candidate_clip['track'], dim=-1)
        det_track_mu, det_track_var = generate_track_gaussian(det_track_embed, det_masks_soft)
        candidate_clip['track_mu'] = det_track_mu
        candidate_clip['track_var'] = det_track_var
        n_dets = det_bbox.size(0)

        # calculate KL divergence for Gaussian distribution of isntances
        kl_divergence = compute_kl_div(det_track_mu, det_track_var)  # value in [[0, +infinite]]
        # from [0, +infinite] to [0, 1]: sim = 1/ (exp(0.1*kl_div))
        # sim = -1 * torch.div(1., torch.exp(0.1 * kl_divergence)).log()

        # Calculate BIoU and MIoU between detected masks and tracked masks for assign IDs
        bbox_ious = jaccard(det_bbox, det_bbox)  # [n_dets, n_prev]
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
        score_threshold = 1.5 + cfg.match_coeff[0] * 0. \
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

        # if an instance is missed in a frame of the clip because of occlusion, blur or other reasons,
        # we have to track it from other frames of the clip. Otherwise, if the instance does disappear in the frame,
        # we hope the masks predicted from other frames should be blank.
        # for the output, [T, N, 4] for boxes, [T, N, d] for mask_coeff, [T, N, h, w, d] for proto_data
        # [T] for frame_id and so on.
        result_clip = {}
        for k in candidate_clip.keys():
            result_clip[k] = []

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

                candidates_single_obj = []
                for idx, base_info_candidate in enumerate(base_info_candidates):
                    if missed_frames[idx]:
                        # to predict masks from selected frame to the missed frame
                        candidate_shift_cur = CandidateShift(net, base_info_candidate, selected_candidate)
                        for k, v in selected_candidate.items():
                            if k not in candidate_shift_cur.keys():
                                candidate_shift_cur[k] = v
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
                        candidates_single_obj.append(candidate_cur)

                # merge those candidates including single object to a clip-level candidate
                candidate_clip_single_obj = Fold_candidates(candidates_single_obj)  # [T, 4]
                for k, v in candidate_clip_single_obj.items():
                    result_clip[k].append(v)                             # [T, 4]

            else:

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

        return result_clip