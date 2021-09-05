import torch
from layers.utils import jaccard, mask_iou, compute_comp_scores, compute_kl_div
from .TF_clip_utils import Fold_candidates_by_order, UnFold_candidate_clip, select_distinct_track
from .track_TF_within_clip import Track_TF_within_clip, Backward_Track_TF_within_clip
from utils import timer
import numpy as np

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)


class Track_TF_Clip(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """
    # TODO: Refactor this whole class away. It needs to go.

    def __init__(self, net, match_coeff, correlation_patch_size, train_maskshift=False, conf_thresh=0.1,
                 track_by_Gaussian=False, train_masks=True):
        self.last_clip_result = None
        self.prev_track = None
        self.net = net
        self.match_coeff = match_coeff
        self.track_by_Gaussian = track_by_Gaussian
        # used for two clips, so the threshold should be small
        self.bbox_dummy_iou = 0.1

    def __call__(self, candidates, img_metas, imgs=None):
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
                print('num_clip is 0', img_metas[0]['video_id'])
                results = candidates

            else:
                # first tracking within a clip and then tracking within clips
                results = self.track_clips(candidates, is_first, img_metas=img_metas, imgs=imgs)

        return results

    def track_clips(self, candidates, is_first, img_metas=None, imgs=None):
        n_frames = len(candidates)

        # Forward_tracking: assign instances IDs for each frame
        candidates = Track_TF_within_clip(self.net, candidates, img_metas, imgs)

        # Backward shift for adding missed objects
        candidates = Backward_Track_TF_within_clip(self.net, candidates, img_metas, imgs)
        candidates_clip = Fold_candidates_by_order(candidates, self.track_by_Gaussian, img_metas)
        remove_blank = True
        if remove_blank:
            threshold = min(1, n_frames//4)
            # whether tracked masks are greater than a small threshold, which removes some false positives
            cond1 = (candidates_clip['mask'].gt(0.5).sum(dim=(2, 3)) > 2).sum(dim=0) > threshold
            # a declining weights (0.8) to remove some false positives that cased by consecutively track to segment
            cond2 = (candidates_clip['score'] > 0.05).sum(dim=0) > threshold
            keep_clip = cond1 & cond2
            for k, v in candidates_clip.items():
                candidates_clip[k] = v[:, keep_clip]

        n_frames, n_dets = candidates_clip['box'].size()[:2]
        if n_dets == 0:
            return [None]*n_frames

        distinct_track = select_distinct_track(candidates_clip)

        # compared bboxes in current frame with bboxes in previous frame to achieve tracking
        if is_first or (not is_first and self.last_clip_result is None):
            # the parameters of Gaussian in last frame as the keyframe
            candidates_clip['box_ids'] = torch.arange(n_dets).view(1, -1, 1).repeat(n_frames, 1, 1)
            self.prev_track = dict()
            for k, v in distinct_track.items():
                self.prev_track[k] = v.clone()
            self.last_clip_result = dict()

        else:

            assert self.last_clip_result is not None
            n_prev = self.prev_track['box'].size(0)
            cur_scores = torch.mean(candidates_clip['score'], dim=0).view(-1, 1)

            # translate match_ids to det_obj_ids, assign new id to new objects
            # update tracking features/bboxes of exisiting object,
            # add tracking features/bboxes of new object
            det_obj_ids = torch.ones(n_dets, dtype=torch.int64) * (-1)
            best_match_scores = torch.ones(n_prev) * (-1)
            best_match_idx = torch.ones(n_prev, dtype=torch.int64) * (-1)

            # find idx of overlapped frames ids between current clip and last clip
            # >>> a, b = torch.arange(2,7), torch.arange(4,9)
            # >>> overlapped_idx = (a.unsqueeze(1) == b).nonzero(as_tuple=False)
            # >>> overlapped_idx
            # tensor([[2, 0],
            #         [3, 1],
            #         [4, 2]])
            last_clip_frame_ids = self.last_clip_result['frame_id'][:, 0]
            cur_clip_frame_ids = candidates_clip['frame_id'][:, 0]
            frame_ids_overlapped_idx = (last_clip_frame_ids.unsqueeze(1) == cur_clip_frame_ids).nonzero(as_tuple=False)
            last_overlapped_idx = frame_ids_overlapped_idx[:, 0]
            cur_overlapped_idx = frame_ids_overlapped_idx[:, 1]

            if last_overlapped_idx.nelement() > 0:
                # whether the tracker only use the detected objects to assign ID on overlapped frames, like KL, MkIoU
                keep_detected = (self.last_clip_result['tracked_mask'][last_overlapped_idx].unsqueeze(1) == 0) * \
                                (candidates_clip['tracked_mask'][cur_overlapped_idx].unsqueeze(-1) == 0)
                num_detected = torch.clamp((keep_detected.sum(dim=0)), min=1)

                if self.track_by_Gaussian:
                    # calculate KL divergence for Gaussian distribution of isntances
                    last_clip_track_mu = self.last_clip_result['track_mu'][last_overlapped_idx]
                    last_clip_track_var = self.last_clip_result['track_var'][last_overlapped_idx]
                    cur_track_mu_over = candidates_clip['track_mu'][cur_overlapped_idx]
                    cur_track_var_over = candidates_clip['track_var'][cur_overlapped_idx]
                    kl_divergence_over = compute_kl_div(last_clip_track_mu, last_clip_track_var,
                                                        cur_track_mu_over, cur_track_var_over)
                    kl_divergence_over[~keep_detected] = 0
                    kl_divergence_over = kl_divergence_over.sum(dim=0) / num_detected
                    kl_divergence_over[keep_detected.sum(dim=0) == 0] = 100

                    # from [0, +infinite] to [0, 1]: sim = 1/ (exp(0.1*kl_div)), threshold=10
                    sim_over = torch.div(1., torch.exp(0.1 * kl_divergence_over))
                    # from [0, +infinite] to [0, 1]: sim = exp(1-kl_div)), threshold  = 5
                    # sim = torch.exp(1 - kl_divergence_over)
                else:
                    last_clip_track_over = self.last_clip_result['track'][last_overlapped_idx]
                    cur_track_over = candidates_clip['track'][cur_overlapped_idx]
                    cos_sim = (cur_track_over.unsqueeze(-2) * last_clip_track_over.unsqueeze(-3)).sum(-1)  # [n_dets, n_prev]
                    sim_over = (cos_sim.mean(dim=0) + 1) / 2  # [0, 1]

                # compute clip-level BIoU and MIoU for objects in two clips to assign IDs
                last_bbox = self.last_clip_result['box'][last_overlapped_idx]  # [T_over, N_prev, 4]
                cur_bbox = candidates_clip['box'][cur_overlapped_idx]  # [T_over, N_cur, 4]
                bbox_ious_over = jaccard(cur_bbox, last_bbox)
                bbox_ious_over[~keep_detected] = 0
                bbox_ious_over = bbox_ious_over.sum(dim=0) / num_detected  # [N_cur,  N_prev]
                last_masks_soft = self.last_clip_result['mask'][last_overlapped_idx]  # [T_over, N_prev, d]
                cur_masks_soft = candidates_clip['mask'][cur_overlapped_idx]  # [T_over, N_cur,  d]
                mask_ious_over = mask_iou(cur_masks_soft.gt(0.5), last_masks_soft.gt(0.5))
                mask_ious_over[~keep_detected] = 0
                mask_ious_over = mask_ious_over.sum(dim=0) / num_detected

                # match the same instances on two overlappled clips,
                # otherwise, it will be viewed as new objects to compute the similarity on all previous instances
                comp_scores_over = sim_over + bbox_ious_over + mask_ious_over
                match_likelihood_last, match_ids_last = torch.max(comp_scores_over, dim=1)
                new_obj_last = match_likelihood_last < 2.8
                match_ids_last[new_obj_last] = -1
                last_box_ids = self.last_clip_result['box_ids'][0].view(-1)
                match_ids_prev = last_box_ids[match_ids_last[match_ids_last > -1]]
                match_idx_prev = torch.arange(n_dets)[match_ids_last > -1]
                det_obj_ids[match_ids_last > -1] = match_ids_prev
                best_match_scores[match_ids_prev] = cur_scores[~new_obj_last].view(-1)
                best_match_idx[match_ids_prev] = match_idx_prev

                for obj_id, obj_idx in zip(match_ids_prev, match_idx_prev):
                    for k, v in self.prev_track.items():
                        if k not in {'box_ids'}:
                            self.prev_track[k][obj_id] = distinct_track[k][obj_idx]

            else:
                new_obj_last = torch.ones(n_dets) > 0
                match_ids_prev = []

            if new_obj_last.sum() > 0:

                if self.track_by_Gaussian:
                    # calculate KL divergence of Gaussian distribution between instances of the current clip
                    # and all instances of previous instances
                    kl_divergence = compute_kl_div(self.prev_track['track_mu'], self.prev_track['track_var'],
                                                   distinct_track['track_mu'], distinct_track['track_var'])
                    sim_dummy = torch.ones(n_dets, 1) * 20.
                    sim = torch.div(1., torch.exp(0.1 * torch.cat([sim_dummy, kl_divergence], dim=-1)))
                else:
                    cos_sim = (distinct_track['track'].unsqueeze(-2) * self.prev_track['track'].unsqueeze(-3)).sum(-1)
                    cos_sim = torch.cat([torch.zeros(n_dets, 1), cos_sim], dim=1)
                    sim = (cos_sim + 1) / 2  # [0, 1]

                bbox_ious = jaccard(distinct_track['box'], self.prev_track['box'])
                mask_ious = mask_iou(distinct_track['mask'].gt(0.5), self.prev_track['mask'].gt(0.5))
                # Compute clip-level lables and scores
                label_delta = (distinct_track['class'].view(-1, 1) == self.prev_track['class'].view(1, -1)).float()

                # compute comprehensive score
                comp_scores = compute_comp_scores(sim[new_obj_last],
                                                  cur_scores[new_obj_last],
                                                  bbox_ious[new_obj_last],
                                                  mask_ious[new_obj_last],
                                                  label_delta[new_obj_last],
                                                  add_bbox_dummy=True,
                                                  bbox_dummy_iou=self.bbox_dummy_iou,
                                                  match_coeff=self.match_coeff)

                new_obj_match_likelihood, new_obj_match_ids = torch.max(comp_scores, dim=1)
                new_obj_last_idx = torch.arange(n_dets)[new_obj_last]

                for idx, match_id in zip(new_obj_last_idx, new_obj_match_ids):
                    if match_id - 1 in match_ids_prev:
                        match_id = 0

                    if match_id == 0:
                        det_obj_ids[idx] = self.prev_track['box'].size(0)
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

            candidates_clip['box_ids'] = det_obj_ids.view(1, -1, 1).repeat(n_frames, 1, 1)

        candidates_output = UnFold_candidate_clip(candidates_clip, remove_blank=False)

        for k, v in candidates_clip.items():
            self.last_clip_result[k] = v.clone()

        return candidates_output
