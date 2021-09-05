import torch
from layers.utils import compute_DIoU,  mask_iou, generate_track_gaussian, compute_comp_scores, compute_kl_div, center_size
from .TF_utils import CandidateShift
from utils import timer
import numpy as np

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)


class Track_TF(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """
    # TODO: Refactor this whole class away. It needs to go.

    def __init__(self, net, match_coeff, correlation_patch_size, train_maskshift=False, conf_thresh=0.1,
                 track_by_Gaussian=False, train_masks=True):
        self.prev_candidate = None
        self.match_coeff = match_coeff
        self.correlation_patch_size = correlation_patch_size
        self.train_maskshift = train_maskshift
        self.conf_thresh = conf_thresh
        self.track_by_Gaussian = track_by_Gaussian
        self.train_masks = train_masks
        self.img_level_keys = ['proto', 'fpn_feat', 'fpn_feat_temp', 'sem_seg']
        if self.track_by_Gaussian:
            self.img_level_keys += 'track'

        self.CandidateShift = CandidateShift(net, self.correlation_patch_size, train_maskshift=self.train_maskshift,
                                             train_masks=self.train_masks, track_by_Gaussian=self.track_by_Gaussian)

    def __call__(self, candidates, imgs_meta, img=None):
        """
        Args:
             loc_data: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]

        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.

            Note that the outputs are sorted only if cross_class_nms is False
        """

        with timer.env('Track_TF'):
            # only support batch_size = 1 for video test
            for batch_idx in range(len(candidates)):
                self.track(candidates[batch_idx], imgs_meta[batch_idx], img=img[batch_idx])

        return candidates

    def track(self, candidate, img_meta, img=None):
        # only support batch_size = 1 for video test
        is_first = img_meta['is_first']
        if is_first:
            self.prev_candidate = None

        if candidate['box'].nelement() == 0:
            candidate['box_ids'] = torch.Tensor()
            return candidate

        assert candidate['box'].nelement() > 0
        # get bbox and class after NMS
        det_bbox = candidate['box']
        det_score, det_labels = candidate['score'], candidate['class']
        if self.train_masks:
            det_masks_soft = candidate['mask']
        if self.track_by_Gaussian:
            candidate['track_mu'], candidate['track_var'] = generate_track_gaussian(candidate['track'].squeeze(0),
                                                                                    det_masks_soft, det_bbox)

        n_dets = det_bbox.size(0)
        candidate['tracked_mask'] = torch.zeros(n_dets)

        # compared bboxes in current frame with bboxes in previous frame to achieve tracking
        if is_first or (not is_first and self.prev_candidate is None):
            det_obj_ids = torch.arange(n_dets)
            # save bbox and features for later matching
            self.prev_candidate = dict()
            for k, v in candidate.items():
                self.prev_candidate[k] = v.clone()

        else:

            assert self.prev_candidate is not None

            # tracked mask: to track masks from previous frames to current frame
            self.CandidateShift(candidate, self.prev_candidate, img=img, img_meta=img_meta)

            # calculate KL divergence for Gaussian distribution of isntances
            n_prev = self.prev_candidate['box'].size(0)
            # Calculate BIoU and MIoU between detected masks and tracked masks for assign IDs
            bbox_ious = compute_DIoU(det_bbox, self.prev_candidate['box'])  # [n_dets, n_prev]
            label_delta = (self.prev_candidate['class'] == det_labels.view(-1, 1)).float()
            if self.train_masks:
                mask_ious = mask_iou(det_masks_soft.gt(0.5).float(),
                                     self.prev_candidate['mask'].gt(0.5))  # [n_dets, n_prev]
            else:
                mask_ious = torch.zeros_like(bbox_ious)

            if self.train_masks and self.track_by_Gaussian:
                kl_divergence = compute_kl_div(self.prev_candidate['track_mu'], self.prev_candidate['track_var'],
                                               candidate['track_mu'], candidate['track_var'])    # value in [[0, +infinite]]
                sim_dummy = torch.ones(n_dets, 1, device=det_bbox.device) * 10.
                # from [0, +infinite] to [0, 1]: sim = 1/ (exp(0.1*kl_div)), threshold  = 10
                sim = torch.div(1., torch.exp(0.1 * torch.cat([sim_dummy, kl_divergence], dim=-1)))
                # from [0, +infinite] to [0, 1]: sim = exp(1-kl_div)), threshold  = 5
                # sim = torch.exp(1 - torch.cat([sim_dummy, kl_divergence], dim=-1))
            else:
                cos_sim = candidate['track'] @ self.prev_candidate['track'].t()  # [n_dets, n_prev], val in [-1, 1]
                cos_sim = torch.cat([torch.zeros(n_dets, 1), cos_sim], dim=1)
                sim = (cos_sim + 1) / 2  # [0, 1]

            # compute comprehensive score
            comp_scores = compute_comp_scores(sim,
                                              det_score.view(-1, 1),
                                              bbox_ious,
                                              mask_ious,
                                              label_delta,
                                              add_bbox_dummy=True,
                                              bbox_dummy_iou=0.3,
                                              match_coeff=self.match_coeff)
            # only need to do that for isntances whose Mask IoU is lower than 0.9
            match_likelihood, match_ids = torch.max(comp_scores, dim=1)

            # translate match_ids to det_obj_ids, assign new id to new objects
            # update tracking features/bboxes of exisiting object,
            # add tracking features/bboxes of new object
            det_obj_ids = torch.ones(n_dets, dtype=torch.int64) * (-1)
            best_match_scores = torch.ones(n_prev) * (-1)
            best_match_idx = torch.ones(n_prev, dtype=torch.int64) * (-1)
            for match_idx, match_id in enumerate(match_ids):
                if match_id == 0:
                    det_obj_ids[match_idx] = self.prev_candidate['box'].size(0)
                    for k, v in self.prev_candidate.items():
                        if k not in self.img_level_keys + ['box_ids']:
                            self.prev_candidate[k] = torch.cat([v, candidate[k][match_idx][None]], dim=0)

                else:
                    # multiple candidate might match with previous object, here we choose the one with
                    # largest comprehensive score
                    obj_id = match_id - 1
                    match_score = match_likelihood[match_idx]
                    if match_score > best_match_scores[obj_id]:
                        if best_match_idx[obj_id] != -1:
                            det_obj_ids[int(best_match_idx[obj_id])] = -1
                        det_obj_ids[match_idx] = obj_id
                        best_match_scores[obj_id] = match_score
                        best_match_idx[obj_id] = match_idx
                        # udpate feature
                        for k, v in self.prev_candidate.items():
                            if k not in self.img_level_keys + ['box_ids']:
                                self.prev_candidate[k][obj_id] = candidate[k][match_idx]

        self.prev_candidate['box_ids'] = torch.arange(self.prev_candidate['box'].size(0))
        candidate['box_ids'] = det_obj_ids

        # Whether add some objects whose masks are tracked form previous frames by Temporal Fusion Module
        cond1 = (self.prev_candidate['tracked_mask'] > 0) & (self.prev_candidate['tracked_mask'] <= 7)
        # a declining weights (0.8) to remove some false positives that cased by consecutively track to segment
        cond2 = self.prev_candidate['score'].clone().detach() > 0.5*self.conf_thresh
        keep_tracked_objs = cond1 & cond2
        if self.train_masks:
            # whether tracked masks are greater than a small threshold, which removes some false positives
            cond3 = self.prev_candidate['mask'].gt(0.5).sum([1, 2]) > 0
            keep_tracked_objs = keep_tracked_objs & cond3
        # Choose objects detected and segmented by frame-level prediction head
        keep_detected_objs = det_obj_ids >= 0

        if keep_detected_objs.sum() + keep_tracked_objs.sum() == 0:
            candidate['box_ids'] = torch.Tensor()
        else:
            for k, v in candidate.items():
                if k not in self.img_level_keys:
                    if keep_detected_objs.sum() == 0:
                        candidate[k] = self.prev_candidate[k][keep_tracked_objs]
                    elif keep_tracked_objs.sum() == 0:
                        candidate[k] = v[keep_detected_objs]
                    else:
                        candidate[k] = torch.cat([v[keep_detected_objs], self.prev_candidate[k][keep_tracked_objs]])

        return candidate
