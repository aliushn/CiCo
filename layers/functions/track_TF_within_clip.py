import torch
from layers.utils import compute_DIoU,  mask_iou, generate_track_gaussian, compute_comp_scores, generate_mask, compute_kl_div, center_size
from .TF_utils import CandidateShift
from utils import timer

from datasets import cfg
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import os

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)


def Track_TF_within_clip(net, candidates, imgs_meta, imgs=None):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """

    with timer.env('Track_TF'):
        prev_candidate = None
        results = []

        # only support batch_size = 1 for video test
        for batch_idx, candidate in enumerate(candidates):
            img_meta = imgs_meta[batch_idx]
            img = imgs[batch_idx]

            # compared bboxes in current frame with bboxes in previous frame to achieve tracking
            if prev_candidate is None:
                if candidate['box'].nelement() == 0:
                    candidate['box_ids'] = torch.Tensor()
                    results.append(candidate)
                    continue
                else:
                    # save bbox and features for later matching
                    det_track_embed = candidate['track']
                    if cfg.track_by_Gaussian:
                        det_track_mu, det_track_var = generate_track_gaussian(det_track_embed.squeeze(0),
                                                                              candidate['mask'],
                                                                              candidate['box'])
                        candidate['track_mu'] = det_track_mu
                        candidate['track_var'] = det_track_var

                    prev_candidate = dict()
                    for k, v in candidate.items():
                        prev_candidate[k] = v.clone()
                    prev_candidate['tracked_mask'] = torch.zeros(prev_candidate['box'].size(0))

            else:

                # tracked mask: to track masks from previous frames to current frame
                prev_candidate_shift = CandidateShift(net, candidate, prev_candidate, img=img, img_meta=img_meta)
                for k, v in prev_candidate_shift.items():
                    prev_candidate[k] = v.clone()
                prev_candidate['tracked_mask'] = prev_candidate['tracked_mask'] + 1

                if candidate['box'].nelement() > 0:
                    # get bbox and class after NMS
                    det_bbox = candidate['box']
                    det_score, det_labels = candidate['score'], candidate['class']
                    # get masks
                    det_masks_soft = candidate['mask']
                    if cfg.use_semantic_segmentation_loss:
                        sem_seg = candidate['sem_seg'].squeeze(0).permute(2, 0, 1).contiguous()

                    det_track_embed = candidate['track']
                    if cfg.track_by_Gaussian and candidate['box'].nelement() > 0:
                        det_track_mu, det_track_var = generate_track_gaussian(det_track_embed.squeeze(0),
                                                                              det_masks_soft,
                                                                              det_bbox)
                        candidate['track_mu'] = det_track_mu
                        candidate['track_var'] = det_track_var

                    # calculate KL divergence for Gaussian distribution of isntances
                    n_dets = det_bbox.size(0)
                    n_prev = prev_candidate['box'].size(0)

                    if cfg.track_by_Gaussian:
                        kl_divergence = compute_kl_div(prev_candidate['track_mu'], prev_candidate['track_var'],
                                                       det_track_mu, det_track_var)    # value in [[0, +infinite]]
                        sim_dummy = torch.ones(n_dets, 1, device=det_bbox.device) * 10  # threshold for kl_divergence = 10
                        # from [0, +infinite] to [0, 1]: sim = 1/ (exp(0.1*kl_div))
                        sim = torch.div(1., torch.exp(0.1 * torch.cat([sim_dummy, kl_divergence], dim=-1)))
                    else:
                        cos_sim = det_track_embed @ prev_candidate['track'].t()  # [n_dets, n_prev], val in [-1, 1]
                        cos_sim = torch.cat([torch.zeros(n_dets, 1), cos_sim], dim=1)
                        sim = (cos_sim + 1) / 2  # [0, 1]

                    # Calculate BIoU and MIoU between detected masks and tracked masks for assign IDs
                    bbox_ious = compute_DIoU(det_bbox, prev_candidate['box'])  # [n_dets, n_prev]
                    mask_ious = mask_iou(det_masks_soft.gt(0.5).float(), prev_candidate['mask'].gt(0.5))  # [n_dets, n_prev]
                    label_delta = (prev_candidate['class'] == det_labels.view(-1, 1)).float()

                    # compute comprehensive score
                    comp_scores = compute_comp_scores(sim,
                                                      det_score.view(-1, 1),
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
                    det_obj_ids = torch.ones(n_dets, dtype=torch.int32) * (-1)
                    best_match_scores = torch.ones(n_prev) * (-1)
                    best_match_idx = torch.ones(n_prev) * (-1)
                    for idx, match_id in enumerate(match_ids):
                        if match_id == 0:
                            det_obj_ids[idx] = prev_candidate['box'].size(0)
                            for k, v in prev_candidate.items():
                                if k not in {'proto', 'fpn_feat', 'track', 'sem_seg', 'tracked_mask'}:
                                    prev_candidate[k] = torch.cat([v, candidate[k][idx][None]], dim=0)
                            prev_candidate['tracked_mask'] = torch.cat([prev_candidate['tracked_mask'],
                                                                    torch.zeros(1)], dim=0)

                        else:
                            # multiple candidate might match with previous object, here we choose the one with
                            # largest comprehensive score
                            obj_id = match_id - 1
                            match_score = match_likelihood[idx]
                            if match_score > best_match_scores[obj_id]:
                                if best_match_idx[obj_id] != -1:
                                    det_obj_ids[int(best_match_idx[obj_id])] = -1
                                det_obj_ids[idx] = obj_id
                                best_match_scores[obj_id] = match_score
                                best_match_idx[obj_id] = idx
                                # udpate feature
                                for k, v in prev_candidate.items():
                                    if k not in {'proto', 'fpn_feat', 'track', 'sem_seg', 'tracked_mask'}:
                                        prev_candidate[k][obj_id] = candidate[k][idx]
                                prev_candidate['tracked_mask'][obj_id] = 0

            result = dict()
            for k, v in prev_candidate.items():
                result[k] = v.clone()

            result['box_ids'] = torch.arange(prev_candidate['box'].size(0))
            results.append(result)

        return results


def Backward_Track_TF_within_clip(net, candidates, imgs_meta, imgs=None):
    """
    Args:
                net: STMask
         candidates: [candidate1, candidate2, ...]
              imgs_meta: [img_meta1, img_meta2, ...]

    Returns:
        output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
        These outputs are in the order: class idx, confidence, bbox coords, and mask.

        Note that the outputs are sorted only if cross_class_nms is False
    """

    with timer.env('Track_TF'):
        prev_candidate = None
        results = []

        # only support batch_size = 1 for video test
        for batch_idx, candidate in enumerate(candidates):
            img_meta = imgs_meta[batch_idx]
            img = imgs[batch_idx]

            if prev_candidate is None:
                # save bbox and features for later matching
                prev_candidate = dict()
                for k, v in candidate.items():
                    prev_candidate[k] = v.clone()

            else:
                n_prev = prev_candidate['box'].size(0)
                n_det = candidate['box'].size(0)
                if n_det == n_prev and candidate['tracked_mask'].sum() == 0:
                    for k, v in candidate.items():
                        prev_candidate[k] = v.clone()

                else:

                    # tracked mask: to track masks from previous frames to current frame
                    prev_candidate_shift = CandidateShift(net, candidate, prev_candidate, img=img, img_meta=img_meta)
                    for k, v in prev_candidate_shift.items():
                        prev_candidate[k] = v.clone()
                    prev_candidate['tracked_mask'] += 1

                    for idx_prev, box_id in enumerate(prev_candidate['box_ids']):
                        if box_id in candidate['box_ids']:
                            idx_det = candidate['box_ids'].tolist().index(box_id.item())
                            # estimated masks from closer frames will be given a priority
                            if candidate['tracked_mask'][idx_det] < prev_candidate['tracked_mask'][idx_prev]:
                                for k, v in candidate.items():
                                    if k in {'score', 'class', 'mask', 'mask_coeff', 'box', 'centerness', 'tracked_mask',
                                             'track_mu', 'track_var'}:
                                        prev_candidate[k][idx_prev] = v[idx_det].clone()

            result = dict()
            for k, v in prev_candidate.items():
                if k in {'box', 'class', 'score', 'centerness', 'mask_coeff', 'mask', 'track_mu', 'track_var',
                         'box_ids', 'tracked_mask'}:
                    result[k] = v.clone()
            results.append(result)

        return results[::-1]

