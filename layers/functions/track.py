import torch
import torch.nn.functional as F
from layers.utils import jaccard, mask_iou, compute_DIoU, generate_mask, compute_comp_scores, generate_rel_coord, \
    generate_track_gaussian, compute_kl_div
from utils import timer

from datasets import cfg

import numpy as np

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)


class Track(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """
    # TODO: Refactor this whole class away. It needs to go.

    def __init__(self, clip_frames=1):
        self.prev_detection = None
        self.clip_frames = clip_frames

    def __call__(self, pred_outs_after_NMS, img_meta):
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
            batch_size = len(pred_outs_after_NMS)

            results = []
            for batch_idx in range(batch_size):
                results.append(self.track(pred_outs_after_NMS[batch_idx],
                                          img_meta[batch_idx*self.clip_frames:(batch_idx+1)*self.clip_frames]))

        return results

    def track(self, detection, img_metas):

        # only support batch_size = 1 for video test
        is_first = img_metas[0]['is_first']
        if is_first:
            self.prev_detection = None

        if detection['class'].nelement() == 0:
            detection['box_ids'] = torch.tensor([], dtype=torch.int64)
            return detection

        # get bbox and class after NMS
        det_bbox = detection['box']
        det_labels = detection['class']  # class
        det_score = detection['score']
        det_masks_soft = detection['mask'] if cfg.train_masks else None
        det_masks = det_masks_soft.gt(0.5).float() if cfg.train_masks else None

        if cfg.track_by_Gaussian:
            detection['track_mu'], detection['track_var'] = generate_track_gaussian(detection['track'].squeeze(0),
                                                                                    det_masks_soft, det_bbox)
            del detection['track']

        # compared bboxes in current frame with bboxes in previous frame to achieve tracking
        if is_first or (not is_first and self.prev_detection is None):
            det_obj_ids = torch.arange(det_bbox.size(0))
            # save bbox and features for later matching
            self.prev_detection = dict()
            for k, v in detection.items():
                if k in {'box', 'mask', 'class', 'track', 'track_mu', 'track_var'}:
                    self.prev_detection[k] = v.clone()

        else:

            assert self.prev_detection is not None
            n_dets = det_bbox.size(0)
            n_prev = self.prev_detection['box'].size(0)
            # only support one image at a time
            if cfg.track_by_Gaussian:
                kl_divergence = compute_kl_div(self.prev_detection['track_mu'], self.prev_detection['track_var'],
                                               detection['track_mu'], detection['track_var'])     # value in [[0, +infinite]]
                sim_dummy = torch.ones(n_dets, 1, device=det_bbox.device) * 10  # threshold for kl_divergence = 10
                # from [0, +infinite] to [0, 1]: sim = 1/ (exp(0.1*kl_div))
                sim = torch.div(1., torch.exp(0.1 * torch.cat([sim_dummy, kl_divergence], dim=-1)))
            else:
                cos_sim = detection['track'] @ self.prev_detection['track'].t()      # [n_dets, n_prev], val in [-1, 1]
                cos_sim = torch.cat([torch.zeros(n_dets, 1), cos_sim], dim=1)
                sim = (cos_sim + 1) / 2  # [0, 1]

            label_delta = (self.prev_detection['class'] == det_labels.view(-1, 1)).float()
            bbox_ious = jaccard(det_bbox[:, :4], self.prev_detection['box'][:, -4:])
            if cfg.train_masks:
                if self.clip_frames > 1:
                    mask_ious = mask_iou(det_masks[..., 0], self.prev_detection['mask'].gt(0.5).float()[..., -1])
                else:
                    mask_ious = mask_iou(det_masks, self.prev_detection['mask'].gt(0.5).float())
            else:
                mask_ious = torch.zeros_like(bbox_ious)
                cfg.match_coeff[1] = 0

            # compute comprehensive score
            comp_scores = compute_comp_scores(sim,
                                              det_score.view(-1, 1),
                                              bbox_ious,
                                              mask_ious,
                                              label_delta,
                                              add_bbox_dummy=True,
                                              bbox_dummy_iou=0.3,
                                              match_coeff=cfg.match_coeff)

            match_likelihood, match_ids = torch.max(comp_scores, dim=1)
            # translate match_ids to det_obj_ids, assign new id to new objects
            # update tracking features/bboxes of exisiting object,
            # add tracking features/bboxes of new object
            det_obj_ids = torch.ones(n_dets, dtype=torch.int32) * (-1)
            best_match_scores = torch.ones(n_prev) * (-1)
            best_match_idx = torch.ones(n_prev) * (-1)
            for idx, match_id in enumerate(match_ids):
                if match_id == 0:
                    det_obj_ids[idx] = self.prev_detection['box'].size(0)
                    for k, v in self.prev_detection.items():
                        self.prev_detection[k] = torch.cat([v, detection[k][idx][None]], dim=0)

                else:
                    # Multiple candidate might match with previous object, here we choose the one with
                    # largest comprehensive score
                    obj_id = match_id - 1
                    match_score = det_score[idx]  # match_likelihood[idx]
                    if match_score > best_match_scores[obj_id]:
                        if best_match_idx[obj_id] != -1:
                            det_obj_ids[int(best_match_idx[obj_id])] = -1
                        det_obj_ids[idx] = obj_id
                        best_match_scores[obj_id] = match_score
                        best_match_idx[obj_id] = idx
                        # udpate feature
                        for k, v in self.prev_detection.items():
                            self.prev_detection[k][obj_id] = detection[k][idx].clone()

        detection['box_ids'] = det_obj_ids

        keep = det_obj_ids >= 0
        for k, v in detection.items():
            if k not in {'proto'}:
                detection[k] = v[keep]

        return detection