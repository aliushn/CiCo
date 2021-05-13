import torch
import torch.nn.functional as F
import torch.distributions as dist
from ..box_utils import jaccard,  mask_iou
from ..track_utils import generate_track_gaussian, compute_comp_scores, display_association_map
from ..mask_utils import generate_mask
from .TF_utils import CandidateShift
from utils import timer

from datasets import cfg

import numpy as np

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)


class Track_TF(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """
    # TODO: Refactor this whole class away. It needs to go.

    def __init__(self):
        self.prev_candidate = None

    def __call__(self, net, candidate, img_meta, key_frame=1, img=None):
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
            result, n_prev_inst = self.track(net, candidate, img_meta, key_frame=key_frame, img=img)
            out = [{'detection': result, 'net': net}]

        return out, n_prev_inst

    def track(self, net, candidate, img_meta, key_frame=1, img=None):
        # only support batch_size = 1 for video test
        is_first = img_meta['is_first']
        if is_first:
            self.prev_candidate = None

        if key_frame == 1 or self.prev_candidate is None:
            if candidate['box'].nelement() == 0:
                detection = {'box': torch.Tensor(), 'mask_coeff': torch.Tensor(), 'class': torch.Tensor(),
                             'score': torch.Tensor(), 'box_ids': torch.Tensor()}
                return detection, 0

        if key_frame == 0 and self.prev_candidate is not None:
            prev_candidate_shift = CandidateShift(net, candidate['fpn_feat'], self.prev_candidate,
                                                  img=img, img_meta=[img_meta])
            for k, v in prev_candidate_shift.items():
                self.prev_candidate[k] = v.clone()
            self.prev_candidate['proto'] = candidate['proto']
            self.prev_candidate['tracked_mask'] = self.prev_candidate['tracked_mask'] + 1

        else:

            assert candidate['box'].nelement() > 0
            # get bbox and class after NMS
            det_bbox = candidate['box']
            det_score, det_labels = candidate['conf'][:, 1:].max(1)  # class
            det_masks_coeff = candidate['mask_coeff']
            proto_data = candidate['proto']
            # det_track_embed = F.normalize(candidate['track'], dim=-1)
            det_track_embed = candidate['track']
            det_track_mu, det_track_var = generate_track_gaussian(det_track_embed, det_bbox)
            candidate['track_mu'] = det_track_mu
            candidate['track_var'] = det_track_var
            del candidate['track']

            n_dets = det_bbox.size(0)
            # get masks
            det_masks = generate_mask(proto_data, cfg.mask_proto_coeff_activation(det_masks_coeff), det_bbox)
            det_masks = det_masks.gt(0.5).float()

            # compared bboxes in current frame with bboxes in previous frame to achieve tracking
            if is_first or (not is_first and self.prev_candidate is None):
                # save bbox and features for later matching
                self.prev_candidate = candidate
                self.prev_candidate['tracked_mask'] = torch.zeros(n_dets)
            else:

                assert self.prev_candidate is not None
                prev_candidate_shift = CandidateShift(net, candidate['fpn_feat'], self.prev_candidate,
                                                      img=img, img_meta=[img_meta])
                for k, v in prev_candidate_shift.items():
                    self.prev_candidate[k] = v.clone()
                self.prev_candidate['proto'] = proto_data
                self.prev_candidate['tracked_mask'] = self.prev_candidate['tracked_mask'] + 1

                n_prev = self.prev_candidate['conf'].size(0)
                # only support one image at a time
                prev_track_mu = self.prev_candidate['track_mu']
                prev_track_var = self.prev_candidate['track_var']

                # save_dir = 'weights/OVIS/weights_r50_kl'
                # display_association_map(det_track_embed, prev_track_mu, prev_track_var, save_dir,
                #                         video_id=img_meta['video_id'], frame_id=img_meta['frame_id'])

                first_term = torch.sum(prev_track_var.log(), dim=-1).unsqueeze(0) \
                             - torch.sum(det_track_var.log(), dim=-1).unsqueeze(1) - det_track_mu.size(1)
                # ([1, n, c] - [n, 1, c]) * [1, n, c] => [n, n] sec_kj = \sum_{i=1}^C (mu_ij-mu_ik)^2 * sigma_i^{-1}
                second_term = torch.sum((prev_track_mu.unsqueeze(0) - det_track_mu.unsqueeze(1)) ** 2 / prev_track_var.unsqueeze(0), dim=-1)
                third_term = torch.mm(det_track_var, 1. / prev_track_var.t())  # [n, n]
                kl_divergence = 0.5 * (first_term + second_term+third_term)  # [1, +infinite]
                sim_dummy = torch.ones(n_dets, 1, device=det_bbox.device) * 1e-3  # threshold for kl_divergence = 5
                sim = torch.cat([sim_dummy, torch.exp(-kl_divergence)], dim=-1)

                bbox_ious = jaccard(det_bbox, self.prev_candidate['box'])
                if cfg.temporal_fusion_module:
                    prev_masks_shift = generate_mask(proto_data,
                                                     cfg.mask_proto_coeff_activation(self.prev_candidate['mask_coeff']),
                                                     self.prev_candidate['box'])
                else:
                    prev_masks_shift = self.prev_candidate['mask']
                prev_masks_shift = prev_masks_shift.gt(0.5).float()  # [n_prev, h, w]
                mask_ious = mask_iou(det_masks, prev_masks_shift)  # [n_dets, n_prev]

                # compute comprehensive score
                prev_det_score, prev_det_labels = self.prev_candidate['conf'][:, 1:].max(1)  # class
                label_delta = (prev_det_labels == det_labels.view(-1, 1)).float()
                comp_scores = compute_comp_scores(sim,
                                                  det_score.view(-1, 1),
                                                  bbox_ious,
                                                  mask_ious,
                                                  label_delta,
                                                  add_bbox_dummy=True,
                                                  bbox_dummy_iou=0.3,
                                                  match_coeff=cfg.match_coeff)
                comp_scores[:, 1:] = comp_scores[:, 1:] * 0.95 ** (self.prev_candidate['tracked_mask'] - 1).view(1, -1)
                match_likelihood, match_ids = torch.max(comp_scores, dim=1)
                # translate match_ids to det_obj_ids, assign new id to new objects
                # update tracking features/bboxes of exisiting object,
                # add tracking features/bboxes of new object
                det_obj_ids = torch.ones(n_dets, dtype=torch.int32) * (-1)
                best_match_scores = torch.ones(n_prev) * (-1)
                best_match_idx = torch.ones(n_prev) * (-1)
                for idx, match_id in enumerate(match_ids):
                    if match_id == 0:
                        det_obj_ids[idx] = self.prev_candidate['conf'].size(0)
                        for k, v in self.prev_candidate.items():
                            if k not in {'proto', 'fpn_feat', 'mask', 'tracked_mask'}:
                                self.prev_candidate[k] = torch.cat([v, candidate[k][idx][None]], dim=0)
                        self.prev_candidate['tracked_mask'] = torch.cat([self.prev_candidate['tracked_mask'],
                                                                         torch.zeros(1)], dim=0)

                    else:
                        # multiple candidate might match with previous object, here we choose the one with
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
                            for k, v in self.prev_candidate.items():
                                if k not in {'proto', 'fpn_feat', 'mask', 'tracked_mask'}:
                                    self.prev_candidate[k][obj_id] = candidate[k][idx]
                            self.prev_candidate['tracked_mask'][obj_id] = 0

            self.prev_candidate['mask'] = generate_mask(self.prev_candidate['proto'],
                                                        cfg.mask_proto_coeff_activation(self.prev_candidate['mask_coeff']),
                                                        self.prev_candidate['box'])

        self.prev_candidate['fpn_feat'] = candidate['fpn_feat']
        # whether add some tracked masks
        cond1 = self.prev_candidate['tracked_mask'] <= 5
        # whether tracked masks are greater than a small threshold, which removes some false positives
        cond2 = self.prev_candidate['mask'].gt_(0.5).sum([1, 2]) > 2
        # a declining weights (0.8) to remove some false positives that cased by consecutively track to segment
        det_score, _ = self.prev_candidate['conf'][:, 1:].max(1)
        cond3 = det_score.clone().detach() > cfg.eval_conf_thresh
        keep = cond1 & cond2 & cond3

        if keep.sum() == 0:
            detection = {'box': torch.Tensor(), 'mask_coeff': torch.Tensor(), 'class': torch.Tensor(),
                         'score': torch.Tensor(), 'box_ids': torch.Tensor()}
        else:
            det_obj_ids = torch.arange(self.prev_candidate['conf'].size(0))
            det_score, det_labels = self.prev_candidate['conf'][keep, 1:].max(1)
            detection = {'box': self.prev_candidate['box'][keep],
                         'mask_coeff': self.prev_candidate['mask_coeff'][keep],
                         'class': det_labels+1, 'score': det_score,
                         'centerness': self.prev_candidate['centerness'][keep],
                         'proto': self.prev_candidate['proto'],
                         'box_ids': det_obj_ids[keep], 'mask': self.prev_candidate['mask'][keep]}

        return detection, keep.sum()
