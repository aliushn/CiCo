import torch
from layers.utils import jaccard,  mask_iou, generate_track_gaussian, compute_comp_scores, generate_mask
from .TF_utils import CandidateShift
from utils import timer

from datasets import cfg, activation_func
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import os

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

    def __call__(self, net, candidate, img_meta, img=None):
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
            result = self.track(net, candidate, img_meta, img=img)

        return result

    def track(self, net, candidate, img_meta, img=None):
        # only support batch_size = 1 for video test
        is_first = img_meta['is_first']
        if is_first:
            self.prev_candidate = None

        if candidate['box'].nelement() == 0:
            return {'box': torch.Tensor(), 'box_ids': torch.Tensor(), 'mask_coeff': torch.Tensor(),
                    'class': torch.Tensor(), 'score': torch.Tensor()}

        assert candidate['box'].nelement() > 0
        # get bbox and class after NMS
        det_bbox = candidate['box']
        det_score, det_labels = candidate['score'], candidate['class']
        det_masks_coeff = candidate['mask_coeff']
        proto_data = candidate['proto']
        # get masks
        det_masks_soft = generate_mask(proto_data, det_masks_coeff, det_bbox)
        candidate['mask'] = det_masks_soft

        det_track_embed = F.normalize(candidate['track'], dim=-1)
        det_track_mu, det_track_var = generate_track_gaussian(det_track_embed, det_masks_soft)
        candidate['track_mu'] = det_track_mu
        candidate['track_var'] = det_track_var

        n_dets = det_bbox.size(0)

        # compared bboxes in current frame with bboxes in previous frame to achieve tracking
        if is_first or (not is_first and self.prev_candidate is None):
            # save bbox and features for later matching
            self.prev_candidate = candidate
            self.prev_candidate['mask'] = det_masks_soft
            self.prev_candidate['tracked_mask'] = torch.zeros(n_dets)
        else:

            assert self.prev_candidate is not None

            # tracked mask: to track masks from previous frames to current frame
            prev_candidate_shift = CandidateShift(net, proto_data, candidate['fpn_feat'], self.prev_candidate,
                                                  img=img, img_meta=[img_meta])
            for k, v in prev_candidate_shift.items():
                self.prev_candidate[k] = v.clone()
            for k, v in self.prev_candidate.items():
                if k in {'proto', 'fpn_feat', 'track', 'sem_seg'}:
                    self.prev_candidate[k] = v
            self.prev_candidate['tracked_mask'] = self.prev_candidate['tracked_mask'] + 1

            # calculate KL divergence for Gaussian distribution of isntances
            n_prev = self.prev_candidate['box'].size(0)
            prev_track_mu = self.prev_candidate['track_mu']
            prev_track_var = self.prev_candidate['track_var']
            first_term = torch.sum(prev_track_var.log(), dim=-1).unsqueeze(0) \
                         - torch.sum(det_track_var.log(), dim=-1).unsqueeze(1) - det_track_mu.size(1)
            # ([1, n, c] - [n, 1, c]) * [1, n, c] => [n, n] sec_kj = \sum_{i=1}^C (mu_ij-mu_ik)^2 * sigma_i^{-1}
            second_term = torch.sum((prev_track_mu.unsqueeze(0) - det_track_mu.unsqueeze(1)) ** 2 / prev_track_var.unsqueeze(0), dim=-1)
            third_term = torch.mm(det_track_var, 1. / prev_track_var.t())  # [n, n]
            kl_divergence = 0.5 * (first_term + second_term + third_term)  # [0, +infinite]
            sim_dummy = torch.ones(n_dets, 1, device=det_bbox.device) * 5  # threshold for kl_divergence = 5
            sim = torch.cat([sim_dummy, kl_divergence], dim=-1) * -1

            # Calculate BIoU and MIoU between detected masks and tracked masks for assign IDs
            bbox_ious = jaccard(det_bbox, self.prev_candidate['box'])  # [n_dets, n_prev]
            mask_ious = mask_iou(det_masks_soft.gt(0.5).float(), self.prev_candidate['mask'].gt(0.5))  # [n_dets, n_prev]

            # outs = torch.cat((det_masks, prev_masks_shift), dim=0).view(-1, det_masks.size(-1))
            # plt.imshow(outs.cpu().numpy())
            # root_dir = ''.join(['/home/lmh/Downloads/VIS/code/OSTMask/weights/YTVIS2019/weights_r50_kl_vos_D_occluded_coeff/vos_mask/',
            #                     str(img_meta['video_id'])])
            # if not os.path.exists(root_dir):
            #     os.makedirs(root_dir)
            # plt.savefig(''.join([root_dir, '/', str(img_meta['frame_id']), '_miou.png']))

            # compute comprehensive score
            label_delta = (self.prev_candidate['class'] == det_labels.view(-1, 1)).float()
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
                    det_obj_ids[idx] = self.prev_candidate['box'].size(0)
                    for k, v in self.prev_candidate.items():
                        if k not in {'proto', 'fpn_feat', 'track', 'sem_seg', 'tracked_mask'}:
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
                            if k not in {'proto', 'fpn_feat', 'track', 'sem_seg', 'tracked_mask'}:
                                self.prev_candidate[k][obj_id] = candidate[k][idx]
                        self.prev_candidate['tracked_mask'][obj_id] = 0

        # whether add some tracked masks
        cond1 = self.prev_candidate['tracked_mask'] <= 5
        # whether tracked masks are greater than a small threshold, which removes some false positives
        cond2 = self.prev_candidate['mask'].gt(0.5).sum([1, 2]) > 10
        # a declining weights (0.8) to remove some false positives that cased by consecutively track to segment
        cond3 = self.prev_candidate['score'].clone().detach() > cfg.eval_conf_thresh
        keep = cond1 & cond2 & cond3

        if keep.sum() == 0:
            detection = {'box': torch.Tensor(), 'box_ids': torch.Tensor(), 'mask_coeff': torch.Tensor(),
                         'class': torch.Tensor(), 'score': torch.Tensor()}
        else:
            det_obj_ids = torch.arange(self.prev_candidate['box'].size(0))
            detection = {'box': self.prev_candidate['box'][keep], 'box_ids': det_obj_ids[keep],
                         'class': self.prev_candidate['class'][keep], 'score': self.prev_candidate['score'][keep],
                         'mask_coeff': self.prev_candidate['mask_coeff'][keep], 'proto': self.prev_candidate['proto'],
                         'mask': self.prev_candidate['mask'][keep]}
            if cfg.train_centerness:
                detection['centerness'] = self.prev_candidate['centerness'][keep]

        return detection
