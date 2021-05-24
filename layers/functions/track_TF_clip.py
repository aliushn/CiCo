import torch
from layers.utils import jaccard,  mask_iou, generate_track_gaussian, compute_comp_scores, generate_mask, compute_kl_div
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

        # assign ID within a clip

        with timer.env('Track'):
            # only support batch_size = 1 for video test
            result, n_prev_inst = self.track(net, candidate, img_meta, key_frame=key_frame, img=img)
            out = [{'detection': result, 'net': net}]

        return out, n_prev_inst

    def track_within_clip(self, net, candidates):
        # only support batch_size = 1 for video test
        len_clip = len(candidates)
        n_obj_per_frame = []
        for i in range(len_clip):
           n_obj_per_frame.append(candidates[i]['box'].nelement())

        if max(n_obj_per_frame) == 0:
            detections = []
            for i in range(len(candidates)):
                detections.append({'box': torch.Tensor(), 'mask_coeff': torch.Tensor(), 'class': torch.Tensor(),
                                   'score': torch.Tensor(), 'box_ids': torch.Tensor()})
            return detections

        sorted_frame_idx = n_obj_per_frame.sort()[1]

        # get masks and mean/ var of Gaussian for tracking
        key_candidate = candidates[sorted_frame_idx[0]]
        key_det_masks_coeff = cfg.mask_proto_coeff_activation(key_candidate['mask_coeff'])
        if cfg.mask_proto_coeff_activation == activation_func.sigmoid:
            # hope to get mask coefficients with L0 sparsity
            key_det_masks_coeff = key_det_masks_coeff * 2 - 1
            key_det_masks_coeff = torch.clamp(key_det_masks_coeff, min=0)
        key_det_masks_soft = generate_mask(key_candidate['proto'], key_det_masks_coeff, key_candidate['box'])
        key_track_embed = F.normalize(key_candidate['track'], dim=-1)
        key_mu, key_var = generate_track_gaussian(key_track_embed, key_det_masks_soft)

        for i in sorted_frame_idx[1:]:
            non_key_candidate = candidates[i]

            # get masks and mean/ var of Gaussian for tracking
            non_key_det_masks_coeff = cfg.mask_proto_coeff_activation(non_key_candidate['mask_coeff'])
            if cfg.mask_proto_coeff_activation == activation_func.sigmoid:
                # hope to get mask coefficients with L0 sparsity
                non_key_det_masks_coeff = non_key_det_masks_coeff * 2 - 1
                non_key_det_masks_coeff = torch.clamp(non_key_det_masks_coeff, min=0)
            non_key_det_masks_soft = generate_mask(non_key_candidate['proto'],
                                                   non_key_det_masks_coeff,
                                                   non_key_candidate['box'])

            non_key_track_embed = F.normalize(non_key_candidate['track'], dim=-1)
            non_key_mu, non_key_var = generate_track_gaussian(non_key_track_embed, non_key_det_masks_soft)
            n_obj_non_key = non_key_mu.size(0)

            kl_divergence = compute_kl_div(non_key_mu, non_key_var, key_mu, key_var)
            sim_dummy = torch.ones(n_obj_non_key, 1) * 5   # threshold for kl_divergence = 5
            sim = torch.cat([sim_dummy, kl_divergence], dim=-1)

            match_likelihood, match_ids = torch.min(sim, dim=1)
            best_match_scores = torch.ones(n_obj_non_key) * 10
            non_key_obj_ids = torch.ones(n_obj_non_key, dtype=torch.int32) * (-1)
            for idx, match_id in enumerate(match_ids):
                if match_id == 0:
                    non_key_obj_ids[idx] = key_mu.size(0)
                    key_mu = torch.cat((key_mu, non_key_mu[idx][None]), dim=0)
                    key_var = torch.cat((key_var, non_key_var[idx][None]), dim=0)
                    for k, v in key_candidate.items():
                        if k not in {'proto', 'fpn_feat', 'mask_coeff', 'box'}:
                            key_candidate[k] = torch.cat([v, non_key_candidate[k][idx][None]], dim=0)

                else:
                    # multiple candidate might match with previous object, here we choose the one with
                    # largest comprehensive score
                    obj_id = match_id - 1
                    match_score = match_likelihood[idx]  # match_likelihood[idx]
                    if match_score < best_match_scores[obj_id]:
                        if (non_key_obj_ids == obj_id).sum() > 0:
                            non_key_obj_ids[non_key_obj_ids == obj_id] = -1
                        non_key_obj_ids[idx] = obj_id
                        best_match_scores[obj_id] = match_score

            candidates[i]['obj_id_clip'] = non_key_obj_ids

            new_obj_idx = match_ids == 0
            if new_obj_idx.sum() > 0:
                corr = correlate_operator(key_candidate['fpn_feat'], non_key_candidate['fpn_feat'],
                                          patch_size=cfg.correlation_patch_size, kernel_size=3, dilation_patch=1)
                # display_correlation_map(fpn_ref, img_meta, idx)
                concatenated_features = F.relu(torch.cat([corr, key_candidate['fpn_feat'],
                                                          non_key_candidate['fpn_feat']], dim=1))

                # extract features on the predicted bbox
                new_obj_box_c = center_size(non_key_candidate['box'][new_obj_idx])
                # we use 1.2 box to crop features
                new_obj_box_crop = point_form(torch.cat([new_obj_box_c[:, :2],
                                                     torch.clamp(new_obj_box_c[:, 2:] * 1.2, min=0, max=1)], dim=1))
                bbox_feat_input = bbox_feat_extractor(concatenated_features, new_obj_box_crop, 7)
                loc_offsets, mask_coeff_offsets = net.TemporalNet(bbox_feat_input)
                box_shift = torch.cat([(loc_offsets[:, :2] * new_obj_box_c[:, 2:] + new_obj_box_c[:, :2]),
                                           torch.exp(loc_offsets[:, 2:]) * new_obj_box_c[:, 2:]], dim=1)
                box_shift = point_form(box_shift)
                mask_coeff_shift = mask_coeff_offsets + non_key_candidate['mask_coeff'][new_obj_idx]

                key_candidate['mask_coeff'] = torch.cat((key_candidate['mask_coeff'], mask_coeff_shift), dim=0)
                key_candidate['box'] = torch.cat((key_candidate['box'], box_shift), dim=0)

        clip_candidate = {}
        for key in key_candidate:
            clip_candidate[key] = torch.stack([candidates[t][key] for t in range(len_clip)], dim=1)  # n * T * H * w

        return clip_candidate
