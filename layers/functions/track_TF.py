import torch
import torch.nn.functional as F
from layers.utils import jaccard,  mask_iou, generate_track_gaussian, compute_comp_scores, compute_kl_div, generate_mask, \
    center_size, point_form
from .TF_utils import CandidateShift
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

    def __init__(self, net, cfg):
        self.prev_candidate = None
        self.cfg = cfg
        self.match_coeff = cfg.MODEL.TRACK_HEADS.MATCH_COEFF
        self.correlation_patch_size = cfg.STMASK.T2S_HEADS.CORRELATION_PATCH_SIZE
        self.conf_thresh = cfg.TEST.NMS_CONF_THRESH
        self.track_by_Gaussian = cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN
        self.proto_coeff_occlu = cfg.MODEL.MASK_HEADS.PROTO_COEFF_OCCLUSION
        self.img_level_keys = ['proto', 'fpn_feat', 'fpn_feat_temp', 'sem_seg']
        if self.track_by_Gaussian:
            self.img_level_keys += ['track']
        self.num_clip_frames = cfg.SOLVER.NUM_CLIP_FRAMES
        self.use_cubic_TF = False

        if self.cfg.STMASK.T2S_HEADS.TEMPORAL_FUSION_MODULE:
            self.CandidateShift = CandidateShift(net, cfg.STMASK.T2S_HEADS.CORRELATION_PATCH_SIZE,
                                                 train_maskshift=cfg.STMASK.T2S_HEADS.TRAIN_MASKSHIFT,
                                                 train_masks=cfg.MODEL.MASK_HEADS.TRAIN_MASKS,
                                                 track_by_Gaussian=self.track_by_Gaussian,
                                                 proto_coeff_occlusion=self.proto_coeff_occlu)

    def __call__(self, candidates, imgs_meta, img=None):
        outputs_aft_track = []
        for batch_idx in range(len(candidates)):
            outputs_aft_track.append(self.track(candidates[batch_idx], imgs_meta[batch_idx], img=img[batch_idx]))

        return outputs_aft_track

    def track(self, outputs_aft_nms, img_meta, img=None, bbox_dummy_iou=0.3):
        # In case inplace operation
        candidate = dict()
        for k, v in outputs_aft_nms.items():
            candidate[k] = v.clone()

        # only support batch_size = 1 for video test
        if isinstance(img_meta, list):
            is_first = img_meta[0]['is_first']
            candidate['img_ids'] = torch.tensor([meta['frame_id'] for meta in img_meta])
        else:
            is_first = img_meta['is_first']
            candidate['img_ids'] = img_meta['frame_id']
        if is_first:
            self.prev_candidate = None

        # generate mean and variance of Gaussian for objects
        det_masks_soft = candidate['mask'] if self.cfg.MODEL.MASK_HEADS.TRAIN_MASKS else None
        if self.track_by_Gaussian:
            candidate['track_mu'], candidate['track_var'] = generate_track_gaussian(candidate['track'], det_masks_soft,
                                                                                    candidate['box'])
        track_dim = candidate['track'].size(-1)

        # compared bboxes in current frame with bboxes in previous frame to achieve tracking
        if is_first or (not is_first and self.prev_candidate is None):
            if candidate['box'].nelement() == 0:
                candidate['box_ids'] = torch.Tensor()
            else:
                candidate['box_ids'] = torch.arange(candidate['box'].size(0))
                candidate['tracked_mask'] = torch.zeros(candidate['box'].size(0))
                # save bbox and features for later matching
                self.prev_candidate = dict()
                for k, v in candidate.items():
                    self.prev_candidate[k] = v.clone()
            return candidate

        else:

            assert self.prev_candidate is not None

            # Tracked mask: to track masks from previous frames to current frame in STMask
            if self.cfg.STMASK.T2S_HEADS.TEMPORAL_FUSION_MODULE:
                self.CandidateShift(candidate, self.prev_candidate, img=img, img_meta=img_meta)
            if self.use_cubic_TF:
                # Tracked masks by temporal coherence: P_t * coeff_{t-1} between two adjacent clips
                prev_boxes_cir = center_size(self.prev_candidate['box_cir'])
                prev_boxes_cir[:, 2:] *= 1.2
                self.prev_candidate['mask'] = generate_mask(candidate['proto'], self.prev_candidate['mask_coeff'],
                                                            point_form(prev_boxes_cir))
                self.prev_candidate['score'] *= 0.9**(self.prev_candidate['mask'].size(1))

            for k, v in candidate.items():
                if k in self.img_level_keys:
                    self.prev_candidate[k] = v.clone()
            self.prev_candidate['tracked_mask'] += 1
            n_prev = self.prev_candidate['box'].size(0)

            # Combine detected masks and tracked masks
            if candidate['box'].nelement() == 0:
                candidate['box_ids'] = torch.Tensor()
                return candidate
            else:
                # get bbox and class after NMS
                det_bbox, det_score = candidate['box'], candidate['score']
                n_dets = det_bbox.size(0)
                candidate['tracked_mask'] = torch.zeros(n_dets)

                match_type = 1
                if match_type == 1:
                    img_over_idx = torch.nonzero(self.prev_candidate['img_ids'].unsqueeze(1)==candidate['img_ids'],
                                                 as_tuple=False)

                    if img_over_idx.nelement() > 0:
                        prev_overlapped_idx = img_over_idx[:, 0]
                        cur_overlapped_idx = img_over_idx[:, 1]
                    else:
                        prev_overlapped_idx, cur_overlapped_idx = [candidate['proto'].size(2)-1], [0]

                    # Calculate BIoU and MIoU between detected masks and previous masks for assign IDs
                    if det_bbox.size(-1) == 4:
                        bbox_ious = jaccard(det_bbox,  self.prev_candidate['box'])
                    else:
                        bbox_ious = torch.stack([jaccard(det_bbox[:, cur_idx*4:(cur_idx+1)*4],
                                                 self.prev_candidate['box'][:, prev_idx*4:(prev_idx+1)*4])
                                                 for prev_idx, cur_idx in zip(prev_overlapped_idx, cur_overlapped_idx)
                                                 ]).mean(dim=0)

                    if self.cfg.MODEL.CLASS_HEADS.TRAIN_INTERCLIPS_CLASS:
                        label_delta = torch.zeros_like(bbox_ious)
                    else:
                        det_labels = candidate['class']
                        label_delta = (self.prev_candidate['class'] == det_labels.view(-1, 1)).float()

                    if self.cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
                        det_masks, prev_masks = det_masks_soft.gt(0.5).float(), self.prev_candidate['mask'].gt(0.5).float()
                        if self.use_cubic_TF:
                            mask_ious = mask_iou(det_masks.permute(1, 0, 2, 3).contiguous(),
                                                 prev_masks.permute(1, 0, 2, 3).contiguous()).mean(dim=0)
                        else:
                            mask_ious = torch.stack([mask_iou(det_masks[:, cur_idx], prev_masks[:, prev_idx])
                                                     for prev_idx, cur_idx in zip(prev_overlapped_idx, cur_overlapped_idx)
                                                     ]).mean(dim=0)

                    else:
                        mask_ious = torch.zeros_like(bbox_ious)

                    if self.cfg.MODEL.MASK_HEADS.TRAIN_MASKS and self.track_by_Gaussian:
                        # Calculate KL divergence for Gaussian distribution of isntances, value in [[0, +infinite]]
                        kl_divergence = compute_kl_div(self.prev_candidate['track_mu'].reshape(-1, track_dim),
                                                       self.prev_candidate['track_var'].reshape(-1, track_dim),
                                                       candidate['track_mu'].reshape(-1, track_dim),
                                                       candidate['track_var'].reshape(-1, track_dim))
                        kl_divergence = kl_divergence.reshape(n_dets, self.num_clip_frames, -1, self.num_clip_frames).mean(1).mean(-1)
                        sim_dummy = torch.ones(n_dets, 1, device=det_bbox.device) * 10.
                        # from [0, +infinite] to [0, 1]: sim = 1/ (exp(0.1*kl_div)), threshold  = 10
                        sim = torch.div(1., torch.exp(0.1 * torch.cat([sim_dummy, kl_divergence], dim=-1)))
                        # from [0, +infinite] to [0, 1]: sim = exp(1-kl_div)), threshold  = 5
                        # sim = torch.exp(1 - torch.cat([sim_dummy, kl_divergence], dim=-1))
                    else:
                        cos_sim = candidate['track'] @ self.prev_candidate['track'].t()   # val in [-1, 1]
                        cos_sim = (cos_sim + 1) / 2                                       # [0, 1]
                        sim = torch.cat([torch.ones(n_dets, 1)*0.5, cos_sim], dim=1)

                    # compute comprehensive score
                    det_box_hw = center_size(det_bbox.reshape(-1, 4)).reshape(n_dets, -1, 4).mean(dim=1)[:, 2:]
                    small_objects = torch.prod(det_box_hw, dim=-1) < 0.1
                    comp_scores = compute_comp_scores(sim,
                                                      det_score.view(-1, 1),
                                                      bbox_ious,
                                                      mask_ious,
                                                      label_delta,
                                                      small_objects=small_objects,
                                                      add_bbox_dummy=True,
                                                      bbox_dummy_iou=bbox_dummy_iou,
                                                      match_coeff=self.match_coeff)

                else:
                    comp_scores = F.normalize(candidate['mask_coeff'], dim=-1) @ F.normalize(self.prev_candidate['mask_coeff'], dim=-1).t()
                    comp_scores = torch.cat([torch.ones(n_dets, 1)*0.5, comp_scores], dim=1)

                match_likelihood, match_ids = torch.max(comp_scores, dim=1)

                # Translate match_ids to det_obj_ids, assign new id to new objects
                # update tracking features/bboxes of exisiting object,
                # add tracking features/bboxes of new object
                candidate['box_ids'] = torch.ones(n_dets, dtype=torch.int64) * (-1)
                best_match_scores = torch.ones(n_prev) * (-1)
                best_match_idx = torch.ones(n_prev, dtype=torch.int64) * (-1)
                for match_idx, match_id in enumerate(match_ids):
                    if match_id == 0:
                        candidate['box_ids'][match_idx] = self.prev_candidate['box'].size(0)
                        for k, v in self.prev_candidate.items():
                            if k not in self.img_level_keys + ['img_ids']:
                                self.prev_candidate[k] = torch.cat([v, candidate[k][match_idx][None]], dim=0)

                    else:
                        # multiple candidate might match with previous object, here we choose the one with
                        # largest comprehensive score
                        obj_id = match_id - 1
                        match_score = match_likelihood[match_idx]
                        if match_score > best_match_scores[obj_id]:
                            if best_match_idx[obj_id] != -1:
                                candidate['box_ids'][int(best_match_idx[obj_id])] = -1
                            candidate['box_ids'][match_idx] = obj_id
                            best_match_scores[obj_id] = match_score
                            best_match_idx[obj_id] = match_idx
                            for k, v in self.prev_candidate.items():
                                if k not in self.img_level_keys + ['img_ids']:
                                    self.prev_candidate[k][obj_id] = candidate[k][match_idx]

                # Whether add some objects whose masks are tracked form previous frames by Temporal Fusion Module
                cond1 = (self.prev_candidate['tracked_mask'] > 0) & (self.prev_candidate['tracked_mask'] <= 5)
                # a declining weights (0.85) to remove some false positives that cased by consecutively track to segment
                cond2 = self.prev_candidate['score'].clone().detach() > self.conf_thresh
                keep_tra = cond1 & cond2
                if self.cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
                    # whether tracked masks are greater than a small threshold, which removes some false positives
                    cond3 = (self.prev_candidate['mask'].gt(0.5).sum([-1, -2]) > 3).sum(-1) > 0
                    keep_tra = keep_tra & cond3.reshape(-1)

                # Choose objects detected and segmented by frame-level prediction head
                keep_det = candidate['box_ids'] >= 0

                if keep_det.sum() + keep_tra.sum() > 0:
                    for k, v in self.prev_candidate.items():
                        if k not in self.img_level_keys+['img_ids']:
                            if keep_det.sum() == 0:
                                candidate[k] = v[keep_tra].clone()
                            elif keep_tra.sum() == 0:
                                candidate[k] = candidate[k][keep_det].clone()
                            else:
                                candidate[k] = torch.cat([candidate[k][keep_det], v[keep_tra].clone()])
                self.prev_candidate['img_ids'] = candidate['img_ids'].clone()
                return candidate
