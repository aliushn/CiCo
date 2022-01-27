import torch
import torch.nn.functional as F
import numpy as np
from layers.utils import jaccard, center_size, point_form, crop, mask_iou, decode, generate_mask, \
    generate_track_gaussian, compute_comp_scores, compute_kl_div
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)


class Track_TF(object):
    """
    At test time, track is the final part of VIS. We expend the tracking of MaskTrack R-CNN as a clip-level tracking.
    (https://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_Video_Instance_Segmentation_ICCV_2019_paper.pdf)
    1. MaskTrack R-CNN combines the similarity of embedding vectors, bou IoU, Mask IoU and classes between objects
    of two frames to compute matching scores.
    2. STMask adds a temporal fusion module to track and detect objects from previous frames to current frame,
    which effectively reduce missed objects for challenging cases. (https://arxiv.org/abs/2104.05606)
    3. CiCo also extend it as clip-level tracking between two clips.
    """
    # TODO: Refactor this whole class away. It needs to go.

    def __init__(self, net, cfg):
        self.net = net
        self.cfg = cfg
        self.prev_candidate = None
        self.match_coeff = cfg.MODEL.TRACK_HEADS.MATCH_COEFF
        self.conf_thresh = cfg.TEST.NMS_CONF_THRESH
        self.train_mask = cfg.MODEL.MASK_HEADS.TRAIN_MASKS
        self.track_by_Gaussian = cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN
        self.use_dynamic_mask = cfg.MODEL.MASK_HEADS.USE_DYNAMIC_MASK
        self.img_level_keys = ['proto', 'fpn_feat', 'fpn_feat_temp', 'sem_seg']
        if self.track_by_Gaussian:
            self.img_level_keys += ['track']
        self.num_clip_frames = cfg.SOLVER.NUM_CLIP_FRAMES

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
            candidate['img_ids'] = torch.tensor([img_meta['frame_id']])
        if is_first:
            self.prev_candidate = None

        # generate mean and variance of Gaussian for objects
        if candidate['box'].nelement() > 0:
            det_masks_soft = candidate['mask'] if self.cfg.MODEL.MASK_HEADS.TRAIN_MASKS else None
            if self.track_by_Gaussian:
                candidate['track_mu'], candidate['track_var'] = generate_track_gaussian(candidate['track'],
                                                                                        masks=det_masks_soft,
                                                                                        boxes=candidate['box'])

        # compared bboxes in current frame with bboxes in previous frame to achieve tracking
        if is_first or (not is_first and self.prev_candidate is None):
            if candidate['box'].nelement() == 0:
                candidate['box_ids'] = torch.Tensor()
            else:
                candidate['box_ids'] = torch.arange(candidate['box'].size(1))
                candidate['tracked_mask'] = torch.zeros(candidate['box'].size(1))
                # save bbox and features for later matching
                self.prev_candidate = dict()
                for k, v in candidate.items():
                    self.prev_candidate[k] = v.clone()
            return candidate

        else:

            assert self.prev_candidate is not None
            # Combine detected masks and tracked masks
            if candidate['box'].nelement() == 0:
                candidate['box_ids'] = torch.Tensor()
                return candidate
            else:
                n_prev = self.prev_candidate['box'].size(1)
                n_dets = candidate['mask'].size(1)
                # get bbox and class after NMS
                det_bbox, det_score, det_labels = candidate['box'], candidate['score'], candidate['class']
                track_dim = candidate['track'].size(-1)
                candidate['tracked_mask'] = torch.zeros(n_dets)
                label_delta = (self.prev_candidate['class'] == det_labels.view(-1, 1)).float()
                if not self.track_by_Gaussian:
                    cos_sim = candidate['track'] @ self.prev_candidate['track'].t()  # val in [-1, 1]
                    cos_sim = (cos_sim + 1) / 2  # val in [0, 1]
                    sim = torch.cat([torch.ones(n_dets, 1) * 0.5, cos_sim], dim=1)

                img_over_idx = torch.nonzero(self.prev_candidate['img_ids'].unsqueeze(1) == candidate['img_ids'],
                                             as_tuple=False)
                if img_over_idx.nelement() > 0:
                    # ------------------------------ overlapping clips ----------------------------------
                    prev_over_idx, cur_over_idx = img_over_idx[:, 0], img_over_idx[:, 1]
                    bbox_ious = jaccard(det_bbox[cur_over_idx].reshape(-1, n_dets, 4),
                                        self.prev_candidate['box'][prev_over_idx].reshape(-1, n_prev, 4))
                    if self.cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
                        prev_masks = self.prev_candidate['mask'][prev_over_idx].gt(.5).float()
                        det_masks = candidate['mask'][cur_over_idx].gt(.5).float()
                        det_masks_flag = det_masks.reshape(len(cur_over_idx), n_dets, -1).sum(-1) > 2
                        prev_masks_flag = prev_masks.reshape(len(prev_over_idx), n_prev, -1).sum(-1) > 2
                        flag = det_masks_flag.unsqueeze(-1) & prev_masks_flag.unsqueeze(-2)
                        mask_ious = mask_iou(det_masks, prev_masks)
                        mask_ious = (mask_ious * flag).sum(0) / torch.clamp(flag.sum(0), min=1)
                        bbox_ious = (bbox_ious * flag).sum(0) / torch.clamp(flag.sum(0), min=1)
                    else:
                        bbox_ious = bbox_ious.mean(0)
                        mask_ious = torch.zeros_like(bbox_ious)

                else:
                    # ---------------------------- non-overlapping clips ---------------------------------
                    # To track masks from previous frames to current frame in STMask
                    if self.cfg.STMASK.T2S_HEADS.TEMPORAL_FUSION_MODULE:
                        self.T2S_Module(candidate, train_masks=self.train_mask, 
                                        track_by_Gaussian=self.track_by_Gaussian)
                    else:
                        # Tracked masks by temporal coherence: P_t * coeff_{t-1} between two adjacent clips
                        self.Naive_Tacker(candidate)
                    
                    # match objects in previous and current clips
                    # Calculate BIoU and MIoU between detected masks and previous masks for assign IDs
                    label_delta = (self.prev_candidate['class'] == det_labels.view(-1, 1)).float()
                    bbox_ious = jaccard(det_bbox.reshape(-1, n_dets, 4)[0],
                                        self.prev_candidate['box'].reshape(-1, n_prev, 4)[-1])
                    if self.cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
                        det_masks, prev_masks = candidate['mask'].gt(.5).float(), self.prev_candidate['mask'].gt(.5).float()
                        det_masks_flag = det_masks.reshape(self.num_clip_frames, n_dets, -1).sum(-1) > 2
                        prev_masks_flag = prev_masks.reshape(self.num_clip_frames, n_prev, -1).sum(-1) > 2
                        flag = det_masks_flag.unsqueeze(-1) & prev_masks_flag.unsqueeze(-2)
                        mask_ious = mask_iou(det_masks, prev_masks)
                        mask_ious = (mask_ious * flag).sum(0) / torch.clamp(flag.sum(0), min=1)

                        if self.track_by_Gaussian:
                            # Calculate KL divergence for Gaussian distribution of instances, value [[0, +infinite]]
                            kl_div = compute_kl_div(self.prev_candidate['track_mu'].reshape(-1, track_dim),
                                                    self.prev_candidate['track_var'].reshape(-1, track_dim),
                                                    candidate['track_mu'].reshape(-1, track_dim),
                                                    candidate['track_var'].reshape(-1, track_dim))
                            kl_div = kl_div.reshape(self.num_clip_frames, n_dets, self.num_clip_frames, n_prev)
                            # remove those objects with blank masks, N1T*N2T
                            flag = det_masks_flag.view(-1, 1) & prev_masks_flag.view(1, -1)
                            flag = flag.reshape(self.num_clip_frames, n_dets, self.num_clip_frames, n_prev)
                            kl_div = (kl_div * flag).sum([0, 2]) / torch.clamp(flag.sum([0, 2]), min=1)
                            sim_dummy = torch.ones(n_dets, 1, device=det_bbox.device) * 10.
                            # from [0, +infinite] to [0, 1]: sim = 1/ (exp(0.1*kl_div)), threshold  = 10
                            sim = torch.div(1., torch.exp(0.1 * torch.cat([sim_dummy, kl_div], dim=-1)))
                            # from [0, +infinite] to [0, 1]: sim = exp(1-kl_div)), threshold  = 5
                            # sim = torch.exp(1 - torch.cat([sim_dummy, kl_div], dim=-1))
                    else:
                        mask_ious = torch.zeros_like(bbox_ious)

                # compute comprehensive score
                det_box_hw = center_size(det_bbox.reshape(-1, 4)).reshape(-1, n_dets, 4).mean(0)[:, 2:]
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
                match_likelihood, match_ids = torch.max(comp_scores, dim=1)

                # Translate match_ids to det_obj_ids, assign new id to new objects
                # update tracking features/bboxes of existing object,
                # add tracking features/bboxes of new object
                candidate['box_ids'] = torch.ones(n_dets, dtype=torch.int64) * (-1)
                best_match_scores = torch.ones(n_prev) * (-1)
                best_match_idx = torch.ones(n_prev, dtype=torch.int64) * (-1)
                for match_idx, match_id in enumerate(match_ids):
                    if match_id == 0:
                        candidate['box_ids'][match_idx] = self.prev_candidate['box'].size(1)
                        for k, v in self.prev_candidate.items():
                            if k not in self.img_level_keys + ['img_ids']:
                                if k in {'box', 'mask', 'track_mu', 'track_var'}:
                                    self.prev_candidate[k] = torch.cat([v, candidate[k][:, match_idx].unsqueeze(1)], dim=1)
                                else:
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
                                    if k in {'box', 'mask', 'track_mu', 'track_var'}:
                                        self.prev_candidate[k][:, obj_id] = candidate[k][:, match_idx]
                                    else:
                                        self.prev_candidate[k][obj_id] = candidate[k][match_idx]

                # Choose objects detected and segmented by frame-level prediction head
                keep_det = candidate['box_ids'] >= 0
                if img_over_idx.nelement() == 0:
                    # Whether add some objects whose masks are tracked form previous frames by Temporal Fusion Module
                    cond1 = (self.prev_candidate['tracked_mask'] > 0) & (self.prev_candidate['tracked_mask'] <= 5)
                    # a declining weights to remove some false positives that cased by consecutively track to segment
                    cond2 = self.prev_candidate['score'].clone().detach() > self.conf_thresh
                    keep_tra = cond1 & cond2
                    if self.cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
                        # whether tracked masks are greater than a small threshold, which removes some false positives
                        cond3 = (self.prev_candidate['mask'].gt(0.5).sum([-1, -2]) > 3).sum(0) > 0
                        keep_tra = keep_tra & cond3.reshape(-1)

                    if keep_det.sum() + keep_tra.sum() > 0:
                        for k, v in self.prev_candidate.items():
                            if k not in self.img_level_keys + ['img_ids']:
                                if k in {'box', 'mask', 'track_mu', 'track_var'}:
                                    candidate[k] = torch.cat([candidate[k][:, keep_det], v[:, keep_tra].clone()], dim=1)
                                else:
                                    candidate[k] = torch.cat([candidate[k][keep_det], v[keep_tra].clone()])
                else:
                    if keep_det.sum() > 0:
                        for k, v in candidate.items():
                            if k not in self.img_level_keys+['img_ids']:
                                if k in {'box', 'mask', 'track_mu', 'track_var'}:
                                    candidate[k] = v[:, keep_det]
                                else:
                                    candidate[k] = v[keep_det]
                self.prev_candidate['img_ids'] = candidate['img_ids'].clone()
                return candidate

    def T2S_Module(self, candidate, train_masks=True, track_by_Gaussian=False, update_track=False):
        if self.prev_candidate['box'].nelement() > 0:
            boxes_dim = self.prev_candidate['box'].dim()
            boxes_ref = self.prev_candidate['box'].reshape(-1, 4).clone()
            pred_boxes_off, pred_mask_coeff_tar = self.net.T2S_Head(self.prev_candidate['fpn_feat'], 
                                                                    candidate['fpn_feat'], boxes_ref)
            if pred_mask_coeff_tar is not None:
                self.prev_candidate['mask_coeff'] = pred_mask_coeff_tar
    
            # Update bounding boxes
            pred_boxes_tar = decode(pred_boxes_off, center_size(boxes_ref))
            self.prev_candidate['box'] = pred_boxes_tar.unsqueeze(0) if boxes_dim == 3 else pred_boxes_tar
    
            # decay_rate = 0.9**self.prev_candidate['tracked_mask'] if 'tracked_mask' in candidate.keys() else 0.9
            self.prev_candidate['score'] *= 0.9
            if train_masks:
                self.prev_candidate['mask'] = generate_mask(candidate['proto'], self.prev_candidate['mask_coeff'], pred_boxes_tar).transpose(0, 1)
    
            if 'frame_id' in candidate.keys():
                self.prev_candidate['frame_id'] = candidate['frame_id']
            if track_by_Gaussian and update_track:
                mu, var = generate_track_gaussian(self.prev_candidate['track'],
                                                  self.prev_candidate['mask'].gt(0.5),
                                                  self.prev_candidate['box'])
                self.prev_candidate['track_mu'], self.prev_candidate['track_var'] = mu, var
    
        for k, v in candidate.items():
            if k in self.img_level_keys:
                self.prev_candidate[k] = v.clone()
        self.prev_candidate['tracked_mask'] += 1

    def Naive_Tacker(self, candidate):
        # Tracked masks by temporal coherence: P_t * coeff_{t-1} between two adjacent clips
        prev_boxes_cir_c = center_size(self.prev_candidate['box_cir'])
        prev_boxes_cir_c[:, 2:] *= 1.2
        prev_boxes_cir_expand = point_form(prev_boxes_cir_c)
        if self.use_dynamic_mask:
            pred_masks = self.net.ProtoNet.DynamicMaskHead(candidate['proto'].permute(2, 3, 0, 1).contiguous(),
                                                           self.prev_candidate['mask_coeff'],
                                                           self.prev_candidate['box_cir'],
                                                           self.prev_candidate['prior_levels'])
            if not self.cfg.MODEL.MASK_HEADS.LOSS_WITH_DICE_COEFF:
                pred_masks = crop(pred_masks.permute(2, 3, 1, 0).contiguous(), prev_boxes_cir_expand)
                pred_masks = pred_masks.permute(2, 3, 0, 1).contiguous()
        else:
            pred_masks = generate_mask(candidate['proto'], self.prev_candidate['mask_coeff'],
                                       prev_boxes_cir_expand).transpose(0, 1)
        self.prev_candidate['mask'] = pred_masks
        self.prev_candidate['score'] *= 0.8 ** self.num_clip_frames
        self.prev_candidate['box_cir'] = self.prev_candidate['box'][-1].clone()
        self.prev_candidate['box'] = self.prev_candidate['box'][-1].unsqueeze(0).repeat(self.num_clip_frames, 1, 1)
        for k, v in candidate.items():
            if k in self.img_level_keys:
                self.prev_candidate[k] = v.clone()
        self.prev_candidate['tracked_mask'] += 1
    