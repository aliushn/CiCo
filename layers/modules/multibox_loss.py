# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers.utils import decode, jaccard, generate_mask, compute_kl_div, generate_track_gaussian, \
    log_sum_exp, DIoU_loss
from .match_samples import match, match_clip
from .losses import LincombMaskLoss, T2SLoss
from fvcore.nn import sigmoid_focal_loss_jit, giou_loss
import random


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, cfg, net, num_classes, pos_threshold, neg_threshold):
        super(MultiBoxLoss, self).__init__()
        self.cfg = cfg
        self.net = net
        self.num_classes = num_classes if self.cfg.MODEL.CLASS_HEADS.USE_FOCAL_LOSS else num_classes + 1
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.clip_frames = cfg.SOLVER.NUM_CLIP_FRAMES
        self.use_cir_boxes = self.cfg.MODEL.PREDICTION_HEADS.CIRCUMSCRIBED_BOXES
        self.expand_proposals_clip = cfg.STR.ST_CONSISTENCY.EXPAND_PROPOSALS_CLIP

        self.T2SLoss = T2SLoss(cfg, net)
        self.LincombMaskLoss = LincombMaskLoss(cfg, net)

    def forward(self, predictions, gt_boxes, gt_labels, gt_masks, gt_ids, num_crowds):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            mask preds, and prior boxes from SSD net.
                loc shape: torch.size(batch_size,num_priors,4)
                conf shape: torch.size(batch_size,num_priors,num_classes)
                masks shape: torch.size(batch_size,num_priors,mask_dim)
                priors shape: torch.size(num_priors,4)
                proto* shape: torch.size(batch_size,mask_h,mask_w,mask_dim)

            gt_boxes (list<tensor>): Ground truth boxes for a batch,
                shape: [batch_size][num_objs,4].
            
            gt_labels (list<tensor>): Ground truth labels for a batch,
                shape: [batch_size][num_objs,1].

            gt_masks (list<tensor>): Ground truth masks for each object in each image,
                shape: [batch_size][num_objs,im_height,im_width]

            gt_ids (list<tensor>): Ground truth ids for each object in each image,
                shape: [batch_size][num_objs,1]

            num_crowds (list<int>): Number of crowd annotations per batch. The crowd
                annotations should be the last num_crowds elements of targets and masks.

            * Only if mask_type == lincomb
        """

        if self.cfg.DATASETS.TYPE not in {'coco', 'det'}:
            # In video domain, we load GT annotations as [n_objs, n_clips*n_clip_frames, ...],
            # So we have to unfold them in the temporal dim (2-th) as [n_objs, n_clips, n_clip_frames, ...]
            if self.cfg.SOLVER.NUM_CLIPS > 1:
                gt_boxes = sum([split_squeeze(boxes, length=self.cfg.SOLVER.NUM_CLIP_FRAMES, dim=1) for boxes in gt_boxes], [])
                if None not in gt_masks:
                    gt_masks = sum([split_squeeze(masks, length=self.cfg.SOLVER.NUM_CLIP_FRAMES, dim=1) for masks in gt_masks], [])
                gt_labels = sum([[labels]*self.cfg.SOLVER.NUM_CLIPS for labels in gt_labels], [])
                gt_ids = sum([[ids]*self.cfg.SOLVER.NUM_CLIPS for ids in gt_ids], [])
                num_crowds = sum([[num]*self.cfg.SOLVER.NUM_CLIPS for num in num_crowds], [])

        losses, ids_t = self.multibox_loss(predictions, gt_boxes, gt_labels, gt_masks, gt_ids, num_crowds)
        for k, v in losses.items():
            if torch.isinf(v) or torch.isnan(v):
                print(k)

        return losses

    def multibox_loss(self, predictions, gt_boxes, gt_labels, gt_masks, gt_ids, num_crowds):
        # ----------------------------------  Prepare Data ----------------------------------
        loc_data = predictions['loc']
        batch_size, num_priors, box_dim = loc_data.size()
        centerness_data = predictions['centerness'] if self.cfg.MODEL.BOX_HEADS.TRAIN_CENTERNESS else None
        track_data = predictions['track'] if self.cfg.MODEL.TRACK_HEADS.TRAIN_TRACK else None
        priors, prior_levels = predictions['priors'], predictions['prior_levels']
        if self.cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE and not self.cfg.MODEL.PREDICTION_HEADS.CIRCUMSCRIBED_BOXES:
            priors = priors.repeat(1, 1, self.clip_frames)
            if self.expand_proposals_clip:
                priors = priors.repeat(batch_size//len(gt_boxes), 1, 1)

        # Match priors (default boxes) and ground truth boxes
        # These tensors will be created with the same device as loc_data
        loc_t = loc_data.new(loc_data.shape)
        conf_t = loc_data.new(batch_size, num_priors).long()
        idx_t = loc_data.new(batch_size, num_priors).long()
        ids_t = loc_data.new(batch_size, num_priors).long() if gt_ids is not None else None  # object ids for tracking

        # Matcher: assign positive samples
        crowd_boxes = None
        for idx in range(len(gt_boxes)):
            # Split the crowd annotations because they come bundled in
            if num_crowds is not None:
                cur_crowds = num_crowds[idx]
                if cur_crowds > 0:
                    split = lambda x: (x[-cur_crowds:], x[:-cur_crowds])
                    crowd_boxes, gt_boxes[idx] = split(gt_boxes[idx])

                    # We don't use the crowd labels or masks
                    _, gt_labels[idx] = split(gt_labels[idx])
                    _, gt_masks[idx] = split(gt_masks[idx])

            if self.cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE:
                if self.expand_proposals_clip:
                    for jdx in range(self.clip_frames):
                        kdx = idx*self.clip_frames + jdx
                        gt_boxes[idx] = match_clip(gt_boxes[idx], gt_labels[idx], gt_ids[idx], priors[idx], loc_data[kdx],
                                                   loc_t, conf_t, idx_t, ids_t, idx, self.pos_threshold, self.neg_threshold,
                                                   use_cir_boxes=self.use_cir_boxes, jdx=jdx)
                else:
                    gt_boxes[idx] = match_clip(gt_boxes[idx], gt_labels[idx], gt_ids[idx], priors[idx], loc_data[idx],
                                               loc_t, conf_t, idx_t, ids_t, idx, self.pos_threshold, self.neg_threshold,
                                               use_cir_boxes=self.use_cir_boxes)
            else:
                gt_ids_cur = gt_ids[idx] if gt_ids is not None else None
                match(self.cfg, gt_boxes[idx], gt_labels[idx], gt_ids_cur, crowd_boxes, priors[idx], loc_data[idx],
                      loc_t, conf_t, idx_t, ids_t, idx, self.pos_threshold, self.neg_threshold)

        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        idx_t = Variable(idx_t, requires_grad=False)
        if gt_ids is not None:
            ids_t = Variable(ids_t, requires_grad=False)

        pos = conf_t > 0
        losses = dict()
        if pos.sum() > 0:
            # ---------------------------  Localization Loss (Smooth L1) -------------------------------------------
            losses_loc, pred_boxes_p, gt_boxes_p = self.loclization_loss(pos, priors, loc_data, loc_t, centerness_data)
            losses.update(losses_loc)
            
            # Split boxes of positive samples frame-by-frame to form a list
            num_objs_per_frame = pos.sum(dim=1).reshape(len(gt_boxes), -1).sum(-1).tolist()
            pred_boxes_p_split = torch.split(pred_boxes_p.reshape(-1, box_dim), num_objs_per_frame, dim=0)
            pred_boxes_p_split = [box.reshape(box.size(0), -1) for box in pred_boxes_p_split]
            gt_boxes_p_split = torch.split(gt_boxes_p.reshape(-1, box_dim), num_objs_per_frame, dim=0)
            gt_boxes_p_split = [box.reshape(box.size(0), -1) for box in gt_boxes_p_split]

            # --------------------------------  Mask Loss  -----------------------------------------------------
            if self.cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
                loss_m = self.LincombMaskLoss(pos, idx_t, pred_boxes_p_split, gt_boxes_p_split, predictions['mask_coeff'],
                                              prior_levels, predictions['proto'], gt_masks, gt_boxes, gt_ids)
                losses.update(loss_m)

                # These losses also don't depend on anchors
                if self.cfg.MODEL.MASK_HEADS.USE_SEMANTIC_SEGMENTATION_LOSS:
                    losses['S'] = self.semantic_segmentation_loss(predictions['sem_seg'], gt_masks, gt_labels)

            # --------------------------------- Confidence loss ------------------------------------------------
            # Focal loss for COCO (crowded objects > 10),
            # Ohem loss for VIS dataset (sparse objects < 6, tooo many negative samples)
            if self.cfg.MODEL.CLASS_HEADS.TRAIN_CLASS:
                conf_t_unfold = conf_t.reshape(-1)
                pos_unfold = pos.float().reshape(-1)
                if self.cfg.MODEL.CLASS_HEADS.TRAIN_INTERCLIPS_CLASS:
                    # Stuff loss
                    stuff_data_unfold = torch.flatten(predictions['conf'])
                    stuff_target = pos_unfold.float()
                    if self.cfg.MODEL.CLASS_HEADS.USE_FOCAL_LOSS:
                        # filter out those samples with IoU threshold between 0.4 and 0.5
                        keep = conf_t_unfold != -1
                        stuff_loss = sigmoid_focal_loss_jit(
                            stuff_data_unfold[keep],
                            stuff_target[keep],
                            alpha=self.cfg.MODEL.CLASS_HEADS.FOCAL_LOSS_ALPHA,
                            gamma=self.cfg.MODEL.CLASS_HEADS.FOCAL_LOSS_GAMMA,
                            reduction='sum')
                        stuff_loss = stuff_loss / pos_unfold.sum()
                    else:
                        neg = self.select_neg_bboxes(stuff_data_unfold, conf_t_unfold, ratio=3, type='stuff')
                        keep = (pos_unfold + neg).gt(0)
                        stuff_loss = F.binary_cross_entropy_with_logits(stuff_data_unfold[keep], stuff_target[keep],
                                                                        reduction='mean')
                    losses['stuff'] = self.cfg.MODEL.CLASS_HEADS.LOSS_ALPHA * stuff_loss

                    # InterClips classification loss or global classification
                    intercla_loss = self.interclips_classification_loss(pos, gt_labels, predictions['conf_feat'], idx_t)
                    losses['CI'] = self.cfg.MODEL.CLASS_HEADS.LOSS_ALPHA * intercla_loss

                else:
                    conf_data_unfold = torch.flatten(predictions['conf'], end_dim=-2)

                    if self.cfg.MODEL.CLASS_HEADS.USE_FOCAL_LOSS:
                        pos_inds = torch.nonzero(conf_t_unfold > 0, as_tuple=False)
                        # prepare one_hot
                        class_target = torch.zeros_like(conf_data_unfold)
                        class_target[pos_inds, conf_t_unfold[pos_inds] - 1] = 1
                        # filter out those samples with IoU threshold between 0.4 and 0.5
                        keep = conf_t_unfold != -1
                        class_loss = sigmoid_focal_loss_jit(
                            conf_data_unfold[keep],
                            class_target[keep],
                            alpha=self.cfg.MODEL.CLASS_HEADS.FOCAL_LOSS_ALPHA,
                            gamma=self.cfg.MODEL.CLASS_HEADS.FOCAL_LOSS_GAMMA,
                            reduction='sum')
                        losses['C_focal'] = self.cfg.MODEL.CLASS_HEADS.LOSS_ALPHA * class_loss / len(pos_inds)
                    else:
                        neg = self.select_neg_bboxes(conf_data_unfold, conf_t_unfold, ratio=3)
                        keep = (pos_unfold + neg).gt(0)
                        class_loss = F.cross_entropy(conf_data_unfold[keep], conf_t_unfold[keep], reduction='mean')
                        losses['C'] = self.cfg.MODEL.CLASS_HEADS.LOSS_ALPHA * class_loss

            # --------------------------------- Tracking loss ------------------------------------------------
            if self.cfg.MODEL.TRACK_HEADS.TRAIN_TRACK:
                if self.cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
                    losses_track = self.track_gauss_loss(track_data, gt_masks, gt_ids, predictions['proto'],
                                                         predictions['mask_coeff'], gt_boxes, pos, ids_t, idx_t)
                else:
                    losses_track = self.track_loss(track_data, pos, ids_t)
                losses.update(losses_track)

            # ------------------------------- Track to segment -----------------------------------------------
            if self.cfg.MODEL.TRACK_HEADS.TRAIN_TRACK and self.cfg.STMASK.T2S_HEADS.TEMPORAL_FUSION_MODULE:
                losses_t2s = self.T2SLoss(predictions['fpn_feat'], gt_boxes, gt_masks, predictions['proto'])
                losses.update(losses_t2s)

        return losses, ids_t

    def select_neg_bboxes(self, conf_data, conf_t, ratio=3, use_most_confident=True, type='conf'):
        '''
        :param conf_data: [n, num_classes] or [n, 1]
        :param conf_t: [n, 1]
        :param type: 'class' or 'stuff'
        :return:
        '''

        if type == 'conf':
            # Compute max conf across batch for hard negative mining
            if use_most_confident:
                # i.e. max(softmax) along classes > 0
                conf_data = F.softmax(conf_data, dim=1)
                loss_c, _ = conf_data[:, 1:].max(dim=1)
            else:
                loss_c = log_sum_exp(conf_data) - conf_data[:, 0]
        else:
            loss_c = conf_data.sigmoid()

        # Hard Negative Mining
        num_pos = (conf_t > 0).sum()
        num_neg = min(ratio * num_pos, conf_t.size(0) - 1)
        loss_c[conf_t > 0] = 0  # filter out pos samples and neutrals
        loss_c[conf_t < 0] = 0  # filter out pos samples and neutrals
        loss_c_sorted, loss_idx = loss_c.sort(descending=True)
        neg = torch.zeros_like(conf_t)
        neg[loss_idx[:num_neg]] = 1
        # print(loss_idx[:num_neg], loss_c_sorted[:num_neg])
        # Just in case there aren't enough negatives, don't start using positives as negatives
        neg[conf_t > 0] = 0  # filter out pos samplers and neutrals
        neg[conf_t < 0] = 0  # filter out pos samplers and neutrals

        return neg
    
    def loclization_loss(self, pos, priors, loc_data, loc_t, centerness_data):
        loc_p = loc_data[pos].reshape(-1, 4)
        loc_t = loc_t[pos].reshape(-1, 4)
        pos_priors = priors[pos].reshape(-1, 4)
        box_loss = F.smooth_l1_loss(loc_p, loc_t, reduction='none').sum(dim=-1)
        losses = {'B': box_loss.mean() * self.cfg.MODEL.BOX_HEADS.LOSS_ALPHA}

        decoded_loc_p = decode(loc_p, pos_priors)
        decoded_loc_t = decode(loc_t, pos_priors)
        if self.cfg.MODEL.PREDICTION_HEADS.USE_DIoU:
            IoU_loss = DIoU_loss(decoded_loc_p, decoded_loc_t, reduction='none')
        else:
            IoU_loss = giou_loss(decoded_loc_p, decoded_loc_t, reduction='none')
        IoU_loss = IoU_loss.reshape(pos.sum(), -1).mean(-1)

        if self.cfg.MODEL.BOX_HEADS.USE_BOXIOU_LOSS:
            losses['BIoU'] = IoU_loss.mean() * self.cfg.MODEL.BOX_HEADS.BIoU_ALPHA

        if centerness_data is not None:
            loss_cn = F.binary_cross_entropy(centerness_data[pos].reshape(-1),
                                             (1 - IoU_loss).detach(), reduction='mean')
            # loss_cn = F.smooth_l1_loss(centerness_data[pos].view(-1), DIoU, reduction='mean')
            losses['center'] = self.cfg.MODEL.BOX_HEADS.CENTERNESS_ALPHA * loss_cn
            
        return losses, decoded_loc_p, decoded_loc_t

    def repulsion_loss(self, pos, idx_t, decoded_loc_p, gt_boxes):
        batch_size = pos.size(0)
        rep_loss = torch.tensor(0.0, device=pos.device)
        n = 0.

        for idx in range(batch_size):
            n_gt_cur = gt_boxes[idx].size(0)
            n_pos_cur = decoded_loc_p.size(0)
            # the overlaps between predicted boxes and ground-truth boxes
            overlaps_cur = jaccard(decoded_loc_p, gt_boxes[idx]).detach()  # [n_pos, n_gt]

            # matched ground-truth bounding box idx with each predicted boxes
            idx_t_cur = idx_t[idx][pos[idx]]
            # to find surrounding boxes, we assign the overlaps of matched idx as -1
            for j in range(pos[idx].sum()):
                overlaps_cur[j, idx_t_cur[j]] = -1

            # Crowd occlusion: BIoU greater than 0.1
            keep = (overlaps_cur > 0.1).view(-1)
            if len(torch.nonzero(keep)) > 0:
                n += 1
                rep_decoded_loc_p = decoded_loc_p.unsqueeze(1).repeat(1, n_gt_cur, 1).view(-1, 4)[keep]
                rep_gt_boxes = gt_boxes[idx].unsqueeze(0).repeat(n_pos_cur, 1, 1).view(-1, 4)[keep]
                rep_IoU = 1. - giou_loss(rep_decoded_loc_p, rep_gt_boxes, reduction='none')
                rep_loss += rep_IoU.mean() * self.cfg.MODEL.BOX_HEADS.BIoU_ALPHA

        return rep_loss / max(n, 1.)

    def track_loss(self, track_data, pos, ids_t=None):
        if self.cfg.DATASETS.TYPE == 'cocovis':
            n_clips = self.cfg.SOLVER.NUM_CLIPS
        else:
            n_clips = track_data.size(0) if self.cfg.SOLVER.NUM_CLIPS == 1 else self.cfg.SOLVER.NUM_CLIPS
        bs = track_data.size(0) // n_clips

        loss = torch.tensor(0., device=track_data.device)
        for i in range(bs):
            pos_cur = pos[i*n_clips:(i+1)*n_clips]
            pos_track_data = track_data[i*n_clips:(i+1)*n_clips][pos_cur]
            cos_sim = pos_track_data @ pos_track_data.t()  # [n_pos, n_ref_pos]
            # Rescale to be between 0 and 1
            cos_sim = (cos_sim + 1) / 2

            pos_ids_t = ids_t[i*n_clips:(i+1)*n_clips][pos_cur]
            inst_eq = (pos_ids_t.view(-1, 1) == pos_ids_t.view(1, -1)).float()

            # If they are the same instance, use cosine distance, else use consine similarity
            # loss += ((1 - cos_sim) * inst_eq + cos_sim * (1 - inst_eq)).sum() / (cos_sim.size(0) * cos_sim.size(1))
            # pos: -log(cos_sim), neg: -log(1-cos_sim)
            cos_sim_diff = torch.clamp(1 - cos_sim, min=1e-10)
            loss_t = -1 * (inst_eq * torch.clamp(cos_sim, min=1e-10).log() + (1 - inst_eq) * cos_sim_diff.log())
            loss += loss_t.mean()

        return {'T': loss / bs * self.cfg.MODEL.TRACK_HEADS.LOSS_ALPHA}

    def track_gauss_loss(self, track_data, gt_masks, gt_ids, proto_data, mask_coeff, gt_boxes, pos, ids_t, idx_t):

        def _prepare_masks_for_track(cfg, pos, ids_t, idx_t, gt_boxes, proto_data, mask_coeff, gt_masks, gt_ids):
            if cfg.MODEL.TRACK_HEADS.CROP_WITH_PRED_MASK:
                pos_ids_t = ids_t[pos]
                # get GT bounding boxes for cropping pred masks
                pos_bboxes = gt_boxes[idx_t[pos]]
                pos_masks = generate_mask(proto_data, mask_coeff[pos], pos_bboxes)
            else:
                pos_ids_t = gt_ids
                pos_masks = gt_masks
                pos_bboxes = gt_boxes
            return pos_masks, pos_bboxes, pos_ids_t

        bs, track_h, track_w, t_dim = track_data.size()
        loss = torch.tensor(0., device=track_data.device)
        n_bs_track, n_clip = 0, 2
        for i in range(bs//n_clip):
            mu, var, obj_ids = [], [], []

            for j in range(n_clip):
                masks_cur, bbox_cur, ids_t_cur = _prepare_masks_for_track(self.cfg, pos[2*i+j],
                                                                          ids_t[2*i+j], idx_t[2*i+j],
                                                                          gt_boxes[2*i+j], proto_data[2*i+j],
                                                                          mask_coeff[2*i+j], gt_masks[2*i+j],
                                                                          gt_ids[2*i+j])
                mu_cur, var_cur = generate_track_gaussian(track_data[2 * i + j], masks_cur, bbox_cur)
                mu.append(mu_cur)
                var.append(var_cur)
                obj_ids.append(ids_t_cur)

            mu, var, obj_ids = torch.cat(mu, dim=0), torch.cat(var, dim=0), torch.cat(obj_ids, dim=0)

            if len(obj_ids) > 1:
                n_bs_track += 1

                # to calculate the kl_divergence for two Gaussian distributions, where c is the dim of variables
                kl_divergence = compute_kl_div(mu, var) + 1e-5  # value in [1e-5, +infinite]

                # We hope the kl_divergence between same instance Ids is small, otherwise the kl_divergence is large.
                inst_eq = (obj_ids.view(-1, 1) == obj_ids.view(1, -1)).float()

                # If they are the same instance, use cosine distance, else use consine similarity
                # pos: log(1+kl), neg: exp(1-kl)
                pre_loss = inst_eq * (1 + 2 * kl_divergence).log() \
                           + (1. - inst_eq) * torch.exp(1 - 0.1 * kl_divergence)
                loss += pre_loss.mean()

        losses = {'T': self.cfg.MODEL.TRACK_HEADS.LOSS_ALPHA * loss / max(n_bs_track, 1)}

        return losses

    def interclips_classification_loss(self, pos, gt_labels, conf_feats, idx_t):
        '''
        :param pos: [bs*n_clips, n_priors]
        :param gt_labels: List[label_1, ...], the total length is n_clips
        :param conf_feats: [bs*n_clips, n_priors, n_clip_frames*d]
        :return:
        '''
        n_clips = self.cfg.SOLVER.NUM_CLIPS
        pos_list = split_squeeze(pos, length=n_clips, dim=0)
        idx_t_list = split_squeeze(idx_t, length=n_clips, dim=0)
        conf_feats_list = split_squeeze(conf_feats, length=n_clips, dim=0)
        bs = len(conf_feats_list)

        losses = torch.tensor(0., device=pos.device)
        for k in range(bs):
            loss_cur = torch.tensor(0., device=pos.device)
            gt_labels_cur = gt_labels[k*n_clips]
            pos_cur = pos_list[k].reshape(-1)                # [n_clips, n_priors]
            idx_t_cur = idx_t_list[k].reshape(-1)[pos_cur]            # [n_clips, n_priors]
            conf_feats_cur = conf_feats_list[k].reshape(-1, conf_feats.size(-1))[pos_cur]

            # Find all positive samples for each instance clip by clip
            for i, label in enumerate(gt_labels_cur):
                keep = idx_t_cur == i
                conf_feats_cur_obj = conf_feats_cur[keep]
                n_samples = keep.sum()
                iters = max(n_samples//n_clips, 1)
                selected_idx = torch.randint(n_samples, (iters, n_clips))
                conf_data = self.net.InterclipsClass(conf_feats_cur_obj[selected_idx.reshape(-1)].reshape(iters, -1))
                loss_cur += F.cross_entropy(conf_data, (label-1).reshape(-1).repeat(iters))

            losses += loss_cur / max(len(gt_labels_cur), 1)
        return losses / bs

    def _mask_iou(self, mask1, mask2):
        intersection = torch.sum(mask1*mask2, dim=(1, 2))
        area1 = torch.sum(mask1, dim=(1, 2))
        area2 = torch.sum(mask2, dim=(1, 2))
        union = (area1 + area2) - intersection
        ret = intersection / union
        return ret

    def semantic_segmentation_loss(self, segment_data, mask_t, class_t, interpolation_mode='bilinear', focal_loss=False):
        '''
        :param segment_data: [bs, h, w, num_class]
        :param mask_t: a list of groundtruth masks
        :param class_t: a list of groundtruth clsses: begin with 1
        :param interpolation_mode:
        :return:
        '''
        # Note num_classes here is without the background class so self.cfg.num_classes-1
        segment_data = segment_data.permute(0, 3, 1, 2).contiguous()
        bs, _, mask_h, mask_w = segment_data.size()
        sem_loss = 0
        # void out of memory so as to calcuate loss for a single image
        for idx in range(bs):
            mask_t_downsample = F.interpolate(mask_t[idx].float().unsqueeze(0), (mask_h, mask_w),
                                              mode=interpolation_mode, align_corners=False).squeeze(0).gt_(0.5)

            # prepare one-hat
            segment_t = torch.zeros_like(segment_data[idx])
            with torch.no_grad():
                for obj_idx, obj_mask_t in enumerate(mask_t_downsample):
                    obj_class = (class_t[idx][obj_idx]-1).long()
                    segment_t[obj_class][obj_mask_t == 1] = 1

            if focal_loss:
                # avoid out of memory so as to calculate semantic loss for a single image
                pre_sem_loss = sigmoid_focal_loss_jit(
                    segment_data[idx],
                    segment_t,
                    alpha=self.cfg.MODEL.CLASS_HEADS.FOCAL_LOSS_ALPHA,
                    gamma=self.cfg.MODEL.CLASS_HEADS.FOCAL_LOSS_GAMMA,
                    reduction="sum"
                )
                sem_loss += pre_sem_loss / torch.clamp(segment_t.sum(), min=1)

            else:
                pre_sem_loss = F.binary_cross_entropy_with_logits(segment_data[idx], segment_t, reduction='sum')
                sem_loss += pre_sem_loss / mask_h / mask_w * 0.1

        return sem_loss / float(bs) * self.cfg.MODEL.MASK_HEADS.USE_SEMANTIC_SEGMENTATION_LOSS


def split_squeeze(x, length=1, dim=-1):
    x_list = list(torch.split(x, length, dim=dim))
    for i in range(len(x_list)):
        x_list[i] = x_list[i].squeeze(dim)
    return x_list
