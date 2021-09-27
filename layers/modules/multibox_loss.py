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
            # In video domain, we load GT annotations as [n_objs, n_clip_frames, ...],
            # So we have to unfold them in the temporal dim (2-th) to do frame-level detection or segmentation
            clip_frames = gt_boxes[0].size(1)
            if self.cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE:
                if not self.cfg.MODEL.PREDICTION_HEADS.CIRCUMSCRIBED_BOXES:
                    predictions['priors'] = predictions['priors'].repeat(1, 1, clip_frames)
            else:
                gt_boxes = sum([split_squeeze(boxes, dim=1) for boxes in gt_boxes], [])
                if None not in gt_masks:
                    gt_masks = sum([split_squeeze(masks, dim=1) for masks in gt_masks], [])
                gt_labels = sum([[labels]*clip_frames for labels in gt_labels], [])
                gt_ids = sum([[ids]*clip_frames for ids in gt_ids], [])
                num_crowds = sum([[num]*clip_frames for num in num_crowds], [])

        losses, ids_t = self.multibox_loss(predictions, gt_boxes, gt_labels, gt_masks, gt_ids, num_crowds)
        for k, v in losses.items():
            if torch.isinf(v) or torch.isnan(v):
                print(k)

        return losses

    def multibox_loss(self, predictions, gt_boxes, gt_labels, gt_masks, gt_ids, num_crowds):
        # ----------------------------------  Prepare Data ----------------------------------
        loc_data = predictions['loc']
        centerness_data = predictions['centerness'] if self.cfg.MODEL.BOX_HEADS.TRAIN_CENTERNESS else None
        conf_data = predictions['conf']
        mask_coeff, proto_data, segm_data = None, None, None
        if self.cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
            mask_coeff, proto_data = predictions['mask_coeff'], predictions['proto']
            if not self.cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE:
                proto_data = [list(torch.split(proto_data[cdx], 1, dim=-2)) for cdx in range(proto_data.size(0))]
                proto_data = torch.stack(sum(proto_data, []), dim=0)
            if self.cfg.MODEL.MASK_HEADS.USE_SEMANTIC_SEGMENTATION_LOSS:
                segm_data = predictions['sem_seg']
        track_data = predictions['track'] if self.cfg.MODEL.TRACK_HEADS.TRAIN_TRACK else None
        priors = predictions['priors']
        prior_levels = predictions['prior_levels']
        batch_size, num_priors, box_dim = loc_data.size()

        # Match priors (default boxes) and ground truth boxes
        # These tensors will be created with the same device as loc_data
        loc_t = loc_data.new(loc_data.shape)
        conf_t = loc_data.new(batch_size, num_priors).long()
        idx_t = loc_data.new(batch_size, num_priors).long()
        ids_t = loc_data.new(batch_size, num_priors).long() if gt_ids is not None else None  # object ids for tracking

        # Matcher: assign positive samples
        crowd_boxes = None
        for idx in range(batch_size):
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
                gt_boxes[idx] = match_clip(self.cfg, gt_boxes[idx], gt_labels[idx], gt_ids[idx], priors[idx], loc_data[idx],
                                           loc_t, conf_t, idx_t, ids_t, idx, self.pos_threshold, self.neg_threshold,
                                           circumscribed_boxes=self.cfg.MODEL.PREDICTION_HEADS.CIRCUMSCRIBED_BOXES)
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
        losses = {}
        if pos.sum() > 0:
            # ---------------------------  Localization Loss (Smooth L1) -------------------------------------------
            losses_loc, pred_boxes_p, gt_boxes_p = self.loclization_loss(pos, priors, loc_data, loc_t, centerness_data)
            losses.update(losses_loc)
            
            # Split boxes of positive samples frame-by-frame to form a list
            num_objs_per_frame = pos.sum(dim=1).reshape(-1).tolist()
            pred_boxes_p_split = torch.split(pred_boxes_p.reshape(-1, box_dim), num_objs_per_frame, dim=0)
            pred_boxes_p_split = [box.reshape(box.size(0), -1) for box in pred_boxes_p_split]
            gt_boxes_p_split = torch.split(gt_boxes_p.reshape(-1, box_dim), num_objs_per_frame, dim=0)
            gt_boxes_p_split = [box.reshape(box.size(0), -1) for box in gt_boxes_p_split]

            # --------------------------------  Mask Loss  -----------------------------------------------------
            if self.cfg.MODEL.MASK_HEADS.TRAIN_MASKS:
                loss_m = self.LincombMaskLoss(pos, idx_t, pred_boxes_p_split, mask_coeff, prior_levels,
                                              proto_data, gt_masks, gt_boxes_p_split, gt_ids)
                losses.update(loss_m)

                # These losses also don't depend on anchors
                if self.cfg.MODEL.MASK_HEADS.USE_SEMANTIC_SEGMENTATION_LOSS:
                    losses['S'] = self.semantic_segmentation_loss(segm_data, gt_masks, gt_labels)

            # --------------------------------- Confidence loss ------------------------------------------------
            # Focal loss for COCO (crowded objects > 10),
            # ohem loss for VIS dataset (sparse objects < 6, tooo many negative samples)
            if self.cfg.MODEL.CLASS_HEADS.TRAIN_CLASS:
                conf_data_unfold = conf_data.reshape(-1, self.num_classes)
                conf_t_unfold = conf_t.reshape(-1)

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
                    keep = (pos.reshape(-1) + neg).gt(0)
                    class_loss = F.cross_entropy(conf_data_unfold[keep], conf_t_unfold[keep], reduction='mean')
                    losses['C'] = self.cfg.MODEL.CLASS_HEADS.LOSS_ALPHA * class_loss

            # --------------------------------- Tracking loss ------------------------------------------------
            if self.cfg.MODEL.TRACK_HEADS.TRAIN_TRACK:
                if self.cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
                    losses_track = self.track_gauss_loss(track_data, gt_masks, gt_ids, proto_data, mask_coeff,
                                                         gt_boxes, pos, ids_t, idx_t)
                else:
                    losses_track = self.track_loss(track_data, pos, ids_t)
                losses.update(losses_track)

            # ------------------------------- Track to segment -----------------------------------------------
            if self.cfg.MODEL.TRACK_HEADS.TRAIN_TRACK and self.cfg.STMASK.T2S_HEADS.TEMPORAL_FUSION_MODULE:
                losses_t2s = self.T2SLoss(predictions['fpn_feat'], gt_boxes, gt_masks, proto_data)
                losses.update(losses_t2s)

        return losses, ids_t

    def select_neg_bboxes(self, conf_data, conf_t, ratio=3, use_most_confident=True):
        '''
        :param conf_data: [n, num_classes]
        :param conf_t: [n, 1]
        :param type: 'class' or 'stuff'
        :return:
        '''

        # Compute max conf across batch for hard negative mining
        if use_most_confident:
            # i.e. max(softmax) along classes > 0
            conf_data = F.softmax(conf_data, dim=1)
            loss_c, _ = conf_data[:, 1:].max(dim=1)
        else:
            loss_c = log_sum_exp(conf_data) - conf_data[:, 0]

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
        losses = dict()
        loc_p = loc_data[pos].reshape(-1, 4)
        loc_t = loc_t[pos].reshape(-1, 4)
        pos_priors = priors[pos].reshape(-1, 4)
        box_loss = F.smooth_l1_loss(loc_p, loc_t, reduction='none').sum(dim=-1)
        losses['B'] = box_loss.mean() * self.cfg.MODEL.BOX_HEADS.LOSS_ALPHA

        decoded_loc_p = decode(loc_p, pos_priors)
        decoded_loc_t = decode(loc_t, pos_priors)
        if self.cfg.MODEL.PREDICTION_HEADS.USE_DIoU:
            IoU_loss = DIoU_loss(decoded_loc_p, decoded_loc_t, reduction='none')
        else:
            IoU_loss = giou_loss(decoded_loc_p, decoded_loc_t, reduction='none')
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
        clip_frames = pos.size(0) if self.cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE else self.cfg.SOLVER.NUM_CLIP_FRAMES
        n_clips = pos.size(0) // clip_frames

        loss = torch.tensor(0., device=track_data.device)
        for i in range(n_clips):
            pos_cur = pos[i*clip_frames:(i+1)*clip_frames]
            pos_track_data = track_data[i*clip_frames:(i+1)*clip_frames][pos_cur]
            cos_sim = pos_track_data @ pos_track_data.t()  # [n_pos, n_ref_pos]
            # Rescale to be between 0 and 1
            cos_sim = (cos_sim + 1) / 2
            cos_sim.triu_(diagonal=1)

            pos_ids_t = ids_t[i*clip_frames:(i+1)*clip_frames][pos_cur]
            inst_eq = (pos_ids_t.view(-1, 1) == pos_ids_t.view(1, -1)).float()

            # If they are the same instance, use cosine distance, else use consine similarity
            # loss += ((1 - cos_sim) * inst_eq + cos_sim * (1 - inst_eq)).sum() / (cos_sim.size(0) * cos_sim.size(1))
            # pos: -log(cos_sim), neg: -log(1-cos_sim)
            cos_sim_diff = torch.clamp(1 - cos_sim, min=1e-10)
            loss_t = -1 * (inst_eq * torch.clamp(cos_sim, min=1e-10).log() + (1 - inst_eq) * cos_sim_diff.log())
            loss += loss_t.triu_(diagonal=1).sum() / max(len(pos_ids_t), 1) / max(len(pos_ids_t)-1, 1)

        return {'T': loss / n_clips * self.cfg.MODEL.TRACK_HEADS.LOSS_ALPHA}

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


def split_squeeze(x, dim=-1):
    x_list = list(torch.split(x, 1, dim=dim))
    for i in range(len(x_list)):
        x_list[i] = x_list[i].squeeze(dim)
    return x_list
