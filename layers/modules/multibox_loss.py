# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers.utils import match, compute_DIoU, encode, decode, center_size, crop, jaccard, point_form, \
    generate_mask, compute_kl_div, generate_track_gaussian, sanitize_coordinates_hw
from .track_to_segment_head import bbox_feat_extractor
from layers.utils.track_utils import correlate_operator
from layers.visualization import display_correlation_map

from datasets import cfg, activation_func
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

    def __init__(self, net, num_classes, pos_threshold, neg_threshold):
        super(MultiBoxLoss, self).__init__()
        self.net = net
        self.num_classes = num_classes

        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    def forward(self, imgs, predictions, gt_bboxes, gt_labels, gt_masks, gt_ids, num_crowds):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            mask preds, and prior boxes from SSD net.
                loc shape: torch.size(batch_size,num_priors,4)
                conf shape: torch.size(batch_size,num_priors,num_classes)
                masks shape: torch.size(batch_size,num_priors,mask_dim)
                priors shape: torch.size(num_priors,4)
                proto* shape: torch.size(batch_size,mask_h,mask_w,mask_dim)

            targets (list<tensor>): Ground truth boxes and labels for a batch,
                shape: [batch_size][num_objs,5] (last idx is the label).

            masks (list<tensor>): Ground truth masks for each object in each image,
                shape: [batch_size][num_objs,im_height,im_width]

            num_crowds (list<int>): Number of crowd annotations per batch. The crowd
                annotations should be the last num_crowds elements of targets and masks.

            * Only if mask_type == lincomb
        """

        loc_data = predictions['loc']
        centerness_data = predictions['centerness'] if cfg.train_centerness else None

        if cfg.train_class:
            conf_data = predictions['conf']
            stuff_data = None
        else:
            conf_data = None
            stuff_data = predictions['stuff']
        mask_coeff = predictions['mask_coeff']
        track_data = F.normalize(predictions['track'], dim=-1) if cfg.train_track else None

        # This is necessary for training on multiple GPUs because
        # DataParallel will cat the priors from each GPU together
        priors = predictions['priors']
        prior_levels = predictions['prior_levels']
        proto_data = predictions['proto']
        sem_data = predictions['sem_seg'] if cfg.use_semantic_segmentation_loss else None

        losses, conf_t, ids_t, idx_t = self.multibox_loss(imgs, loc_data, conf_data, stuff_data, mask_coeff,
                                                          centerness_data, proto_data, priors, prior_levels, sem_data,
                                                          gt_bboxes, gt_labels,  gt_masks, gt_ids, num_crowds)

        if cfg.train_track and cfg.use_temporal_info:

            if cfg.temporal_fusion_module:
                gt_bboxes_ref, gt_bboxes_next = gt_bboxes[::2], gt_bboxes[1::2]
                gt_ids_ref, gt_ids_next = gt_ids[::2], gt_ids[1::2]
                gt_masks_ref, gt_masks_next = gt_masks[::2], gt_masks[1::2]
                bb_feat_ref = predictions['fpn_feat'][::2].contiguous()
                bb_feat_next = predictions['fpn_feat'][1::2].contiguous()
                assert cfg.forward_flow or cfg.backward_flow

                if not cfg.maskshift_loss:

                    if cfg.forward_flow:
                        losses_shift_for = self.t2s_single_loss(bb_feat_ref, bb_feat_next,
                                                                gt_bboxes_ref, gt_bboxes_next,
                                                                gt_ids_ref, gt_ids_next)

                    if cfg.backward_flow:
                        # backward flow: corr from next frame to reference frame
                        losses_shift_back = self.t2s_single_loss(bb_feat_next, bb_feat_ref,
                                                                 gt_bboxes_next, gt_bboxes_ref,
                                                                 gt_ids_next, gt_ids_ref)

                else:
                    loc_data_ref = loc_data[::2].detach()
                    loc_data_next = loc_data[1::2].detach()
                    ids_t_ref = ids_t[::2]
                    ids_t_next = ids_t[1::2]
                    mask_coeff_ref = mask_coeff[::2].detach()
                    mask_coeff_next = mask_coeff[1::2].detach()
                    proto_data_ref = proto_data[::2].detach()
                    proto_data_next = proto_data[1::2].detach()
                    if cfg.forward_flow:
                        losses_shift_for = self.track_to_segment_loss(bb_feat_ref, bb_feat_next, loc_data_ref, ids_t_ref,
                                                                      mask_coeff_ref, proto_data_next,
                                                                      priors[0], gt_bboxes_ref, gt_bboxes_next,
                                                                      gt_ids_ref, gt_ids_next, gt_masks_next)

                    if cfg.backward_flow:
                        losses_shift_back = self.track_to_segment_loss(bb_feat_next, bb_feat_ref, loc_data_next, ids_t_next,
                                                                       mask_coeff_next, proto_data_ref,
                                                                       priors[0], gt_bboxes_next, gt_bboxes_ref,
                                                                       gt_ids_next, gt_ids_ref, gt_masks_ref)

                losses_shift = dict()
                if cfg.forward_flow and cfg.backward_flow:
                    for k, v in losses_shift_for.items():
                        losses_shift[k] = 0.5 * (losses_shift_for[k] + losses_shift_back[k])

                elif cfg.forward_flow:
                    for k, v in losses_shift_for.items():
                        losses_shift[k] = v
                else:
                    for k, v in losses_shift_back.items():
                        losses_shift[k] = v

                losses.update(losses_shift)

        if cfg.train_track:
            losses_clip = self.track_loss(track_data, gt_masks, gt_ids, proto_data, mask_coeff,
                                          gt_bboxes, loc_data, priors, conf_t, ids_t, idx_t)
            losses.update(losses_clip)

        for k, v in losses.items():
            if torch.isinf(v) or torch.isnan(v):
                print(k)

        return losses

    def split(self, x):
        x1, x2 = torch.split(x, [1, 1], dim=-1)
        return x1.squeeze(-1), x2.squeeze(-1)

    def multibox_loss(self, imgs, loc_data, conf_data, stuff_data, mask_data, centerness_data, proto_data, priors,
                      prior_levels, segm_data, gt_bboxes, gt_labels, gt_masks, gt_ids, num_crowds):
        batch_size, num_priors = loc_data.size()[0:2]

        # Match priors (default boxes) and ground truth boxes
        # These tensors will be created with the same device as loc_data
        loc_t = loc_data.new(batch_size, num_priors, 4)
        conf_t = loc_data.new(batch_size, num_priors).long()
        idx_t = loc_data.new(batch_size, num_priors).long()

        if gt_ids is not None:
            ids_t = loc_data.new(batch_size, num_priors).long()  # pids for tracking
        else:
            ids_t = None

        # assign positive samples
        crowd_boxes = None
        for idx in range(batch_size):
            # Split the crowd annotations because they come bundled in
            if num_crowds is not None:
                cur_crowds = num_crowds[idx]
                if cur_crowds > 0:
                    split = lambda x: (x[-cur_crowds:], x[:-cur_crowds])
                    crowd_boxes, gt_bboxes[idx] = split(gt_bboxes[idx])

                    # We don't use the crowd labels or masks
                    _, gt_labels[idx] = split(gt_labels[idx])
                    _, gt_masks[idx] = split(gt_masks[idx])

            gt_ids_cur = gt_ids[idx] if gt_ids is not None else None
            match(gt_bboxes[idx], gt_labels[idx], gt_ids_cur, crowd_boxes, priors[idx], loc_data[idx], loc_t, conf_t,
                  idx_t, ids_t, idx, self.pos_threshold, self.neg_threshold)

        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        idx_t = Variable(idx_t, requires_grad=False)
        if gt_ids is not None:
            ids_t = Variable(ids_t, requires_grad=False)

        pos = conf_t > 0
        losses = {}
        if len(torch.nonzero(pos)) > 0:
            # Localization Loss (Smooth L1)
            if cfg.train_boxes:
                loc_p = loc_data[pos].view(-1, 4)
                loc_t = loc_t[pos].view(-1, 4)
                pos_priors = priors[pos].view(-1, 4)

                box_loss = F.smooth_l1_loss(loc_p, loc_t, reduction='none').sum(dim=-1)
                losses['B'] = box_loss.mean() * cfg.bbox_alpha

                decoded_loc_p = torch.clamp(decode(loc_p, pos_priors, cfg.use_yolo_regressors), min=0, max=1)
                decoded_loc_t = torch.clamp(decode(loc_t, pos_priors, cfg.use_yolo_regressors), min=0, max=1)
                if cfg.use_DIoU:
                    IoU_loss = 1. - compute_DIoU(decoded_loc_p, decoded_loc_t).diag()
                else:
                    IoU_loss = giou_loss(decoded_loc_p, decoded_loc_t, reduction='none')

                if cfg.use_boxiou_loss:
                    losses['BIoU'] = IoU_loss.mean() * cfg.BIoU_alpha

                if cfg.use_repulsion_loss:
                    # add repulsion loss for crowd occlusion
                    losses['Rep'] = self.repulsion_loss(pos, idx_t, decoded_loc_p, gt_bboxes)

                if centerness_data is not None:
                    loss_cn = F.binary_cross_entropy(centerness_data[pos].view(-1), (1 - IoU_loss).detach(), reduction='mean')
                    # loss_cn = F.smooth_l1_loss(centerness_data[pos].view(-1), DIoU, reduction='mean')
                    losses['center'] = cfg.center_alpha * loss_cn

            # Mask Loss
            if cfg.train_masks:
                loss_m = self.lincomb_mask_loss(pos, idx_t, loc_data, mask_data, priors, prior_levels, proto_data,
                                                gt_masks, gt_bboxes)
                losses.update(loss_m)

            # Confidence loss
            if cfg.train_class:
                conf_data_f = conf_data.reshape(-1, self.num_classes)
                conf_t_f = conf_t.reshape(-1)
                pos_inds = torch.nonzero((pos.reshape(-1)).float())
                # prepare one_hot
                class_target = torch.zeros_like(conf_data_f)
                class_target[pos_inds, conf_t_f[pos_inds] - 1] = 1

                if cfg.use_focal_loss:
                    # filter out those samples with IoU threshold between 0.4 and 0.5
                    keep = conf_t_f != -1
                    class_loss = sigmoid_focal_loss_jit(
                        conf_data_f[keep],
                        class_target[keep],
                        alpha=cfg.focal_loss_alpha,
                        gamma=cfg.focal_loss_gamma,
                        reduction='sum'
                    )

                    losses['C'] = cfg.conf_alpha * class_loss / len(pos_inds)
                else:
                    neg = self.select_neg_bboxes(conf_data_f, conf_t_f, type='class')
                    keep = (pos.reshape(-1) + neg).gt(0)
                    class_loss = F.binary_cross_entropy_with_logits(conf_data_f[keep], class_target[keep], reduction='mean')

                    losses['C'] = cfg.conf_alpha * class_loss

            else:
                # TODO: needs to design the loss
                losses['C'] = self.ohem_stuff_loss(stuff_data, conf_t)

            # These losses also don't depend on anchors
            if cfg.use_semantic_segmentation_loss:
                losses['S'] = self.semantic_segmentation_loss(segm_data, gt_masks, gt_labels)

        return losses, conf_t, ids_t, idx_t

    def select_neg_bboxes(self, conf_data, conf_t, type):
        '''
        :param conf_data: [n, num_classes]
        :param conf_t: [n, 1]
        :param type: 'class' or 'stuff'
        :return:
        '''

        # Compute max conf across batch for hard negative mining
        if type == 'stuff':
            loss_c = torch.sigmoid(conf_data).view(-1)
        else:
            loss_c, _ = conf_data.sigmoid().max(dim=1)

        # Hard Negative Mining
        num_pos = (conf_t > 0).sum()
        num_neg = min(3 * num_pos, conf_t.size(0) - 1)
        loss_c[conf_t > 0] = 0  # filter out pos samples and neutrals
        loss_c[conf_t < 0] = 0  # filter out pos samples and neutrals
        _, loss_idx = loss_c.sort(descending=True)
        neg = torch.zeros(conf_t.size(), device=conf_t.get_device())
        neg[loss_idx[:num_neg]] = 1
        # Just in case there aren't enough negatives, don't start using positives as negatives
        neg[conf_t > 0] = 0  # filter out pos samplers and neutrals
        neg[conf_t < 0] = 0  # filter out pos samplers and neutrals

        return neg

    def ohem_conf_loss(self, conf_data, conf_t):
        """
        Focal loss but using sigmoid like the original paper.
        Note: To make things mesh easier, the network still predicts 41 class confidences in this mode.
              Because retinanet originally only predicts 80, we simply just don't use conf_data[..., 0]
        """
        conf_t_f = conf_t.view(-1)
        conf_data_f = conf_data.view(-1, self.num_classes)

        pos = (conf_t > 0).float()
        neg = self.select_neg_bboxes(conf_data, conf_t, type='vis')
        keep = (pos + neg).gt(0)
        use_conf_t = conf_t[keep]
        use_conf_data = conf_data[keep]

        class_target = torch.zeros_like(conf_data_f)
        pos_inds = torch.nonzero(pos.float())
        class_target[pos_inds, conf_t_f[pos_inds] - 1] = 1

        loss = F.binary_cross_entropy(use_conf_data, use_conf_t, reduction='none')

        return cfg.conf_alpha * loss.sum() / (self.negpos_ratio + 1)

    def repulsion_loss(self, pos, idx_t, decoded_loc_p, gt_bboxes):
        batch_size = pos.size(0)
        rep_loss = torch.tensor(0.0, device=pos.device)
        n = 0.

        for idx in range(batch_size):
            n_gt_cur = gt_bboxes[idx].size(0)
            n_pos_cur = decoded_loc_p.size(0)
            # the overlaps between predicted boxes and ground-truth boxes
            overlaps_cur = jaccard(decoded_loc_p, gt_bboxes[idx]).detach()  # [n_pos, n_gt]

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
                rep_gt_boxes = gt_bboxes[idx].unsqueeze(0).repeat(n_pos_cur, 1, 1).view(-1, 4)[keep]
                rep_IoU = 1. - giou_loss(rep_decoded_loc_p, rep_gt_boxes, reduction='none')
                rep_loss += rep_IoU.mean() * cfg.BIoU_alpha

        return rep_loss / max(n, 1.)

    def t2s_single_loss(self, bb_feat_ref, bb_feat_next, gt_bboxes_ref, gt_bboxes_next,
                        gt_ids_ref, gt_ids_next):
        feat_h, feat_w = bb_feat_ref.size()[2:]

        # forward flow: corr from reference frame to next frame
        corr_forward = correlate_operator(bb_feat_ref, bb_feat_next,
                                          patch_size=cfg.correlation_patch_size,
                                          kernel_size=3)
        t2s_in_feats_for = torch.cat((bb_feat_ref, corr_forward, bb_feat_next), dim=1)

        loss_B_shift = torch.tensor(0., device=bb_feat_ref.device)
        loss_BIoU_shift = torch.tensor(0., device=bb_feat_ref.device)
        bs = len(gt_bboxes_ref)
        for i in range(bs):
            # select instances that exists in both two frames
            same_ids_cur = [id for id in gt_ids_ref[i] if id in gt_ids_next[i]]

            if len(same_ids_cur) == 0:
                continue

            # extract features on the predicted bbox
            idx_ref = [gt_ids_ref[i].tolist().index(id) for id in same_ids_cur]
            idx_next = [gt_ids_next[i].tolist().index(id) for id in same_ids_cur]

            with torch.no_grad():
                bboxes_ref_cur = gt_bboxes_ref[i][idx_ref]
                bboxes_next_cur = gt_bboxes_next[i][idx_next]
                bboxes_ref_reg_gt = encode(bboxes_next_cur, center_size(bboxes_ref_cur))

            # extract features from bounding boxes
            bboxes_feats_ref_crop = bbox_feat_extractor(t2s_in_feats_for[i].unsqueeze(0),
                                                        bboxes_ref_cur, feat_h, feat_w, 7)
            # display_correlation_map(bboxes_feats_ref_crop[:, 256:377])
            if cfg.maskshift_loss:
                bboxes_ref_reg, mask_coeff_ref_reg = self.net.TemporalNet(bboxes_feats_ref_crop)
            else:
                bboxes_ref_reg = self.net.TemporalNet(bboxes_feats_ref_crop)

            # B_shift loss
            pre_loss = F.smooth_l1_loss(bboxes_ref_reg, bboxes_ref_reg_gt, reduction='none').sum(dim=-1)
            loss_B_shift += pre_loss.mean()

            # BIoU_shift loss
            bboxes_next_tracked = decode(bboxes_ref_reg, center_size(bboxes_ref_cur), cfg.use_yolo_regressors)
            if cfg.use_DIoU:
                loss_BIoU_shift += (1. - compute_DIoU(bboxes_next_tracked, bboxes_next_cur).diag()).mean()
            else:
                loss_BIoU_shift += giou_loss(bboxes_next_tracked, bboxes_next_cur, reduction='mean')

        losses = {'BIoU_shift': loss_BIoU_shift / bs * cfg.BIoU_alpha,
                  'B_shift': loss_B_shift / bs * cfg.bbox_alpha}

        return losses

    def track_to_segment_loss(self, bb_feat_ref, bb_feat_next, loc_ref, ids_t_ref, mask_coeff_ref, proto_data_next,
                              priors, gt_bboxes_ref, gt_bboxes_next, gt_ids_ref, gt_ids_next,
                              gt_masks_next, interpolation_mode='bilinear'):
        feat_h, feat_w = bb_feat_ref.size()[2:]

        corr = correlate_operator(bb_feat_ref, bb_feat_next,
                                  patch_size=cfg.correlation_patch_size,
                                  kernel_size=3)
        concat_feat = torch.cat((bb_feat_ref, corr, bb_feat_next), dim=1)

        loss_B_shift = torch.tensor(0., device=mask_coeff_ref.device)
        loss_mask_shift = torch.tensor(0., device=mask_coeff_ref.device)
        bs = loc_ref.size(0)
        for i in range(bs):

            # select instances that exists in both two frames
            gt_ids_ref_cur = gt_ids_ref[i]
            gt_ids_next_cur = gt_ids_next[i]
            gt_bboxes_reg = torch.zeros_like(loc_ref[i])
            ids_t_ref_cur = ids_t_ref[i].clone()

            for j, id in enumerate(gt_ids_ref_cur):
                if id in gt_ids_next_cur:
                    keep_inst = ids_t_ref_cur == id
                    # calculate tracking regression values between two frames of bounding boxes
                    with torch.no_grad():
                        gt_bbox_ref_cur = gt_bboxes_ref[i][j].view(1, 4)
                        gt_bbox_next_cur = gt_bboxes_next[i][gt_ids_next_cur == id].view(1, 4)
                        gt_bboxes_reg_cur = encode(gt_bbox_next_cur, center_size(gt_bbox_ref_cur))
                        gt_bboxes_reg[keep_inst] = gt_bboxes_reg_cur.repeat(keep_inst.sum(), 1)

                else:
                    ids_t_ref_cur[ids_t_ref[i] == id] = 0

            pos = ids_t_ref_cur > 0
            n_pos = len(torch.nonzero(pos))

            if n_pos == 0:
                continue

            # extract features on the predicted bbox
            loc_p = loc_ref[i][pos].view(-1, 4).detach()
            priors_p = priors[pos].view(-1, 4)
            bbox_p = decode(loc_p, priors_p, cfg.use_yolo_regressors)
            bbox_feats = bbox_feat_extractor(concat_feat[i].unsqueeze(0), bbox_p, feat_h, feat_w, 7)
            if cfg.maskshift_loss:
                bbox_reg, shift_mask_coeff = self.net.TemporalNet(bbox_feats)
            else:
                bbox_reg = self.net.TemporalNet(bbox_feats)
            pre_loss_B = F.smooth_l1_loss(bbox_reg, gt_bboxes_reg[pos], reduction='none').sum(dim=-1)
            loss_B_shift += pre_loss_B.mean()

            if cfg.maskshift_loss:
                # create mask_gt and bbox_gt in the reference frame
                cur_pos_ids_t = ids_t_ref_cur[pos]
                with torch.no_grad():
                    pos_idx_t = [gt_ids_next[i].tolist().index(id) for id in cur_pos_ids_t]
                    bbox_t_next = gt_bboxes_next[i][pos_idx_t]
                    mask_t_next = gt_masks_next[i][pos_idx_t].float()

                # generate mask coeff shift mask: \sum coeff_tar * proto_ref
                tar_mask_coeff = mask_coeff_ref[i, pos] + shift_mask_coeff
                pred_masks = generate_mask(proto_data_next[i], tar_mask_coeff, bbox_t_next)

                mask_gt_h, mask_gt_w = mask_t_next.size()[:2]
                upsampled_pred_masks = F.interpolate(pred_masks.unsqueeze(0).float(),
                                                     (mask_gt_h, mask_gt_w),
                                                     mode=interpolation_mode, align_corners=False).squeeze(0)

                pre_loss = F.binary_cross_entropy(torch.clamp(upsampled_pred_masks, 0, 1), mask_t_next,
                                                  reduction='none')

                if cfg.mask_proto_crop:
                    pos_get_csize = center_size(bbox_t_next)
                    gt_box_width = torch.clamp(pos_get_csize[:, 2], min=1e-4, max=1) * mask_gt_w
                    gt_box_height = torch.clamp(pos_get_csize[:, 3], min=1e-4, max=1) * mask_gt_h
                    pre_loss = pre_loss.sum(dim=(1, 2)) / gt_box_width / gt_box_height
                    loss_mask_shift += torch.mean(pre_loss)
                else:
                    loss_mask_shift += torch.mean(pre_loss) / mask_gt_h / mask_gt_w

        losses = {'B_shift': loss_B_shift / bs * cfg.bbox_alpha}
        if cfg.maskshift_loss:
            losses['M_shift'] = loss_mask_shift / bs * cfg.maskshift_alpha

        return losses

    def prepare_masks_for_track(self, conf_t, ids_t, loc_data, priors, idx_t, gt_bboxes,
                                proto_data, mask_coeff, gt_masks, gt_ids):

        if cfg.track_crop_with_pred_mask:
            pos = conf_t > 0
            pos_ids_t = ids_t[pos]

            # get predicted or GT bounding boxes for cropping pred masks
            if cfg.track_crop_with_pred_box:
                pos_bboxes = decode(loc_data[pos], priors[pos], cfg.use_yolo_regressors)
            else:
                pos_bboxes = gt_bboxes[idx_t[pos]]

            pos_masks = generate_mask(proto_data, mask_coeff[pos], pos_bboxes)

        else:
            pos_ids_t = gt_ids
            pos_masks = gt_masks
            pos_bboxes = gt_bboxes

        return pos_masks, pos_bboxes, pos_ids_t

    def track_loss(self, track_data, gt_masks, gt_ids, proto_data, mask_coeff, gt_bboxes,
                   loc_data, priors, conf_t, ids_t, idx_t):
        bs, track_h, track_w, t_dim = track_data.size()
        loss = torch.tensor(0., device=track_data.device)

        n_bs_track, n_clip = 0, 2
        for i in range(bs//n_clip):
            mu, var, obj_ids = [], [], []

            for j in range(n_clip):

                masks_cur, bbox_cur, ids_t_cur = self.prepare_masks_for_track(conf_t[2*i+j],
                                                                              ids_t[2*i+j], loc_data[2*i+j],
                                                                              priors[2*i+j], idx_t[2*i+j],
                                                                              gt_bboxes[2*i+j], proto_data[2*i+j],
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

        losses = {'T': cfg.track_alpha * loss / max(n_bs_track, 1)}

        return losses

    def coeff_sparse_loss(self, coeffs):
        """
        coeffs:  should be size [num_pos, num_coeffs]
        """
        torch.abs(coeffs).sum(1)

    def coeff_diversity_loss(self, weights, coeffs, instance_t):
        """
        coeffs     should be size [num_pos, num_coeffs]
        instance_t should be size [num_pos] and be values from 0 to num_instances-1
        """

        instance_t = instance_t.view(-1)  # juuuust to make sure

        coeffs_norm = F.normalize(coeffs, dim=1)
        cos_sim = torch.mm(coeffs_norm, coeffs_norm.t())

        inst_eq = (instance_t[:, None].expand_as(cos_sim) == instance_t[None, :].expand_as(cos_sim)).float()

        # Rescale to be between 0 and 1
        cos_sim = (cos_sim + 1) / 2

        # If they're the same instance, use cosine distance, else use cosine similarity
        cos_sim_diff = torch.clamp(1 - cos_sim, min=1e-10)
        loss = -1 * (torch.clamp(cos_sim, min=1e-10).log() * inst_eq + cos_sim_diff.log() * (1 - inst_eq))
        weights = weights.view(-1, 1) * weights.view(1, -1)

        # Only divide by num_pos once because we're summing over a num_pos x num_pos tensor
        # and all the losses will be divided by num_pos at the end, so just one extra time.
        return cfg.mask_proto_coeff_diversity_alpha * (weights * loss).sum()

    def lincomb_mask_loss(self, pos, idx_t, loc_data, mask_coeff, priors, prior_levels, proto_data,
                          masks_gt, bboxes_gt, interpolation_mode='bilinear'):

        bs = mask_coeff.size(0)
        loss_bce, loss_m_occluded = torch.tensor(0., device=pos.device), torch.tensor(0., device=pos.device)
        if cfg.mask_dice_coefficient:
            loss_dice = torch.tensor(0., device=pos.device)
        if cfg.use_dynamic_mask:
            cfg.mask_proto_coeff_occlusion = False

        for idx in range(bs):
            cur_pos = pos[idx]
            if len(torch.nonzero(cur_pos)) > 0:

                pos_idx_t = idx_t[idx, cur_pos]
                masks_gt_cur = masks_gt[idx]
                N_gt, H_gt, W_gt = masks_gt_cur.size()
                proto_masks = proto_data[idx]                   # [mask_h, mask_w, 32]
                proto_coeff = mask_coeff[idx, cur_pos, :]       # [n_pos, 32 or 196]
                # get mask loss for ono-occluded part, [n_pos, 3, h, w]
                if N_gt > 1 and cfg.mask_proto_coeff_occlusion:
                    masks_gt_cur3 = torch.zeros(N_gt, 3, H_gt, W_gt)
                    masks_gt_cur3[:, 1] = masks_gt_cur
                    # assign non-target objects as 2, then target objects as 1, background as 0
                    masks_gt_cur_all = masks_gt_cur.sum(dim=0)
                    masks_gt_cur3[:, 0] = (masks_gt_cur_all == 0).repeat(N_gt, 1, 1).float()
                    for m_idx in range(masks_gt_cur.size(0)):
                        target_object = masks_gt_cur[m_idx] > 0
                        non_target_objects = (masks_gt_cur_all > 0) * ~target_object
                        masks_gt_cur3[m_idx, 2][non_target_objects] = 1
                    mask_t = masks_gt_cur3[pos_idx_t]

                else:
                    mask_t = masks_gt_cur[pos_idx_t]            # [n_pos, mask_h, mask_w]

                if cfg.mask_proto_crop or cfg.use_dynamic_mask:
                    # Note: this is in point-form
                    if cfg.mask_proto_crop_with_pred_box:
                        pos_gt_box_t = decode(loc_data[idx, cur_pos], priors[idx, cur_pos],
                                              cfg.use_yolo_regressors).detach()
                    else:
                        pos_gt_box_t = bboxes_gt[idx][pos_idx_t]

                    pos_gt_box_t_c = center_size(pos_gt_box_t)
                    pos_gt_box_t_c[:, 2:] *= 1.2
                    pos_gt_box_t = point_form(torch.clamp(pos_gt_box_t_c, min=1e-5, max=1))
                else:
                    pos_gt_box_t = None

                if not cfg.use_dynamic_mask:
                    pred_masks = generate_mask(proto_masks, proto_coeff, pos_gt_box_t)
                else:
                    fpn_levels = prior_levels[idx, cur_pos]
                    pred_masks = self.net.DynamicMaskHead(proto_masks.permute(2, 0, 1).contiguous().unsqueeze(0),
                                                          proto_coeff, pos_gt_box_t, fpn_levels)
                    if cfg.mask_proto_crop:
                        _, pred_masks = crop(pred_masks.permute(1, 2, 0).contiguous(), pos_gt_box_t)
                        pred_masks = pred_masks.permute(2, 0, 1).contiguous()

                if cfg.use_dynamic_mask or not cfg.mask_proto_coeff_occlusion:
                    if cfg.mask_loss_with_ori_size:
                        # [n, h, w] => [n, 1, h, w]
                        pred_masks = F.interpolate(pred_masks.unsqueeze(1).float(), (H_gt, W_gt),
                                                   mode=interpolation_mode, align_corners=False).squeeze(1)
                        pred_masks = torch.clamp(pred_masks, min=0, max=1)
                    else:
                        mask_t = F.interpolate(mask_t.unsqueeze(1).float(), (pred_masks.size(-2), pred_masks.size(-1)),
                                               mode=interpolation_mode, align_corners=False).squeeze(1)

                    if cfg.mask_dice_coefficient:
                        pre_loss_dice = self.dice_coefficient(pred_masks, mask_t.float())
                    else:
                        # activation_funcunction is sigmoid
                        pre_loss_bce = F.binary_cross_entropy(pred_masks, mask_t.float(), reduction='none')

                else:
                    if cfg.mask_loss_with_ori_size:
                        # [n, h, w, 3] => [n, 3, h, w]
                        pred_masks = pred_masks.permute(0, 3, 1, 2).contiguous()
                        pred_masks = F.interpolate(pred_masks.float(), (H_gt, W_gt),
                                                   mode=interpolation_mode, align_corners=False)
                        pred_masks = torch.clamp(pred_masks, min=0, max=1)
                    else:
                        mask_t = F.interpolate(mask_t.float(), (pred_masks.size(-2), pred_masks.size(-1)),
                                               mode=interpolation_mode, align_corners=False)

                    if cfg.mask_dice_coefficient:
                        cur_loss = []
                        for i in range(3):
                            cur_loss.append(self.dice_coefficient(pred_masks[:, i], mask_t.float()[:, i]))
                        pre_loss_dice = torch.stack(cur_loss, dim=0).mean(dim=0)
                    else:
                        pre_loss_bce = F.binary_cross_entropy(pred_masks,
                                                              mask_t.float(), reduction='none').sum(dim=1)
                if cfg.mask_dice_coefficient:
                    loss_dice += pre_loss_dice.mean()
                else:
                    if cfg.mask_proto_crop:
                        pos_gt_box_w = torch.clamp(pos_gt_box_t_c[:, 2], min=1e-4, max=1) * W_gt
                        pos_gt_box_h = torch.clamp(pos_gt_box_t_c[:, 3], min=1e-4, max=1) * H_gt
                        pre_loss_bce = pre_loss_bce.sum(dim=(1, 2)) / pos_gt_box_w / pos_gt_box_h
                    else:
                        pre_loss_bce = pre_loss_bce.sum(dim=(1, 2)) / H_gt / W_gt

                    keep = (torch.isinf(pre_loss_bce) + torch.isnan(pre_loss_bce)) > 0
                    if keep.sum() > 0:
                        print(pre_loss_bce)
                    loss_bce += pre_loss_bce[~keep].mean()

        if cfg.mask_dice_coefficient:
            losses = {'M_dice': loss_dice * cfg.mask_alpha / max(bs, 1)}
        else:
            losses = {'M_bce': loss_bce * cfg.mask_alpha / max(bs, 1)}

        return losses

    def dice_coefficient(self, x, target):
        eps = 1e-5
        n_inst = x.size(0)
        x = x.reshape(n_inst, -1)
        target = target.reshape(n_inst, -1)
        intersection = (x * target).sum(dim=1)
        union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
        loss = 1. - (2 * intersection / union)
        return loss

    def _mask_iou(self, mask1, mask2):
        intersection = torch.sum(mask1*mask2, dim=(1, 2))
        area1 = torch.sum(mask1, dim=(1, 2))
        area2 = torch.sum(mask2, dim=(1, 2))
        union = (area1 + area2) - intersection
        ret = intersection / union
        return ret

    def semantic_segmentation_loss(self, segment_data, mask_t, class_t, interpolation_mode='bilinear'):
        '''
        :param segment_data: [bs, h, w, num_class]
        :param mask_t: a list of groundtruth masks
        :param class_t: a list of groundtruth clsses: begin with 1
        :param interpolation_mode:
        :return:
        '''
        # Note num_classes here is without the background class so cfg.num_classes-1
        segment_data = segment_data.permute(0, 3, 1, 2).contiguous()
        bs, _, mask_h, mask_w = segment_data.size()
        sem_loss = 0
        # void out of memory so as to calcuate loss for a single image
        for idx in range(bs):
            mask_t_downsample = F.interpolate(mask_t[idx].unsqueeze(0), (mask_h, mask_w),
                                              mode=interpolation_mode, align_corners=False).squeeze(0).gt_(0.5)

            # prepare one-hat
            segment_t = torch.zeros_like(segment_data[idx])
            with torch.no_grad():
                for obj_idx, obj_mask_t in enumerate(mask_t_downsample):
                    obj_class = (class_t[idx][obj_idx]-1).long()
                    segment_t[obj_class][obj_mask_t == 1] = 1

            # avoid out of memory so as to calculate semantic loss for a single image
            pre_sem_loss = sigmoid_focal_loss_jit(
                            segment_data[idx],
                            segment_t,
                            alpha=cfg.focal_loss_alpha,
                            gamma=cfg.focal_loss_gamma,
                            reduction="sum"
                        )
            sem_loss += pre_sem_loss / segment_t.sum()

        return sem_loss / float(bs) * cfg.semantic_segmentation_alpha
