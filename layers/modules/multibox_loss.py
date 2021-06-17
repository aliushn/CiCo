# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers.utils import compute_DIoU, match, log_sum_exp, encode, decode, center_size, crop, jaccard, point_form, \
    generate_mask, compute_kl_div, generate_track_gaussian, sanitize_coordinates_hw
from .track_to_segment_head import bbox_feat_extractor
from layers.utils.track_utils import correlate_operator
from layers.visualization import display_correlation_map

from datasets import cfg, activation_func


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

    def __init__(self, net, num_classes, pos_threshold, neg_threshold, negpos_ratio):
        super(MultiBoxLoss, self).__init__()
        self.net = net
        self.num_classes = num_classes

        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.negpos_ratio = negpos_ratio

        # If you output a proto mask with this area, your l1 loss will be l1_alpha
        # Note that the area is relative (so 1 would be the entire image)
        self.l1_expected_area = 20 * 20 / 70 / 70
        self.l1_alpha = 0.1

        if cfg.use_class_balanced_conf:
            self.class_instances = None
            self.total_instances = 0

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
        proto_data = predictions['proto']
        sem_data = predictions['sem_seg'] if cfg.use_semantic_segmentation_loss else None

        losses, conf_t, ids_t, idx_t = self.multibox_loss(imgs, loc_data, conf_data, stuff_data, mask_coeff,
                                                          centerness_data, proto_data, priors, sem_data,
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

            elif cfg.use_FEELVOS:

                bb_feat_ref = F.normalize(predictions['fpn_feat'][::2].contiguous(), dim=1)
                bb_feat_next = F.normalize(predictions['fpn_feat'][1::2].contiguous(), dim=1)
                gt_masks_next = gt_masks[1::2]
                gt_ids_next = gt_ids[1::2]
                conf_t_ref = conf_t[::2].detach()
                idx_t_ref = idx_t[::2]
                ids_t_ref = ids_t[::2]
                gt_bboxes_ref = gt_bboxes[::2]
                gt_masks_ref = gt_masks[::2]
                losses_shift = self.VOS_loss(bb_feat_next, gt_masks_next, gt_ids_next, bb_feat_ref, conf_t_ref, idx_t_ref,
                                             ids_t_ref, gt_bboxes_ref, gt_masks_ref)
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

    def multibox_loss(self, imgs, loc_data, conf_data, stuff_data, mask_data, centerness_data, proto_data, priors, segm_data,
                      gt_bboxes, gt_labels, gt_masks, gt_ids, num_crowds):
        batch_size, num_priors = loc_data.size()[0:2]

        # Match priors (default boxes) and ground truth boxes
        # These tensors will be created with the same device as loc_data
        loc_t = loc_data.new(batch_size, num_priors, 4)
        gt_boxes_t = loc_data.new(batch_size, num_priors, 4)
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
            match(self.pos_threshold, self.neg_threshold, gt_bboxes[idx], gt_labels[idx], gt_ids_cur, crowd_boxes,
                  priors[idx], loc_data[idx], loc_t, conf_t, idx_t, ids_t, idx)

            gt_boxes_t[idx] = gt_bboxes[idx][idx_t[idx]]

        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        idx_t = Variable(idx_t, requires_grad=False)
        if gt_ids is not None:
            ids_t = Variable(ids_t, requires_grad=False)

        pos = conf_t > 0
        total_num_pos = pos.data.sum()
        num_pos_per_img = []
        for idx in range(batch_size):
            num_pos_per_img.append(pos[idx].sum().long())

        losses = {}

        # design weights for all pos samples in per img
        IoU_split = torch.ones(int(total_num_pos.tolist()), device=loc_data.device).split(num_pos_per_img)
        pos_weights_per_img = [IoU_cur / torch.clamp(IoU_cur.sum(), min=1) for IoU_cur in IoU_split]
        pos_weights = torch.cat(pos_weights_per_img, dim=0)

        # Localization Loss (Smooth L1)
        if cfg.train_boxes:
            loc_p = loc_data[pos].view(-1, 4)
            loc_t = loc_t[pos].view(-1, 4)
            pos_priors = priors[pos].view(-1, 4)

            losses['B'] = (pos_weights.view(-1, 1) * F.smooth_l1_loss(loc_p, loc_t,
                                                                      reduction='none')).sum() * cfg.bbox_alpha

            if cfg.use_boxiou_loss:
                decoded_loc_p = decode(loc_p, pos_priors, cfg.use_yolo_regressors)
                DIoU = compute_DIoU(decoded_loc_p, gt_boxes_t[pos]).diag().view(-1)
                # DIoU in [-1, 1], 0.5(DIoU+1) in [0, 1], - In( 0.5*(1+DIoU) ) in [0, +infinte]
                DIoU_loss = - torch.log(torch.clamp(0.5 * (1 + DIoU), min=1e-5))
                losses['BIoU'] = (pos_weights * DIoU_loss).sum() * cfg.bboxiou_alpha
                # add repulsion loss for crowd occlusion
                if cfg.use_repulsion_loss:
                    losses['Rep'] = self.repulsion_loss(pos, idx_t, decoded_loc_p, gt_bboxes)

        # Mask Loss
        if cfg.train_masks:
            loss = self.lincomb_mask_loss(pos_weights_per_img, pos, idx_t, loc_data, mask_data,
                                          priors, proto_data, gt_masks, gt_boxes_t)
            losses.update(loss)

            if cfg.mask_proto_loss is not None:
                if cfg.mask_proto_loss == 'l1':
                    losses['P'] = torch.mean(torch.abs(proto_data)) / self.l1_expected_area * self.l1_alpha
                elif cfg.mask_proto_loss == 'disj':
                    losses['P'] = -torch.mean(torch.max(F.log_softmax(proto_data, dim=-1), dim=-1)[0])

        # Confidence loss
        if cfg.train_class:
            if cfg.use_sigmoid_focal_loss:
                loss_c = self.focal_conf_sigmoid_loss(conf_data, conf_t)
            else:
                loss_c = self.ohem_conf_loss(pos_weights, conf_data, conf_t, centerness_data, loc_data, priors, gt_boxes_t)
            losses.update(loss_c)
        else:
            loss_c = self.ohem_stuff_loss(stuff_data, conf_t)
            losses.update(loss_c)

        # These losses also don't depend on anchors
        if cfg.use_semantic_segmentation_loss:
            losses['S'] = self.semantic_segmentation_loss(segm_data, gt_masks, gt_labels, gt_bboxes)

        # Divide all losses by the number of positives.
        # Don't do it for loss[P] because that doesn't depend on the anchors.
        # loss[P] and loss [IC] have been divided the number of positives
        for k in losses:
            losses[k] /= batch_size
        # Loss Key:
        #  - B: Box Localization Loss
        #  - C: Class Confidence Loss
        #  - M: Mask Loss
        #  - T: Tracking Loss
        #  - P: Prototype Loss
        #  - D: Coefficient Diversity Loss
        #  - S: Semantic Segmentation Loss
        return losses, conf_t, ids_t, idx_t

    def repulsion_loss(self, pos, idx_t, decoded_loc_p, gt_bboxes):
        batch_size = pos.size(0)
        rep_loss = torch.tensor(0.0, device=pos.device)
        n = 0.

        for idx in range(batch_size):
            n_gt_cur = gt_bboxes[idx].size(0)
            n_pos_cur = decoded_loc_p.size(0)
            # the overlaps between predicted boxes and ground-truth boxes
            overlaps_cur = jaccard(decoded_loc_p, gt_bboxes[idx])  # [n_pos, n_gt]

            # matched ground-truth bounding box idx with each predicted boxes
            idx_t_cur = idx_t[idx][pos[idx]]
            # to find surrounding boxes, we assign the overlaps of matched idx as -1
            for j in range(pos[idx].sum()):
                overlaps_cur[j, idx_t_cur[j]] = -1

            # Crowd occlusion: BIoU greater than 0.1
            keep = (overlaps_cur > 0.1).view(-1)
            if keep.sum() > 0:
                n += 1
                rep_decoded_loc_p = decoded_loc_p.unsqueeze(1).repeat(1, n_gt_cur, 1).view(-1, 4)[keep]
                rep_gt_boxes = gt_bboxes[idx].unsqueeze(0).repeat(n_pos_cur, 1, 1).view(-1, 4)[keep]

                rep_DIoU = compute_DIoU(rep_decoded_loc_p, rep_gt_boxes).diag().view(-1)
                # rep_DIoU in [-1, 1], 0.5 * (1-rep_DIoU) in [0, 1]
                # rep_loss = -In (0.5*(1-DIoU))
                rep_DIoU_loss = - torch.log(torch.clamp(0.5 * (1 - rep_DIoU), min=1e-5))
                rep_loss += rep_DIoU_loss.mean() * 0.2 * cfg.bboxiou_alpha

        return rep_loss / max(n, 1.) * batch_size

    def VOS_loss(self, bb_feat_next, gt_masks_next, gt_ids_next, bb_feat_ref, conf_t_ref, idx_t_ref, ids_t_ref,
                 gt_bboxes_ref, gt_masks_ref):
        loss = torch.tensor(0.0, device=conf_t_ref.device)
        bs = bb_feat_next.size(0)
        mask_gt_h, mask_gt_w = gt_masks_next[0][0].size()
        upsampled_bb_feat_next = F.interpolate(bb_feat_next.float(), (int(mask_gt_h/4), int(mask_gt_w/4)),
                                               mode='bilinear', align_corners=False)
        attention_next = self.net.VOS_attention(upsampled_bb_feat_next)

        for i in range(bs):
            pos = conf_t_ref[i] > 0
            pos_idx_ref = idx_t_ref[i, pos]
            pos_ids_ref = ids_t_ref[i, pos]
            bbox_t_ref = gt_bboxes_ref[i][pos_idx_ref]

            # if an instance disappears in the next frame, we hope the vos head can output a mask with all 0
            masks_t_next = torch.zeros((pos.sum(), mask_gt_h, mask_gt_w), device=idx_t_ref.device)
            for id in pos_ids_ref.unique():
                if id in gt_ids_next[i]:
                    idx_next = torch.where(gt_ids_next[i] == id)[0]
                    masks_t_next[pos_ids_ref == id] = gt_masks_next[i][idx_next].float()

            feat_h, feat_w = bb_feat_ref.size()[2:]
            masks_t_ref = gt_masks_ref[i][pos_idx_ref].float()
            downsampled_masks_t_ref = F.interpolate(masks_t_ref.unsqueeze(0).float(), (feat_h, feat_w),
                                                    mode='bilinear', align_corners=False).squeeze(0)

            pred_masks_next = []
            for j in range(pos.sum()):
                corr_cur = correlate_operator(bb_feat_next[i].unsqueeze(0),
                                              (bb_feat_ref[i] * downsampled_masks_t_ref[j].unsqueeze(0)).unsqueeze(0),
                                              patch_size=cfg.correlation_patch_size, kernel_size=3)
                corr_cur = torch.max(corr_cur, dim=1)[0]  # [1, h/16, w/16]
                vos_feat = bb_feat_next[i] * corr_cur
                upsampled_vos_feat = F.interpolate(vos_feat.unsqueeze(0), (int(mask_gt_h/4), int(mask_gt_w/4)),
                                                   mode='bilinear', align_corners=False)   # [1, h/4, w/4]
                pred_vos = self.net.VOS_head(upsampled_vos_feat)
                pred_masks_next.append(pred_vos * attention_next[i].unsqueeze(0))
            pred_masks_next = torch.cat(pred_masks_next, dim=1)
            upsampled_pred_masks_next = F.interpolate(pred_masks_next.float(), (mask_gt_h, mask_gt_w),
                                                      mode='bilinear', align_corners=False).squeeze(0)
            upsampled_pred_masks_next = cfg.mask_proto_mask_activation(upsampled_pred_masks_next)  # [n_ref, h, w]

            if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                pre_loss = F.binary_cross_entropy(torch.clamp(upsampled_pred_masks_next, 0, 1), masks_t_next, reduction='none')
            else:
                pre_loss = F.smooth_l1_loss(upsampled_pred_masks_next, masks_t_next, reduction='none')

            # only consider the region of the predicted reference bbox
            box_t_ref_c = center_size(bbox_t_ref)
            # we use 1.2 box to crop features
            box_next_crop_c = torch.cat([box_t_ref_c[:, :2],
                                         torch.clamp(box_t_ref_c[:, 2:] * 1.5, min=0, max=1)], dim=1)
            box_next_crop = point_form(box_next_crop_c)
            _, pre_loss_crop = crop(pre_loss.permute(1, 2, 0).contiguous(), box_next_crop)  # [h, w, n_ref]
            gt_box_width = box_next_crop_c[:, 2] * mask_gt_w
            gt_box_width = torch.clamp(gt_box_width, min=1)
            gt_box_height = box_next_crop_c[:, 3] * mask_gt_h
            gt_box_height = torch.clamp(gt_box_height, min=1)
            pre_loss_crop = pre_loss_crop.sum(dim=(0, 1)) / gt_box_width / gt_box_height
            loss += pre_loss_crop.mean()

        return {'M_shift': cfg.maskshift_alpha * loss / bs}

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

            bboxes_ref_cur = gt_bboxes_ref[i][idx_ref]
            bboxes_next_cur = gt_bboxes_next[i][idx_next]

            # extract features from bounding boxes
            bboxes_feats_ref_crop = bbox_feat_extractor(t2s_in_feats_for[i].unsqueeze(0),
                                                        bboxes_ref_cur,
                                                        feat_h, feat_w, 7)
            # display_correlation_map(bboxes_feats_ref_crop[:, 256:377])
            if cfg.maskshift_loss:
                bboxes_ref_reg, mask_coeff_ref_reg = self.net.TemporalNet(bboxes_feats_ref_crop)
            else:
                bboxes_ref_reg = self.net.TemporalNet(bboxes_feats_ref_crop)
            bboxes_ref_reg_gt = encode(bboxes_next_cur, center_size(bboxes_ref_cur))
            pre_loss = F.smooth_l1_loss(bboxes_ref_reg, bboxes_ref_reg_gt, reduction='none').sum(dim=1)
            loss_B_shift += pre_loss.mean()

            # BIoU loss
            bboxes_next_tracked = decode(bboxes_ref_reg, center_size(bboxes_ref_cur))
            DIoU_tracked_ref = compute_DIoU(bboxes_next_tracked, bboxes_next_cur).diag().view(-1)
            DIoU_tracked_ref_loss = - torch.log(torch.clamp(0.5 * (1 + DIoU_tracked_ref), min=1e-5))
            loss_BIoU_shift += DIoU_tracked_ref_loss.mean()

        losses = {'BIoU_shift': loss_BIoU_shift / bs * cfg.bboxiou_alpha,
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
                    gt_bbox_ref_cur = gt_bboxes_ref[i][j].view(1, 4)
                    gt_bbox_next_cur = gt_bboxes_next[i][gt_ids_next_cur == id].view(1, 4)
                    gt_bboxes_reg_cur = encode(gt_bbox_next_cur, center_size(gt_bbox_ref_cur))
                    gt_bboxes_reg[keep_inst] = gt_bboxes_reg_cur.repeat(keep_inst.sum(), 1)

                else:
                    ids_t_ref_cur[ids_t_ref[i] == id] = 0

            pos = ids_t_ref_cur > 0
            n_pos = pos.sum()

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
            pre_loss_B = F.smooth_l1_loss(bbox_reg, gt_bboxes_reg[pos], reduction='none').sum(dim=1)
            loss_B_shift += pre_loss_B.mean()

            if cfg.maskshift_loss:
                # create mask_gt and bbox_gt in the reference frame
                cur_pos_ids_t = ids_t_ref_cur[pos]
                pos_idx_t = [gt_ids_next[i].tolist().index(id) for id in cur_pos_ids_t]
                bbox_t_next = gt_bboxes_next[i][pos_idx_t]
                mask_t_next = gt_masks_next[i][pos_idx_t].permute(1, 2, 0).contiguous().float()

                # generate mask coeff shift mask: \sum coeff_tar * proto_ref
                tar_mask_coeff = mask_coeff_ref[i, pos] + shift_mask_coeff
                pred_masks = generate_mask(proto_data_next[i], tar_mask_coeff, bbox_t_next)

                mask_gt_h, mask_gt_w = mask_t_next.size()[:2]
                upsampled_pred_masks = F.interpolate(pred_masks.unsqueeze(0).float(),
                                                     (mask_gt_h, mask_gt_w),
                                                     mode=interpolation_mode, align_corners=False).squeeze(0)
                upsampled_pred_masks = upsampled_pred_masks.permute(1, 2, 0).contiguous()  # [mask_h, mask_w, n_pos]

                if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                    pre_loss = F.binary_cross_entropy(torch.clamp(upsampled_pred_masks, 0, 1), mask_t_next,
                                                      reduction='none')
                else:
                    pre_loss = F.smooth_l1_loss(upsampled_pred_masks, mask_t_next, reduction='none')

                if cfg.mask_proto_crop:
                    pos_get_csize = center_size(bbox_t_next)
                    gt_box_width = torch.clamp(pos_get_csize[:, 2], min=1e-4, max=1) * mask_gt_w
                    gt_box_height = torch.clamp(pos_get_csize[:, 3], min=1e-4, max=1) * mask_gt_h
                    pre_loss = pre_loss.sum(dim=(0, 1)) / gt_box_width / gt_box_height
                    loss_mask_shift += torch.mean(pre_loss)
                else:
                    loss_mask_shift += torch.mean(pre_loss) / mask_gt_h / mask_gt_w

        losses = {'B_shift': loss_B_shift / bs * cfg.bbox_alpha}
        if cfg.maskshift_loss:
            losses['M_shift'] = loss_mask_shift / bs * cfg.maskshift_alpha

        return losses

    def prepare_masks_for_track(self, h, w, conf_t, ids_t, loc_data, priors, idx_t, gt_bboxes,
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

        n_bs_track, n_bs_class, n_clip = torch.zeros(1), torch.zeros(1), 2
        for i in range(bs//n_clip):
            mu, var, obj_ids = [], [], []

            for j in range(n_clip):

                masks_cur, bbox_cur, ids_t_cur = self.prepare_masks_for_track(track_h, track_w, conf_t[2*i+j],
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
                cur_weights = torch.ones(mu.size(0))
                loss_weights = cur_weights.view(-1, 1) @ cur_weights.view(1, -1)
                loss_weights = torch.triu(loss_weights, diagonal=1) + torch.triu(loss_weights.t(), diagonal=1).t()

                # If they are the same instance, use cosine distance, else use consine similarity
                # pos: log(kl_divergence+1), neg: 4 / exp(0.1*kl_divergence)
                pre_loss = inst_eq * (kl_divergence + 1).log() \
                           + (1. - inst_eq) * torch.exp(-1 * kl_divergence)
                loss += (pre_loss * loss_weights).sum() / loss_weights.sum()

        losses = {'T': cfg.track_alpha * loss / torch.clamp(n_bs_track, min=1)}

        return losses

    def instance_conf_loss(self, conf_data, ref_conf_data, conf_t, ref_conf_t, ids_t, ref_ids_t):
        batch_size, num_priors, n_classes = conf_data.size()
        loss = 0

        for idx in range(batch_size):
            # calculate the instance confident in a single frame: the mean of the bboxes with same instance id
            inst_pt, inst_conf_t, inst_id = self.get_inst_conf(conf_data[idx],  conf_t[idx], ids_t[idx])
            ref_inst_pt, ref_inst_conf_t, ref_inst_id = self.get_inst_conf(ref_conf_data[idx],
                                                                           ref_conf_t[idx], ref_ids_t[idx])

            # get the instance confident cross multi frames by mean them in different frames
            all_inst_id = torch.unique(torch.cat([inst_id, ref_inst_id]))
            all_inst_pt, all_inst_conf_t = [], []
            for i, id in enumerate(all_inst_id):
                used_idx = inst_id == id
                used_ref_idx = ref_inst_id == id
                if id in inst_id and id in ref_inst_id:
                    inst_pt_id = (inst_pt[used_idx] + ref_inst_pt[used_ref_idx]) / 2.0
                    inst_conf_t_id = inst_conf_t[used_idx]
                elif id in inst_id:
                    inst_pt_id = inst_pt[used_idx]
                    inst_conf_t_id = inst_conf_t[used_idx]
                else:
                    inst_pt_id = ref_inst_pt[used_ref_idx]
                    inst_conf_t_id = ref_inst_conf_t[used_ref_idx]
                all_inst_pt.append(inst_pt_id)
                all_inst_conf_t.append(inst_conf_t_id)

            # get the confident loss of instances in the a short subset (includes 2 frames here)
            loss += F.cross_entropy(torch.cat(all_inst_pt, dim=0), torch.cat(all_inst_conf_t), reduction='sum') / all_inst_id.size(0)

        return loss / batch_size

    def get_inst_conf(self, conf_data,  conf_t, ids_t):
        # put bboxes with same instances ids into a set, called instance confidence
        pos = (conf_t > 0)
        pos_pt = F.softmax(conf_data[pos], -1)  # [n_pos, 41]
        pos_ids_t = ids_t[pos]
        pos_conf_t = conf_t[pos]

        # merge those bboxes with same instances into a subset
        # use the conf data's mean of all bboxes in the subset as the conf data of the instance in the subset
        inst_id, inv_id = torch.unique(pos_ids_t, sorted=True, return_inverse=True)
        inst_pt = [pos_pt[inv_id == i].sum(dim=0) / (inv_id == i).sum() for i in range(len(inst_id))]
        inst_conf_t = [pos_conf_t[inv_id == i][0].view(1) for i in range(len(inst_id))]

        return torch.stack(inst_pt, dim=0), torch.cat(inst_conf_t), inst_id

    def select_neg_bboxes(self, conf_data, conf_t, type):
        if len(conf_t.size()) == 2 or len(conf_data.size()) == 3:
            conf_t = conf_t.view(-1)                          # [batch_size*num_priors]
            conf_data = conf_data.view(conf_t.size(0), -1)  # [batch_size*num_priors, num_classes]

        # Compute max conf across batch for hard negative mining
        if type == 'stuff':
            loss_c = torch.sigmoid(conf_data).view(-1)
        else:
            if cfg.ohem_use_most_confident:
                conf_data = F.softmax(conf_data, dim=1)
                loss_c, _ = conf_data[:, 1:].max(dim=1)
            else:
                loss_c = log_sum_exp(conf_data) - conf_data[:, 0]

        # Hard Negative Mining
        num_pos = (conf_t > 0).sum()
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=conf_t.size()[0] - 1)
        loss_c[conf_t > 0] = 0  # filter out pos samples and neutrals
        loss_c[conf_t < 0] = 0  # filter out pos samples and neutrals
        _, loss_idx = loss_c.sort(descending=True)
        neg = torch.zeros(conf_t.size(), device=conf_t.get_device())
        neg[loss_idx[:num_neg]] = 1
        # Just in case there aren't enough negatives, don't start using positives as negatives
        neg[conf_t > 0] = 0  # filter out pos samplers and neutrals
        neg[conf_t < 0] = 0  # filter out pos samplers and neutrals

        return neg

    def ohem_conf_loss(self, pos_weights, conf_data, conf_t, centerness_data, loc, priors, gt_boxes_t):
        """
        Focal loss but using sigmoid like the original paper.
        Note: To make things mesh easier, the network still predicts 41 class confidences in this mode.
              Because retinanet originally only predicts 80, we simply just don't use conf_data[..., 0]
        """
        batch_size = conf_t.size(0)
        conf_t = conf_t.view(-1)
        conf_data = conf_data.view(-1,  self.num_classes)

        pos = (conf_t > 0).float()
        neg = self.select_neg_bboxes(conf_data, conf_t, type='vis')
        keep = (pos + neg).gt(0)
        use_conf_t = conf_t[keep]
        use_conf_data = conf_data[keep]
        num_neg = (neg > 0).sum()
        neg_weights = torch.ones(num_neg, device=pos_weights.device) / num_neg * self.negpos_ratio * batch_size
        loss_weights = torch.cat([pos_weights, neg_weights])

        loss = F.cross_entropy(use_conf_data, use_conf_t, reduction='none')
        losses = {'C': cfg.conf_alpha * (loss_weights * loss).sum() / (self.negpos_ratio + 1)}

        if centerness_data is not None:
            pos = pos.gt(0)
            decoded_loc = decode(loc.view(-1, 4)[pos], priors.view(-1, 4)[pos], cfg.use_yolo_regressors)
            DIoU = compute_DIoU(decoded_loc, gt_boxes_t.view(-1, 4)[pos]).diag().view(-1)
            loss_cn = F.smooth_l1_loss(centerness_data.view(-1)[pos], DIoU, reduction='none')
            losses['center'] = cfg.center_alpha * (pos_weights * loss_cn).sum()

        return losses

    def ohem_stuff_loss(self, stuff_data, conf_t):
        """
        Focal loss but using sigmoid like the original paper.
        Note: To make things mesh easier, the network still predicts 81 class confidences in this mode.
              Because retinanet originally only predicts 80, we simply just don't use conf_data[..., 0]
        """
        batch_size = stuff_data.size(0)
        conf_t = conf_t.view(-1)
        # set pos samplers as 1, neg samplers as 0, neutrals as -1
        conf_t[conf_t > 0] = 1
        stuff_data = stuff_data.view(-1)

        pos = (conf_t > 0).float()
        neg = self.select_neg_bboxes(stuff_data, conf_t, type='stuff')
        keep = (pos + neg).gt(0)
        use_stuff_t = conf_t[keep].float()
        use_stuff_data = stuff_data[keep]

        loss = F.binary_cross_entropy_with_logits(use_stuff_data, use_stuff_t, reduction='none')
        losses = {'stuff': cfg.stuff_alpha * batch_size * loss.mean()}

        return losses

    def focal_conf_loss(self, conf_data, conf_t):
        conf_t = conf_t.view(-1)
        conf_data = conf_data.view(-1, self.num_classes)
        pos = (conf_t > 0).float()
        neg = self.select_neg_bboxes(conf_data, conf_t)
        keep = (pos + neg).gt(0)

        # Confidence Loss Including Positive and Negative Examples
        # adapted from focal loss to reduce the effect of class unbalance
        conf_p = conf_data[keep].view(-1, self.num_classes)
        logpt = F.log_softmax(conf_p, dim=-1)
        use_conf_t = conf_t[keep].view(-1)
        logpt = logpt.gather(1, use_conf_t.unsqueeze(-1)).view(-1)
        pt = logpt.exp()
        at = (1 - cfg.focal_loss_alpha) * pos.float() + cfg.focal_loss_alpha * neg.float()
        at = at[keep].view(-1)
        loss_c = -1 * at * (1 - pt) ** cfg.focal_loss_gamma * logpt

        return cfg.conf_alpha * loss_c.sum()

    def focal_conf_sigmoid_loss(self, conf_data, conf_t):
        """
        Focal loss but using sigmoid like the original paper.
        Note: To make things mesh easier, the network still predicts 81 class confidences in this mode.
              Because retinanet originally only predicts 80, we simply just don't use conf_data[..., 0]
        """

        bs, _, num_classes = conf_data.size()

        conf_t = conf_t.view(-1)  # [batch_size*num_priors]
        conf_data = conf_data.view(-1, num_classes)  # [batch_size*num_priors, num_classes]

        # Ignore neutral samples (class < 0)
        keep = (conf_t >= 0).float()
        conf_t[conf_t < 0] = 0  # can't mask with -1, so filter that out

        # Compute a one-hot embedding of conf_t
        # From https://github.com/kuangliu/pytorch-retinanet/blob/master/utils.py
        conf_one_t = torch.eye(num_classes, device=conf_t.get_device())[conf_t]
        conf_pm_t = conf_one_t * 2 - 1  # -1 if background, +1 if forground for specific class

        logpt = F.logsigmoid(conf_data * conf_pm_t)  # note: 1 - sigmoid(x) = sigmoid(-x)
        pt = logpt.exp()

        at = cfg.focal_loss_alpha * conf_one_t + (1 - cfg.focal_loss_alpha) * (1 - conf_one_t)
        at[..., 0] = 0  # Set alpha for the background class to 0 because sigmoid focal loss doesn't use it

        loss = -at * (1 - pt) ** cfg.focal_loss_gamma * logpt
        loss = keep * loss.sum(dim=-1)

        losses = {'C': cfg.conf_alpha * loss.sum() / keep.sum() * bs}

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
        num_pos = coeffs.size(0)
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

    def lincomb_mask_loss(self, pos_weights_per_img, pos, idx_t, loc_data, mask_coeff,
                          priors, proto_data, masks_gt, gt_box_t, interpolation_mode='bilinear'):

        loss_m, loss_m_occluded, loss_miou = 0, 0, 0

        for idx in range(mask_coeff.size(0)):
            cur_pos = pos[idx]
            if cur_pos.sum() == 0:
                continue

            if cfg.mask_proto_crop:
                # Note: this is in point-form
                if cfg.mask_proto_crop_with_pred_box:
                    pos_pred_box = decode(loc_data[idx, cur_pos], priors[idx, cur_pos],
                                          cfg.use_yolo_regressors).detach()
                    pos_gt_box_t = pos_pred_box
                else:
                    pos_gt_box_t = gt_box_t[idx, cur_pos]

                pos_gt_box_t = center_size(pos_gt_box_t)
                pos_gt_box_t[:, 2:] *= 1.2
                pos_gt_box_t = point_form(pos_gt_box_t)
                pos_gt_box_t = torch.clamp(pos_gt_box_t, min=1e-5, max=1)

            pos_idx_t = idx_t[idx, cur_pos]
            mask_t = masks_gt[idx][pos_idx_t].float()       # [n, mask_h, mask_w]
            proto_masks = proto_data[idx]                   # [mask_h, mask_w, 32]
            proto_coeff = mask_coeff[idx, cur_pos, :]         # [num_pos, 32, Class+1]

            # get mask loss for ono-occluded part
            # pred_masks = generate_mask(proto_masks, proto_coef, pos_gt_box_t)
            mask_gt_h, mask_gt_w = mask_t.size()[1:]
            if cfg.mask_proto_crop:
                pred_masks = generate_mask(proto_masks, proto_coeff, pos_gt_box_t)
            else:
                pred_masks = generate_mask(proto_masks, proto_coeff)
            upsampled_pred_masks = F.interpolate(pred_masks.unsqueeze(0).float(),
                                                 (mask_gt_h, mask_gt_w),
                                                 mode=interpolation_mode, align_corners=False).squeeze(0)

            if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                pre_loss = F.binary_cross_entropy(torch.clamp(upsampled_pred_masks, 0, 1), mask_t, reduction='none')
            else:
                pre_loss = F.smooth_l1_loss(upsampled_pred_masks, mask_t, reduction='none')

            if cfg.mask_proto_crop:
                pos_gt_box_wh = center_size(pos_gt_box_t)[:, :2]
                pos_gt_box_w = torch.clamp(pos_gt_box_wh[:, 0] * mask_gt_w, min=1)
                pos_gt_box_h = torch.clamp(pos_gt_box_wh[:, 1] * mask_gt_h, min=1)
                pre_loss = pre_loss.sum(dim=(1, 2)) / pos_gt_box_w / pos_gt_box_h
                loss_m += torch.sum(pos_weights_per_img[idx] * pre_loss)
            else:
                loss_m += torch.sum(pos_weights_per_img[idx] * pre_loss) / mask_gt_h / mask_gt_w

        losses = {'M': loss_m * cfg.mask_alpha}

        if cfg.use_maskiou_loss:
            losses['MIoU'] = loss_miou * cfg.maskiou_alpha

        return losses

    def _mask_iou(self, mask1, mask2):
        intersection = torch.sum(mask1*mask2, dim=(1, 2))
        area1 = torch.sum(mask1, dim=(1, 2))
        area2 = torch.sum(mask2, dim=(1, 2))
        union = (area1 + area2) - intersection
        ret = intersection / union
        return ret

    def mask_iou_loss(self, net, maskiou_targets):
        maskiou_net_input, maskiou_t, label_t = maskiou_targets

        maskiou_p = net.maskiou_net(maskiou_net_input)

        label_t = label_t[:, None]
        maskiou_p = torch.gather(maskiou_p, dim=1, index=label_t).view(-1)

        loss_i = F.smooth_l1_loss(maskiou_p, maskiou_t, reduction='sum')

        return loss_i * cfg.maskiou_alpha

    def semantic_segmentation_loss(self, segment_data, mask_t, class_t, box_t, interpolation_mode='bilinear'):
        # Note num_classes here is without the background class so cfg.num_classes-1
        segment_data = segment_data.permute(0, 3, 1, 2).contiguous()
        batch_size, num_classes, mask_h, mask_w = segment_data.size()
        mask_t_h, mask_t_w = mask_t[0].size()[-2:]

        upsampled_segment_data = F.interpolate(segment_data, (mask_t_h, mask_t_w),
                                               mode=interpolation_mode, align_corners=False)
        upsampled_segment_data_act = torch.clamp(F.softmax(upsampled_segment_data, dim=1), min=1e-3)

        # Construct Semantic Segmentation
        segment_t = torch.zeros((batch_size, mask_t_h, mask_t_w), requires_grad=False).type(torch.int64)
        weight = torch.zeros((batch_size, mask_t_h, mask_t_w), requires_grad=False)
        for idx in range(batch_size):
            cur_class_t = class_t[idx]
            cur_box_t = box_t[idx]

            with torch.no_grad():
                cur_mask_t = mask_t[idx].float()
                cur_mask_t_bg = cur_mask_t.sum(dim=0) == 0
                for obj_idx, obj_mask_t in enumerate(cur_mask_t):
                    segment_t[idx, obj_mask_t > 0] = cur_class_t[obj_idx]
                    cur_segment_data = upsampled_segment_data_act[idx, cur_class_t[obj_idx]]
                    # Construct weight for pixels with objects
                    weight[idx, obj_mask_t > 0] = 0.1 * torch.div(1., cur_segment_data[obj_mask_t > 0])

                    # Construct weight (sucha as 0.1) for pixels with background
                    cur_box_t_c = center_size(cur_box_t[obj_idx].view(1, -1))
                    cur_box_t_c[:, 2:] *= 1.2
                    cur_box_t_p = torch.clamp(point_form(cur_box_t_c), min=1e-5, max=1)
                    cur_weight_mask, _ = crop(weight[idx].unsqueeze(-1), cur_box_t_p)
                    keep_bg = cur_weight_mask.squeeze(-1).bool() & cur_mask_t_bg
                    weight[idx, keep_bg] = 0.1

        pre_loss = F.cross_entropy(upsampled_segment_data, segment_t, reduction='none')
        loss_s = (pre_loss * weight.unsqueeze(1)).sum() / weight.sum()

        return loss_s * cfg.semantic_segmentation_alpha
