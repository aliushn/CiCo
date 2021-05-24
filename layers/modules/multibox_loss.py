# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers.utils import match, log_sum_exp, decode, center_size, crop, jaccard, point_form, generate_mask
from .track_to_segment_head import bbox_feat_extractor
from layers.utils.track_utils import correlate_operator

from datasets import cfg, mask_type, activation_func


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

    def forward(self, predictions, gt_bboxes, gt_labels, gt_masks, gt_ids):
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
        if cfg.mask_coeff_for_occluded:
            mask_occluded_coeff = predictions['mask_occluded_coeff']
        else:
            mask_occluded_coeff = None
        track_data = F.normalize(predictions['track'], dim=-1) if cfg.train_track else None

        # This is necessary for training on multiple GPUs because
        # DataParallel will cat the priors from each GPU together
        priors = predictions['priors']
        proto_data = predictions['proto']
        if cfg.use_semantic_segmentation_loss:
            sem_data = predictions['sem_seg'].permute(0, 3, 1, 2).contiguous()
        else:
            sem_data = None

        losses, conf_t, ids_t, idx_t = self.multibox_loss(loc_data, conf_data, stuff_data, mask_coeff, mask_occluded_coeff,
                                                          centerness_data, proto_data, priors, sem_data,
                                                          gt_bboxes, gt_labels,  gt_masks, gt_ids)

        if cfg.use_temporal_info:

            if cfg.temporal_fusion_module:
                bb_feat_ref = predictions['fpn_feat'][::2].contiguous()
                bb_feat_next = predictions['fpn_feat'][1::2].contiguous()
                corr = correlate_operator(bb_feat_next, bb_feat_ref, patch_size=cfg.correlation_patch_size,
                                          kernel_size=3)
                loc_data_ref = loc_data[::2].detach()
                ids_t_ref = ids_t[::2]
                mask_coeff_ref = mask_coeff[::2].detach()
                proto_data_next = proto_data[1::2].detach()
                t2s_in_feats = torch.cat((corr, bb_feat_ref, bb_feat_next), dim=1)
                losses_shift = self.track_to_segment_loss(t2s_in_feats,
                                                          loc_data_ref, ids_t_ref, mask_coeff_ref, proto_data_next,
                                                          priors[0], gt_bboxes, gt_ids, gt_masks)
                losses.update(losses_shift)

            elif cfg.use_FEELVOS:
                bb_feat_ref = predictions['fpn_feat'][::2].contiguous()
                bb_feat_next = predictions['fpn_feat'][1::2].contiguous()
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
            losses_clip = self.track_loss(sem_data, gt_labels, track_data, gt_masks, gt_ids, proto_data, mask_coeff,
                                          gt_bboxes, loc_data, priors, conf_t, ids_t, idx_t)
            losses.update(losses_clip)

        for k, v in losses.items():
            if torch.isinf(v) or torch.isnan(v):
                print(k)

        return losses

    def split(self, x):
        x1, x2 = torch.split(x, [1, 1], dim=-1)
        return x1.squeeze(-1), x2.squeeze(-1)

    def multibox_loss(self, loc_data, conf_data, stuff_data, mask_data, mask_occluded_data,
                      centerness_data, proto_data, priors, segm_data, gt_bboxes, gt_labels, gt_masks, gt_ids):
        batch_size, num_priors = loc_data.size()[0:2]

        # Match priors (default boxes) and ground truth boxes
        # These tensors will be created with the same device as loc_data
        loc_t = loc_data.new(batch_size, num_priors, 4)
        gt_boxes_t = loc_data.new(batch_size, num_priors, 4)
        conf_t = loc_data.new(batch_size, num_priors).long()
        idx_t = loc_data.new(batch_size, num_priors).long()
        ids_t = loc_data.new(batch_size, num_priors).long()  # pids for tracking

        # assign positive samples
        for idx in range(batch_size):
            match(self.pos_threshold, self.neg_threshold, gt_bboxes[idx], gt_labels[idx], gt_ids[idx], priors[idx],
                  loc_data[idx], loc_t, conf_t, idx_t, ids_t, idx)

            gt_boxes_t[idx] = gt_bboxes[idx][idx_t[idx]]

        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        idx_t = Variable(idx_t, requires_grad=False)
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

            if cfg.use_boxiou_loss:
                decoded_loc_p = decode(loc_p, pos_priors, cfg.use_yolo_regressors)
                DIoU = self.get_DIoU(decoded_loc_p, gt_boxes_t[pos])
                losses['BIoU'] = (pos_weights * (1-DIoU)).sum() * cfg.bboxiou_alpha
            else:
                losses['B'] = (pos_weights.view(-1, 1) * F.smooth_l1_loss(loc_p, loc_t,
                                                                          reduction='none')).sum() * cfg.bbox_alpha

        # Mask Loss
        if cfg.train_masks:
            loss = self.lincomb_mask_loss(pos_weights_per_img, pos, idx_t, ids_t, loc_data, mask_data, mask_occluded_data,
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
            losses['S'] = self.semantic_segmentation_loss(segm_data, gt_masks, gt_labels)

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

    def get_DIoU(self, pos_pred_boxes, pos_gt_boxes):
        # calculate bbox IoUs
        IoU = jaccard(pos_gt_boxes, pos_pred_boxes).diag().view(-1)

        # calculate the diagonal length of the smallest enclosing box
        x_label = torch.cat([pos_pred_boxes[:, ::2], pos_gt_boxes[:, ::2]], dim=1)
        y_label = torch.cat([pos_pred_boxes[:, 1::2], pos_gt_boxes[:, 1::2]], dim=1)
        c2 = (x_label.max(dim=1)[0] - x_label.min(dim=1)[0])**2 + (y_label.max(dim=1)[0] - y_label.min(dim=1)[0])**2
        c2 = torch.clamp(c2, min=1e-10)

        # get the distance of centres between pred_bbox and gt_bbox
        pos_pred_boxes_c = pos_pred_boxes[:, :2] / 2 + pos_pred_boxes[:, 2:] / 2
        pos_gt_boxes_c = pos_gt_boxes[:, :2] / 2 + pos_gt_boxes[:, 2:] / 2
        d2 = ((pos_pred_boxes_c - pos_gt_boxes_c)**2).sum(dim=1)

        # DIoU
        DIoU = IoU - d2/c2

        return DIoU

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
            # bbox_t_ref = gt_bboxes_ref[i][pos_idx_ref]

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
                upsampled_corr_cur = F.interpolate(corr_cur.float(), (int(mask_gt_h/4), int(mask_gt_w/4)),
                                                   mode='bilinear', align_corners=False)
                upsampled_corr_cur = torch.max(upsampled_corr_cur, dim=1)[0]  # [1, h, w]
                norm_upsampled_corr_cur = upsampled_corr_cur / torch.clamp(upsampled_corr_cur.max(), min=1e-5)
                vos_feat = upsampled_bb_feat_next[i] * norm_upsampled_corr_cur
                pred_masks_next.append(self.net.VOS_head(vos_feat.unsqueeze(0)) * attention_next[i].unsqueeze(0))
            pred_masks_next = torch.cat(pred_masks_next, dim=1)
            upsampled_pred_masks_next = F.interpolate(pred_masks_next.float(), (mask_gt_h, mask_gt_w),
                                                      mode='bilinear', align_corners=False).squeeze(0)
            upsampled_pred_masks_next = cfg.mask_proto_mask_activation(upsampled_pred_masks_next)  # [n_ref, h, w]

            if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                pre_loss = F.binary_cross_entropy(torch.clamp(upsampled_pred_masks_next, 0, 1), masks_t_next, reduction='none')
            else:
                pre_loss = F.smooth_l1_loss(upsampled_pred_masks_next, masks_t_next, reduction='none')

            # only consider the region of the predicted reference bbox
            # box_t_ref_c = center_size(bbox_t_ref)
            # we use 1.2 box to crop features
            # box_next_crop_c = torch.cat([box_t_ref_c[:, :2],
            #                              torch.clamp(box_t_ref_c[:, 2:] * 1.5, min=0, max=1)], dim=1)
            # box_next_crop = point_form(box_next_crop_c)
            # print(pre_loss.permute(1, 2, 0).contiguous().size(), box_next_crop.size())
            # _, pre_loss_crop = crop(pre_loss.permute(1, 2, 0).contiguous(), box_next_crop)  # [h, w, n_ref]
            # gt_box_width = box_next_crop_c[:, 2] * mask_gt_w
            # gt_box_width = torch.clamp(gt_box_width, min=1)
            # gt_box_height = box_next_crop_c[:, 3] * mask_gt_h
            # gt_box_height = torch.clamp(gt_box_height, min=1)
            # pre_loss_crop = pre_loss_crop.sum(dim=(0, 1)) / gt_box_width / gt_box_height
            loss += pre_loss.mean()

        return {'M_shift': cfg.maskshift_alpha * loss / bs}

    def track_to_segment_loss(self, concat_feat, loc_ref, ids_t_ref, mask_coeff_ref, proto_data_next,
                              priors, gt_bboxes, gt_ids, gt_masks, interpolation_mode='bilinear'):
        # pair data: first frame as reference frame, second frame as next frame
        gt_bboxes_ref, gt_bboxes_next = gt_bboxes[::2], gt_bboxes[1::2]
        gt_ids_ref, gt_ids_next = gt_ids[::2], gt_ids[1::2]
        gt_masks_ref, gt_masks_next = gt_masks[::2], gt_masks[1::2]

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
                    gt_bbox_ref_cur = center_size(gt_bboxes_ref[i][j].view(1, 4))
                    gt_bbox_next_cur = center_size(gt_bboxes_next[i][gt_ids_next_cur == id].view(1, 4))
                    gt_bboxes_reg_cur = torch.cat([(gt_bbox_next_cur[:, :2] - gt_bbox_ref_cur[:, :2]) / gt_bbox_ref_cur[:, 2:],
                                                   (gt_bbox_next_cur[:, 2:] / gt_bbox_ref_cur[:, 2:]).log()           ], dim=1)
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
            bbox_p_c = center_size(bbox_p)
            bbox_p_crop = point_form(torch.cat([bbox_p_c[:, :2],
                                                torch.clamp(bbox_p_c[:, 2:] * 1.2, min=0, max=1)], dim=1))
            bbox_feats = bbox_feat_extractor(concat_feat[i].unsqueeze(0), bbox_p_crop, 7)
            bbox_reg, shift_mask_coeff = self.net.TemporalNet(bbox_feats)
            pre_loss_B = F.smooth_l1_loss(bbox_reg, gt_bboxes_reg[pos], reduction='none').sum(1)
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

        losses = {'B_shift': loss_B_shift / bs * cfg.boxshift_alpha}
        if cfg.maskshift_loss:
            losses['M_shift'] = loss_mask_shift / bs * cfg.maskshift_alpha

        return losses

    def prepare_masks_for_track(self, h, w, conf_t, ids_t, loc_data, priors, idx_t, gt_bboxes,
                                proto_data, mask_coeff, gt_masks, gt_ids, gt_labels):

        if cfg.track_crop_with_pred_mask:
            pos = conf_t > 0
            pos_ids_t = ids_t[pos]

            # get predicted or GT bounding boxes for cropping pred masks
            if cfg.track_crop_with_pred_box:
                pos_bboxes = decode(loc_data[pos], priors[pos], cfg.use_yolo_regressors)
            else:
                pos_bboxes = gt_bboxes[idx_t[pos]]

            pos_masks = generate_mask(proto_data, mask_coeff[pos], pos_bboxes)
            pos_labels = conf_t[pos]

        else:
            pos_ids_t = gt_ids
            pos_masks = gt_masks
            pos_labels = gt_labels

        # resize pos_masks_ref and pos_masks_next to keep the same size with track data
        downsampled_pos_masks = F.interpolate(pos_masks.unsqueeze(1).float(), (h, w),
                                              mode='bilinear', align_corners=False).gt(0.5)

        # remove objects whose pixels of masks in the resolution (track_h, track_w) less than 1
        # because Gaussian distribution needs at least two pixels to build the means and variances
        keep = (downsampled_pos_masks.sum(dim=(2, 3)) > 1).squeeze(1)

        if len(keep) > 0:
            return downsampled_pos_masks[keep], pos_ids_t[keep], pos_labels[keep]
        else:
            return None, None, None

    def track_loss(self, segment_data, gt_labels, track_data, gt_masks, gt_ids, proto_data, mask_coeff, gt_bboxes,
                   loc_data, priors, conf_t, ids_t, idx_t):
        bs, track_h, track_w, t_dim = track_data.size()
        n_class = segment_data.size(1)
        loss = torch.tensor(0., device=track_data.device)
        loss_c_clip = torch.tensor(0., device=track_data.device)
        loss_c_obj = torch.tensor(0., device=track_data.device)
        # [bs, t_dim, h, w]
        track_data = track_data.permute(0, 3, 1, 2).contiguous()

        n_bs_track, n_bs_class, n_clip = torch.zeros(1), torch.zeros(1), 2
        for i in range(bs//n_clip):
            mu, var, obj_ids, obj_labels = [], [], [], []
            segment_data_clip = []

            for j in range(n_clip):
                masks_cur, ids_t_cur, labels_cur = self.prepare_masks_for_track(track_h, track_w, conf_t[2*i+j],
                                                                                ids_t[2*i+j], loc_data[2*i+j],
                                                                                priors[2*i+j], idx_t[2*i+j],
                                                                                gt_bboxes[2*i+j], proto_data[2*i+j],
                                                                                mask_coeff[2*i+j], gt_masks[2*i+j],
                                                                                gt_ids[2*i+j], gt_labels[2*i+j])

                if masks_cur is not None:
                    track_data_cur = track_data[2*i+j].unsqueeze(0) * masks_cur             # [n, t_dim, h, w]
                    mu_cur = track_data_cur.sum(dim=(2, 3)) / masks_cur.sum(dim=(2, 3))     # [n, t_dim]
                    var_cur = torch.sum((track_data_cur - mu_cur.view(-1, t_dim, 1, 1)) ** 2, dim=(2, 3))
                    mu.append(mu_cur)
                    var.append(var_cur)
                    obj_ids.append(ids_t_cur)
                    obj_labels.append(labels_cur)

                    if not cfg.train_class and cfg.use_semantic_segmentation_loss:
                        segment_data_cur = segment_data[2*i+j].unsqueeze(0) * masks_cur                  # [n, c, h, w]
                        # average pooling for all pixels that belong to an object
                        segment_data_clip.append(segment_data_cur.sum(dim=(2, 3)) / masks_cur.sum(dim=(2, 3)))  # [n, c]

            mu, var, obj_ids = torch.cat(mu, dim=0), torch.cat(var, dim=0), torch.cat(obj_ids, dim=0)
            obj_labels = torch.cat(obj_labels, dim=0)

            if len(obj_ids) > 1:
                n_bs_track += 1

                # to calculate the kl_divergence for two Gaussian distributions, where c is the dim of variables
                # D_kl(p||q) = 0.5(log(torch.norm(Sigma_q) / torch.norm(Sigma_p)) - c
                #                  + (mu_p-mu_q)^T * torch.inverse(Sigma_q) * (mu_p-mu_q))
                #                  + torch.trace(torch.inverse(Sigma_q) * Sigma_p))
                log_sum_var = torch.log(var).sum(dim=-1, keepdim=True)
                first_term = log_sum_var - log_sum_var.t() - t_dim
                third_term = torch.mm(1./var, var.t())
                # ([1, n, c] - [n, 1, c]) * [1, n, c] => [n, n]
                second_term = torch.sum((mu.unsqueeze(0) - mu.unsqueeze(1))**2 / var.unsqueeze(0), dim=-1)
                kl_divergence = 0.5 * (first_term + second_term + third_term) + 1 + 1e-5  # [1, +infinite

                # We hope the kl_divergence between same instance Ids is small, otherwise the kl_divergence is large.
                inst_eq = (obj_ids.view(-1, 1) == obj_ids.view(1, -1)).float()
                cur_weights = torch.ones(mu.size(0))
                loss_weights = cur_weights.view(-1, 1) @ cur_weights.view(1, -1)
                loss_weights = torch.triu(loss_weights, diagonal=1) + torch.triu(loss_weights.t(), diagonal=1).t()

                # If they are the same instance, use cosine distance, else use consine similarity
                # pos: log(kl_divergence), neg: exp(-2*kl_divergence)
                pre_loss = inst_eq * kl_divergence.log() + (1. - inst_eq) * torch.exp(-1*kl_divergence.sqrt())
                loss += (pre_loss * loss_weights).sum() / loss_weights.sum()

            if len(obj_ids) > 0:
                n_bs_class += 1

                if not cfg.train_class and cfg.use_semantic_segmentation_loss:
                    conf_data_clip = torch.cat(segment_data_clip, dim=0)
                    obj_ids_unique = torch.unique(obj_ids)
                    conf_data_clip_unique, obj_labels_unique = [], []
                    for id in obj_ids_unique:
                        conf_data_clip_unique.append(torch.mean(conf_data_clip[id == obj_ids].view(-1, n_class), dim=0))
                        obj_labels_unique.append(obj_labels[id == obj_ids][0].view(1))

                    # Calsses do not includes backgroud, because we only consider masks of positive samples
                    loss_c_clip += F.cross_entropy(torch.stack(conf_data_clip_unique),
                                                   torch.cat(obj_labels_unique)-1, reduction='mean')
                    loss_c_obj += F.cross_entropy(conf_data_clip, obj_labels-1, reduction='mean')

        losses = {'T': cfg.track_alpha * loss / torch.clamp(n_bs_track, min=1)}
        if not cfg.train_class and cfg.use_semantic_segmentation_loss:
            losses['C_sem'] = cfg.conf_alpha * (loss_c_clip + loss_c_obj) / torch.clamp(n_bs_class, min=1)

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
        Note: To make things mesh easier, the network still predicts 81 class confidences in this mode.
              Because retinanet originally only predicts 80, we simply just don't use conf_data[..., 0]
        """
        batch_size = conf_t.size(0)
        conf_t = conf_t.view(-1)
        conf_data = conf_data.view(-1,  self.num_classes)

        pos = (conf_t > 0).float()
        neg = self.select_neg_bboxes(conf_data, conf_t)
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
            DIoU = self.get_DIoU(decoded_loc, gt_boxes_t.view(-1, 4)[pos])
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

    def lincomb_mask_loss(self, pos_weights_per_img, pos, idx_t, ids_t, loc_data, mask_coeff, mask_occluded_coeff,
                          priors, proto_data, masks_gt, gt_box_t,
                          interpolation_mode='bilinear'):

        loss_m, loss_m_occluded, loss_miou = 0, 0, 0
        loss_d = 0  # Coefficient diversity loss
        proto_coef_clip, pos_ids_clip, weights_clip = [], [], []

        for idx in range(mask_coeff.size(0)):
            cur_pos = pos[idx]
            pos_idx_t = idx_t[idx, cur_pos]
            pos_ids_t = ids_t[idx, cur_pos]
            pos_pred_box = decode(loc_data[idx, cur_pos], priors[idx, cur_pos], cfg.use_yolo_regressors).detach()

            if cfg.mask_proto_crop:
                # Note: this is in point-form
                if cfg.mask_proto_crop_with_pred_box:
                    pos_gt_box_t = pos_pred_box
                else:
                    pos_gt_box_t = gt_box_t[idx, cur_pos]

                pos_gt_box_t = center_size(pos_gt_box_t)
                pos_gt_box_t[:, 2:] *= 1.2
                pos_gt_box_t = point_form(pos_gt_box_t)
                pos_gt_box_t = torch.clamp(pos_gt_box_t, min=1e-5, max=1)

            if pos_idx_t.size(0) == 0:
                continue

            mask_t = masks_gt[idx][pos_idx_t].float()       # [n, mask_h, mask_w]
            proto_masks = proto_data[idx]                   # [mask_h, mask_w, 32]
            proto_coeff = mask_coeff[idx, cur_pos, :]         # [num_pos, 32]

            if cfg.mask_proto_coeff_diversity_loss:
                proto_coef_clip.append(proto_coeff)
                pos_ids_clip.append(pos_ids_t)
                weights_clip.append(pos_weights_per_img[idx])
                if (idx + 1) % 2 == 0:
                    proto_coef_clip = torch.cat(proto_coef_clip, dim=0)
                    pos_ids_clip = torch.cat(pos_ids_clip, dim=0)
                    weights_clip = torch.cat(weights_clip, dim=0)
                    loss_d += self.coeff_diversity_loss(weights_clip, proto_coef_clip, pos_ids_clip)
                    proto_coef_clip, pos_ids_clip, weights_clip = [], [], []

            # get mask loss for ono-occluded part
            # pred_masks = generate_mask(proto_masks, proto_coef, pos_gt_box_t)
            mask_gt_h, mask_gt_w = mask_t.size()[1:]
            pred_masks = generate_mask(proto_masks, proto_coeff)
            upsampled_pred_masks = F.interpolate(pred_masks.unsqueeze(0).float(),
                                                 (mask_gt_h, mask_gt_w),
                                                 mode=interpolation_mode, align_corners=False).squeeze(0)

            if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                pre_loss = F.binary_cross_entropy(torch.clamp(upsampled_pred_masks, 0, 1), mask_t, reduction='none')
            else:
                pre_loss = F.smooth_l1_loss(upsampled_pred_masks, mask_t, reduction='none')

            if cfg.mask_proto_crop:
                upsampled_crop_masks, _ = crop(upsampled_pred_masks.permute(1, 2, 0).contiguous(),
                                               pos_gt_box_t)  # [mask_h, mask_w, n]
                upsampled_crop_masks = upsampled_crop_masks.permute(2, 0, 1).contiguous()  # [n_masks, h, w]
                weights = upsampled_crop_masks / torch.clamp(upsampled_crop_masks.sum(dim=(1, 2), keepdim=True), min=1)

                if cfg.mask_proto_crop_outside:
                    upsampled_crop_masks_outside = (upsampled_crop_masks == 0).float()
                    weights_outside = upsampled_crop_masks_outside / torch.clamp(upsampled_crop_masks_outside.sum(dim=(1, 2), keepdim=True), min=1)
                    weights = weights + weights_outside

                pre_loss = (pre_loss * weights).sum(dim=(1, 2))
                loss_m += torch.sum(pos_weights_per_img[idx] * pre_loss)
            else:
                loss_m += torch.sum(pos_weights_per_img[idx] * pre_loss) / mask_gt_h / mask_gt_w

            # get mask loss for occluded part by other instances
            if cfg.mask_coeff_for_occluded:
                # build GT masks which are occluded by other instances
                mask_gt_all = torch.sum(masks_gt[idx], dim=0, keepdim=True).permute(1, 2, 0).contiguous()
                mask_t_occluded = mask_gt_all - mask_t
                _, mask_t_occluded = crop(mask_t_occluded, pos_gt_box_t)

                proto_coeff_occluded = mask_occluded_coeff[idx, cur_pos, :]  # [num_pos, 32]
                pred_masks_occluded = generate_mask(proto_masks, proto_coeff_occluded, pos_gt_box_t)
                upsampled_pred_masks_occluded = F.interpolate(pred_masks_occluded.unsqueeze(0).float(),
                                                              (mask_gt_h, mask_gt_w),
                                                              mode=interpolation_mode, align_corners=False).squeeze(0)
                upsampled_pred_masks_occluded = upsampled_pred_masks_occluded.permute(1, 2, 0).contiguous()

                if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                    pre_loss_occluded = F.binary_cross_entropy(torch.clamp(upsampled_pred_masks_occluded, 0, 1),
                                                               mask_t_occluded, reduction='none')
                else:
                    pre_loss_occluded = F.smooth_l1_loss(upsampled_pred_masks_occluded,
                                                         mask_t_occluded, reduction='none')

                if cfg.mask_proto_crop:
                    pre_loss_occluded = (pre_loss_occluded * weights).sum(dim=(1, 2))
                    loss_m_occluded += torch.sum(pos_weights_per_img[idx] * pre_loss_occluded)
                else:
                    loss_m_occluded += torch.sum(pos_weights_per_img[idx] * pre_loss_occluded) / mask_gt_h / mask_gt_w

            if cfg.use_maskiou_loss:
                upsampled_pred_masks = upsampled_pred_masks.gt(0.5).float()
                maskiou_t = self._mask_iou(upsampled_pred_masks, mask_t)
                loss_miou += torch.mean(1 - maskiou_t)  # [n_pos]

        losses = {'M': loss_m * cfg.mask_alpha}
        if cfg.mask_coeff_for_occluded:
            losses['M_occluded'] = loss_m_occluded * cfg.mask_occluded_alpha

        if cfg.use_maskiou_loss:
            losses['MIoU'] = loss_miou * cfg.maskiou_alpha

        if cfg.mask_proto_coeff_diversity_loss:
            losses['D'] = loss_d

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

    def semantic_segmentation_loss(self, segment_data, mask_t, class_t, interpolation_mode='bilinear'):
        # Note num_classes here is without the background class so cfg.num_classes-1
        batch_size, num_classes, mask_h, mask_w = segment_data.size()
        mask_t_h, mask_t_w = mask_t[0].size()[-2:]
        # mask_h, mask_w = 2*mask_h, 2*mask_w
        loss_s = 0

        upsampled_segment_data = F.interpolate(segment_data, (mask_t_h, mask_t_w),
                                               mode=interpolation_mode, align_corners=False)
        for idx in range(batch_size):
            cur_segment = upsampled_segment_data[idx]
            cur_class_t = class_t[idx]-1

            with torch.no_grad():
                cur_mask_t = mask_t[idx].float()
                # Construct Semantic Segmentation
                segment_t = torch.zeros_like(cur_segment, requires_grad=False)
                for obj_idx in range(cur_mask_t.size(0)):
                    segment_t[cur_class_t[obj_idx]] = torch.max(segment_t[cur_class_t[obj_idx]],
                                                                cur_mask_t[obj_idx])

            loss_s += F.binary_cross_entropy_with_logits(cur_segment, segment_t, reduction='sum')

        return loss_s / mask_t_h / mask_t_w * cfg.semantic_segmentation_alpha
