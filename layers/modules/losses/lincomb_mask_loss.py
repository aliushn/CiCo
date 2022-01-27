import torch
import torch.nn.functional as F
from ...utils import center_size, point_form, generate_mask, jaccard, crop, sanitize_coordinates, circum_boxes, mask_iou


class LincombMaskLoss(object):
    def __init__(self, cfg, net):
        super(LincombMaskLoss, self).__init__()
        self.cfg = cfg
        self.net = net
    
    def __call__(self, pos, idx_t, pred_boxes_pos, boxes_t_pos, mask_coeff, prior_levels, proto_data,
                 masks_gt, boxes_gt, obj_ids_gt, labels_gt):
        '''
        Compute instance segmentation loss proposed by YOLACT, which combines linearly mask coefficients and prototypes.
        Please link here https://github.com/dbolya/yolact for more information.
        :param            pos: [bs, n_anchors], positive samples flag, which is 1 if positive samples else 0
        :param          idx_t: [bs, n_anchors], store the matched index of predicted boxes and ground-truth boxes
        :param pred_boxes_pos: List: bs * [n_pos_frame, clip_frames, 4], the predicted boxes of positive samples
        :param    boxes_t_pos: List: bs * [n_pos, 4]
        :param     mask_coeff: [bs, n_anchors, n_mask], mask coefficients, a predicted cubic box of a clip share
                               same mask coefficients due to temporal redundancy
        :param   prior_levels: [bs, n_anchors]
        :param     proto_data: [bs, h, w, n_mask], or [bs, h, w, n_frames, n_mask] prototypes
        :param       masks_gt: List: bs * [n_objs, n_frames, h, w]
        :param     obj_ids_gt:
        :param      labels_gt: ground-truth labels
        :return:
        '''

        bs, h, w = proto_data.size()[:3]
        loss, loss_intraclass = torch.tensor(0., device=idx_t.device), torch.tensor(0., device=idx_t.device)
        sparse_loss = torch.tensor(0., device=idx_t.device)
        loss_coeff_div = torch.tensor(0., device=idx_t.device)
        T_out = mask_coeff.size(0) // bs

        for i in range(bs):
            if pred_boxes_pos[i].size(0) > 0:
                ind_range = range(i*T_out, (i+1)*T_out)
                pos_cur = pos[ind_range] > 0

                # deal with the i-th clip
                if pos_cur.sum() > 0:
                    idx_t_cur = idx_t[ind_range][pos_cur]
                    mask_coeff_cur = mask_coeff[ind_range][pos_cur].reshape(-1, mask_coeff.size(-1))
                    # If the input is given frame by frame style, for example boxes_gt [n_objs, 4],
                    # please expand them in temporal axis like: [n_objs, 1, 4] to better coding
                    masks_gt_cur = masks_gt[i].unsqueeze(1) if masks_gt[i].dim() == 3 else masks_gt[i]
                    labels_gt_cur = labels_gt[i]
                    if not self.cfg.MODEL.MASK_HEADS.LOSS_WITH_OIR_SIZE:
                        masks_gt_cur = F.interpolate(masks_gt_cur.float(), (h, w),
                                                     mode='bilinear', align_corners=False)
                        masks_gt_cur.gt_(0.5)

                    with torch.no_grad():
                        N_gt, n_frames, H_gt, W_gt = masks_gt_cur.size()
                        # Mask loss for non-occluded part, [n_pos, 2, h, w]
                        masks_gt_cur_intraclass = masks_gt_cur.clone()
                        # Assign non-target objects as 1, then target objects as 0
                        for m_idx in range(N_gt):
                            keep_intraclass = labels_gt_cur == labels_gt_cur[m_idx]
                            intraclass_objects = masks_gt_cur[keep_intraclass].sum(dim=0) > 0
                            masks_gt_cur_intraclass[m_idx][intraclass_objects] = 1

                    mask_t = masks_gt_cur[idx_t_cur]  # [n_pos, n_frames, mask_h, mask_w]
                    mask_t_intraclass = masks_gt_cur_intraclass[idx_t_cur]

                    # if self.cfg.MODEL.MASK_HEADS.PROTO_CROP or self.cfg.MODEL.MASK_HEADS.USE_DYNAMIC_MASK:
                    # Only use circumscribed boxes to crop mask
                    if self.cfg.MODEL.MASK_HEADS.PROTO_CROP_WITH_PRED_BOX:
                        boxes_cur = circum_boxes(pred_boxes_pos[i].reshape(pred_boxes_pos[i].size(0), -1).detach())
                    else:
                        boxes_cur = circum_boxes(boxes_t_pos[i].reshape(boxes_t_pos[i].size(0), -1))

                    boxes_cur_c = center_size(boxes_cur)
                    # for small objects, we set width and height of bounding boxes is 0.1
                    boxes_cur_c[:, 2:] = torch.clamp(boxes_cur_c[:, 2:]*1.5, min=0.1)
                    boxes_cur_expand = point_form(boxes_cur_c)

                    # Mask coefficients sparse loss
                    sparse_loss += self.coeff_sparse_loss(mask_coeff_cur)

                    # Mask coefficients diversity loss
                    if self.cfg.MODEL.MASK_HEADS.PROTO_COEFF_DIVERSITY_LOSS:
                        instance_t = idx_t_cur if obj_ids_gt[i] is None else obj_ids_gt[i][idx_t_cur]
                        loss_coeff_div += self.coeff_diversity_loss(mask_coeff_cur, instance_t)

                    # Mask: linear combination between mask coefficients and prototypes
                    if not self.cfg.MODEL.MASK_HEADS.USE_DYNAMIC_MASK:
                        pred_masks = generate_mask(proto_data[i], mask_coeff_cur, boxes_cur_expand)
                        pred_masks = pred_masks.reshape(-1, n_frames, pred_masks.size(-2), pred_masks.size(-1))
                    else:
                        fpn_levels = prior_levels[i][pos_cur.view(-1)]
                        pred_masks = self.net.ProtoNet.DynamicMaskHead(proto_data[i].permute(2, 3, 0, 1).contiguous(),
                                                                       mask_coeff_cur, boxes_cur, fpn_levels)
                        if not self.cfg.MODEL.MASK_HEADS.LOSS_WITH_DICE_COEFF:
                            pred_masks = crop(pred_masks.permute(2, 3, 1, 0).contiguous(), boxes_cur_expand)
                            pred_masks = pred_masks.permute(3, 2, 0, 1).contiguous()

                    if self.cfg.MODEL.MASK_HEADS.LOSS_WITH_OIR_SIZE:  # [n, T, h, w]
                        pred_masks = F.interpolate(pred_masks.float(), (H_gt, W_gt), mode='bilinear',
                                                   align_corners=True)
                        pred_masks = torch.clamp(pred_masks, min=0, max=1)

                    if self.cfg.MODEL.MASK_HEADS.LOSS_WITH_DICE_COEFF:
                        pre_loss = self.dice_coefficient(pred_masks, mask_t)
                    else:
                        # activation_function is sigmoid
                        pre_loss = F.binary_cross_entropy(pred_masks, mask_t, reduction='none')

                    if not self.cfg.MODEL.MASK_HEADS.LOSS_WITH_DICE_COEFF:
                        if self.cfg.MODEL.MASK_HEADS.PROTO_CROP:
                            x1, x2 = sanitize_coordinates(boxes_cur[:, 0], boxes_cur[:, 2], pred_masks.size(-1), 0, cast=True)
                            y1, y2 = sanitize_coordinates(boxes_cur[:, 1], boxes_cur[:, 3], pred_masks.size(-2), 0, cast=True)
                            pos_gt_box_w, pos_gt_box_h = torch.clamp(x2-x1, min=1), torch.clamp(y2-y1, min=1)
                            pre_loss = pre_loss.sum(dim=(-1, -2, -3)) / pos_gt_box_w/pos_gt_box_h/n_frames
                        else:
                            with torch.no_grad():
                                # Design weights for background and objects pixels
                                masks_gt_weights = masks_gt_cur.new_ones(N_gt, n_frames, H_gt, W_gt) * (1./H_gt/W_gt)
                                boxes_gt_cir_cur = circum_boxes(boxes_gt[i].reshape(N_gt, -1))
                                tar_obj_idx = crop(masks_gt_cur.new_ones(H_gt, W_gt, N_gt), boxes_gt_cir_cur)
                                tar_obj_idx = tar_obj_idx.permute(2, 0, 1).contiguous().unsqueeze(1)
                                masks_gt_weights += 1./N_gt/torch.sum(tar_obj_idx, dim=(-2, -1), keepdim=True) * tar_obj_idx
                            pre_loss = (pre_loss * masks_gt_weights[idx_t_cur]).sum(dim=(-1, -2, -3)) / n_frames

                    loss += pre_loss.mean()

                    # add a segmentation loss to distinguish intra-class objects
                    # pre_loss_intraclass = F.binary_cross_entropy(pred_masks[mask_t_intraclass == 1],
                    #                                              mask_t[[mask_t_intraclass == 1]], reduction='none')
                    # loss_intraclass += pre_loss_intraclass.mean()

        loss = loss / max(bs, 1) * self.cfg.MODEL.MASK_HEADS.LOSS_ALPHA * min(n_frames, 3.)
        losses = {'M_dice': 2*loss} if self.cfg.MODEL.MASK_HEADS.LOSS_WITH_DICE_COEFF else {'M_bce': loss}
        # losses['M_intraclass'] = loss_intraclass / max(bs, 1) * self.cfg.MODEL.MASK_HEADS.LOSS_ALPHA
        # losses['M_l1'] = sparse_loss / max(bs, 1)
        if self.cfg.MODEL.MASK_HEADS.PROTO_COEFF_DIVERSITY_LOSS:
            losses['M_coeff'] = loss_coeff_div / max(bs, 1)
    
        return losses

    def coeff_sparse_loss(self, coeffs):
        return 0.0001 * torch.abs(coeffs).sum(dim=-1).mean()

    def dice_coefficient(self, x, target):
        eps = 1e-5
        n_inst = x.size(0)
        x = x.reshape(n_inst, -1)
        target = target.reshape(n_inst, -1)
        intersection = (x * target).sum(dim=1)
        union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
        loss = 1. - (2 * intersection / union)
        return loss

    def coeff_diversity_loss(self, coeffs, instance_t):
        """
        coeffs     should be size [num_pos, num_coeffs]
        instance_t should be size [num_pos] and be values from 0 to num_instances-1
        box_t      should be size [num_pos, 4]
        """
    
        coeffs_norm = F.normalize(coeffs, dim=1)
        cos_sim = torch.mm(coeffs_norm, coeffs_norm.t())
        # Rescale to be between 0 and 1
        cos_sim = (cos_sim + 1) / 2
    
        instance_t = instance_t.view(-1)  # juuuust to make sure
        inst_eq = (instance_t[:, None].expand_as(cos_sim) == instance_t[None, :].expand_as(cos_sim)).float()
    
        # If they're the same instance, use cosine distance, else use cosine similarity
        cos_sim_diff = torch.clamp(1 - cos_sim, min=1e-10)
        loss = -1 * (torch.clamp(cos_sim, min=1e-10).log() * inst_eq + cos_sim_diff.log() * (1-inst_eq))
        loss.triu_(diagonal=1)
    
        # Only divide by num_pos once because we're summing over a num_pos x num_pos tensor
        # and all the losses will be divided by num_pos at the end, so just one extra time.
        num = (loss.size(0)-1)*loss.size(0) // 2
        return self.cfg.MODEL.MASK_HEADS.PROTO_COEFF_DIVERSITY_ALPHA*loss.sum() / max(num, 1)

    def proto_divergence_loss(self, protos, masks, boxes, threshold=0.5):
        '''
        :param protos: [bs, h, w, n_frames, n_mask]
        :param masks: ground-truth masks, List: bs * [n_objs, n_frames, h, w]
        :param boxes: ground-truth boxes, List: bs * [n_objs, n_frames, 4]
        :return:
        '''
        bs, h, w, T, M = protos.size()
        # From [0, +infinite] to [0, 1], 1.5 ==> 0.5
        protos = 1 - torch.exp(-0.5*protos)

        losses = torch.tensor(0., device=protos.device)
        count = 0
        for i in range(bs):
            masks_cur = masks[i].unsqueeze(1) if masks[i].dim() == 3 else masks[i]
            masks_cur = F.interpolate(masks_cur.float(), (h, w), mode='bilinear', align_corners=False)
            masks_cur.gt_(0.5)
            n_objs = boxes[i].size(0)
            boxes_cir_cur = circum_boxes(boxes[i].reshape(n_objs, -1)).unsqueeze(1).repeat(1,M,1)

            # Compute response score between each prototype and all instances
            response_score = []
            for n in range(n_objs):
                protos_obj = crop(protos[i], boxes_cir_cur[n])
                response_score.append(mask_iou(protos_obj.permute(2,3,0,1).contiguous(), masks_cur[n].unsqueeze(1)).mean(0))
            response_score = torch.cat(response_score, dim=-1)

            flag, fired_masks = torch.zeros(M, device=protos.device), []
            if boxes[i].dim() == 3:
                iou = jaccard(boxes[i].permute(1, 0, 2).contiguous(), boxes[i].permute(1,0,2).contiguous()).mean(0)
            else:
                iou = jaccard(boxes[i], boxes[i])
            for m in range(M):
                sorted_score, idx = response_score[m].reshape(-1).sort(descending=True)
                fired_idx = idx[sorted_score > threshold]
                if fired_idx.nelement() > 0:
                    flag[m] = 1
                    # If there are overlapping objects, set regions of instances with lower response scores as 0
                    if fired_idx.nelement() > 1:
                        iou_fired = iou[fired_idx][:, fired_idx]
                        iou_fired = torch.triu(iou_fired, diagonal=1)
                        iou_fired_max, _ = torch.max(iou_fired, dim=0)
                        fired_idx = fired_idx[iou_fired_max <= 0.1]
                    fired_masks.append(masks_cur[fired_idx].sum(0))

            if flag.sum() > 0:
                count += 1
                losses += self.dice_coefficient(protos[i][..., flag == 1].permute(3, 2, 0, 1).contiguous(),
                                                torch.stack(fired_masks, dim=0)).mean()

            return losses / max(count, 1) * self.cfg.MODEL.MASK_HEADS.PROTO_DIVERGENCE_LOSS_ALPHA





