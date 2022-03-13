import torch
import torch.nn.functional as F
from ...utils import center_size, point_form, generate_mask, jaccard, crop, sanitize_coordinates, circum_boxes, mask_iou


class LincombMaskLoss(object):
    def __init__(self, cfg, net):
        super(LincombMaskLoss, self).__init__()
        self.cfg = cfg
        self.net = net
        self.clip_frames = cfg.SOLVER.NUM_CLIP_FRAMES
        self.decay_weights = 0.9**(torch.arange(self.clip_frames).view(1, -1)
                                   - torch.arange(self.clip_frames).view(-1, 1)).abs()

    def __call__(self, pos, idx_t, pred_boxes_pos, boxes_t_pos, mask_coeff, prior_levels, proto_data,
                 masks_gt, boxes_gt, obj_ids_gt, labels_gt):
        '''
        Compute instance segmentation loss proposed by YOLACT or CondInst. The former combines linearly
        mask coefficients and prototypes, while the latter feeds mask coefficients and prototypes into
        a small FCN with three dynamic filters. Please link here https://github.com/dbolya/yolact or
        https://github.com/aim-uofa/AdelaiDet/blob/master/configs/CondInst/README.md for more information.
        Args:
                        pos: [bs, n_anchors], positive samples flag, which is 1 if positive samples else 0
                      idx_t: [bs, n_anchors], store the matched index of predicted boxes and ground-truth boxes
             pred_boxes_pos: List: bs * [n_pos_frame, clip_frames, 4], the predicted boxes of positive samples
                boxes_t_pos: List: bs * [n_pos, 4]
                 mask_coeff: [bs, n_anchors, n_mask], mask coefficients, a predicted cubic box of a clip share
                                   same mask coefficients due to temporal redundancy
               prior_levels: [bs, n_anchors]
                 proto_data: [bs, h, w, n_mask], or [bs, h, w, n_frames, n_mask] prototypes
                   masks_gt: List: bs * [n_objs, n_frames, h, w]
                 obj_ids_gt:
                  labels_gt: ground-truth labels
        :return: a dict with mask loss
        '''

        bs, h, w = proto_data.size()[:3]
        loss = torch.tensor(0., device=idx_t.device)
        loss_coeff_div = torch.tensor(0., device=idx_t.device)
        T_out = mask_coeff.size(0) // bs

        for i in range(bs):
            # deal with the i-th clip or the i-th frame
            if pred_boxes_pos[i].size(0) > 0:
                ind_range = range(i*T_out, (i+1)*T_out)
                pos_cur = pos[ind_range] > 0
                if pos_cur.sum() > 0:
                    matched_frame_idx, _ = torch.nonzero(pos_cur, as_tuple=True)
                    decay_weights = self.decay_weights[matched_frame_idx].cuda()
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

                    # Only use circumscribed boxes to crop mask
                    boxes_cur = circum_boxes(pred_boxes_pos[i].reshape(pred_boxes_pos[i].size(0), -1).detach()) \
                        if self.cfg.MODEL.MASK_HEADS.PROTO_CROP_WITH_PRED_BOX else \
                        circum_boxes(boxes_t_pos[i].reshape(boxes_t_pos[i].size(0), -1))
                    boxes_cur_c = center_size(boxes_cur)
                    # For small objects, we set width and height of bounding boxes as 0.1
                    boxes_cur_c[:, 2:] = torch.clamp(boxes_cur_c[:, 2:]*1.5, min=0.1)
                    boxes_cur_expand = point_form(boxes_cur_c)

                    # Mask coefficients diversity loss
                    if self.cfg.MODEL.MASK_HEADS.PROTO_COEFF_DIVERSITY_LOSS:
                        instance_t = idx_t_cur if obj_ids_gt[i] is None else obj_ids_gt[i][idx_t_cur]
                        loss_coeff_div += self.coeff_diversity_loss(mask_coeff_cur, instance_t)

                    if self.cfg.MODEL.MASK_HEADS.USE_DYNAMIC_MASK:
                        # Mask generation by CondInst
                        fpn_levels = prior_levels[i][pos_cur.view(-1)]
                        pred_masks = self.net.DynamicMaskHead(proto_data[i].permute(2, 3, 0, 1).contiguous(),
                                                              mask_coeff_cur, boxes_cur, fpn_levels)
                        if not self.cfg.MODEL.MASK_HEADS.LOSS_WITH_DICE_COEFF:
                            pred_masks = crop(pred_masks.permute(2, 3, 1, 0).contiguous(), boxes_cur_expand)
                            pred_masks = pred_masks.permute(3, 2, 0, 1).contiguous()
                    else:
                        # Mask generation by Yolact: linear combination between mask coefficients and prototypes
                        pred_masks = generate_mask(proto_data[i], mask_coeff_cur, boxes_cur_expand)
                        pred_masks = pred_masks.reshape(-1, n_frames, pred_masks.size(-2), pred_masks.size(-1))

                    if self.cfg.MODEL.MASK_HEADS.LOSS_WITH_OIR_SIZE:  # [n, T, h, w]
                        # Upsample masks to the input size of frames to compute mask loss,
                        # which needs more memory but can obtain better performance
                        pred_masks = F.interpolate(pred_masks.float(), (H_gt, W_gt), mode='bilinear',
                                                   align_corners=True)
                        pred_masks = torch.clamp(pred_masks, min=0, max=1)

                    if self.cfg.MODEL.MASK_HEADS.LOSS_WITH_DICE_COEFF:
                        pre_loss = self.dice_coefficient(pred_masks, mask_t)
                    else:
                        pre_loss = F.binary_cross_entropy(pred_masks, mask_t, reduction='none')

                        # Crop masks to compute loss
                        if self.cfg.MODEL.MASK_HEADS.PROTO_CROP:
                            x1, x2 = sanitize_coordinates(boxes_cur[:, 0], boxes_cur[:, 2],
                                                          pred_masks.size(-1), 0, cast=True)
                            y1, y2 = sanitize_coordinates(boxes_cur[:, 1], boxes_cur[:, 3],
                                                          pred_masks.size(-2), 0, cast=True)
                            pos_gt_box_area = torch.clamp(x2-x1, min=1) * torch.clamp(y2-y1, min=1)
                            pre_loss = pre_loss.sum(dim=(-1, -2)) / pos_gt_box_area.view(-1, 1)
                        else:
                            with torch.no_grad():
                                # Design weights for background and objects pixels
                                masks_gt_weights = masks_gt_cur.new_ones(N_gt, n_frames, H_gt, W_gt) * (1./H_gt/W_gt)
                                boxes_gt_cir_cur = circum_boxes(boxes_gt[i].reshape(N_gt, -1))
                                tar_obj_idx = crop(masks_gt_cur.new_ones(H_gt, W_gt, N_gt), boxes_gt_cir_cur)
                                tar_obj_idx = tar_obj_idx.permute(2, 0, 1).contiguous().unsqueeze(1)
                                masks_gt_weights += 1./N_gt/torch.sum(tar_obj_idx, dim=(-2, -1), keepdim=True)*tar_obj_idx
                            pre_loss = (pre_loss * masks_gt_weights[idx_t_cur]).sum(dim=(-1, -2))

                        # decay_weights：as the interval increases, the weight decreases.
                        pre_loss = (decay_weights * pre_loss).sum(-1) / decay_weights.sum(-1)

                    loss += pre_loss.mean()

        loss = loss / max(bs, 1) * self.cfg.MODEL.MASK_HEADS.LOSS_ALPHA * min(n_frames, 5.)
        losses = {'M_dice': 2*loss} if self.cfg.MODEL.MASK_HEADS.LOSS_WITH_DICE_COEFF else {'M_bce': loss}
        if self.cfg.MODEL.MASK_HEADS.PROTO_COEFF_DIVERSITY_LOSS:
            losses['M_coeff'] = loss_coeff_div / max(bs, 1)
    
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




