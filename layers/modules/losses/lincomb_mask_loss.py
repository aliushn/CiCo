import torch
import torch.nn.functional as F
from ...utils import center_size, point_form, generate_mask, jaccard, crop, sanitize_coordinates, circum_boxes


class LincombMaskLoss(object):
    def __init__(self, cfg, net):
        super(LincombMaskLoss, self).__init__()
        self.cfg = cfg
        self.net = net
    
    def __call__(self, pos, idx_t, pred_boxes_pos, mask_coeff, prior_levels, proto_data,
                 masks_gt, boxes_t_pos, obj_ids_gt):
        '''
        Compute instance segmentation loss proposed by YOLACT, which combines linearly mask coefficients and prototypes.
        Please link here https://github.com/dbolya/yolact for more information.
        :param          pos: [bs, n_anchors], positive samples flag, which is 1 if positive samples else 0
        :param        idx_t: [bs, n_anchors], store the matched index of predicted boxes and ground-truth boxes
        :param   pred_boxes: List: bs * [n_pos_frame, clip_frames, 4], the predicted boxes of positive samples
        :param   mask_coeff: [bs, n_anchors, n_mask], mask coefficients, a predicted cubic box of a clip share
                                   same mask coefficients due to temporal redundancy
        :param prior_levels: [bs, n_anchors]
        :param   proto_data: [bs, h, w, n_mask], or [bs, h, w, n_frames, n_mask] prototypes
        :param     masks_gt: List: bs * [n_objs, n_frames, h, w]
        :param  boxes_t_pos: List: bs * [n_pos, 4]
        :param   obj_ids_gt:
        :return:
        '''

        bs, h, w = proto_data.size()[:3]
        if proto_data.dim() == 4:
            proto_data = proto_data.unsqueeze(3)
        # Only use circumscribed boxes to crop mask
        pred_boxes_pos = [circum_boxes(boxes) for boxes in pred_boxes_pos]
        boxes_t_pos = [circum_boxes(boxes) for boxes in boxes_t_pos]
        loss, loss_m_occluded = torch.tensor(0., device=idx_t.device), torch.tensor(0., device=idx_t.device)
        loss_coeff_div = torch.tensor(0., device=idx_t.device)
    
        for i in range(bs):
            pos_cur = pos[i] > 0
    
            # deal with the i-th clip
            if pos_cur.sum() > 0:
                idx_t_cur = idx_t[i, pos_cur]
                mask_coeff_cur = mask_coeff[i, pos_cur]
                fpn_levels = prior_levels[i, pos_cur]
                # If the input is given frame by frame style, for example bboxes_gt [n_objs, 4],
                # please expand them in temporal axis like: [n_objs, 1, 4] to better coding
                masks_gt_cur = masks_gt[i].unsqueeze(1) if masks_gt[i].dim() == 3 else masks_gt[i]
                if not self.cfg.MODEL.MASK_HEADS.LOSS_WITH_OIR_SIZE:
                    masks_gt_cur = F.interpolate(masks_gt_cur.float(), (h, w),
                                                 mode='bilinear', align_corners=True)
                    masks_gt_cur.gt_(0.5)
                N_gt, n_frames, H_gt, W_gt = masks_gt_cur.size()

                # get mask loss for ono-occluded part, [n_pos, 3, h, w]
                if N_gt > 1 and self.cfg.MODEL.MASK_HEADS.PROTO_COEFF_OCCLUSION:
                    masks_gt_cur3 = torch.zeros(N_gt, 3, n_frames, H_gt, W_gt)
                    masks_gt_cur3[:, 1] = masks_gt_cur
                    # assign non-target objects as 2, then target objects as 1, background as 0
                    masks_gt_cur_all = masks_gt_cur.sum(dim=0)
                    masks_gt_cur3[:, 0] = (masks_gt_cur_all == 0).repeat(N_gt, 1, 1, 1).float()
                    for m_idx in range(masks_gt_cur.size(0)):
                        target_object = masks_gt_cur[m_idx] > 0
                        non_target_objects = (masks_gt_cur_all > 0) * ~target_object
                        masks_gt_cur3[m_idx, 2][non_target_objects] = 1
                    mask_t = masks_gt_cur3[idx_t_cur]

                else:
                    mask_t = masks_gt_cur[idx_t_cur]                       # [n_pos, n_frames, mask_h, mask_w]
    
                if self.cfg.MODEL.MASK_HEADS.PROTO_COEFF_DIVERSITY_LOSS:
                    instance_t = idx_t_cur if obj_ids_gt[i] is None else obj_ids_gt[i][idx_t_cur]
                    loss_coeff_div += self.coeff_diversity_loss(mask_coeff_cur, instance_t, boxes_t_pos[i])

                if self.cfg.MODEL.MASK_HEADS.PROTO_CROP or self.cfg.MODEL.MASK_HEADS.USE_DYNAMIC_MASK:
                    # Note: this is in point-form
                    boxes_cur = pred_boxes_pos[i].detach() if self.cfg.MODEL.MASK_HEADS.PROTO_CROP_WITH_PRED_BOX \
                        else boxes_t_pos[i]
                    boxes_cur_c = center_size(boxes_cur)
                    boxes_cur_c[:, 2:] *= 1.2
                    # for small objects, we set width and height of bounding boxes is 0.1
                    boxes_cur_c[:, 2:] = torch.clamp(boxes_cur_c[:, 2:], min=0.1)
                    boxes_cur = point_form(boxes_cur_c)
                else:
                    boxes_cur = None

                if not self.cfg.MODEL.MASK_HEADS.USE_DYNAMIC_MASK:
                    pred_masks = generate_mask(proto_data[i], mask_coeff_cur, boxes_cur)
                else:
                    pred_masks = self.net.DynamicMaskHead(proto_data[i].permute(3, 0, 1, 2).contiguous().unsqueeze(0),
                                                          mask_coeff_cur, boxes_cur, fpn_levels)
                    if self.cfg.MODEL.MASK_HEADS.PROTO_CROP:
                        pred_masks = crop(pred_masks.permute(1, 2, 0).contiguous(), boxes_cur)
                        pred_masks = pred_masks.permute(2, 0, 1).contiguous()

                if self.cfg.MODEL.MASK_HEADS.USE_DYNAMIC_MASK or not self.cfg.MODEL.MASK_HEADS.PROTO_COEFF_OCCLUSION:
                    if self.cfg.MODEL.MASK_HEADS.LOSS_WITH_OIR_SIZE:
                        # [n, T, h, w] 
                        pred_masks = F.interpolate(pred_masks.float(), (H_gt, W_gt), mode='bilinear', align_corners=True)
                        pred_masks = torch.clamp(pred_masks, min=0, max=1)

                    if self.cfg.MODEL.MASK_HEADS.LOSS_WITH_DICE_COEFF:
                        pre_loss = self.dice_coefficient(pred_masks, mask_t)
                    else:
                        # activation_function is sigmoid
                        pre_loss = F.binary_cross_entropy(pred_masks, mask_t, reduction='none')

                else:
                    if self.cfg.MODEL.MASK_HEADS.LOSS_WITH_OIR_SIZE:
                        # [n, T, h, w, 3] => [n, 3, T, h, w]
                        pred_masks = pred_masks.permute(0, 4, 1, 2, 3).contiguous()
                        pred_masks = F.interpolate(pred_masks.float(), (1, H_gt, W_gt), mode='trilinear', 
                                                   align_corners=True)
                        pred_masks = torch.clamp(pred_masks, min=0, max=1)

                    if self.cfg.MODEL.MASK_HEADS.LOSS_WITH_DICE_COEFF:
                        cur_loss = []
                        for k in range(3):
                            cur_loss.append(self.dice_coefficient(pred_masks[:, k], mask_t[:, k]))
                        pre_loss = torch.stack(cur_loss, dim=0).mean(dim=0)
                    else:
                        pre_loss = F.binary_cross_entropy(pred_masks, mask_t, reduction='none').sum(dim=1)

                if not self.cfg.MODEL.MASK_HEADS.LOSS_WITH_DICE_COEFF:
                    if self.cfg.MODEL.MASK_HEADS.PROTO_CROP:
                        x1, x2 = sanitize_coordinates(boxes_cur[:, 0], boxes_cur[:, 2], pred_masks.size(-1), 0, cast=True)
                        y1, y2 = sanitize_coordinates(boxes_cur[:, 1], boxes_cur[:, 3], pred_masks.size(-2), 0, cast=True)
                        pos_gt_box_w, pos_gt_box_h = torch.clamp(x2-x1, min=1), torch.clamp(y2-y1, min=1)
                        pre_loss = pre_loss.sum(dim=(-1, -2, -3)) / pos_gt_box_w/pos_gt_box_h/n_frames
                    else:
                        pre_loss = pre_loss.sum(dim=(-1, -2, -3)) / pred_masks.size(-1)/pred_masks.size(-2)/n_frames

                loss += pre_loss.mean()
        loss = loss / bs * self.cfg.MODEL.MASK_HEADS.LOSS_ALPHA
        losses = {'M_dice': loss} if self.cfg.MODEL.MASK_HEADS.LOSS_WITH_DICE_COEFF else {'M_bce': loss}
        if self.cfg.MODEL.MASK_HEADS.PROTO_COEFF_DIVERSITY_LOSS:
            losses['M_coeff'] = loss_coeff_div / max(bs, 1)
    
        return losses

    def coeff_diversity_loss(self, coeffs, instance_t, box_t):
        """
        coeffs     should be size [num_pos, num_coeffs]
        instance_t should be size [num_pos] and be values from 0 to num_instances-1
        box_t      should be size [num_pos, 4]
        """
    
        coeffs_norm = F.normalize(self.cfg.mask_proto_coeff_activation(coeffs), dim=1)
        cos_sim = torch.mm(coeffs_norm, coeffs_norm.t())
        # Rescale to be between 0 and 1
        cos_sim = (cos_sim + 1) / 2
    
        instance_t = instance_t.view(-1)  # juuuust to make sure
        inst_eq = (instance_t[:, None].expand_as(cos_sim) == instance_t[None, :].expand_as(cos_sim))
    
        box_iou = jaccard(box_t, box_t)
        box_iou[inst_eq] = 0
        spatial_neighbours = box_iou > 0.05
    
        # If they're the same instance, use cosine distance, else use cosine similarity
        cos_sim_diff = torch.clamp(1 - cos_sim, min=1e-10)
        loss = -1 * (torch.clamp(cos_sim, min=1e-10).log() * inst_eq.float() + cos_sim_diff.log() * spatial_neighbours.float())
    
        # Only divide by num_pos once because we're summing over a num_pos x num_pos tensor
        # and all the losses will be divided by num_pos at the end, so just one extra time.
        return self.cfg.MODEL.MASK_HEADS.PROTO_COEFF_DIVERSITY_ALPHA*loss.sum() / max(inst_eq.sum() + spatial_neighbours.sum(), 1)

    def dice_coefficient(self, x, target):
        eps = 1e-5
        n_inst = x.size(0)
        x = x.reshape(n_inst, -1)
        target = target.reshape(n_inst, -1)
        intersection = (x * target).sum(dim=1)
        union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
        loss = 1. - (2 * intersection / union)
        return loss