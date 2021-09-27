import torch
import torch.nn.functional as F
import torch.nn as nn
from ...utils import encode, center_size, point_form, decode, generate_mask, correlate_operator, DIoU_loss
from ...visualization_temporal import display_correlation_map_patch
from ..track_to_segment_head import bbox_feat_extractor
from fvcore.nn import giou_loss


class T2SLoss(nn.Module):
    def __init__(self, cfg, net):
        super(T2SLoss, self).__init__()
        self.cfg = cfg
        self.net = net
        self.clip_frames = cfg.SOLVER.NUM_CLIP_FRAMES

        self.forward_flow = cfg.STMASK.T2S_HEADS.FORWARD_FLOW
        self.backward_flow = cfg.STMASK.T2S_HEADS.BACKWARD_FLOW
        if not self.forward_flow and not self.backward_flow:
            self.forward_flow = True

    def forward(self, features, gt_boxes_unflod, gt_masks_unfold, prototypes):
        '''
        Track to segment Loss
        :param features: features used to compute correlation of two frame
        :param gt_boxes_unflod: List[torch.tensor], [a1, a2, b1, b2, c1,c2]
        :param gt_masks_unfold: List[torch.tensor]
        :param prototypes: List[torch.tensor], a list of prototypes
        :return:
        '''

        losses_shift = {'B_shift': [], 'BIoU_shift': []}
        if self.cfg.STMASK.T2S_HEADS.TRAIN_MASKSHIFT:
            losses_shift['M_shift'] = []
        # Complie temporal fusion module of STMask in CVPR2021  https://github.com/MinghanLi/STMask
        for cdx in range(self.clip_frames-1):
            gt_boxes_ref, gt_boxes_tar = gt_boxes_unflod[cdx::self.clip_frames], gt_boxes_unflod[cdx+1::self.clip_frames]
            gt_masks_ref, gt_masks_tar = gt_masks_unfold[cdx::self.clip_frames], gt_masks_unfold[cdx+1::self.clip_frames]
            bb_feat_ref = features[cdx::self.clip_frames].contiguous()
            bb_feat_tar = features[cdx+1::self.clip_frames].contiguous()
            prototypes_ref, prototypes_tar = prototypes[cdx::self.clip_frames], prototypes[cdx+1::self.clip_frames]
            if self.forward_flow:
                losses_shift_for = self.track_to_segment_loss(bb_feat_ref, bb_feat_tar, gt_boxes_tar, gt_masks_tar,
                                                              gt_boxes_ref, prototypes_tar)
                for k, v in losses_shift_for.items():
                    losses_shift[k].append(v)

            if self.backward_flow:
                losses_shift_back = self.track_to_segment_loss(bb_feat_tar, bb_feat_ref, gt_boxes_ref, gt_masks_ref,
                                                               gt_boxes_tar, prototypes_ref)
                for k, v in losses_shift_back.items():
                    losses_shift[k].append(v)

            for k, v in losses_shift.items():
                losses_shift[k] = torch.stack(v).mean()

        return losses_shift

    def track_to_segment_loss(self, bb_feat_ref, bb_feat_tar, gt_boxes_tar, gt_masks_tar, gt_boxes_ref, prototypes_tar,
                              interpolation_mode='bilinear'):
        '''
        Compute displacements of bounding boxes and mask coefficients between two adjacent frames
        :param bb_feat_ref: features in reference frame used to compute correlation
        :param bb_feat_tar: features in target frame used to compute correlation
        :param gt_boxes_ref: bs * List[torch.tensor], ground-truth bounding boxes in the reference frame
        :param prototypes_tar: [bs, h, w, 1, n_mask], prototypes of matched samples in target frame
        :param gt_boxes_tar: a list of ground-truth bounding boxes
        :param gt_masks_tar: a list of ground-truth masks
        :param interpolation_mode: 'bilinear'
        :return:
        '''

        bs, _, feat_h, feat_w = bb_feat_ref.size()
        # Compute teh correlation of features of two adjacent frames, inspired by Optical Flow
        corr = correlate_operator(bb_feat_ref, bb_feat_tar,
                                  patch_size=self.cfg.STMASK.T2S_HEADS.CORRELATION_PATCH_SIZE,
                                  kernel_size=1)
        concat_feat = torch.cat((bb_feat_ref, corr, bb_feat_tar), dim=1)

        loss_B_shift = torch.tensor(0., device=bb_feat_ref.device)
        loss_BIoU_shift = torch.tensor(0., device=bb_feat_ref.device)
        loss_mask_shift = torch.tensor(0., device=bb_feat_ref.device)
        for i in range(bs):
            if gt_boxes_ref[i].nelement() == 0:
                continue

            # Compute ground-truth displacements for instances that exists in both two frames,
            # if some instances disappear in one of frames, the displacements should be 0
            gt_boxes_tar_c = center_size(gt_boxes_tar[i])
            gt_boxes_ref_c = center_size(gt_boxes_ref[i])
            valid_flag = ((gt_boxes_tar_c[:, 2:] > 0).sum(dim=-1) == 2) & ((gt_boxes_ref_c[:, 2:]>0).sum(dim=-1)==2)
            if valid_flag.sum() > 0:
                # Extract features on the predicted bbox, and align cropped features of bounding boxes as 7*7
                cur_gt_boxes_ref_c = center_size(gt_boxes_ref[i][valid_flag])
                cur_gt_boxes_ref_c[:, 2:] *= 1.2
                cur_gt_boxes_ref = torch.clamp(point_form(cur_gt_boxes_ref_c), min=0, max=1)
                boxes_feats = bbox_feat_extractor(concat_feat[i], cur_gt_boxes_ref, feat_h, feat_w, 7)
                # display_correlation_map_patch(boxes_feats[:, 256:377])
                if self.cfg.STMASK.T2S_HEADS.TRAIN_MASKSHIFT:
                    shift_boxes_reg, shift_mask_coeff_reg = self.net.TemporalNet(boxes_feats)

                    # ----------- Compute Maskshift loss ---------------
                    # Generate masks between shifted mask coefficients and prototypes in the key frame
                    pred_masks_tar = generate_mask(prototypes_tar[i], shift_mask_coeff_reg,
                                                   gt_boxes_tar[i][valid_flag]).squeeze(1)
                    h, w = pred_masks_tar.size()[-2:]
                    ds_masks_tar_gt_tar = F.interpolate(gt_masks_tar[i][valid_flag].unsqueeze(0).float(), (h, w),
                                                        mode=interpolation_mode,
                                                        align_corners=False).squeeze(0)
                    ds_masks_tar_gt_tar.gt_(0.5)
                    pre_loss = F.binary_cross_entropy(pred_masks_tar, ds_masks_tar_gt_tar, reduction='none')
                    if self.cfg.MODEL.MASK_HEADS.PROTO_CROP:
                        boxes_tar_gt_cur_c = center_size(gt_boxes_tar[i][valid_flag])
                        gt_box_width = torch.clamp(boxes_tar_gt_cur_c[:, 2]*w, min=1)
                        gt_box_height = torch.clamp(boxes_tar_gt_cur_c[:, 3]*h, min=1)
                        pre_loss = pre_loss.sum(dim=(1, 2)) / gt_box_width / gt_box_height
                        loss_mask_shift += pre_loss.mean()
                    else:
                        loss_mask_shift += pre_loss.mean() / h / w

                else:
                    shift_boxes_reg = self.net.TemporalNet(boxes_feats)

                # ----------------- Compute Boxshift loss ----------------------
                shift_boxes_reg_gt = encode(gt_boxes_tar[i][valid_flag], center_size(gt_boxes_ref[i][valid_flag]))
                pre_loss_B = F.smooth_l1_loss(shift_boxes_reg, shift_boxes_reg_gt, reduction='none').sum(dim=-1)
                loss_B_shift += pre_loss_B.mean()

                # BIoU_shift loss
                shift_boxes_tar = decode(shift_boxes_reg, center_size(gt_boxes_ref[i][valid_flag]))
                if self.cfg.MODEL.PREDICTION_HEADS.USE_DIoU:
                    loss_BIoU_shift += DIoU_loss(shift_boxes_tar, gt_boxes_tar[i][valid_flag], reduction='mean')
                else:
                    loss_BIoU_shift += giou_loss(shift_boxes_tar, gt_boxes_tar[i][valid_flag], reduction='mean')

        losses = {'B_shift': loss_B_shift / bs * self.cfg.MODEL.BOX_HEADS.LOSS_ALPHA,
                  'BIoU_shift': loss_BIoU_shift / bs * self.cfg.MODEL.BOX_HEADS.BIoU_ALPHA}
        if self.cfg.STMASK.T2S_HEADS.TRAIN_MASKSHIFT:
            losses['M_shift'] = loss_mask_shift / bs * self.cfg.STMASK.T2S_HEADS.MASKSHIFT_ALPHA

        return losses