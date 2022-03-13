import torch
import torch.nn.functional as F
import torch.nn as nn
from ...utils import encode, center_size, generate_mask


class T2SLoss(nn.Module):
    '''Track-to-segment Loss
    To compile temporal fusion module of STMask in CVPR2021 (https://github.com/MinghanLi/STMask)
    '''
    def __init__(self, cfg, net):
        super(T2SLoss, self).__init__()
        self.cfg = cfg
        self.net = net
        self.clip_frames = cfg.SOLVER.NUM_CLIP_FRAMES*cfg.SOLVER.NUM_CLIPS
        self.n_clips = cfg.SOLVER.NUM_CLIPS

        self.forward_flow = cfg.STMASK.T2S_HEADS.FORWARD_FLOW
        self.backward_flow = cfg.STMASK.T2S_HEADS.BACKWARD_FLOW
        if not self.forward_flow and not self.backward_flow:
            self.forward_flow = True

    def forward(self, features, gt_boxes_unflod, gt_masks_unfold, prototypes):
        '''
        To compile temporal fusion module of STMask in CVPR2021 (https://github.com/MinghanLi/STMask)
        :param features: features used to compute correlation of two frame
        :param gt_boxes_unflod: List[torch.tensor], [a1, a2, b1, b2, c1,c2]
        :param gt_masks_unfold: List[torch.tensor]
        :param prototypes: List[torch.tensor], a list of prototypes
        :return:
        '''

        losses_shift = {'B_T2S': []}
        if self.cfg.STMASK.T2S_HEADS.TRAIN_MASKSHIFT:
            losses_shift['M_T2S'] = []

        for cdx in range(self.n_clips-1):
            # Note that our dataloader in datasets/ytvos.py has automatically added the missed boxes and masks 
            # by zero values for those frames where some objects disappear. 
            # If your dataloader does not include the function, you can modify your dataloader 
            # or link boxes with same instance IDs from the paired frames.
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

    def track_to_segment_loss(self, feat_ref, feat_tar, boxes_tar, masks_tar, boxes_ref, prototypes_tar,
                              interpolation_mode='bilinear'):
        '''
        Compute displacements of bounding boxes and mask coefficients between two adjacent frames
        :param feat_ref: features in reference frame used to compute correlation
        :param feat_tar: features in target frame used to compute correlation
        :param boxes_ref: bs * List[torch.tensor], ground-truth bounding boxes in the reference frame
        :param prototypes_tar: [bs, h, w, 1, n_mask], prototypes of matched samples in target frame
        :param boxes_tar: a list of ground-truth bounding boxes
        :param masks_tar: a list of ground-truth masks
        :param interpolation_mode: 'bilinear'
        :return:
        '''
        loss_B_shift = torch.tensor(0., device=feat_ref.device)
        loss_BIoU_shift = torch.tensor(0., device=feat_ref.device)
        loss_mask_shift = torch.tensor(0., device=feat_ref.device)
        bs = feat_ref.size(0)
        for i in range(bs):
            if boxes_ref[i].nelement() == 0:
                continue

            # Compute ground-truth displacements for instances that exists in both two frames,
            # Only consider those instances that appear in both two frames
            boxes_ref_c, boxes_tar_c = center_size(boxes_ref[i]), center_size(boxes_tar[i])
            keep = ((boxes_tar_c[:, 2:] > 0) & (boxes_ref_c[:, 2:] > 0)).sum(-1) == 2

            # Obtain boxes offsets between two frames and mask coefficients
            pred_boxes_off, pred_mask_coeff_tar = self.net.T2S_Head(feat_ref[i], feat_tar[i], boxes_ref[i][keep])

            # ----------------- Compute Boxshift loss ----------------------
            gt_boxes_off = encode(boxes_tar[i][keep], center_size(boxes_ref[i][keep]))
            pre_loss_B = F.smooth_l1_loss(pred_boxes_off, gt_boxes_off, reduction='none').sum(dim=-1)
            loss_B_shift += pre_loss_B.mean()

            # BIoU_shift loss
            # pred_boxes_tar = decode(pred_boxes_off, center_size(boxes_ref[i]))
            # IoU_loss = DIoU_loss if self.cfg.MODEL.PREDICTION_HEADS.USE_DIoU else giou_loss()
            # loss_BIoU_shift += IoU_loss(pred_boxes_tar, boxes_tar[i][keep], reduction='mean')

            if keep.sum() > 0:
                if pred_mask_coeff_tar is not None:
                    # ----------- Compute Maskshift loss ---------------
                    # Generate masks between shifted mask coefficients and prototypes in the key frame
                    pred_masks_tar = generate_mask(prototypes_tar[i], pred_mask_coeff_tar,
                                                   boxes_tar[i][keep]).squeeze(1)
                    h, w = pred_masks_tar.size()[-2:]
                    ds_masks_tar = F.interpolate(masks_tar[i][keep].unsqueeze(0).float(), (h, w),
                                                 mode=interpolation_mode,
                                                 align_corners=False).squeeze(0).gt_(0.5)
                    pre_loss = F.binary_cross_entropy(pred_masks_tar, ds_masks_tar, reduction='none')
                    if self.cfg.MODEL.MASK_HEADS.PROTO_CROP:
                        boxes_tar_cur_c = center_size(boxes_tar[i][keep])
                        boxes_width = torch.clamp(boxes_tar_cur_c[:, 2]*w, min=1)
                        boxes_height = torch.clamp(boxes_tar_cur_c[:, 3]*h, min=1)
                        pre_loss = pre_loss.sum(dim=(1, 2)) / boxes_width / boxes_height
                        loss_mask_shift += pre_loss.mean()
                    else:
                        loss_mask_shift += pre_loss.mean() / h / w

        losses = {'B_T2S': loss_B_shift / bs * self.cfg.MODEL.BOX_HEADS.LOSS_ALPHA}
                  # 'BIoU_T2S': loss_BIoU_shift / bs * self.cfg.MODEL.BOX_HEADS.BIoU_ALPHA}
        if self.cfg.STMASK.T2S_HEADS.TRAIN_MASKSHIFT:
            losses['M_T2S'] = loss_mask_shift / bs * self.cfg.STMASK.T2S_HEADS.MASKSHIFT_ALPHA

        return losses