import torch
import torch.nn.functional as F
from layers.utils import center_size, point_form, decode, generate_mask, correlate_operator, generate_track_gaussian,\
    mask_iou, crop
from layers.modules import bbox_feat_extractor
from datasets import cfg
from utils import timer
import matplotlib.pyplot as plt
import os
from ..visualization_temporal import display_box_shift, display_correlation_map_patch, display_correlation_map


def CandidateShift(net, candidate, ref_candidate, img=None, img_meta=None, update_track=False,
                   display=False):
        """
        The function try to shift the candidates of reference frame to that of target frame.
        The most important step is to shift the bounding box of reference frame to that of target frame
        :param net: network
        :param candidate: ptoto_data [1, h, w, 32] the proto of masks for the next frame
        :param ref_candidate: the candidate dictionary that includes 'box', 'conf', 'mask_coeff', 'track' items.
        :return: candidates on the target frame
        """

        proto_next = candidate['proto'].squeeze(0)
        fpn_feat_next = candidate['fpn_feat']
        fpn_feat_ref = ref_candidate['fpn_feat']
        sem_seg_next = candidate['sem_seg'].squeeze(0) if 'sem_seg' in candidate.keys() else None
        ref_candidate['tracked_mask'] += 1

        if ref_candidate['box'].size(0) > 0:

            if cfg.temporal_fusion_module:
                # we only use the features in the P3 layer to perform correlation operation
                corr = correlate_operator(fpn_feat_ref, fpn_feat_next,
                                          patch_size=cfg.correlation_patch_size,
                                          kernel_size=3)
                # display_correlation_map(corr, img_meta)
                concat_features = torch.cat([fpn_feat_ref, corr, fpn_feat_next], dim=1)

                # extract features on the predicted bbox
                if cfg.t2s_with_roialign:
                    # Align cropped features of bounding boxes as 7*7
                    feat_h, feat_w = fpn_feat_ref.size()[2:]
                    bbox_feat_input = bbox_feat_extractor(concat_features, ref_candidate['box'], feat_h, feat_w, 7)
                else:
                    # Use entire features for temporal fusion module to predict bounding box offsets between frames
                    bbox_feat_input = crop(concat_features.repeat(ref_candidate['box'].size(0), 1, 1, 1).permute(2,3,1,0).contiguous(),
                                           ref_candidate['box']).permute(3,2,0,1).contiguous()
                if cfg.train_mask and cfg.maskshift_loss:
                    loc_ref_shift, mask_coeff_shift = net.TemporalNet(bbox_feat_input)
                    ref_candidate['mask_coeff'] += mask_coeff_shift
                else:
                    loc_ref_shift = net.TemporalNet(bbox_feat_input)
                ref_candidate['box'] = decode(loc_ref_shift, center_size(ref_candidate['box']))

                # display = True
                if display:
                    display_correlation_map_patch(bbox_feat_input[:, 256:377], img_meta=img_meta)
                    # display_box_shift(box_ref, box_ref_shift, img_meta=img_meta, img_gpu=img)

                # decay_rate = 0.9 ** ref_candidate['tracked_mask'] if 'tracked_mask' in candidate.keys() else 0.9
                ref_candidate['score'] *= 0.9
                if cfg.train_masks:
                    if cfg.mask_proto_coeff_occlusion:
                        pred_masks_all = generate_mask(proto_next, ref_candidate['mask_coeff'], ref_candidate['box'])
                        pred_masks_all = F.softmax(pred_masks_all, dim=-1)
                        pred_masks_next = pred_masks_all[:, :, :, 1]
                        pred_masks_next_non_target = pred_masks_all[:, :, :, -1]
                    else:
                        pred_masks_next = generate_mask(proto_next, ref_candidate['mask_coeff'], ref_candidate['box'])

            else:

                # FEELVOS
                # we only use the features in the P3 layer to perform correlation operation
                # display_correlation_map(fpn_ref, img_meta, idx)

                fpn_feat_next = F.normalize(fpn_feat_next, dim=1)
                fpn_feat_ref = F.normalize(ref_candidate['fpn_feat'], dim=1)
                ref_class = ref_candidate['class']
                n_ref = ref_candidate['box'].size(0)
                mask_pred_h, mask_pred_w = proto_next.size()[:2]
                feat_h, feat_w = fpn_feat_ref.size()[2:]
                downsampled_pred_masks_ref = F.interpolate(ref_candidate['mask'].unsqueeze(0).float(), (feat_h, feat_w),
                                                           mode='bilinear', align_corners=False).squeeze(0).gt(0.3)
                corr, sem_seg_next_obj, attention_next = [], [], []
                for j in range(n_ref):
                    corr.append(correlate_operator(fpn_feat_next,
                                                   fpn_feat_ref * downsampled_pred_masks_ref[j].view(1, 1, feat_h, feat_w),
                                                   patch_size=cfg.correlation_patch_size, kernel_size=3, dilation_patch=1))
                    if sem_seg_next is not None:
                        sem_seg_next_obj.append(sem_seg_next[:, :, ref_class[j]-1])

                if sem_seg_next is not None:
                    sem_seg_next_obj = torch.stack(sem_seg_next_obj, dim=0)

                corr = torch.cat(corr, dim=0)
                upsampled_corr = F.interpolate(corr.float(), (mask_pred_h, mask_pred_w),
                                               mode='bilinear', align_corners=False)
                upsampled_corr, _ = torch.max(upsampled_corr, dim=1)  # [n_ref, h, w]

                if cfg.use_FEELVOS:
                    upsampled_fpn_feat_next = F.interpolate(fpn_feat_next.float(), (mask_pred_h, mask_pred_w),
                                                            mode='bilinear', align_corners=False)
                    attention_next = net.VOS_attention(upsampled_fpn_feat_next)

                    pred_masks_next = []
                    for j in range(n_ref):
                        # norm_upsampled_corr = upsampled_corr[j] / torch.clamp(upsampled_corr[j].max(), min=1e-5)
                        vos_feat = upsampled_fpn_feat_next * upsampled_corr[j].unsqueeze(0).unsqueeze(1)
                        vos_mask_temp = net.VOS_head(vos_feat) * attention_next  # [1, 1, mask_pred_h, mask_pred_w]
                        vos_mask_temp = cfg.mask_proto_mask_activation(vos_mask_temp.squeeze(0))  # .gt_(0.5)
                        pred_masks_next.append(vos_mask_temp)
                    pred_masks_next = torch.cat(pred_masks_next, dim=0)
                else:
                    max_val = upsampled_corr.view(n_ref, -1).max(dim=-1)[0]
                    upsampled_corr = upsampled_corr / max_val.view(-1, 1, 1)
                    # heat_map = torch.zeros_like(upsampled_corr)
                    heat_map = torch.clamp(upsampled_corr - 0.1, min=0)
                    pred_masks_next_ori = generate_mask(proto_next, ref_candidate['mask_coeff'])
                    pred_masks_next = pred_masks_next_ori * heat_map

            if cfg.train_mask:
                ref_candidate['mask'] = pred_masks_next
                if cfg.mask_proto_coeff_occlusion:
                    ref_candidate['mask_non_target'] = pred_masks_next_non_target

            if 'frame_id' in candidate.keys():
                ref_candidate['frame_id'] = candidate['frame_id']
            if cfg.track_by_Gaussian and update_track:
                ref_candidate['track_mu'], ref_candidate['track_var'] = generate_track_gaussian(ref_candidate['track'],
                                                                                                pred_masks_next.gt(0.5),
                                                                                                ref_candidate['box'])
        return ref_candidate


def merge_candidates(candidate, ref_candidate_clip_shift):
    merged_candidate = {}
    for k, v in candidate.items():
        merged_candidate[k] = v.clone()

    for ref_candidate in ref_candidate_clip_shift:
        if ref_candidate['box'].nelement() > 0:
            for k, v in merged_candidate.items():
                if k not in {'proto', 'fpn_feat'}:
                    merged_candidate[k] = torch.cat([v.clone(), ref_candidate[k].clone()], dim=0)

    return merged_candidate



