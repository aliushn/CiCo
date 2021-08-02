import torch
import torch.nn.functional as F
from layers.utils import center_size, point_form, decode, generate_mask, correlate_operator, generate_track_gaussian, mask_iou
from layers.modules import bbox_feat_extractor
from datasets import cfg
from utils import timer
import matplotlib.pyplot as plt
import os
from ..visualization import display_box_shift, display_correlation_map


def CandidateShift(net, candidate, ref_candidate, img=None, img_meta=None, update_track=False,
                   remove_FP=False, display=False):
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

        ref_candidate_shift = {}
        for k, v in candidate.items():
            if k in {'proto', 'fpn_feat', 'sem_seg'}:
                ref_candidate_shift[k] = v
        if cfg.track_by_Gaussian:
            ref_candidate_shift['track'] = candidate['track']

        sem_seg_next = candidate['sem_seg'].squeeze(0) if 'sem_seg' in candidate.keys() else None

        if ref_candidate['box'].size(0) > 0:

            if cfg.temporal_fusion_module:
                # we only use the features in the P3 layer to perform correlation operation
                corr = correlate_operator(fpn_feat_ref, fpn_feat_next,
                                          patch_size=cfg.correlation_patch_size,
                                          kernel_size=3)
                # display_correlation_map(fpn_ref, img_meta, idx)
                concatenated_features = torch.cat([fpn_feat_ref, corr, fpn_feat_next], dim=1)

                # extract features on the predicted bbox
                box_ref = ref_candidate['box'].clone()
                feat_h, feat_w = fpn_feat_ref.size()[2:]
                bbox_feat_input = bbox_feat_extractor(concatenated_features, box_ref, feat_h, feat_w, 7)
                if cfg.maskshift_loss:
                    loc_ref_shift, mask_coeff_shift = net.TemporalNet(bbox_feat_input)
                    ref_candidate_shift['mask_coeff'] = ref_candidate['mask_coeff'].clone() + mask_coeff_shift
                else:
                    loc_ref_shift = net.TemporalNet(bbox_feat_input)
                    ref_candidate_shift['mask_coeff'] = ref_candidate['mask_coeff'].clone()
                box_ref_shift = decode(loc_ref_shift, center_size(box_ref))

                # display = True
                if display:
                    display_correlation_map(bbox_feat_input[:, 256:377], img_meta=img_meta)
                    display_box_shift(box_ref, box_ref_shift, img_meta=img_meta, img_gpu=img)

                ref_candidate_shift['box'] = box_ref_shift.clone()
                # if 'frame_id' in candidate.keys():
                #     decay_rate = 0.8 ** abs(ref_candidate['frame_id'].item() - candidate['frame_id'].item())
                # else:
                #     decay_rate = 0.8
                ref_candidate_shift['score'] = ref_candidate['score'].clone() * 0.9

                if cfg.mask_proto_coeff_occlusion:
                    pred_masks_all = generate_mask(proto_next, ref_candidate_shift['mask_coeff'], box_ref_shift)
                    pred_masks_all = F.softmax(pred_masks_all, dim=-1)
                    pred_masks_next = pred_masks_all[:, :, :, 1]
                    pred_masks_next_non_target = pred_masks_all[:, :, :, -1]
                else:
                    pred_masks_next = generate_mask(proto_next, ref_candidate_shift['mask_coeff'], box_ref_shift)

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

            if remove_FP:
                _, sort_idx = ref_candidate['tracked_mask'].sort(descending=True)
                miou = mask_iou(pred_masks_next[sort_idx].gt(0.5), pred_masks_next[sort_idx].gt(0.5))
                miou.triu_(diagonal=1)
                miou_max, _ = torch.max(miou, dim=1)
                FP_idx = sort_idx[miou_max > 0.85]
                ref_candidate['tracked_mask'][FP_idx] = 100
                ref_candidate_shift['tracked_mask'] = ref_candidate['tracked_mask'].clone()

            ref_candidate_shift['mask'] = pred_masks_next
            if cfg.mask_proto_coeff_occlusion:
                ref_candidate_shift['mask_non_target'] = pred_masks_next_non_target

            if 'frame_id' in candidate.keys():
                ref_candidate_shift['frame_id'] = candidate['frame_id']
            if cfg.track_by_Gaussian and update_track:
                mu_next, var_next = generate_track_gaussian(ref_candidate_shift['track'], pred_masks_next.gt(0.5),
                                                            box_ref_shift)
                ref_candidate_shift['track_mu'] = mu_next
                ref_candidate_shift['track_var'] = var_next

            if remove_FP:
                for k, v in ref_candidate_shift.items():
                    if k not in {'proto', 'fpn_feat', 'track', 'sem_seg', 'tracked_mask'}:
                        ref_candidate_shift[k][FP_idx] = 0

        return ref_candidate_shift


def generate_candidate(predictions):
    batch_Size = predictions['loc'].size(0)

    candidate = []
    for i in range(batch_Size):
        candidate_cur = {}
        if cfg.train_class:
            conf_data = predictions['conf'][i].t().contiguous()
            scores, _ = torch.max(conf_data, dim=0)
        else:
            scores = predictions['stuff'][i].view(-1)

        if 'centerness' in predictions.keys():
            scores *= predictions['centerness'][i].view(-1)

        keep = (scores > cfg.eval_conf_thresh)
        for k, v in predictions.items():
            if k in {'proto', 'fpn_feat', 'sem_seg'}:
                candidate_cur[k] = v[i].unsqueeze(0)
            elif k == 'track':
                if cfg.track_by_Gaussian:
                    candidate_cur[k] = v[i].unsqueeze(0)
                else:
                    candidate_cur[k] = v[i][keep]
            else:
                candidate_cur[k] = v[i][keep]

        candidate_cur['box'] = decode(candidate_cur['loc'], candidate_cur['priors'])
        candidate.append(candidate_cur)

    return candidate


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



