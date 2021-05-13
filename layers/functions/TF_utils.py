import torch
import torch.nn.functional as F
from layers.box_utils import jaccard, center_size, point_form, decode, crop, mask_iou
from layers.mask_utils import generate_mask
from layers.modules import correlate, bbox_feat_extractor
from layers.visualization import display_box_shift, display_correlation_map
from datasets import cfg
from utils import timer


def CandidateShift(net, fpn_feat_next, ref_candidate, img=None, img_meta=None):
        """
        The function try to shift the candidates of reference frame to that of target frame.
        The most important step is to shift the bounding box of reference frame to that of target frame
        :param net: network
        :param ref_candidate: the candidate dictionary that includes 'box', 'conf', 'mask_coeff', 'track' items.
        :param correlation_patch_size: the output size of roialign
        :return: candidates on the target frame
        """
        # ref_candidate_shift = {}
        # for k, v in ref_candidate.items():
        #     ref_candidate_shift[k] = v.clone()

        if ref_candidate['box'].size(0) == 0:
            ref_candidate_shift = {'box': torch.tensor([])}

        else:
            # we only use the features in the P3 layer to perform correlation operation
            fpn_feat_ref = ref_candidate['fpn_feat']
            corr = correlate(fpn_feat_ref, fpn_feat_next, patch_size=cfg.correlation_patch_size, kernel_size=3)
            # display_correlation_map(fpn_ref, img_meta, idx)

            if cfg.temporal_fusion_module:
                concatenated_features = F.relu(torch.cat([corr, fpn_feat_ref, fpn_feat_next], dim=1))

                # extract features on the predicted bbox
                box_ref_c = center_size(ref_candidate['box'])
                # we use 1.2 box to crop features
                box_ref_crop = point_form(torch.cat([box_ref_c[:, :2],
                                                     torch.clamp(box_ref_c[:, 2:] * 1.2, min=0, max=1)], dim=1))
                bbox_feat_input = bbox_feat_extractor(concatenated_features, box_ref_crop, 7)
                loc_ref_shift, mask_coeff_shift = net.TemporalNet(bbox_feat_input)
                box_ref_shift = torch.cat([(loc_ref_shift[:, :2] * box_ref_c[:, 2:] + box_ref_c[:, :2]),
                                           torch.exp(loc_ref_shift[:, 2:]) * box_ref_c[:, 2:]], dim=1)
                box_ref_shift = point_form(box_ref_shift)

                ref_candidate_shift = {'box': box_ref_shift.clone()}
                ref_candidate_shift['conf'] = ref_candidate['conf'].clone() * 0.8
                ref_candidate_shift['mask_coeff'] = ref_candidate['mask_coeff'].clone() + mask_coeff_shift

            else:

                # FEELVOS
                n_ref = ref_candidate['box'].size(0)
                proto_ref = ref_candidate['proto']
                mask_coeff_ref = cfg.mask_proto_coeff_activation(ref_candidate['mask_coeff'])
                mask_pred_h, mask_pred_w = proto_ref.size()[:2]
                upsampled_fpn_feat_next = F.interpolate(fpn_feat_next.float(), (mask_pred_h, mask_pred_w),
                                                        mode='bilinear', align_corners=False)
                upsampled_corr = F.interpolate(corr.float(), (mask_pred_h, mask_pred_w),
                                               mode='bilinear', align_corners=False)

                pred_masks_ref = generate_mask(proto_ref, mask_coeff_ref, ref_candidate['box'])
                pred_masks_ref = pred_masks_ref.gt(0.5).float()
                pred_masks_next = []
                for j in range(n_ref):
                    vos_feat = torch.cat([upsampled_fpn_feat_next, upsampled_corr,
                                          pred_masks_ref[j].unsqueeze(0).unsqueeze(1)], dim=1)
                    pred_masks_next.append(net.VOS_head(vos_feat).squeeze(0))  # [1, mask_pred_h, mask_pred_w]
                ref_candidate_shift = {'mask': cfg.mask_proto_mask_activation(torch.cat(pred_masks_next, dim=0))}

        return ref_candidate_shift


def generate_candidate(predictions):
    batch_Size = predictions['loc'].size(0)
    candidate = []
    prior_data = predictions['priors'].squeeze(0)
    for i in range(batch_Size):
        loc_data = predictions['loc'][i]
        conf_data = predictions['conf'][i]

        candidate_cur = {'fpn_feat': predictions['fpn_feat'][i].unsqueeze(0)}

        with timer.env('Detect'):
            decoded_boxes = decode(loc_data, prior_data)

            conf_data = conf_data.t().contiguous()
            conf_scores, _ = torch.max(conf_data[1:, :], dim=0)

            keep = (conf_scores > cfg.eval_conf_thresh)
            candidate_cur['proto'] = predictions['proto'][i]
            candidate_cur['conf'] = conf_data[:, keep].t()
            candidate_cur['box'] = decoded_boxes[keep, :]
            candidate_cur['mask_coeff'] = predictions['mask_coeff'][i][keep, :]
            candidate_cur['track'] = predictions['track'][i]
            if cfg.train_centerness:
                candidate_cur['centerness'] = predictions['centerness'][i][keep].view(-1)

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



