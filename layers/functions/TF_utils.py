import torch
import torch.nn.functional as F
from layers.utils import center_size, point_form, decode, crop, generate_mask, correlate_operator
from layers.modules import bbox_feat_extractor
from datasets import cfg
from utils import timer
import matplotlib.pyplot as plt
import os


def CandidateShift(net, fpn_feat_next, ref_candidate, img=None, img_meta=None):
        """
        The function try to shift the candidates of reference frame to that of target frame.
        The most important step is to shift the bounding box of reference frame to that of target frame
        :param net: network
        :param ref_candidate: the candidate dictionary that includes 'box', 'conf', 'mask_coeff', 'track' items.
        :param correlation_patch_size: the output size of roialign
        :return: candidates on the target frame
        """

        if ref_candidate['box'].size(0) == 0:
            ref_candidate_shift = {'box': torch.tensor([])}

        else:

            if cfg.temporal_fusion_module:
                # we only use the features in the P3 layer to perform correlation operation
                fpn_feat_ref = ref_candidate['fpn_feat']
                corr = correlate_operator(fpn_feat_next, fpn_feat_ref,
                                          patch_size=cfg.correlation_patch_size, kernel_size=3, dilation_patch=1)
                # display_correlation_map(fpn_ref, img_meta, idx)
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
                # we only use the features in the P3 layer to perform correlation operation
                fpn_feat_ref = ref_candidate['fpn_feat']
                # display_correlation_map(fpn_ref, img_meta, idx)
                n_ref = ref_candidate['box'].size(0)
                proto_ref = ref_candidate['proto']
                mask_coeff_ref = cfg.mask_proto_coeff_activation(ref_candidate['mask_coeff'])
                mask_pred_h, mask_pred_w = proto_ref.size()[:2]
                upsampled_fpn_feat_next = F.interpolate(fpn_feat_next.float(), (mask_pred_h, mask_pred_w),
                                                        mode='bilinear', align_corners=False)
                attention_next = net.VOS_attention(upsampled_fpn_feat_next)

                pred_masks_ref = generate_mask(proto_ref, mask_coeff_ref, ref_candidate['box'])
                pred_masks_ref = pred_masks_ref.gt(0.5).float()
                seg_type = 2
                if seg_type == 1:
                    corr = correlate_operator(fpn_feat_next, fpn_feat_ref,
                                              patch_size=cfg.correlation_patch_size, kernel_size=3, dilation_patch=1)
                    upsampled_corr = F.interpolate(corr.float(), (mask_pred_h, mask_pred_w),
                                                   mode='bilinear', align_corners=False)
                    upsampled_corr = torch.max(upsampled_corr, dim=1)[0]  # [1, h, w]
                    # extract features on the predicted bbox
                    box_ref_c = center_size(ref_candidate['box'])
                    # we use 1.2 box to crop features
                    box_ref_crop = point_form(torch.cat([box_ref_c[:, :2],
                                                         torch.clamp(box_ref_c[:, 2:] * 1.5, min=0, max=1)], dim=1))
                    _, upsampled_corr = crop(upsampled_corr.permute(1, 2, 0).contiguous().repeat(1, 1, n_ref),
                                             box_ref_crop.view(n_ref, -1))
                    upsampled_corr = upsampled_corr.permute(2, 0, 1).contiguous()  # [n_ref, h, w]

                elif seg_type == 2:
                    feat_h, feat_w = fpn_feat_ref.size()[2:]
                    downsampled_pred_masks_ref = F.interpolate(pred_masks_ref.unsqueeze(0).float(), (feat_h, feat_w),
                                                               mode='bilinear', align_corners=False).squeeze(0)
                    corr = []
                    for j in range(n_ref):
                        corr.append(correlate_operator(fpn_feat_next,
                                                       fpn_feat_ref * downsampled_pred_masks_ref[j].view(1, 1, feat_h, feat_w),
                                                       patch_size=cfg.correlation_patch_size, kernel_size=3))
                    corr = torch.cat(corr, dim=0)
                    upsampled_corr = F.interpolate(corr.float(), (mask_pred_h, mask_pred_w),
                                                   mode='bilinear', align_corners=False)
                    upsampled_corr = torch.max(upsampled_corr, dim=1)[0]  # [n_ref, h, w]

                pred_masks_next = []
                for j in range(n_ref):
                    norm_upsampled_corr = upsampled_corr[j] / torch.clamp(upsampled_corr[j].max(), min=1e-5)
                    vos_feat = upsampled_fpn_feat_next * norm_upsampled_corr.unsqueeze(0).unsqueeze(1)
                    vos_mask_temp = net.VOS_head(vos_feat) * attention_next  # [1, 1, mask_pred_h, mask_pred_w]
                    vos_mask_temp = cfg.mask_proto_mask_activation(vos_mask_temp.squeeze(0)).gt_(0.5)
                    pred_masks_next.append(vos_mask_temp)
                pred_masks_next = torch.cat(pred_masks_next, dim=0)
                # _, pred_masks_next_ = crop(pred_masks_next.permute(1, 2, 0).contiguous(), box_ref_crop)
                # pred_masks_next = pred_masks_next.permute(2, 0, 1).contiguous()  # [n_ref, h, w]
                ref_candidate_shift = {'mask': pred_masks_next}

                # temp = torch.max(upsampled_corr.view(n_ref, -1), dim=1)[0].view(n_ref, 1, 1)
                # plt.imshow(torch.stack((upsampled_corr / temp, pred_masks_ref,
                #                        pred_masks_next), dim=2).view(n_ref*mask_pred_h, -1).cpu().numpy())
                # root_dir = ''.join(['/home/lmh/Downloads/VIS/code/OSTMask/weights/YTVIS2019/weights_r50_kl_vos_C/vos_mask/',
                #                     str(img_meta[0]['video_id'])])
                # if not os.path.exists(root_dir):
                #     os.makedirs(root_dir)
                # plt.savefig(''.join([root_dir, '/', str(img_meta[0]['frame_id']), '_+.png']))

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
            if cfg.mask_coeff_for_occluded:
                candidate_cur['mask_occluded_coeff'] = predictions['mask_occluded_coeff'][i][keep, :]

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



