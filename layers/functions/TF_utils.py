import torch
import torch.nn.functional as F
from layers.utils import center_size, point_form, decode, crop, generate_mask, correlate_operator
from layers.modules import bbox_feat_extractor
from datasets import cfg, activation_func
from utils import timer
import matplotlib.pyplot as plt
import os


def CandidateShift(net, proto_next, fpn_feat_next, ref_candidate, img=None, img_meta=None):
        """
        The function try to shift the candidates of reference frame to that of target frame.
        The most important step is to shift the bounding box of reference frame to that of target frame
        :param net: network
        :param proto_next: the proto of masks for the next frame
        :param fpn_feat_next: fpn_feat of the next frame [256, h, w]
        :param ref_candidate: the candidate dictionary that includes 'box', 'conf', 'mask_coeff', 'track' items.
        :return: candidates on the target frame
        """

        fpn_feat_next = fpn_feat_next.unsqueeze(0)
        fpn_feat_ref = ref_candidate['fpn_feat'].unsqueeze(0)

        if ref_candidate['box'].size(0) == 0:
            ref_candidate_shift = {'box': torch.tensor([])}

        else:

            if cfg.temporal_fusion_module:
                # we only use the features in the P3 layer to perform correlation operation
                corr = correlate_operator(fpn_feat_next, fpn_feat_ref,
                                          patch_size=cfg.correlation_patch_size, kernel_size=3, dilation_patch=1)
                # display_correlation_map(fpn_ref, img_meta, idx)
                concatenated_features = F.relu(torch.cat([corr, fpn_feat_next, fpn_feat_ref], dim=1))

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
                ref_candidate_shift['score'] = ref_candidate['score'].clone() * 0.8
                ref_candidate_shift['mask_coeff'] = ref_candidate['mask_coeff'].clone() + mask_coeff_shift
                ref_candidate_shift['mask'] = generate_mask(proto_next, ref_candidate_shift['mask_coeff'], box_ref_shift)

            else:

                # FEELVOS
                # we only use the features in the P3 layer to perform correlation operation
                # display_correlation_map(fpn_ref, img_meta, idx)
                n_ref = ref_candidate['box'].size(0)
                mask_pred_h, mask_pred_w = proto_next.size()[:2]
                feat_h, feat_w = fpn_feat_ref.size()[2:]
                downsampled_pred_masks_ref = F.interpolate(ref_candidate['mask'].unsqueeze(0).float(), (feat_h, feat_w),
                                                           mode='bilinear', align_corners=False).squeeze(0).gt(0.5)
                corr, attention_next = [], []
                for j in range(n_ref):
                    corr.append(correlate_operator(fpn_feat_next,
                                                   fpn_feat_ref * downsampled_pred_masks_ref[j].view(1, 1, feat_h, feat_w),
                                                   patch_size=cfg.correlation_patch_size, kernel_size=3))
                corr = torch.cat(corr, dim=0)
                upsampled_corr = F.interpolate(corr.float(), (mask_pred_h, mask_pred_w),
                                               mode='bilinear', align_corners=False)
                upsampled_corr = torch.max(upsampled_corr, dim=1)[0]  # [n_ref, h, w]

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
                    heat_map = torch.zeros_like(upsampled_corr)
                    heat_map[upsampled_corr > 0.5] = 1
                    pred_masks_next = generate_mask(proto_next, ref_candidate['mask_coeff'])
                    pred_masks_next = pred_masks_next * heat_map

                ref_candidate_shift = {'mask': pred_masks_next}

                # plt.axis('off')
                # plt.imshow((upsampled_corr[0] / upsampled_corr[0].max()).cpu().numpy())
                # root_dir = ''.join(['/home/lmh/Downloads/VIS/code/OSTMask/weights/YTVIS2019/weights_r50_kl_vos_D_occluded_coeff/corr/',
                #                     str(img_meta[0]['video_id'])])
                # if not os.path.exists(root_dir):
                #     os.makedirs(root_dir)
                # plt.savefig(''.join([root_dir, '/', str(img_meta[0]['frame_id']), '.png']))
                # plt.clf()
                # plt.imshow(attention_next.view(mask_pred_h, mask_pred_w).cpu().numpy())
                # root_dir = ''.join(['/home/lmh/Downloads/VIS/code/OSTMask/weights/YTVIS2019/weights_r50_kl_vos_D_occluded_coeff/vos_mask/',
                #                     str(img_meta[0]['video_id'])])
                # plt.savefig(''.join([root_dir, '/', str(img_meta[0]['frame_id']), '_att.png']))
                # pred_masks_next_ori = generate_mask(proto_next, mask_coeff_ref)
                # pred_masks_next_ori = pred_masks_next_ori.permute(1, 0, 2).contiguous().view(mask_pred_h, -1).gt(0.5)
                # pred_masks_next = pred_masks_next.permute(1, 0, 2).contiguous().view(mask_pred_h, -1).gt(0.5)
                # plt.imshow(torch.cat([pred_masks_next_ori, pred_masks_next]).cpu().numpy())
                # plt.axis('off')
                # root_dir = ''.join(['/home/lmh/Downloads/VIS/code/OSTMask/weights/YTVIS2019/weights_r50_kl_vos_D_occluded_coeff/vos_mask_min/',
                #                     str(img_meta[0]['video_id'])])
                # if not os.path.exists(root_dir):
                #     os.makedirs(root_dir)
                # plt.savefig(''.join([root_dir, '/', str(img_meta[0]['frame_id']), '_ori_vs_ori*corr.png']))
                # plt.clf()

        return ref_candidate_shift


def generate_candidate(predictions):
    batch_Size = predictions['loc'].size(0)
    priors = predictions['priors'].view(-1, 4)  # [bs, num_anchors*h*w, 4]
    del predictions['priors']

    candidate = []
    for i in range(batch_Size):
        candidate_cur = {}
        if cfg.train_class:
            conf_data = predictions['conf'][i].t().contiguous()
            scores, _ = torch.max(conf_data[1:, :], dim=0)
        else:
            scores = predictions['stuff'][i].view(-1)

        keep = (scores > cfg.eval_conf_thresh)
        for k, v in predictions.items():
            if k in {'proto', 'track', 'fpn_feat', 'sem_seg'}:
                candidate_cur[k] = v[i]
            else:
                candidate_cur[k] = v[i][keep]

        candidate_cur['box'] = decode(candidate_cur['loc'], priors[keep])
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



