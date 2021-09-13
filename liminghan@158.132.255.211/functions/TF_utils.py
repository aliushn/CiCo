import torch
import torch.nn.functional as F
from layers.utils import center_size, decode, generate_mask, correlate_operator, generate_track_gaussian
from layers.modules import bbox_feat_extractor
from ..visualization_temporal import display_box_shift, display_correlation_map_patch, display_correlation_map


class CandidateShift(object):
    def __init__(self, net, correlation_patch_size=5, train_maskshift=False, train_masks=True, track_by_Gaussian=False,
                 update_track=False, mask_proto_coeff_occlusion=False):
        """
        The function try to shift the candidates of reference frame to that of target frame.
        The most important step is to shift the bounding box of reference frame to that of target frame
        :param net: network
        :param candidate: ptoto_data [1, h, w, 32] the proto of masks for the next frame
        :param ref_candidate: the candidate dictionary that includes 'box', 'conf', 'mask_coeff', 'track' items.
        :return: candidates on the target frame
        """
        self.net = net
        self.correlation_patch_size = correlation_patch_size
        self.train_maskshift = train_maskshift
        self.train_masks = train_masks
        self.track_by_Gaussian = track_by_Gaussian
        self.mask_proto_coeff_occlusion = mask_proto_coeff_occlusion
        self.update_track = update_track

    def __call__(self, candidate, ref_candidate, img=None, img_meta=None):
        fpn_feat_next = candidate['fpn_feat']
        fpn_feat_ref = ref_candidate['fpn_feat']
        ref_candidate['tracked_mask'] += 1

        if ref_candidate['box'].size(0) > 0:
            # we only use the features in the P3 layer to perform correlation operation
            corr = correlate_operator(fpn_feat_ref, fpn_feat_next,
                                      patch_size=self.correlation_patch_size,
                                      kernel_size=3)
            # display_correlation_map(corr, img_meta)
            concat_features = torch.cat([fpn_feat_ref, corr, fpn_feat_next], dim=1)
            # Align cropped features of bounding boxes as 7*7
            feat_h, feat_w = fpn_feat_ref.size()[2:]
            bbox_feat_input = bbox_feat_extractor(concat_features, ref_candidate['box'], feat_h, feat_w, 7)

            if self.train_masks and self.train_maskshift:
                loc_ref_shift, mask_coeff_shift = self.net.TemporalNet(bbox_feat_input)
                ref_candidate['mask_coeff'] += mask_coeff_shift
            else:
                loc_ref_shift = self.net.TemporalNet(bbox_feat_input)
            ref_candidate['box'] = decode(loc_ref_shift, center_size(ref_candidate['box']))

            display = False
            if display:
                display_correlation_map_patch(bbox_feat_input[:, 256:377], img_meta=img_meta)
                # display_box_shift(box_ref, box_ref_shift, img_meta=img_meta, img_gpu=img)

            # decay_rate = 0.9**ref_candidate['tracked_mask'] if 'tracked_mask' in candidate.keys() else 0.9
            ref_candidate['score'] *= 0.9
            if self.train_masks:
                proto_next = candidate['proto'].squeeze(0)
                if self.mask_proto_coeff_occlusion:
                    pred_masks_all = generate_mask(proto_next, ref_candidate['mask_coeff'], ref_candidate['box'])
                    pred_masks_all = F.softmax(pred_masks_all, dim=-1)
                    ref_candidate['mask'] = pred_masks_all[:, :, :, 1]
                    ref_candidate['mask_non_target'] = pred_masks_all[:, :, :, -1]
                else:
                    ref_candidate['mask'] = generate_mask(proto_next, ref_candidate['mask_coeff'], ref_candidate['box'])

            if 'frame_id' in candidate.keys():
                ref_candidate['frame_id'] = candidate['frame_id']
            if self.track_by_Gaussian and self.update_track:
                ref_candidate['track_mu'], ref_candidate['track_var'] = generate_track_gaussian(ref_candidate['track'],
                                                                                                ref_candidate['mask'].gt(0.5),
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



