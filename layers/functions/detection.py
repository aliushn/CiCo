import torch
import torch.nn.functional as F
from ..utils import jaccard, crop, point_form, center_size, decode, circum_boxes, generate_mask, mask_iou


class Detect(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """

    # TODO: Refactor this whole class away. It needs to go.

    def __init__(self, cfg):
        self.cfg = cfg
        self.train_masks = cfg.MODEL.MASK_HEADS.TRAIN_MASKS
        self.use_dynamic_mask = cfg.MODEL.MASK_HEADS.USE_DYNAMIC_MASK
        self.nms_with_miou = cfg.TEST.NMS_WITH_MIoU
        self.nms_with_biou = cfg.TEST.NMS_WITH_BIoU
        self.use_focal_loss = cfg.MODEL.CLASS_HEADS.USE_FOCAL_LOSS
        self.num_classes = cfg.DATASETS.NUM_CLASSES
        self.top_k = min(cfg.TEST.DETECTIONS_PER_IMG * cfg.SOLVER.NUM_CLIP_FRAMES, 300)
        self.nms_thresh = cfg.TEST.NMS_IoU_THRESH
        if self.nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = cfg.TEST.NMS_CONF_THRESH
        self.clip_frames = cfg.SOLVER.NUM_CLIP_FRAMES
        self.PHL_kernel_size = self.cfg.CiCo.CPH.LAYER_KERNEL_SIZE[0]
        self.PHL_stride = self.cfg.CiCo.CPH.LAYER_STRIDE[0]
        self.use_cico = cfg.CiCo.ENGINE
        self.img_level_keys = ['proto', 'fpn_feat', 'fpn_feat_temp', 'sem_seg']
        if cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
            self.img_level_keys += ['track']

        self.use_cross_class_nms = True if cfg.display else False

    def __call__(self, net, predictions):
        """
        Args:
                    net:  network is needed for the controller of dynamic filters in CondInst
            predictions: (tensor) Shape: Conf preds from conf layers

        Returns:
            The output is a dictionary: including scores, class idx, bbox coords, and mask.
            In addition, for linking objects in tracking, prototypes and mask parameters are stored.
        """

        assert isinstance(predictions, dict), \
            'The inputs of detection must be a dictionary!!'

        result = []
        batch_size = predictions['proto'].size(0)
        T_out = predictions['loc'].size(0) // batch_size
        fpn_levels = predictions['prior_levels']
        predictions['priors'] = predictions['priors'].repeat(T_out, 1, 1)
        predictions['prior_levels'] = predictions['prior_levels'].repeat(T_out, 1)

        for i in range(batch_size):
            ind = range(i*T_out, (i+1)*T_out)
            candidate_cur = dict()
            scores, _ = torch.max(predictions['conf'][ind], dim=-1) if self.use_focal_loss else \
                torch.max(predictions['conf'][ind, :, 1:], dim=-1)

            # Remove proposals whose confidences are lower than the threshold
            scores = scores.reshape(-1)
            keep = scores > self.conf_thresh
            idx_out = torch.arange(len(scores))[keep]
            if len(idx_out) > self.top_k:
                _, idx_sorted = scores[keep].sort(0, descending=True)
                idx_out = idx_out[idx_sorted[:self.top_k]]
            for k, v in predictions.items():
                candidate_cur[k] = v[i].clone() if k in self.img_level_keys else \
                    v[ind].reshape(len(scores), -1)[idx_out].clone()

            if len(idx_out) == 0:
                out_aft_nms = self.return_empty_out()
            else:
                # Perform nms for only the max scoring class that isn't background (class 0)
                T = candidate_cur['loc'].size(-1) // 4
                priors = candidate_cur['priors'].repeat(1, T).reshape(-1, 4)
                boxes = decode(candidate_cur['loc'].reshape(-1, 4), priors)
                candidate_cur['box'] = boxes.reshape(-1, T, 4)
                candidate_cur['box_cir'] = circum_boxes(boxes.reshape(-1, T*4))

                # Since mask generation usually needs huge storage compared with bounding boxes,
                # we design three naive conditions to filter some unuseful boxes out
                # Filter1: remove bounding boxes whose IoU > 0.9 with other boxes
                if not self.use_focal_loss:
                    # Collapse all the classes into 1
                    candidate_cur['conf'] = candidate_cur['conf'][..., 1:]
                scores, classes = candidate_cur['conf'].max(dim=1)
                scores = candidate_cur['centerness'].mean(-1).reshape(-1) * scores \
                    if 'centerness' in candidate_cur.keys() else scores
                _, idx = scores.sort(0, descending=True)
                # Compute the pairwise IoU between the bounding boxes and circumscribed boxes of objects
                boxes_idx = candidate_cur['box'][idx].transpose(0, 1)
                biou = jaccard(boxes_idx, boxes_idx)                               # [T, n_objs, n_objs]
                boxes_cir_idx = candidate_cur['box_cir'][idx]
                biou_cir = jaccard(boxes_cir_idx, boxes_cir_idx)                   # [n_objs, n_objs]
                iou = 0.5 * biou.mean(0) + 0.5 * biou_cir
                iou = torch.triu(iou, diagonal=1)
                iou_max, _ = torch.max(iou, dim=0)

                # Filter2: remove bounding boxes whose center beyond images
                n_objs, T, _ = candidate_cur['box'][idx].size()
                boxes_c = center_size(candidate_cur['box'][idx].reshape(-1, 4)).reshape(n_objs, T, 4)
                keep = (((boxes_c[..., :2] > 0) & (boxes_c[..., :2] < 1)).sum(-1) == 2).sum(dim=-1) >= max(T//2, 1)
                keep = keep & (iou_max <= 0.9)
                boxes_c = boxes_c[keep]
                biou, biou_cir = biou[:, keep][..., keep], biou_cir[keep][:, keep]
                for k, v in candidate_cur.items():
                    if k not in self.img_level_keys:
                        candidate_cur[k] = v[idx[keep]]

                # Clip-level instance segmentation generation by a small FCN or linear combination
                if self.train_masks:
                    proto_data, masks_coeff = candidate_cur['proto'], candidate_cur['mask_coeff']
                    boxes_cir_c = center_size(candidate_cur['box_cir'])
                    if self.use_dynamic_mask:
                        # FCN of CondInst https://github.com/aim-uofa/AdelaiDet/blob/master/configs/CondInst/README.md
                        boxes_cir_c[:, 2:] = torch.clamp(boxes_cir_c[:, 2:] * 1.5, min=0.1)
                        boxes_cir_expand = point_form(boxes_cir_c)
                        pred_masks = net.DynamicMaskHead(proto_data.permute(2, 3, 0, 1), masks_coeff,
                                                         candidate_cur['box_cir'], fpn_levels)
                        if not self.cfg.MODEL.MASK_HEADS.LOSS_WITH_DICE_COEFF:
                            pred_masks = crop(pred_masks.permute(2, 3, 1, 0).contiguous(), boxes_cir_expand)
                            pred_masks = pred_masks.permute(2, 3, 0, 1).contiguous()
                        pred_masks = pred_masks.transpose(0, 1)

                    else:
                        # Linear combination of Yolact https://github.com/dbolya/yolact
                        boxes_cir_c[:, 2:] = torch.clamp(boxes_cir_c[:, 2:] * 1.1, min=0.1)
                        boxes_cir_expand = point_form(boxes_cir_c)
                        boxes_crop = boxes_cir_expand if self.cfg.MODEL.MASK_HEADS.PROTO_CROP else None
                        pred_masks = generate_mask(proto_data, masks_coeff.reshape(-1, proto_data.size(-1)),
                                                   boxes_crop)
                    candidate_cur['mask'] = pred_masks

                    # Filter3: remove bounding boxes whose masks is blank or mask_area/box_area < 0.1
                    h, w = pred_masks.size()[-2:]
                    mask_area = pred_masks.gt(0.5).float().sum(dim=[-1, -2])
                    non_empty_mask = (mask_area > 5).sum(dim=-1) > 0
                    # Compute cover_rate = mask_area / box_area
                    cover_rate = mask_area / (h * w) / (boxes_c[..., 2] * boxes_c[..., 3])
                    non_low_cover = (cover_rate > 0.1).sum(dim=-1) > 0
                    keep_mask = non_empty_mask & non_low_cover
                    biou, biou_cir = biou[:, keep_mask][..., keep_mask], biou_cir[keep_mask][:, keep_mask]
                    for k, v in candidate_cur.items():
                        if k not in self.img_level_keys:
                            candidate_cur[k] = v[keep_mask]

                if candidate_cur['box'].nelement() == 0:
                    out_aft_nms = self.return_empty_out()
                else:
                    if self.use_cross_class_nms:
                        out_aft_nms = self.cc_fast_nms(candidate_cur, biou, biou_cir)
                    else:
                        out_aft_nms = self.fast_nms(candidate_cur, biou, biou_cir)

            for k, v in candidate_cur.items():
                if k in self.img_level_keys:
                    out_aft_nms[k] = v.clone()

            result.append(out_aft_nms)

        return result

    def cc_fast_nms(self, candidate, biou, biou_cir):
        '''
        :param candidate:
        :param biou: [T, N, N]
        :param biou_cir: [N, N]
        :return:
        '''
        if not self.train_masks:
            iou = 0.5 * biou.mean(0) + 0.5 * biou_cir
        else:
            masks_idx = candidate['mask'].gt(0.5).float().transpose(0, 1)
            flag = masks_idx.sum(dim=[-1, -2]) > 2                          # [T, n]
            flag = flag.unsqueeze(-1) * flag.unsqueeze(-2)

            biou = (biou * flag).sum(0) / flag.sum(0)
            iou = 0.5 * biou + 0.5 * biou_cir

            if self.nms_with_miou:
                # In case of out of memory when the length of the clip T >= 5, [T, n, n]
                m_iou = torch.stack([mask_iou(masks_idx[i], masks_idx[i]) for i in range(masks_idx.size(0))], dim=0)
                # m_iou = mask_iou(masks_idx, masks_idx)
                coincident_mask = ((m_iou > 0.9) & flag).sum(0) > 0
                m_iou = (m_iou * flag).sum(0) / flag.sum(0)
                # Calculate similarity of mask coefficients
                masks_coeff_idx = F.normalize(candidate['mask_coeff'], dim=-1)
                sim = masks_coeff_idx @ masks_coeff_idx.t()
                iou = 0.5*iou + 0.4*m_iou + 0.1*sim if self.use_cico else 0.5*iou + 0.5*m_iou
                iou[coincident_mask] = 1

        # Zero out the lower triangle of the cosine similarity matrix and diagonal
        iou = torch.triu(iou, diagonal=1)

        # Now that everything in the diagonal and below is zeroed out, if we take the max
        # of the IoU matrix along the columns, each column will represent the maximum IoU
        # between this element and every element with a higher score than this element.
        iou_max, _ = torch.max(iou, dim=0)

        # Now just filter out the ones greater than the threshold, i.e., only keep boxes that
        # don't have a higher scoring box that would supress it in normal NMS.
        idx_out = iou_max <= self.nms_thresh

        scores, classes = candidate['conf'].max(dim=1)
        out_after_NMS = {'score': scores[idx_out], 'class': classes[idx_out]+1}
        for k, v in candidate.items():
            if k in {'box', 'mask', 'track', 'box_cir', 'mask_coeff', 'prior_levels'}:
                out_after_NMS[k] = v[idx_out]

        return out_after_NMS

    def fast_nms(self, candidate, biou, biou_cir, second_threshold: bool = True):
        '''
        :param candidate: a dict
        :param biou: IoU of bounding boxes [T, N, N]
        :param biou_cir: IoU of circumscribed boxes on an instance throughout a video clip, [N, N]
        :param second_threshold: remove objects whose scores is lower than the threshold
        :return:
        '''
        scores = candidate['conf'].t()                                # [n_classes, n_dets]
        rescores = candidate['centerness'].mean(-1).reshape(1, -1) * scores \
            if 'centerness' in candidate.keys() else scores
        rescores, idx = rescores.sort(1, descending=True)             # [n_classes, n_dets]
        num_classes, n_objs = idx.size()

        # Objects may disappear in some frames of a video clip due to occlusion or new-coming objects.
        # Thus, BIoU and MIoU are based on these frames where object masks are not blank
        biou_cir = biou_cir[idx.reshape(-1)].reshape(-1, n_objs, n_objs)
        if not self.train_masks:
            biou = torch.stack([biou.mean(0)[idx[c]][:, idx[c]] for c in range(num_classes)], dim=0)
            iou = 0.5 * biou + 0.5 * biou_cir
        else:
            masks = candidate['mask'].gt(0.5).float().transpose(0, 1)
            T, _, h, w = masks.size()
            # flag: object masks is blank or not
            flag = masks.sum(dim=[-1, -2]) > 5
            flag = flag.unsqueeze(-1) & flag.unsqueeze(-2)                                 # [T, n_dets, n_dets]
            biou = (biou * flag).sum(0) / torch.clamp(flag.sum(0), min=1)
            biou = torch.stack([biou[idx[c]][:, idx[c]] for c in range(num_classes)])
            iou = 0.5 * biou + 0.5 * biou_cir

            if self.nms_with_miou:
                # In case of out of memory when the clip length T >= 5,
                # we first compute miou frame by frame and stack them along temporal dim
                miou_ori = torch.stack([mask_iou(masks[t], masks[t]) for t in range(T)])
                miou = (miou_ori * flag).sum(0) / torch.clamp(flag.sum(0), min=1)
                # two objects with miou > 0.9, set miou of the one with lower score as 1 to filter out
                miou[((miou_ori > 0.9) & flag).sum(0) > 0] = 1
                m_iou = torch.stack([miou[idx[c]][:, idx[c]] for c in range(num_classes)])
                iou = iou * 0.5 + m_iou * 0.5

        iou_triu = iou.triu(diagonal=1)
        iou_max, _ = iou_triu.max(dim=1)                                                      # [n_classes, n_dets]
        # Now just filter out the ones higher than the threshold
        keep = iou_max <= self.nms_thresh                                                     # [n_classes, n_dets]
        if second_threshold:
            keep *= (rescores > self.conf_thresh)

        # Assign each kept detection to its corresponding score and class
        scores = torch.stack([scores[c][idx[c]] for c in range(num_classes)], dim=0)[keep]
        _, sorted_idx = scores.sort(0, descending=True)
        classes = torch.arange(num_classes, device=scores.device)[:, None].expand_as(keep)[keep][sorted_idx]

        out_after_NMS = {'class': classes+1, 'score': scores[sorted_idx]}
        for k, v in candidate.items():
            if k in {'mask', 'box'}:
                v = v[idx.reshape(-1)].reshape(num_classes, n_objs, -1)[keep][sorted_idx]
                out_after_NMS[k] = v.reshape(-1, T, h, w) if k == 'mask' else v.reshape(-1, T, 4)
            elif k in {'track', 'box_cir', 'mask_coeff', 'prior_levels'}:
                out_after_NMS[k] = v[idx.reshape(-1)].view(num_classes, n_objs, -1)[keep][sorted_idx]

        return out_after_NMS

    def return_empty_out(self):
        out_aft_nms = {'box': torch.Tensor(), 'score': torch.Tensor(), 'class': torch.Tensor()}
        if self.train_masks:
            out_aft_nms['mask_coeff'] = torch.Tensor()
            out_aft_nms['mask'] = torch.Tensor()

        return out_aft_nms