import torch
import torch.nn.functional as F
from layers.utils import jaccard, mask_iou, crop, generate_mask, compute_DIoU, center_size, decode, circum_boxes
from utils import timer


class Detect(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """

    # TODO: Refactor this whole class away. It needs to go.
    # TODO: conf without background, using sigmoid() !!!! Important

    def __init__(self, cfg):
        self.cfg = cfg
        self.train_masks = cfg.MODEL.MASK_HEADS.TRAIN_MASKS
        self.use_dynamic_mask = cfg.MODEL.MASK_HEADS.USE_DYNAMIC_MASK
        self.mask_coeff_occlusion = cfg.MODEL.MASK_HEADS.PROTO_COEFF_OCCLUSION
        self.nms_with_miou = cfg.TEST.NMS_WITH_MIoU
        self.use_focal_loss = cfg.MODEL.CLASS_HEADS.USE_FOCAL_LOSS
        self.num_classes = cfg.DATASETS.NUM_CLASSES
        self.top_k = cfg.TEST.DETECTIONS_PER_IMG
        self.nms_thresh = cfg.TEST.NMS_IoU_THRESH
        if self.nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = cfg.TEST.NMS_CONF_THRESH

        self.use_cross_class_nms = True
        self.use_fast_nms = True
        self.cubic_mode = cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE
        if cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
            self.img_level_keys = {'proto', 'fpn_feat', 'fpn_feat_temp', 'sem_seg', 'track'}
        else:
            self.img_level_keys = {'proto', 'fpn_feat', 'fpn_feat_temp', 'sem_seg'}

    def __call__(self, net, predictions):
        """
        Args:
                    net:  network
            predictions: (tensor) Shape: Conf preds from conf layers
        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.

            Note that the outputs are sorted only if cross_class_nms is False
        """

        with timer.env('Detect'):
            result = []
            batch_size = predictions['loc'].size(0)
            for i in range(batch_size):
                candidate_cur = dict()
                scores, scores_idx = torch.max(predictions['conf'][i], dim=1) if self.use_focal_loss else \
                    torch.max(predictions['conf'][i, :, 1:], dim=-1)
                if 'centerness' in predictions. keys():
                    scores = scores * predictions['centerness'][i].mean(-1).reshape(-1)

                # Remove proposals whose confidences are lower than the threshold
                keep = scores > self.conf_thresh
                idx_out = torch.arange(len(scores))[keep]
                if len(idx_out) > self.top_k:
                    _, idx_sorted = scores[keep].sort(0, descending=True)
                    idx_out = idx_out[idx_sorted[:self.top_k]]
                for k, v in predictions.items():
                    if k in self.img_level_keys:
                        candidate_cur[k] = v[i]
                    else:
                        candidate_cur[k] = v[i][idx_out]

                if len(idx_out) == 0:
                    candidate_cur['box'] = torch.Tensor()
                    candidate_cur['box_cir'] = torch.Tensor()
                    candidate_cur['mask'] = torch.Tensor()
                else:
                    dim_boxes = candidate_cur['loc'].size(-1)
                    priors = candidate_cur['priors'].repeat(1, dim_boxes//4).reshape(-1, 4)
                    candidate_cur['box'] = decode(candidate_cur['loc'].reshape(-1, 4), priors).reshape(-1, dim_boxes)
                    boxes_cir = circum_boxes(candidate_cur['box'])
                    candidate_cur['box_cir'] = boxes_cir
                    if self.train_masks:
                        proto_data, masks_coeff = candidate_cur['proto'], candidate_cur['mask_coeff']
                        if self.use_dynamic_mask:
                            det_masks_soft = net.DynamicMaskHead(proto_data.permute(2, 0, 1).unsqueeze(0), masks_coeff, boxes_cir)
                            _, pred_masks = crop(det_masks_soft.permute(1, 2, 0).contiguous(), boxes_cir)
                            candidate_cur['mask'] = pred_masks.permute(2, 0, 1).contiguous()
                        else:
                            if self.cfg.MODEL.MASK_HEADS.PROTO_CROP:
                                det_masks_all = generate_mask(proto_data, masks_coeff, boxes_cir,
                                                              proto_coeff_occlusion=self.mask_coeff_occlusion)
                            else:
                                det_masks_all = generate_mask(proto_data, masks_coeff,
                                                              proto_coeff_occlusion=self.mask_coeff_occlusion)
                            if self.mask_coeff_occlusion:
                                candidate_cur['mask'] = det_masks_all[:, 0] - det_masks_all[:, 1]
                            else:
                                # [n_objs, T, H, W]
                                candidate_cur['mask'] = det_masks_all

                result.append(self.detect(candidate_cur))

        return result

    def detect(self, candidate):
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        if candidate['box'].nelement() == 0:
            out_aft_nms = {'box': torch.Tensor(), 'class': torch.Tensor(), 'score': torch.Tensor()}
            if self.train_masks:
                out_aft_nms['mask_coeff'] = torch.Tensor()
                out_aft_nms['mask'] = torch.Tensor()

        else:

            if self.use_cross_class_nms:
                out_aft_nms = self.cc_fast_nms(candidate, self.nms_thresh, self.top_k)
            else:
                # Prepare data
                boxes = candidate['box']
                scores = candidate['conf']
                if not self.use_focal_loss:
                    scores = scores[..., 1:]
                mask_coeff, masks = None, None
                if self.train_masks:
                    mask_coeff = candidate['mask_coeff']
                    masks = candidate['mask']
                track_data = candidate['track'] if 'track' in candidate.keys() else None
                priors, prior_levels = candidate['priors'], candidate['prior_levels']
                out_aft_nms = self.fast_nms(boxes, mask_coeff, masks, scores, track_data, prior_levels,
                                            self.nms_thresh, self.top_k)

        for k, v in candidate.items():
            if k in self.img_level_keys:
                out_aft_nms[k] = v.clone()

        return out_aft_nms

    def cc_fast_nms(self, candidate, iou_threshold: float = 0.5, top_k: int = 100):
        with timer.env('Detect'):
            # Remove bounding boxes whose center beyond images or mask = 0
            boxes_c = center_size(candidate['box'].reshape(-1, 4))
            keep = ((boxes_c[:, :2] > 0) & (boxes_c[:, :2] < 1)).reshape(candidate['box'].size(0), -1)
            keep = keep.sum(dim=-1) >= keep.size(-1)//2
            if self.train_masks:
                non_empty_mask = (candidate['mask'].gt(0.5).sum(dim=[-1, -2]) > 5).sum(dim=-1) > 0
                keep = keep & non_empty_mask
            for k, v in candidate.items():
                if k not in self.img_level_keys:
                    candidate[k] = v[keep]

            if candidate['box'].nelement() == 0:
                out_after_NMS = {'box': torch.Tensor(), 'class': torch.Tensor(), 'score': torch.Tensor()}
                if self.train_masks:
                    out_after_NMS['mask_coeff'] = torch.Tensor()
                    out_after_NMS['mask'] = torch.Tensor()
                return out_after_NMS

            # Collapse all the classes into 1
            scores = candidate['conf']
            if not self.use_focal_loss:
                scores = scores[..., 1:]
            rescores = candidate['centerness'].mean(-1).reshape(-1, 1) * scores if 'centerness' in candidate.keys() else scores
            rescores, classes = rescores.max(dim=1)
            _, idx = rescores.sort(0, descending=True)
            idx = idx[:top_k]

            # Compute the pairwise IoU between the boxes
            n_objs = len(idx)
            boxes_idx = candidate['box'][idx].reshape(n_objs, -1, 4).permute(1,0,2).contiguous()
            box_iou = jaccard(boxes_idx, boxes_idx).mean(0)
            boxes_cir_idx = candidate['box_cir'][idx]
            box_cir_iou = jaccard(boxes_cir_idx, boxes_cir_idx)
            iou = 0.5*box_iou + 0.5*box_cir_iou

            if self.train_masks and self.nms_with_miou:
                masks_idx = candidate['mask'][idx].gt(0.5).float().permute(1,0,2,3).contiguous()
                flag = masks_idx.sum(dim=[-1, -2]) > 0
                m_iou = mask_iou(masks_idx, masks_idx)
                m_iou = m_iou.sum(dim=0) / flag.sum(dim=0)
                # Calculate similarity of mask coefficients
                masks_coeff_idx = F.normalize(candidate['mask_coeff'][idx], dim=-1)
                sim = masks_coeff_idx @ masks_coeff_idx.t()
                iou = 0.4*iou + 0.4*m_iou + 0.2*sim if self.cubic_mode else 0.5*iou + 0.5*m_iou
                # iou = 0.8*m_iou + 0.2*sim if self.cubic_mode else m_iou

            # Zero out the lower triangle of the cosine similarity matrix and diagonal
            iou = torch.triu(iou, diagonal=1)

            # Now that everything in the diagonal and below is zeroed out, if we take the max
            # of the IoU matrix along the columns, each column will represent the maximum IoU
            # between this element and every element with a higher score than this element.
            iou_max, _ = torch.max(iou, dim=0)

            # Now just filter out the ones greater than the threshold, i.e., only keep boxes that
            # don't have a higher scoring box that would supress it in normal NMS.
            idx_out = idx[iou_max <= iou_threshold]

            out_after_NMS = {'class': classes[idx_out]+1, 'score': scores[idx_out, classes[idx_out]]}
            for k, v in candidate.items():
                if k not in self.img_level_keys and k not in {'conf', 'loc'}:
                    out_after_NMS[k] = v[idx_out]

            return out_after_NMS

    def fast_nms(self, boxes, masks_coeff, proto_data, scores, track_data, centerness_scores, prior_levels,
                 iou_threshold: float = 0.5, top_k: int = 200,
                 second_threshold: bool = True):

        if centerness_scores is not None:
            centerness_scores = centerness_scores.view(-1, 1)
            scores = scores * centerness_scores.t()

        scores, idx = scores.sort(1, descending=True)  # [num_classes, num_dets]
        idx = idx[:, :top_k].contiguous()
        scores = scores[:, :top_k]

        if len(idx) == 0:
            out_after_NMS = {'box': torch.Tensor(), 'class': torch.Tensor(), 'score': torch.Tensor()}
            if self.train_masks:
                out_after_NMS['mask_coeff'] = torch.Tensor()
            if centerness_scores is not None:
                out_after_NMS['centerness'] = torch.Tensor()

        else:
            num_classes, num_dets = idx.size()
            # TODO: double check repeated bboxes, mask_coeff, track_data...
            boxes = boxes[idx.view(-1), :]
            masks_coeff = masks_coeff[idx.view(-1), :]
            if self.use_DIoU:
                iou = compute_DIoU(boxes, boxes)
                iou_threshold *= 0.85
            else:
                iou = jaccard(boxes, boxes)  # [num_classes, num_dets, num_dets]
            if self.train_masks and self.nms_with_miou:
                det_masks = generate_mask(proto_data, masks_coeff, boxes).view(num_classes, num_dets, proto_data.size(0), proto_data.size(1))
                det_masks = det_masks.gt(0.5).float()
                m_iou = mask_iou(det_masks, det_masks)  # [n_class, num_dets, num_dets]
                iou = iou * 0.5 + m_iou * 0.5

            iou.triu_(diagonal=1)
            iou_max, _ = iou.max(dim=1)  # [num_classes, num_dets]

            # Now just filter out the ones higher than the threshold
            keep = (iou_max <= iou_threshold)  # [num_classes, num_dets]

            # We should also only keep detections over the confidence threshold, but at the cost of
            # maxing out your detection count for every image, you can just not do that. Because we
            # have such a minimal amount of computation per detection (matrix mulitplication only),
            # this increase doesn't affect us much (+0.2 mAP for 34 -> 33 fps), so we leave it out.
            # However, when you implement this in your method, you should do this second threshold.
            if second_threshold:
                keep *= (scores > self.conf_thresh)

            # Assign each kept detection to its corresponding class
            classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
            classes = classes[keep]

            boxes = boxes.view(num_classes, num_dets, -1)[keep]
            scores = scores[keep]

            out_after_NMS = {'box': boxes, 'class': classes, 'score': scores}
            if self.train_masks:
                out_after_NMS['mask_coeff'] = masks_coeff.view(num_classes, num_dets, -1)[keep]
            if track_data is not None:
                out_after_NMS['track'] = track_data[idx.view(-1), :].view(num_classes, num_dets, -1)[keep]
            if centerness_scores is not None:
                out_after_NMS['centerness'] = centerness_scores[idx.view(-1), :].view(num_classes, num_dets, -1)[keep]

        return out_after_NMS

