import torch
from layers.utils import jaccard, mask_iou, crop, generate_mask, compute_DIoU, center_size, decode
from utils import timer
import torch.nn.functional as F


class Detect(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """

    # TODO: Refactor this whole class away. It needs to go.
    # TODO: conf without background, using sigmoid() !!!! Important

    def __init__(self, num_classes, top_k, conf_thresh, nms_thresh, use_focal_loss=True,
                 train_masks=True, use_dynamic_mask=False, mask_coeff_occlusion=False,
                 nms_with_miou=False, use_DIoU=True, track_by_Gaussian=False):
        self.use_DIoU = use_DIoU
        self.train_masks = train_masks
        self.use_dynamic_mask = use_dynamic_mask
        self.mask_coeff_pcclusion = mask_coeff_occlusion
        self.nms_with_miou = nms_with_miou
        self.use_focal_loss = use_focal_loss
        self.num_classes = num_classes
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh

        self.use_cross_class_nms = True
        self.use_fast_nms = True
        if track_by_Gaussian:
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
            clip_frames = predictions['loc'].size(-1) // 4
            if 'centerness' in predictions.keys():
                predictions['conf'] *= predictions['centerness'].mean(dim=-1).unsqueeze(-1)
            for i in range(batch_size):
                candidate_cur = dict()
                scores, scores_idx = torch.max(predictions['conf'][i], dim=1) if self.use_focal_loss else torch.max(predictions['conf'][i, :, 1:], dim=-1)

                keep = scores > self.conf_thresh
                # print(cfg.eval_conf_thresh, keep.sum())
                # print(scores_idx[keep].unique())
                for k, v in predictions.items():
                    if k in self.img_level_keys:
                        candidate_cur[k] = v[i*clip_frames:(i+1)*clip_frames]
                    else:
                        candidate_cur[k] = v[i][keep]

                result.append(self.detect(net, candidate_cur))

        return result

    def detect(self, net, candidate):
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        # Prepare data
        scores = candidate['conf']
        if not self.use_focal_loss:
            scores = scores[..., 1:]

        clip_frames = candidate['loc'].size(-1) // 4
        boxes = decode(candidate['loc'].reshape(-1, 4), candidate['priors'].repeat(1, clip_frames).reshape(-1, 4))
        boxes = boxes.reshape(-1, 4*clip_frames)
        mask_coeff, proto_data = None, None
        if self.train_masks:
            mask_coeff = candidate['mask_coeff']
            proto_data = candidate['proto']
        track_data = candidate['track'] if 'track' in candidate.keys() else None
        prior_levels = candidate['prior_levels']

        if boxes.size(0) == 0:
            out_aft_nms = {'box': boxes, 'class': torch.Tensor(), 'score': torch.Tensor()}
            if self.train_masks:
                out_aft_nms['mask_coeff'] = mask_coeff
                out_aft_nms['mask'] = torch.Tensor()

        else:

            if self.use_cross_class_nms:
                out_aft_nms = self.cc_fast_nms(net, boxes, mask_coeff, proto_data, scores, track_data,
                                               prior_levels, self.nms_thresh, self.top_k)
            else:
                out_aft_nms = self.fast_nms(net, boxes, mask_coeff, proto_data, scores, track_data,
                                            prior_levels, self.nms_thresh, self.top_k)

        for k, v in candidate.items():
            if k in self.img_level_keys:
                out_aft_nms[k] = v.clone()

        return out_aft_nms

    def cc_fast_nms(self, net, boxes, masks_coeff, proto_data, scores, track_data, prior_levels,
                    iou_threshold: float = 0.5, top_k: int = 100):
        with timer.env('Detect'):
            clip_frames = boxes.size(-1) // 4
            iou_threshold /= clip_frames

            if self.train_masks:
                if self.use_dynamic_mask:
                    det_masks_soft = net.DynamicMaskHead(proto_data.permute(2, 0, 1).unsqueeze(0), masks_coeff, boxes)
                    _, pred_masks = crop(det_masks_soft.permute(1, 2, 0).contiguous(), boxes)
                    det_masks_soft = pred_masks.permute(2, 0, 1).contiguous()
                else:
                    if self.mask_coeff_pcclusion:
                        det_masks_all = F.softmax(generate_mask(proto_data, masks_coeff), dim=-1)
                        _, det_masks_all = crop(det_masks_all.permute(1, 2, 3, 0).contiguous(), boxes)
                        det_masks_all = det_masks_all.permute(3, 0, 1, 2).contiguous()
                        det_masks_soft = det_masks_all[:, :, :, 1]
                        det_masks_soft_non_target = det_masks_all[:, :, :, -1]
                    else:
                        det_masks_soft = [generate_mask(proto_data[cdx], masks_coeff, boxes[:, cdx*4:(cdx+1)*4]) for cdx in range(clip_frames)]
                        det_masks_soft = torch.stack(det_masks_soft, dim=-1)

            # Remove bounding boxes whose center beyond images or mask = 0
            boxes_c = center_size(boxes.reshape(-1, 4))
            keep = ((boxes_c[:, :2] > 0) & ((boxes_c[:, :2] < 1))).reshape(-1, 2*clip_frames).sum(dim=-1) == 2*clip_frames
            if self.train_masks:
                non_empty_mask = (det_masks_soft.gt(0.5).sum(dim=[1, 2]) > 5).sum(dim=-1) == clip_frames
                keep = keep & non_empty_mask
            boxes = boxes[keep]
            scores = scores[keep]
            if self.train_masks:
                masks_coeff = masks_coeff[keep]
                det_masks_soft = det_masks_soft[keep]

            if boxes.nelement() == 0:
                out_after_NMS = {'box': torch.Tensor(), 'class': torch.Tensor(), 'score': torch.Tensor()}
                if self.train_masks:
                    out_after_NMS['mask_coeff'] = torch.Tensor()
                    out_after_NMS['mask'] = torch.Tensor()
                return out_after_NMS

            # Collapse all the classes into 1
            scores, classes = scores.max(dim=1)  # [n_dets]
            _, idx = scores.sort(0, descending=True)
            idx = idx[:top_k]

            # Compute the pairwise IoU between the boxes
            boxes_idx = boxes[idx]
            if self.use_DIoU:
                box_iou = [compute_DIoU(boxes_idx[:, cdx*4:(cdx+1)*4], boxes_idx[:, cdx*4:(cdx+1)*4]) for cdx in range(clip_frames)]
            else:
                box_iou = [jaccard(boxes_idx[:, cdx*4:(cdx+1)*4], boxes_idx[:, cdx*4:(cdx+1)*4]) for cdx in range(clip_frames)]
            iou = [torch.stack(box_iou, dim=-1).mean(dim=-1)]

            if self.train_masks and self.nms_with_miou:
                det_masks_idx = det_masks_soft[idx].gt(0.5).float()
                m_iou = [mask_iou(det_masks_idx[..., cdx], det_masks_idx[..., cdx]) for cdx in range(clip_frames)]
                iou.append(torch.stack(m_iou, dim=-1).mean(dim=-1))

            # Zero out the lower triangle of the cosine similarity matrix and diagonal
            iou = torch.stack(iou, dim=0).mean(dim=0)
            iou = torch.triu(iou, diagonal=1)

            # Now that everything in the diagonal and below is zeroed out, if we take the max
            # of the IoU matrix along the columns, each column will represent the maximum IoU
            # between this element and every element with a higher score than this element.
            iou_max, _ = torch.max(iou, dim=0)

            # Now just filter out the ones greater than the threshold, i.e., only keep boxes that
            # don't have a higher scoring box that would supress it in normal NMS.
            idx_out = idx[iou_max <= iou_threshold]

            boxes = boxes[idx_out]
            classes = classes[idx_out] + 1
            scores = scores[idx_out]

            out_after_NMS = {'box': boxes, 'class': classes, 'score': scores}
            if self.train_masks:
                out_after_NMS['mask_coeff'] = masks_coeff[idx_out]
                out_after_NMS['mask'] = det_masks_soft[idx_out] if det_masks_soft.size(-1) > 1 else det_masks_soft[idx_out].squeeze(-1)
            if track_data is not None:
                out_after_NMS['track'] = track_data[idx_out]

            return out_after_NMS

    def fast_nms(self, net, boxes, masks_coeff, proto_data, scores, track_data, centerness_scores, prior_levels,
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
