import torch
from layers.utils import jaccard, mask_iou, crop, generate_mask, compute_DIoU, center_size
from utils import timer
from datasets import cfg
import torch.nn.functional as F


class Detect_TF(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """

    # TODO: Refactor this whole class away. It needs to go.
    # TODO: conf without background, using sigmoid() !!!! Important

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh

        self.use_cross_class_nms = True
        self.use_fast_nms = True

    def __call__(self, net, candidate):
        """
        Args:
            candidate: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_priors, num_classes]
        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.

            Note that the outputs are sorted only if cross_class_nms is False
        """

        with timer.env('Detect'):
            result = []
            for i in range(len(candidate)):
                result.append(self.detect(net, candidate[i]))

        return result

    def detect(self, net, candidate):
        """ Perform nms for only the max scoring class that isn't background (class 0) """

        scores = candidate['conf'].t() if cfg.train_class else candidate['stuff'].view(-1)
        boxes = candidate['box']
        mask_coeff = candidate['mask_coeff']
        proto_data = candidate['proto'].squeeze(0)
        sem_data = candidate['sem_seg'].squeeze(0) if cfg.use_semantic_segmentation_loss else None  # [h, w, num_class]
        centerness_scores = candidate['centerness'] if cfg.train_centerness else None
        track_data = candidate['track'] if cfg.train_track and not cfg.track_by_Gaussian else None

        if boxes.size(0) == 0:
            out_aft_nms = {'box': boxes, 'mask_coeff': mask_coeff, 'class': torch.Tensor(), 'score': torch.Tensor(),
                           'mask': torch.Tensor()}
            if centerness_scores is not None:
                out_aft_nms['centerness'] = torch.Tensor()

        else:
            # if cfg.use_DIoU:
            #     if cfg.nms_as_miou:
            #         (0.5 + 0.5 * 0.9) * nms_thresh
                    # nms_thresh = 0.95 * self.nms_thresh
                # else:
                #     nms_thresh = 0.9 * self.nms_thresh
            # else:
            nms_thresh = self.nms_thresh

            if centerness_scores is not None:
                scores = scores * centerness_scores.view(-1)

            if self.use_cross_class_nms:
                out_aft_nms = self.cc_fast_nms(net, boxes, mask_coeff, proto_data, scores, track_data,
                                               sem_data, centerness_scores, nms_thresh, self.top_k)
            else:
                out_aft_nms = self.fast_nms(net, boxes, mask_coeff, proto_data, scores, track_data,
                                            sem_data, centerness_scores, nms_thresh, self.top_k)

        for k, v in candidate.items():
            if k in {'fpn_feat', 'proto', 'sem_seg'}:
                out_aft_nms[k] = v

        if cfg.train_track and cfg.track_by_Gaussian:
            out_aft_nms['track'] = candidate['track']

        return out_aft_nms

    def cc_fast_nms(self, net, boxes, masks_coeff, proto_data, scores, track_data, sem_data, centerness_scores,
                    iou_threshold: float = 0.5, top_k: int = 100):
        # Collapse all the classes into 1
        if cfg.train_class:
            scores, classes = scores.max(dim=0)   # [n_dets]

        if cfg.use_dynamic_mask:
            det_masks_soft = net.DynamicMaskHead(proto_data.permute(2, 0, 1).unsqueeze(0), masks_coeff, boxes)
            if cfg.mask_proto_crop:
                _, pred_masks = crop(det_masks_soft.permute(1, 2, 0).contiguous(), boxes)
                det_masks_soft = pred_masks.permute(2, 0, 1).contiguous()
        else:
            if cfg.mask_proto_coeff_occlusion:
                det_masks_all = generate_mask(proto_data, masks_coeff)
                det_masks_all = F.softmax(det_masks_all, dim=-1)
                _, det_masks_all = crop(det_masks_all.permute(1, 2, 3, 0).contiguous(), boxes)
                det_masks_all = det_masks_all.permute(3, 0, 1, 2).contiguous()
                det_masks_soft = det_masks_all[:, :, :, 1]
                det_masks_soft_non_target = det_masks_all[:, :, :, -1]
            else:
                det_masks_soft = generate_mask(proto_data, masks_coeff, boxes)

        # if area_mask / area_box < 0.2 and score < 0.2, the box is likely to be wrong.
        h, w = det_masks_soft.size()[1:]
        boxes_c = center_size(boxes)
        area_box = boxes_c[:, 2] * boxes_c[:, 3] * h * w
        area_mask = det_masks_soft.gt(0.5).float().sum(dim=(1, 2))
        area_rate = area_mask / area_box

        _, idx = scores.sort(0, descending=True)
        # keep_idx = (area_rate > 0.2)[idx]
        idx = idx[:top_k]

        if len(idx) == 0:
            out_after_NMS = {'box': torch.Tensor(), 'mask_coeff': torch.Tensor(), 'class': torch.Tensor(),
                             'score': torch.Tensor(), 'mask': torch.Tensor()}
            if centerness_scores is not None:
                out_after_NMS['centerness'] = torch.Tensor()
            if cfg.mask_proto_coeff_occlusion:
                out_after_NMS['mask_non_target'] = torch.Tensor()
            if track_data is not None:
                out_after_NMS['track'] = torch.Tensor()

        else:

            # Compute the pairwise IoU between the boxes
            boxes_idx = boxes[idx]
            if cfg.use_DIoU:
                iou = compute_DIoU(boxes_idx, boxes_idx)
            else:
                iou = jaccard(boxes_idx, boxes_idx)
            det_masks_idx = det_masks_soft[idx].gt(0.5).float()

            if cfg.nms_as_miou:
                m_iou = mask_iou(det_masks_idx, det_masks_idx)
                iou = iou * 0.5 + m_iou * 0.5

            # Zero out the lower triangle of the cosine similarity matrix and diagonal
            iou = torch.triu(iou, diagonal=1)

            # Now that everything in the diagonal and below is zeroed out, if we take the max
            # of the IoU matrix along the columns, each column will represent the maximum IoU
            # between this element and every element with a higher score than this element.
            iou_max, _ = torch.max(iou, dim=0)

            # Now just filter out the ones greater than the threshold, i.e., only keep boxes that
            # don't have a higher scoring box that would supress it in normal NMS.
            idx_out = idx[iou_max <= iou_threshold]

            boxes = boxes[idx_out]
            masks_coeff = masks_coeff[idx_out]
            det_masks_soft = det_masks_soft[idx_out]
            det_masks = det_masks_soft.gt(0.5).float()

            if cfg.train_class:
                classes = classes[idx_out] + 1
                scores = scores[idx_out]
            else:
                # For a specific instance, we only take the fired area into account to calculate its category
                # Step 1: multiply instances' masks and semantic segmentation to filter fired area
                # step 2: mean of all pixels in the fired area as its classification confidence
                sem_data = sem_data.permute(2, 0, 1).contiguous().unsqueeze(0)
                sem_data = (sem_data * det_masks.unsqueeze(1)).gt(0.3).float()
                MIoU = mask_iou(sem_data[:, 1:], det_masks.unsqueeze(1))  # [n, n_class, 1]
                # MIou_sorted,  MIou_sorted_idx = MIoU.sort(1, descending=True)
                # print(MIou_sorted[:, :3].reshape(-1), MIou_sorted_idx[:, :3].reshape(-1), scores[idx_out])
                max_miou, classes = MIoU.max(dim=1)
                classes = classes.view(-1) + 1
                scores = max_miou.view(-1)

            out_after_NMS = {'box': boxes, 'mask_coeff': masks_coeff, 'class': classes, 'score': scores,
                             'mask': det_masks_soft}
            if track_data is not None:
                out_after_NMS['track'] = track_data[idx_out]

            if cfg.train_centerness:
                out_after_NMS['centerness'] = centerness_scores[idx_out]
            if cfg.mask_proto_coeff_occlusion:
                out_after_NMS['mask_non_target'] = det_masks_soft_non_target[idx_out]

        return out_after_NMS

    def fast_nms(self, boxes, masks_coeff, proto_data, scores, track_data, sem_data, centerness_scores,
                 iou_threshold: float = 0.5, top_k: int = 200,
                 second_threshold: bool = True):

        if centerness_scores is not None:
            centerness_scores = centerness_scores.view(-1, 1)
            scores = scores * centerness_scores.t()

        if cfg.train_class:
            scores = scores[1:]
        else:
            h, w = sem_data.size()[2:]
            det_masks = generate_mask(proto_data, masks_coeff, boxes)
            # resize pos_masks_ref and pos_masks_next to keep the same size with track data
            downsampled_det_masks = F.interpolate(det_masks.unsqueeze(1).float(), (h, w),
                                                  mode='bilinear', align_corners=False).gt(0.5)
            scores_conf = (sem_data * downsampled_det_masks).sum(dim=(2, 3)) / downsampled_det_masks.sum(dim=(2, 3))
            scores = (scores * scores_conf).t()

        scores, idx = scores.sort(1, descending=True)  # [num_classes, num_dets]
        idx = idx[:, :top_k].contiguous()
        scores = scores[:, :top_k]

        if len(idx) == 0:
            out_after_NMS = {'box': torch.Tensor(), 'mask_coeff': torch.Tensor(), 'class': torch.Tensor(),
                             'score': torch.Tensor()}
            if centerness_scores is not None:
                out_after_NMS['centerness'] = torch.Tensor()

        else:
            num_classes, num_dets = idx.size()
            # TODO: double check repeated bboxes, mask_coeff, track_data...
            boxes = boxes[idx.view(-1), :]
            masks_coeff = masks_coeff[idx.view(-1), :]
            if cfg.use_DIoU:
                iou = compute_DIoU(boxes, boxes)
            else:
                iou = jaccard(boxes, boxes)  # [num_classes, num_dets, num_dets]
            if cfg.nms_as_miou:
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
            masks_coeff = masks_coeff.view(num_classes, num_dets, -1)[keep]
            scores = scores[keep]

            out_after_NMS = {'box': boxes, 'mask_coeff': masks_coeff, 'class': classes, 'score': scores}
            if track_data is not None:
                out_after_NMS['track'] = track_data[idx.view(-1), :].view(num_classes, num_dets, -1)[keep]
            if centerness_scores is not None:
                out_after_NMS['centerness'] = centerness_scores[idx.view(-1), :].view(num_classes, num_dets, -1)[keep]

        return out_after_NMS

