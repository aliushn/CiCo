import torch
from layers.utils import jaccard, mask_iou, crop, generate_mask
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

    def __call__(self, candidate):
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
                result.append(self.detect(candidate[i]))

        return result

    def detect(self, candidate):
        """ Perform nms for only the max scoring class that isn't background (class 0) """

        scores = candidate['conf'].t() if cfg.train_class else candidate['stuff'].view(-1)
        boxes = candidate['box']
        mask_coeff = candidate['mask_coeff']
        proto_data = candidate['proto'].squeeze(0)
        sem_data = candidate['sem_seg'].squeeze(0) if cfg.use_semantic_segmentation_loss else None  # [h, w, num_class]
        centerness_scores = candidate['centerness'] if cfg.train_centerness else None

        if boxes.size(0) == 0:
            out_aft_nms = {'box': boxes, 'mask_coeff': mask_coeff, 'class': torch.Tensor(), 'score': torch.Tensor()}

        else:
            if self.use_cross_class_nms:
                out_aft_nms = self.cc_fast_nms(boxes, mask_coeff, proto_data, scores,
                                               sem_data, centerness_scores, self.nms_thresh, self.top_k)
            else:
                out_aft_nms = self.fast_nms(boxes, mask_coeff, proto_data, scores, sem_data, centerness_scores,
                                            self.nms_thresh, self.top_k)

        for k, v in candidate.items():
            if k in {'fpn_feat', 'proto', 'track', 'sem_seg'}:
                out_aft_nms[k] = v

        return out_aft_nms

    def cc_fast_nms(self, boxes, masks_coeff, proto_data, scores, sem_data, centerness_scores,
                    iou_threshold: float = 0.5, top_k: int = 200):
        # Collapse all the classes into 1
        if cfg.train_class:
            scores, classes = scores[1:].max(dim=0)   # [n_dets]

        if centerness_scores is not None:
            scores = scores * centerness_scores.view(1, -1)

        _, idx = scores.sort(0, descending=True)
        idx = idx[:top_k]

        if len(idx) == 0:
            out_after_NMS = {'box': torch.Tensor(), 'mask_coeff': torch.Tensor(), 'class': torch.Tensor(),
                             'score': torch.Tensor()}

        else:
            det_masks_soft = generate_mask(proto_data, masks_coeff, boxes)

            # Compute the pairwise IoU between the boxes
            boxes_idx = boxes[idx]
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

            if cfg.train_class:
                classes = classes[idx_out]
                scores = scores[idx_out]
            else:
                # For a specific instance, we only take the fired area into account to calculate its category
                # Step 1: multiply instances' masks and semantic segmentation to filter fired area
                # step 2: mean of all pixels in the fired area as its classification confidence
                sem_data = sem_data.permute(2, 0, 1).contiguous().unsqueeze(0)
                sem_data = (sem_data * det_masks_soft.gt(0.5).float().unsqueeze(1)).gt(0.3).float()
                MIoU = mask_iou(sem_data[:, 1:], det_masks_soft.unsqueeze(1))  # [n, n_class, 1]
                # MIou_sorted,  MIou_sorted_idx = MIoU.sort(1, descending=True)
                # print(MIou_sorted[:, :3].reshape(-1), MIou_sorted_idx[:, :3].reshape(-1), scores[idx_out])
                max_miou, classes = MIoU.max(dim=1)
                classes = classes.view(-1) + 1
                scores = max_miou.view(-1)

            out_after_NMS = {'box': boxes, 'mask_coeff': masks_coeff, 'class': classes, 'score': scores,
                             'mask': det_masks_soft}

            if cfg.train_centerness:
                out_after_NMS['centerness'] = centerness_scores[idx_out]

        return out_after_NMS

    def fast_nms(self, boxes, masks_coeff, proto_data, scores, sem_data, centerness_scores,
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

        else:
            num_classes, num_dets = idx.size()
            iou = jaccard(boxes, boxes)  # [num_classes, num_dets, num_dets]

            if cfg.nms_as_miou:
                boxes = boxes[idx.view(-1), :]
                masks_coeff = masks_coeff[idx.view(-1), :]
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
            if centerness_scores is not None:
                out_after_NMS['centerness'] = centerness_scores[idx.view(-1), :].view(num_classes, num_dets, -1)[keep]

        return out_after_NMS

