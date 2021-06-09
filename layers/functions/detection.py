import torch
import torch.nn.functional as F
from layers.utils import decode, jaccard, mask_iou, crop, generate_mask, compute_DIoU, center_size
from utils import timer

from datasets import cfg

import numpy as np

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)

from utils.cython_nms import nms as cnms


class Detect(object):
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

    def __call__(self, predictions, net):
        """
        Args:
             loc_data: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_priors, num_classes]
            mask_data: (tensor) Mask preds from mask layers
                Shape: [batch, num_priors, mask_dim]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors, 4]
            proto_data: (tensor) If using mask_type.lincomb, the prototype masks
                Shape: [batch, mask_h, mask_w, mask_dim]
        
        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.

            Note that the outputs are sorted only if cross_class_nms is False
        """

        loc_data   = predictions['loc']
        # [bs, h*w*a, n_class] or [bs, h*w*a, 1]
        scores = predictions['conf'] if cfg.train_class else predictions['stuff']
        sem_data = predictions['sem_seg'] if cfg.use_semantic_segmentation_loss else None  # [bs, h, w, num_class]
        centerness_data = predictions['centerness'] if cfg.train_centerness else None
        proto_data = predictions['proto']
        mask_data  = predictions['mask_coeff']
        prior_data = predictions['priors'].squeeze(0)

        results = []

        with timer.env('Detect'):
            batch_size = loc_data.size(0)
            num_priors = prior_data.size(0)

            scores = scores.view(batch_size, num_priors, -1).transpose(2, 1).contiguous()

            for batch_idx in range(batch_size):
                decoded_boxes = decode(loc_data[batch_idx], prior_data)
                centerness_data_cur = centerness_data[batch_idx] if centerness_data is not None else None
                sem_data_cur = sem_data[batch_idx] if sem_data is not None else None

                result = self.detect(scores[batch_idx], decoded_boxes, centerness_data_cur,
                                     mask_data[batch_idx], proto_data[batch_idx], sem_data_cur)

                for k in predictions.keys():
                    if k in {'proto', 'track'}:
                        result[k] = predictions[k][batch_idx]
                
                results.append(result)

        return results

    def detect(self, scores, decoded_boxes, center_data, mask_data, proto_data, sem_data):
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        assert cfg.train_class or cfg.use_semantic_segmentation_loss, \
            'The training process should include train_class or train_stuff.'

        if cfg.train_class:
            cur_scores = scores[1:, :]
            conf_scores, _ = torch.max(cur_scores, dim=0)
            keep = conf_scores > self.conf_thresh
            scores = cur_scores[:, keep]
        else:
            keep = scores > self.conf_thresh
            scores = scores[keep]

        if center_data is not None:
            center_data = center_data[keep].view(-1)

        boxes = decoded_boxes[keep, :]
        masks_coeff = mask_data[keep, :]

        if boxes.size(0) == 0:
            out_after_NMS = {'box': boxes, 'mask_coeff': masks_coeff, 'class': torch.Tensor(), 'score': torch.Tensor(),
                             'mask': torch.Tensor()}

        else:

            if self.use_fast_nms:
                if self.use_cross_class_nms:
                    out_after_NMS = self.cc_fast_nms(boxes, masks_coeff, proto_data, scores, sem_data, center_data,
                                                     self.nms_thresh, self.top_k)
                else:
                    out_after_NMS = self.fast_nms(boxes, masks_coeff, proto_data, scores, sem_data,
                                                  self.nms_thresh, self.top_k)
            else:
                out_after_NMS = self.traditional_nms(boxes, masks_coeff, scores, self.nms_thresh, self.conf_thresh)

        return out_after_NMS

    def cc_fast_nms(self, boxes, masks_coeff, proto_data, scores, sem_data, centerness_scores,
                    iou_threshold: float = 0.5, top_k: int = 200):
        if cfg.train_class:
            # Collapse all the classes into 1
            scores, classes = scores.max(dim=0)

        if centerness_scores is not None:
            scores = scores * centerness_scores

        det_masks_soft = generate_mask(proto_data, masks_coeff, boxes)

        # if area_box / area_mask < 0.2 and score < 0.2, the box is likely to be wrong.
        h, w = det_masks_soft.size()[1:]
        boxes_c = center_size(boxes)
        area_box = boxes_c[:, 2] * boxes_c[:, 3] * h * w
        area_mask = det_masks_soft.gt(0.5).float().sum(dim=(1, 2))
        area_rate = area_mask / area_box

        _, idx = (scores * area_rate).sort(0, descending=True)
        # keep_idx = (area_rate > 0.2)[idx]
        idx = idx[:top_k]

        if len(idx) == 0:
            out_after_NMS = {'box': torch.Tensor(), 'mask_coeff': torch.Tensor(), 'class': torch.Tensor(),
                             'score': torch.Tensor(), 'mask': torch.Tensor()}
            return out_after_NMS

        # Compute the pairwise IoU between the boxes
        boxes_idx = boxes[idx]
        iou = compute_DIoU(boxes_idx, boxes_idx)
        if cfg.nms_as_miou:
            det_masks_idx = det_masks_soft[idx].gt(0.5).float()  # [n, h, w]
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
        scores = scores[idx_out]
        det_masks_soft = det_masks_soft[idx_out]

        if cfg.train_class:
            classes = classes[idx_out] + 1
        else:
            sem_data = sem_data.unsqueeze(-1).permute(3, 2, 0, 1).contiguous()  # [1, n_class+1, h, w]
            sem_data = (sem_data * det_masks_soft.gt(0.5).float().unsqueeze(1)).gt(0.3).float()
            MIoU = mask_iou(sem_data[:, 1:], det_masks_soft.unsqueeze(1))  # [n, n_class, 1]
            max_miou, classes = MIoU.max(dim=1)
            classes = classes.view(-1)

        out_after_NMS = {'box': boxes, 'mask_coeff': masks_coeff, 'class': classes, 'score': scores,
                         'mask': det_masks_soft}
        if centerness_scores is not None:
            out_after_NMS['centerness'] = centerness_scores[idx_out]

        return out_after_NMS

    def coefficient_nms(self, coeffs, scores, cos_threshold=0.9, top_k=400):
        _, idx = scores.sort(0, descending=True)
        idx = idx[:top_k]
        coeffs_norm = F.normalize(coeffs[idx], dim=1)

        # Compute the pairwise cosine similarity between the coefficients
        cos_similarity = coeffs_norm @ coeffs_norm.t()
        
        # Zero out the lower triangle of the cosine similarity matrix and diagonal
        cos_similarity.triu_(diagonal=1)

        # Now that everything in the diagonal and below is zeroed out, if we take the max
        # of the cos similarity matrix along the columns, each column will represent the
        # maximum cosine similarity between this element and every element with a higher
        # score than this element.
        cos_max, _ = torch.max(cos_similarity, dim=0)

        # Now just filter out the ones higher than the threshold
        idx_out = idx[cos_max <= cos_threshold]
        
        return idx_out, idx_out.size(0)

    def fast_nms(self, boxes, masks_coeff, proto_data, scores, sem_data,
                 iou_threshold:float=0.5, top_k:int=200, second_threshold:bool=True):

        det_masks_soft = generate_mask(proto_data, masks_coeff, boxes)
        det_masks = det_masks_soft.gt(0.5).float()
        mask_h, mask_w = det_masks.size()[1:]

        if not cfg.train_class:
            # For a specific instance, we only take the fired area into account to calculate its category
            # Step 1: multiply instances' masks and semantic segmentation to filter fired area
            # step 2: mean of all pixels in the fired area as its classification confidence
            sem_data = sem_data.permute(2, 0, 1).contiguous().unsqueeze(0)
            sem_data = (sem_data * det_masks.unsqueeze(1)).gt(0.3).float()
            miou_scores = mask_iou(sem_data[:, 1:], det_masks.unsqueeze(1))  # [n, n_class]
            scores = scores.view(1, -1) * miou_scores.t()
            # MIou_sorted,  MIou_sorted_idx = MIoU.sort(1, descending=True)
            # print(MIou_sorted[:, :3].reshape(-1), MIou_sorted_idx[:, :3].reshape(-1), scores[idx_out])
            # max_miou, classes = MIoU.max(dim=1)
            # classes = classes.view(-1) + 1
            # scores = max_miou.view(-1)

        num_classes, num_dets = scores.size()
        scores, idx = scores.sort(1, descending=True)  # [num_classes, num_dets]
        idx = idx[:, :top_k].contiguous()
        scores = scores[:, :top_k]

        boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
        det_masks_soft = det_masks_soft[idx.view(-1)].view(num_classes, num_dets, mask_h, mask_w)

        iou = jaccard(boxes, boxes)                  # [num_classes, num_dets, num_dets]
        if cfg.nms_as_miou:
            det_masks = det_masks_soft.gt(0.5).float()
            m_iou = mask_iou(det_masks, det_masks)   # [num_class, num_dets, num_dets]
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

        boxes = boxes[keep]
        masks_coeff = masks_coeff[keep]
        scores = scores[keep]
        masks = det_masks_soft[keep]

        # Only keep the top cfg.max_num_detections highest scores across all classes
        scores, idx = scores.sort(0, descending=True)
        idx = idx[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]
        classes = classes[idx] + 1
        boxes = boxes[idx]
        masks_coeff = masks_coeff[idx]
        masks = masks[idx]

        out_after_NMS = {'box': boxes, 'mask_coeff': masks_coeff, 'class': classes, 'score': scores,
                         'mask': det_masks_soft}

        return out_after_NMS

    def traditional_nms(self, boxes, masks, scores, iou_threshold=0.5, conf_thresh=0.05):
        num_classes = scores.size(0)

        idx_lst = []
        cls_lst = []
        scr_lst = []

        # Multiplying by max_size is necessary because of how cnms computes its area and intersections
        boxes = boxes * cfg.max_size

        for _cls in range(num_classes):
            cls_scores = scores[_cls, :]
            conf_mask = cls_scores > conf_thresh
            idx = torch.arange(cls_scores.size(0), device=boxes.device)

            cls_scores = cls_scores[conf_mask]
            idx = idx[conf_mask]

            if cls_scores.size(0) == 0:
                continue
            
            preds = torch.cat([boxes[conf_mask], cls_scores[:, None]], dim=1).cpu().numpy()
            keep = cnms(preds, iou_threshold)
            keep = torch.Tensor(keep, device=boxes.device).long()

            idx_lst.append(idx[keep])
            cls_lst.append(keep * 0 + _cls)
            scr_lst.append(cls_scores[keep])
        
        idx     = torch.cat(idx_lst, dim=0)
        classes = torch.cat(cls_lst, dim=0)
        scores  = torch.cat(scr_lst, dim=0)

        scores, idx2 = scores.sort(0, descending=True)
        idx2 = idx2[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]

        idx = idx[idx2]
        classes = classes[idx2]

        # Undo the multiplication above
        return boxes[idx] / cfg.max_size, masks[idx], classes, scores
