import torch
import torch.nn.functional as F
from layers.utils import jaccard, mask_iou, crop, generate_mask, point_form, center_size, decode, circum_boxes


class Detect(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """

    # TODO: Refactor this whole class away. It needs to go.

    def __init__(self, cfg, display=False):
        self.cfg = cfg
        self.train_masks = cfg.MODEL.MASK_HEADS.TRAIN_MASKS
        self.use_dynamic_mask = cfg.MODEL.MASK_HEADS.USE_DYNAMIC_MASK
        self.nms_with_miou = cfg.TEST.NMS_WITH_MIoU
        self.nms_with_biou = cfg.TEST.NMS_WITH_BIoU
        self.use_focal_loss = cfg.MODEL.CLASS_HEADS.USE_FOCAL_LOSS
        self.num_classes = cfg.DATASETS.NUM_CLASSES
        self.top_k = cfg.TEST.DETECTIONS_PER_IMG
        self.nms_thresh = cfg.TEST.NMS_IoU_THRESH
        if self.nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = cfg.TEST.NMS_CONF_THRESH
        self.clip_frames = cfg.SOLVER.NUM_CLIP_FRAMES

        if display:  # or cfg.DATASETS.TYPE == 'coco':
            self.use_cross_class_nms = True
        else:
            self.use_cross_class_nms = False
        self.cubic_mode = cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE
        self.img_level_keys = ['proto', 'fpn_feat', 'fpn_feat_temp', 'sem_seg']
        if cfg.MODEL.TRACK_HEADS.TRACK_BY_GAUSSIAN:
            self.img_level_keys += ['track']

    def __call__(self, net, predictions, inference=False):
        """
        Args:
                    net:  network
            predictions: (tensor) Shape: Conf preds from conf layers
        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.

            Note that the outputs are sorted only if cross_class_nms is False
        """

        assert isinstance(predictions, dict)

        result = []
        batch_size = predictions['proto'].size(0)
        T_out = predictions['loc'].size(0)//batch_size
        fpn_levels = predictions['prior_levels']
        predictions['priors'] = predictions['priors'].repeat(T_out, 1, 1)
        predictions['prior_levels'] = predictions['prior_levels'].repeat(T_out, 1)

        for i in range(batch_size):
            ind = range(i*T_out, (i+1)*T_out)
            candidate_cur = dict()
            if self.cfg.MODEL.CLASS_HEADS.TRAIN_INTERCLIPS_CLASS:
                scores = predictions['conf'][i].reshape(-1)
            else:
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
                if k in self.img_level_keys:
                    candidate_cur[k] = v[i].clone()
                else:
                    candidate_cur[k] = v[ind].reshape(len(scores), -1)[idx_out].clone()

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
                    boxes_cir_c = center_size(boxes_cir)
                    # for small objects, we set width and height of bounding boxes is 0.1
                    boxes_cir_c[:, 2:] = torch.clamp(boxes_cir_c[:, 2:] * 1.2, min=0.1)
                    boxes_cir_expand = point_form(boxes_cir_c)

                    if self.use_dynamic_mask:
                        pred_masks = net.ProtoNet.DynamicMaskHead(proto_data.permute(2, 3, 0, 1), masks_coeff,
                                                                  boxes_cir, fpn_levels)
                        if not self.cfg.MODEL.MASK_HEADS.LOSS_WITH_DICE_COEFF:
                            pred_masks = crop(pred_masks.permute(2, 3, 1, 0).contiguous(), boxes_cir_expand)
                            pred_masks = pred_masks.permute(3, 2, 0, 1).contiguous()

                    else:
                        boxes_crop = boxes_cir_expand if self.cfg.MODEL.MASK_HEADS.PROTO_CROP else None
                        pred_masks = generate_mask(proto_data, masks_coeff.reshape(-1, proto_data.size(-1)),
                                                   boxes_crop)
                    candidate_cur['mask'] = pred_masks

            result.append(self.detect(candidate_cur))

        return result

    def detect(self, candidate):
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        if candidate['box'].nelement() == 0:
            out_aft_nms = self.return_empty_out()

        else:
            # Remove bounding boxes whose center beyond images or mask = 0
            boxes_c = center_size(candidate['box'].reshape(-1, 4))
            keep = ((boxes_c[:, :2] > 0) & (boxes_c[:, :2] < 1)).reshape(candidate['box'].size(0), -1)
            keep = keep.sum(dim=-1) >= keep.size(-1)//2
            if self.train_masks:
                non_empty_mask = (candidate['mask'].gt(0.5).sum(dim=[-1, -2]) > 1).sum(dim=-1) > 0
                keep = keep & non_empty_mask
            for k, v in candidate.items():
                if k not in self.img_level_keys:
                    candidate[k] = v[keep]

            if candidate['box'].nelement() == 0:
                out_aft_nms = self.return_empty_out()
            else:
                if self.use_cross_class_nms:
                    out_aft_nms = self.cc_fast_nms(candidate, self.nms_thresh, self.top_k)
                else:
                    out_aft_nms = self.fast_nms(candidate, self.nms_thresh, self.top_k)

        for k, v in candidate.items():
            if k in self.img_level_keys:
                out_aft_nms[k] = v.clone()

        return out_aft_nms

    def cc_fast_nms(self, candidate, iou_threshold: float = 0.5, top_k: int = 100):
        if self.cfg.MODEL.CLASS_HEADS.TRAIN_INTERCLIPS_CLASS:
            scores = candidate['conf'].reshape(-1)
            rescores = candidate['centerness'].mean(-1).reshape(-1)*scores if 'centerness' in candidate.keys() else scores
        else:
            # Collapse all the classes into 1
            scores = candidate['conf']
            if not self.use_focal_loss:
                scores = scores[..., 1:]
            rescores = candidate['centerness'].mean(-1).reshape(-1, 1) * scores if 'centerness' in candidate.keys() else scores
            rescores, classes = rescores.max(dim=1)

        _, idx = rescores.sort(0, descending=True)
        idx = idx[:top_k]
        n_objs = len(idx)

        # Compute the pairwise IoU between the boxes
        boxes_idx = candidate['box'][idx].reshape(n_objs, -1, 4).permute(1, 0, 2).contiguous()
        box_iou = jaccard(boxes_idx, boxes_idx).mean(0)
        boxes_cir_idx = candidate['box_cir'][idx]
        box_cir_iou = jaccard(boxes_cir_idx, boxes_cir_idx)
        iou = 0.5*box_iou + 0.5*box_cir_iou

        if self.train_masks and self.nms_with_miou:
            masks_idx = candidate['mask'][idx].gt(0.5).float().permute(1, 0, 2, 3).contiguous()
            flag = masks_idx.sum(dim=[-1, -2]) > 0
            # In case of out of memory when the length of the clip T >= 5
            m_iou = torch.stack([mask_iou(masks_idx[i], masks_idx[i]) for i in range(masks_idx.size(0))], dim=0)
            # m_iou = mask_iou(masks_idx, masks_idx)
            m_iou = m_iou.sum(dim=0) / flag.sum(dim=0)
            # Calculate similarity of mask coefficients
            masks_coeff_idx = F.normalize(candidate['mask_coeff'][idx], dim=-1)
            sim = masks_coeff_idx @ masks_coeff_idx.t()
            iou = 0.5*iou + 0.4*m_iou + 0.1*sim if self.cubic_mode else 0.5*iou + 0.5*m_iou
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

        if self.cfg.MODEL.CLASS_HEADS.TRAIN_INTERCLIPS_CLASS:
            out_after_NMS = {'score': rescores[idx_out]}
        else:
            out_after_NMS = {'score': scores[idx_out, classes[idx_out]], 'class': classes[idx_out]+1}
        for k, v in candidate.items():
            if k not in self.img_level_keys and k not in {'conf', 'loc'}:
                out_after_NMS[k] = v[idx_out]

        return out_after_NMS

    def fast_nms(self, candidate, iou_threshold: float = 0.5, top_k: int = 200, second_threshold: bool = True):
        # Collapse all the classes into 1
        scores = candidate['conf'].t()
        if not self.use_focal_loss:
            scores = scores[1:]
        rescores = candidate['centerness'].mean(-1).reshape(1, -1) * scores if 'centerness' in candidate.keys() else scores
        rescores, idx = rescores.sort(1, descending=True)                                 # [num_classes, num_dets]
        idx = idx[:, :top_k].contiguous()
        rescores = rescores[:, :top_k].contiguous()

        iou = None
        num_classes, n_objs = idx.size()
        if self.nms_with_biou or not self.nms_with_miou:
            boxes_idx = candidate['box'].reshape(n_objs, -1, 4).permute(1, 0, 2).contiguous()
            box_iou_ori = jaccard(boxes_idx, boxes_idx).mean(0)
            box_iou = torch.stack([box_iou_ori[idx[c]][:, idx[c]] for c in range(idx.size(0))], dim=0)
            # [num_classes, num_dets, num_dets]
            boxes_cir_idx = candidate['box_cir'][idx.view(-1)].reshape(-1, n_objs, 4)
            box_cir_iou = jaccard(boxes_cir_idx, boxes_cir_idx)
            iou = 0.5 * box_iou + 0.5 * box_cir_iou
        if self.train_masks and self.nms_with_miou:
            m_iou = []
            masks = candidate['mask'].gt(0.5).float().permute(1, 0, 2, 3).contiguous()
            # In case of out of memory when the length of the clip T >= 5
            miou_ori = torch.stack([mask_iou(masks[i], masks[i]) for i in range(masks.size(0))])
            for c in range(idx.size(0)):
                masks_idx = masks[:, idx[c]]
                flag = (masks_idx.sum(dim=[-1, -2]) > 0).sum(dim=0)
                m_iou.append(miou_ori[:, idx[c]][:, :, idx[c]].sum(dim=0) / flag)
            m_iou = torch.stack(m_iou, dim=0)          # [n_class, num_dets, num_dets]
            iou = iou * 0.5 + m_iou * 0.5 if iou is not None else m_iou

        iou.triu_(diagonal=1)
        iou_max, _ = iou.max(dim=1)                    # [num_classes, num_dets]
        # Now just filter out the ones higher than the threshold
        keep = iou_max <= iou_threshold                # [num_classes, num_dets]

        # We should also only keep detections over the confidence threshold, but at the cost of
        # maxing out your detection count for every image, you can just not do that. Because we
        # have such a minimal amount of computation per detection (matrix mulitplication only),
        # this increase doesn't affect us much (+0.2 mAP for 34 -> 33 fps), so we leave it out.
        # However, when you implement this in your method, you should do this second threshold.
        if second_threshold:
            keep *= (rescores > 0.5*self.conf_thresh)

        # Assign each kept detection to its corresponding class
        classes = torch.arange(num_classes, device=scores.device)[:, None].expand_as(keep)[keep]
        scores = torch.stack([scores[c][idx[c]] for c in range(idx.size(0))], dim=0)[keep]

        out_after_NMS = {'class': classes+1, 'score': scores}
        for k, v in candidate.items():
            if k not in self.img_level_keys and k not in {'conf', 'loc'}:
                if k == 'mask':
                    T, h, w = v.size()[1:]
                    out_after_NMS[k] = v[idx.view(-1)].view(num_classes, n_objs, -1)[keep].reshape(-1, T, h, w)
                else:
                    out_after_NMS[k] = v[idx.view(-1)].view(num_classes, n_objs, -1)[keep]

        return out_after_NMS

    def return_empty_out(self):
        out_aft_nms = {'box': torch.Tensor(), 'score': torch.Tensor()}
        if not self.cfg.MODEL.CLASS_HEADS.TRAIN_INTERCLIPS_CLASS:
            out_aft_nms['class'] = torch.Tensor()
        else:
            out_aft_nms['conf_feat'] = torch.Tensor()
        if self.train_masks:
            out_aft_nms['mask_coeff'] = torch.Tensor()
            out_aft_nms['mask'] = torch.Tensor()
        return out_aft_nms