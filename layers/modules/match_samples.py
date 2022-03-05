import torch
from ..utils import point_form, decode, center_size, jaccard, compute_DIoU, encode, circum_boxes


def match(cfg, bbox, labels, ids, crowd_boxes, priors, loc_data, loc_t, conf_t, idx_t, ids_t, idx,
          pos_thresh=0.5, neg_thresh=0.3):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        pos_thresh: (float) IoU > pos_thresh ==> positive.
        neg_thresh: (float) IoU < neg_thresh ==> negative.
        bbox: (tensor) Ground truth boxes, Shape: [num_obj, 4].  [x1, y1, x2, y2]
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        ids: (tensor) the instance ids of each gt bbox
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4]. [cx,cy,w,h]
        loc_data: (tensor) The predicted bbox regression coordinates for this batch. [\delta x, \delta y,  \delta w,  \delta h]
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds. Note: -1 means neutral.
        idx_t: (tensor) Tensor to be filled w/ the index of the matched gt box for each prior.
        ids_t: (tensor) Tensor to be filled w/ the ids of the matched gt instance for each prior.
        idx: (int) current batch index.
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # Remove some error annotations with width <= 0 or height <= 0
    labels = labels.reshape(-1)
    bbox_c = center_size(bbox)
    valid = (bbox_c[:, 2] > 0) & (bbox_c[:, 3] > 0)
    if (valid == 0).sum() > 0:
        valid_bbox = bbox[valid]
        valid_idx = torch.nonzero(valid, as_tuple=False).reshape(-1)
    else:
        valid_bbox = bbox
    small_objs_idx = torch.nonzero(center_size(valid_bbox)[:, 2] * center_size(valid_bbox)[:, 3] < 0.01, as_tuple=False)

    # decoded_priors => [x1, y1, x2, y2]
    decoded_boxes = decode(loc_data, priors)
    decoded_priors = torch.clamp(point_form(priors), min=0, max=1)

    # Size [num_objects, num_priors]
    if cfg.MODEL.PREDICTION_HEADS.USE_PREDICTION_MATCHING:
        if cfg.MODEL.PREDICTION_HEADS.USE_DIoU:
            overlaps = 0.5*compute_DIoU(valid_bbox, decoded_boxes) + 0.5*compute_DIoU(valid_bbox, decoded_priors)
            pos_thresh, neg_thresh = 0.85 * pos_thresh, 0.85 * neg_thresh
        else:
            overlaps = 0.5*jaccard(valid_bbox, decoded_boxes) + 0.5*jaccard(valid_bbox, decoded_priors)
    else:
        if cfg.MODEL.PREDICTION_HEADS.USE_DIoU:
            overlaps = compute_DIoU(valid_bbox, decoded_priors)
            pos_thresh, neg_thresh = 0.85 * pos_thresh, 0.85 * neg_thresh
        else:
            overlaps = jaccard(valid_bbox, decoded_priors)

    # Size [num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0)
    if small_objs_idx.nelement() > 0:
        for i in small_objs_idx:
            best_truth_overlap[best_truth_idx == i] *= 1.1

    # if a bbox is matched with two or more groundtruth boxes, this bbox will be assigned as -1
    thresh = 0.5 * (pos_thresh + neg_thresh)
    multiple_bbox = (overlaps > pos_thresh).sum(dim=0) > 1
    best_truth_overlap[multiple_bbox] = thresh

    # We want to ensure that each gt gets used at least once so that we don't
    # waste any training data. In order to do that, find the max overlap anchor
    # with each gt, and force that anchor to use that gt.
    for _ in range(overlaps.size(0)):
        # Find j, the gt with the highest overlap with a prior
        # In effect, this will loop through overlaps.size(0) in a "smart" order,
        # always choosing the highest overlap first.
        best_prior_overlap, best_prior_idx = overlaps.max(1)
        j = best_prior_overlap.max(0)[1]

        # Find i, the highest overlap anchor with this gt
        i = best_prior_idx[j]

        # Set all other overlaps with i to be -1 so that no other gt uses it
        overlaps[:, i] = -1
        # Set all other overlaps with j to be -1 so that this loop never uses j again
        overlaps[j, :] = -1

        # Overwrite i's score to be 2 so it doesn't get thresholded ever
        best_truth_overlap[i] = 2
        # Set the gt to be used for i to be j, overwriting whatever was there
        best_truth_idx[i] = j

    keep_pos = best_truth_overlap > pos_thresh
    keep_neg = best_truth_overlap < neg_thresh
    # print(bbox.size(0), keep_pos.sum())
    # print('idx:', best_truth_idx[keep_pos])

    valid_best_truth_idx = valid_idx[best_truth_idx] if (valid == 0).sum() > 0 else best_truth_idx
    matches = bbox[valid_best_truth_idx]            # Shape: [num_priors,4]  [x1, y1, x2, y2]
    conf = labels[valid_best_truth_idx]             # Shape: [num_priors]
    conf[~keep_pos] = -1                            # label as neutral
    conf[keep_neg] = 0                              # label as background

    # Deal with crowd annotations for COCO
    if crowd_boxes is not None and cfg.MODEL.PREDICTION_HEADS.CROWD_IoU_THRESHOLD < 1:
        # Size [num_priors, num_crowds]
        crowd_overlaps = jaccard(decoded_priors, crowd_boxes, iscrowd=True)
        # Size [num_priors]
        best_crowd_overlap, best_crowd_idx = crowd_overlaps.max(1)
        # Set non-positives with crowd iou of over the threshold to be neutral.
        if conf.size() == best_crowd_overlap.size():
            conf[(conf <= 0) & (best_crowd_overlap > cfg.MODEL.PREDICTION_HEADS.CROWD_IoU_THRESHOLD)] = -1
        else:
            print('The tensors a and b should have same shape: ', conf.size(), best_crowd_overlap.size())

    loc = encode(matches, priors)  # [cx, cy, w, h]
    nonvalid_loc = (torch.isinf(loc).sum(-1) + torch.isnan(loc).sum(-1)) > 0
    if nonvalid_loc.sum() > 0:
        print('Number of Inf or Nan in loc_t when matching samples:', loc[nonvalid_loc])
        conf[nonvalid_loc] = -1
    # filter out those predicted boxes with inf or nan
    nonvalid_loc_data = (torch.isinf(loc_data).sum(-1) + torch.isnan(loc_data).sum(-1)) > 0
    if nonvalid_loc_data.sum() > 0:
        print('Number of Inf or Nan in loc_data when matching samples:', len(torch.nonzero(nonvalid_loc_data)))
        conf[nonvalid_loc_data] = -1

    loc_t[idx]  = loc                                            # [num_priors,4] encoded offsets to learn
    if conf_t[idx].size() != conf.size():
        print(conf_t[idx].size(), conf.size(), labels.size(), best_truth_overlap.size())
    conf_t[idx] = conf                                           # [num_priors] top class label for each prior
    idx_t[idx]  = valid_best_truth_idx.view(-1)                  # [num_priors] indices for lookup
    if ids is not None:
        ids_t[idx] = ids[valid_best_truth_idx]


def match_clip(gt_boxes, gt_labels, gt_obj_ids, priors, loc_data, loc_t, conf_t, idx_t,
               obj_ids_t, kdx, jdx, pos_thresh=0.5, neg_thresh=0.3, use_cir_boxes=False, ind_range=None):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        pos_thresh: (float) IoU > pos_thresh ==> positive.
        neg_thresh: (float) IoU < neg_thresh ==> negative.
        gt_boxes: (tensor) Ground truth boxes, Shape: [num_obj, num_frames, 4].  [x1, y1, x2, y2]
        gt_labels: (tensor) All the class labels for the image, Shape: [num_obj].
        gt_obj_ids: (tensor) the instance ids of each gt bbox
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,num_frames, 4]. [cx,cy,w,h]
        loc_data: (tensor) The predicted bbox regression coordinates for this batch. [\delta x, \delta y,  \delta w,  \delta h]
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds. Note: -1 means neutral.
        idx_t: (tensor) Tensor to be filled w/ the index of the matched gt box for each prior.
        obj_ids_t: (tensor) Tensor to be filled w/ the ids of the matched gt instance for each prior.
        kdx: (int) current batch index.
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    gt_labels = gt_labels.long()
    n_proposals, n_clip_frames = loc_data.size(0), gt_boxes.size(1)

    ind_range = [jdx] if ind_range is None else ind_range
    gt_boxes_hw = gt_boxes[:, jdx, 2:]-gt_boxes[:, jdx, :2]
    # Must occur in the current central frame
    valid = (gt_boxes_hw > 0).sum(dim=-1) == 2
    n_objs = valid.sum()
    if n_objs > 0:
        valid_idx = torch.nonzero(valid, as_tuple=False).reshape(-1)
        valid_gt_boxes = gt_boxes[valid]

        # Introducing smallest enclosing boxes of multiple boxes in the clip to define positive/negative samples
        # gt_boxes_cir = circum_boxes(valid_gt_boxes[:, ind_range].reshape(n_objs, -1))
        gt_boxes_cir = valid_gt_boxes[:, jdx]
        for ind in ind_range:
            # If objects with fast motion, temporal coherence degrades, may mix features from other objects => remove
            keep_cir = jaccard(gt_boxes_cir, valid_gt_boxes[:, ind]).diag() > 0.5
            if keep_cir.sum() > 0:
                gt_boxes_cir[keep_cir] = circum_boxes(torch.cat([gt_boxes_cir[keep_cir],
                                                                 valid_gt_boxes[keep_cir, ind]], dim=-1))

        # decoded_priors [cx, cy, w, h] => [x1, y1, x2, y2]
        decoded_priors = point_form(priors[:, :4])
        overlaps_clip_cir = jaccard(gt_boxes_cir, decoded_priors)
        best_truth_overlap_cir, best_truth_idx_cir = overlaps_clip_cir.max(dim=0)

        # if a bbox is matched with two or more ground truth boxes, this bbox will be assigned as -1
        thresh = 0.5 * (pos_thresh + neg_thresh)
        multiple_bbox = (overlaps_clip_cir > pos_thresh).sum(dim=0) > 1
        best_truth_overlap_cir[multiple_bbox] = thresh
        for _ in range(n_objs):
            # Find j, the gt with the highest overlap with a prior
            # In effect, this will loop through overlaps.size(0) in a "smart" order,
            # always choosing the highest overlap first.
            best_prior_overlap_ext, best_prior_idx_ext = overlaps_clip_cir.max(1)
            j = best_prior_overlap_ext.max(0)[1]

            # Find i, the highest overlap anchor with this gt
            i = best_prior_idx_ext[j]
            if best_truth_overlap_cir[i] >= thresh*0.5 or best_truth_overlap_cir.max() < pos_thresh:
                # Set all other overlaps with i to be -1 so that no other gt uses it
                overlaps_clip_cir[:, i] = -1
                # Set all other overlaps with j to be -1 so that this loop never uses j again
                overlaps_clip_cir[j, :] = -1

                # Overwrite i's score to be 2 so it doesn't get thresholded ever
                best_truth_overlap_cir[i] = 2
                # Set the gt to be used for i to be j, overwriting whatever was there
                best_truth_idx_cir[i] = j

        # Small objects use smaller threshold in order to have more postive samples
        gt_boxes_c = center_size(valid_gt_boxes[:, ind_range].reshape(-1, 4)).reshape(n_objs, -1, 4)[..., 2:]
        small_objs_idx = (torch.prod(gt_boxes_c, dim=-1) < 0.01).sum(dim=1) >= min(len(ind_range)//2, 1)
        small_objs_idx = torch.nonzero(small_objs_idx.float(), as_tuple=False)
        if small_objs_idx.nelement() > 0:
            for i in small_objs_idx:
                best_truth_overlap_cir[best_truth_idx_cir == i] *= 1.05

        # Define positive sample, negative samples and neutral
        keep_pos_cir = best_truth_overlap_cir > pos_thresh
        keep_neg_cir = best_truth_overlap_cir < neg_thresh
        # print(ind_range, keep_pos_cir.sum(), n_objs)
        # print(best_truth_idx_cir[keep_pos_cir], best_truth_overlap_cir[keep_pos_cir])

        best_truth_idx_cir_pos = best_truth_idx_cir[keep_pos_cir]
        conf_t[kdx][keep_pos_cir] = gt_labels[valid][best_truth_idx_cir_pos]   # label as positive samples
        conf_t[kdx][keep_neg_cir] = 0                                          # label as background

        if use_cir_boxes:
            matches = circum_boxes(valid_gt_boxes.reshape(n_objs, -1))[best_truth_idx_cir_pos]
            loc_pos = encode(matches, priors[keep_pos_cir, :4])
        else:
            matches = valid_gt_boxes[best_truth_idx_cir_pos]
            loc_pos = encode(matches.reshape(-1, 4), priors[keep_pos_cir].reshape(-1, 4)).reshape(keep_pos_cir.sum(), -1)

        # Please Triple check to avoid Nan and Inf
        keep = (torch.isinf(loc_pos).sum(-1) + torch.isnan(loc_pos).sum(-1)) > 0
        if keep.sum() > 0:
            print('Inf or Nan occur in loc_t when matching samples:', loc_pos[keep])
            pos_ind = torch.arange(n_proposals)[keep_pos_cir]
            conf_t[kdx][pos_ind[keep]] = -1
        # Filter out those predicted boxes with inf or nan
        keep = (torch.isinf(loc_data).sum(-1) + torch.isnan(loc_data).sum(-1)) > 0
        if keep.sum() > 0:
            print('Num of Inf or Nan in loc_data when matching samples:', len(torch.nonzero(keep, as_tuple=False)))
            conf_t[kdx][keep] = -1

        loc_t[kdx, keep_pos_cir] = loc_pos
        idx_t[kdx] = valid_idx[best_truth_idx_cir].view(-1)
        if gt_obj_ids is not None:
            obj_ids_t[kdx] = gt_obj_ids[valid][best_truth_idx_cir]