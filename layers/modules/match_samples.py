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
    bbox_c = center_size(bbox)
    keep = (bbox_c[:, 2] > 0) & (bbox_c[:, 3] > 0)
    valid_bbox = bbox[keep]
    valid_idx = torch.nonzero(keep, as_tuple=False).reshape(-1)
    small_objs_idx = torch.nonzero(bbox_c[:, 2] * bbox_c[:, 3] < 0.01, as_tuple=False)

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
    # print(idx, keep_pos.sum())
    # print('idx:', best_truth_idx[keep_pos])
    # print('labels:', labels[best_truth_idx[keep_pos]])

    valid_best_truth_idx = valid_idx[best_truth_idx]
    matches = bbox[valid_best_truth_idx]            # Shape: [num_priors,4]  [x1, y1, x2, y2]
    conf = labels[valid_best_truth_idx]             # Shape: [num_priors]
    conf[~keep_pos] = -1                      # label as neutral
    conf[keep_neg] = 0                        # label as background

    # Deal with crowd annotations for COCO
    if crowd_boxes is not None and cfg.MODEL.PREDICTION_HEADS.CROWD_IoU_THRESHOLD < 1:
        # Size [num_priors, num_crowds]
        crowd_overlaps = jaccard(decoded_priors, crowd_boxes, iscrowd=True)
        # Size [num_priors]
        best_crowd_overlap, best_crowd_idx = crowd_overlaps.max(1)
        # Set non-positives with crowd iou of over the threshold to be neutral.
        conf[(conf <= 0) & (best_crowd_overlap > cfg.MODEL.PREDICTION_HEADS.CROWD_IoU_THRESHOLD)] = -1

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
    conf_t[idx] = conf                                           # [num_priors] top class label for each prior
    idx_t[idx]  = valid_best_truth_idx.view(-1)                  # [num_priors] indices for lookup
    if ids is not None:
        ids_t[idx] = ids[valid_best_truth_idx]


def match_clip(cfg, gt_boxes, gt_labels, gt_obj_ids, priors, loc_data, loc_t, conf_t, idx_t,
               obj_ids_t, idx, pos_thresh=0.5, neg_thresh=0.3, circumscribed_boxes=False):
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
        idx: (int) current batch index.
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """

    n_anchors = loc_data.size(0)
    n_objs, n_clip_frames, _ = gt_boxes.size()
    
    # A instance may disappear in some frames of the clip due to occlusion or fast object/camera motion.
    # Thus we have to distinguish whether an instance exists in a frame, called pos_frames.
    # If the anchor is matched ground-truth bounding boxes at least one frame, called pos_clip.
    # When clip_frames=1, pos_clip (clip-level) is same as pos_frames (frame-level)
    # if the object disappears in the clip due to occlusion or other cases, the box is supplemented
    # from the most adjacent frames
    missed_flag = ((gt_boxes[:, :, 2:]-gt_boxes[:, :, :2]) <= 0).sum(dim=-1) > 1
    exist_missed_objects = missed_flag.sum(dim=-1) > 0
    if exist_missed_objects.sum() > 0:
        for kdx, missed in enumerate(exist_missed_objects):
            if missed:
                missed_cdx = torch.arange(n_clip_frames)[missed_flag[kdx, :] == 1]
                occur_cdx = torch.arange(n_clip_frames)[missed_flag[kdx, :] == 0]
                if occur_cdx.nelement() > 0:
                    supp_cdx = torch.abs(missed_cdx.reshape(-1, 1) - occur_cdx.reshape(1, -1)).min(dim=-1)[1]
                    gt_boxes[kdx, missed_cdx] = gt_boxes[kdx, occur_cdx[supp_cdx]]

    # ------- Introducing smallest enclosing boxes of multiple boxes in the clip to define positive/negative samples
    # if the number of frames is odd, use the central 3 frames as target frames, else 2 frames
    if n_clip_frames % 2 == 0:
        gt_boxes_central_unfold = gt_boxes[:, n_clip_frames//2-1:n_clip_frames//2+1].reshape(n_objs, -1)
    else:
        gt_boxes_central_unfold = gt_boxes[:, n_clip_frames//2-1:n_clip_frames//2+2].reshape(n_objs, -1)

    gt_circumscribed_box = circum_boxes(gt_boxes_central_unfold)

    # compute iou for each object
    # iou = torch.stack([jaccard(gt_boxes[i], gt_boxes[i]) for i in range(n_objs)], dim=0).triu_(1)
    # divergence = iou.sum(dim=(1, 2)) / max(n_clip_frames-1, 1) / max(n_clip_frames-2, 1)

    # decoded_priors [cx, cy, w, h] => [x1, y1, x2, y2]
    # TODO: add cfg.MODEL.PREDICTION_HEADS.USE_PREDICTION_MATCHING???
    decoded_priors = point_form(priors[:, :4])
    overlaps_clip_cir = jaccard(gt_circumscribed_box, decoded_priors)
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

        # Set all other overlaps with i to be -1 so that no other gt uses it
        overlaps_clip_cir[:, i] = -1
        # Set all other overlaps with j to be -1 so that this loop never uses j again
        overlaps_clip_cir[j, :] = -1

        # Overwrite i's score to be 2 so it doesn't get thresholded ever
        best_truth_overlap_cir[i] = 2
        # Set the gt to be used for i to be j, overwriting whatever was there
        best_truth_idx_cir[i] = j
    
    # Small objects use smaller threshold in order to have more postive samples
    gt_boxes_c = center_size(gt_boxes.reshape(-1, 4)).reshape(n_objs, n_clip_frames, 4)
    small_objs_idx = ((gt_boxes_c[..., 2] * gt_boxes_c[..., 3] < 0.01).sum(dim=1) >= n_clip_frames//2).float()
    small_objs_idx = torch.nonzero(small_objs_idx, as_tuple=False)
    if small_objs_idx.nelement() > 0:
        for i in small_objs_idx:
            best_truth_overlap_cir[best_truth_idx_cir == i] *= 1.1
    # Define positive sample, negative samples and neutral
    keep_pos_cir = best_truth_overlap_cir > pos_thresh
    keep_neg_cir = best_truth_overlap_cir < neg_thresh
    # print(keep_pos_cir.sum(), n_objs)
    # print(divergence)
    # print(best_truth_idx_cir[keep_pos_cir], best_truth_overlap_cir[keep_pos_cir])

    if not circumscribed_boxes:
        # ------ Mean of multiple boxes in the clip
        # Size [num_objects, num_priors]
        decoded_boxes = decode(loc_data.reshape(-1, 4), priors.reshape(-1, 4)).reshape(n_anchors, -1, 4)
        overlaps_frames = torch.zeros(n_clip_frames, n_objs, n_anchors)
        for cdx in range(n_clip_frames):
            if cfg.MODEL.PREDICTION_HEADS.USE_PREDICTION_MATCHING:
                if cfg.MODEL.PREDICTION_HEADS.USE_DIoU:
                    overlaps_frames[cdx] = 0.5*compute_DIoU(gt_boxes[:, cdx], decoded_boxes[:, cdx]) \
                                           + 0.5*compute_DIoU(gt_boxes[:, cdx], decoded_priors)
                else:
                    overlaps_frames[cdx] = 0.5*jaccard(gt_boxes[:, cdx], decoded_boxes[:, cdx]) \
                                           + 0.5*jaccard(gt_boxes[:, cdx], decoded_priors)
            else:
                overlaps_frames[cdx] = compute_DIoU(gt_boxes[:, cdx], decoded_priors) if cfg.MODEL.PREDICTION_HEADS.USE_DIoU \
                    else jaccard(gt_boxes[:, cdx], decoded_priors)

        # if 1) average iou of the entire clip between ground truth boxes an predicted boxes should greater than pos_thresh,
        #    2) each iou of ground truth and predicted boxes on a single frame should greater than 0.5 * pos,
        # here we assume the instance moves slowly between adjacent frames,
        # which only needs an anchor to predict the regression of bounding boxes.
        overlaps_clip_ave = overlaps_frames.mean(dim=0)
        best_truth_overlap_ave, best_truth_idx_ave = overlaps_clip_ave.max(0)
        best_truth_overlap_ind = overlaps_frames[:, best_truth_idx_ave, range(n_anchors)]
        keep_half_pos_ind = (best_truth_overlap_ind > (0.5 * pos_thresh)).sum(dim=0) == n_clip_frames
        keep_pos_ave = (best_truth_overlap_ave > pos_thresh) & keep_half_pos_ind
        keep_neg_ind = (best_truth_overlap_ind < neg_thresh).sum(dim=0) >= n_clip_frames
        keep_neg_ave = (best_truth_overlap_ave < neg_thresh) & keep_neg_ind[best_truth_idx_ave]
    
        best_truth_idx = best_truth_idx_ave
        best_truth_idx[keep_pos_cir] = best_truth_idx_cir[keep_pos_cir]
        keep_pos = keep_pos_cir | keep_pos_ave
        keep_neg = keep_neg_cir & keep_neg_ave
        # print(keep_pos.sum(), n_objs)
        # print(best_truth_idx_ave[keep_pos_ave], best_truth_overlap_ave[keep_pos_ave])
    else:
        best_truth_idx = best_truth_idx_cir
        keep_pos = keep_pos_cir
        keep_neg = keep_neg_cir

    conf = loc_data.new_ones(n_anchors).long() * -1                   # label as neutral
    conf[keep_pos] = gt_labels[best_truth_idx[keep_pos]]              # label as positive samples
    conf[keep_neg] = 0                                                # label as background

    if not circumscribed_boxes:
        # [n_anchors, 4*n_clip_frames]
        matches = gt_boxes.reshape(n_objs, -1)[best_truth_idx]                         # [n_clip, num_priors, 4], [x1, y1, x2, y2]
        loc_pos = encode(matches[keep_pos].reshape(-1, 4), priors[keep_pos].reshape(-1, 4)).reshape(-1, 4*n_clip_frames)
    else:
        matches = gt_circumscribed_box[best_truth_idx]
        loc_pos = encode(matches[keep_pos], priors[keep_pos, :4])
    # triple check to avoid Nan and Inf
    keep = (torch.isinf(loc_pos).sum(-1) + torch.isnan(loc_pos).sum(-1)) > 0
    if keep.sum() > 0:
        print('Inf or Nan occur in loc_t when matching samples:', loc_pos[keep])
        pos_ind = torch.arange(n_anchors)[keep_pos]
        conf[pos_ind[keep]] = -1
    # filter out those predicted boxes with inf or nan
    keep = (torch.isinf(loc_data).sum(-1) + torch.isnan(loc_data).sum(-1)) > 0
    if keep.sum() > 0:
        print('Num of Inf or Nan in loc_data when matching samples:', len(torch.nonzero(keep)))
        conf[keep] = -1
    loc_t[idx, keep_pos] = loc_pos                      # [num_priors, 4*clip_frames] encoded offsets to learn
    conf_t[idx] = conf                                           # [num_priors] top class label for each prior
    idx_t[idx] = best_truth_idx.view(-1)
    if gt_obj_ids is not None:
        obj_ids_t[idx] = gt_obj_ids[best_truth_idx]

    return gt_boxes
