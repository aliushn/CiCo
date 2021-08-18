# -*- coding: utf-8 -*-
import torch
from datasets import STD, MEANS, cfg
from ..visualization_temporal import display_pos_smaples

@torch.jit.script
def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


@torch.jit.script
def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted cx, cy, w, h form of boxes.
    """
    return torch.cat(( (boxes[:, 2:] + boxes[:, :2])/2,     # cx, cy
                        boxes[:, 2:] - boxes[:, :2]  ), 1)  # w, h

@torch.jit.script
def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [n,A,4]. [x,y,x2,y2]
      box_b: (tensor) bounding boxes, Shape: [n,B,4]. [x,y,x2,y2]
    Return:
      (tensor) intersection area, Shape: [n,A,B].
    """
    n = box_a.size(0)

    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min((box_a[:, :, 2:]).unsqueeze(2).expand(n, A, B, 2),
                       (box_b[:, :, 2:]).unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, :, 0] * inter[:, :, :, 1]


def jaccard(box_a, box_b, iscrowd:bool=False):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [n,A,4] [x1,y1, x2, y2]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [n,B,4] [x1,y1,x2,y2]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)                                               # [n, A, B]
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0])
              * (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [n, A, B]
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0])
              * (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [n, A, B]
    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out.squeeze(0)


def change(gt, priors):
    """
    Compute the d_change metric proposed in Box2Pix:
    https://lmb.informatik.uni-freiburg.de/Publications/2018/UB18/paper-box2pix.pdf
    
    Input should be in point form (xmin, ymin, xmax, ymax).

    Output is of shape [num_gt, num_priors]
    Note this returns -change so it can be a drop in replacement for 
    """
    num_priors = priors.size(0)
    num_gt     = gt.size(0)

    gt_w = (gt[:, 2] - gt[:, 0])[:, None].expand(num_gt, num_priors)
    gt_h = (gt[:, 3] - gt[:, 1])[:, None].expand(num_gt, num_priors)

    gt_mat =     gt[:, None, :].expand(num_gt, num_priors, 4)
    pr_mat = priors[None, :, :].expand(num_gt, num_priors, 4)

    diff = gt_mat - pr_mat
    diff[:, :, 0] /= gt_w
    diff[:, :, 2] /= gt_w
    diff[:, :, 1] /= gt_h
    diff[:, :, 3] /= gt_h

    return -torch.sqrt((diff ** 2).sum(dim=2) )


def match_clip(bbox_list, labels_list, obj_ids_list, priors, loc_data, loc_t, conf_t, idx_t, obj_ids_t, idx,
               pos_thresh=0.5, neg_thresh=0.3):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        pos_thresh: (float) IoU > pos_thresh ==> positive.
        neg_thresh: (float) IoU < neg_thresh ==> negative.
        bbox_list: (tensor) Ground truth boxes, Shape: [num_obj, 4].  [x1, y1, x2, y2]
        labels_list: (tensor) All the class labels for the image, Shape: [num_obj].
        obj_ids_list: (tensor) the instance ids of each gt bbox
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4]. [cx,cy,w,h]
        loc_data: (tensor) The predicted bbox regression coordinates for this batch. [\delta x, \delta y,  \delta w,  \delta h]
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds. Note: -1 means neutral.
        idx_t: (tensor) Tensor to be filled w/ the index of the matched gt box for each prior.
        obj_ids_t: (tensor) Tensor to be filled w/ the ids of the matched gt instance for each prior.
        idx: (int) current batch index.
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """

    # decoded_priors [cx, cy, w, h] => [x1, y1, x2, y2]
    decoded_priors = point_form(priors)

    # Unique object ids in the clip
    obj_ids_unique = torch.cat(obj_ids_list).unique()
    n_clip_frames, n_objs, n_anchors = len(bbox_list), len(obj_ids_unique), loc_data.size(-2)

    # Merge boxes with same object id in the clip
    obj_ids_list2clip_idx = []
    objs_clip_flag = loc_data.new_zeros(n_clip_frames, n_objs).long()
    bbox_clip = loc_data.new_zeros(n_clip_frames, n_objs, 4)
    labels_clip = loc_data.new_ones(n_objs).long() * -1
    for cdx in range(n_clip_frames):
        obj_ids_cdx = [obj_ids_unique.tolist().index(id) for id in obj_ids_list[cdx]]
        obj_ids_list2clip_idx.append(obj_ids_cdx)
        # Due to occlusion or other cases, an object may disappear in some frames of the clip, we need a flag
        objs_clip_flag[cdx, obj_ids_cdx] = 1
        bbox_clip[cdx, obj_ids_cdx] = bbox_list[cdx]
        labels_clip[obj_ids_cdx] = labels_list[cdx]

    # Remove error annotations with width = 0 or height = 0
    bbox_clip_c = center_size(bbox_clip.reshape(-1, 4))
    bbox_clip_c[:, 2:] = torch.clamp(bbox_clip_c[:, 2:], min=1e-7)
    bbox_clip = point_form(bbox_clip_c).reshape(n_clip_frames, n_objs, 4)

    assert cfg.clip_prediction_with_individual_box or cfg.clip_prediction_with_external_box
    if cfg.clip_prediction_with_external_box:
        # ------- Introducing external boxes of multiple boxes in the clip to define postive and negative samples
        bbox_clip_unfold = bbox_clip.permute(1, 0, 2).contiguous().reshape(n_objs, -1)
        external_box = torch.stack([bbox_clip_unfold[:, ::2].min(-1)[0], bbox_clip_unfold[:, 1::2].min(-1)[0],
                                    bbox_clip_unfold[:, ::2].max(-1)[0], bbox_clip_unfold[:, 1::2].max(-1)[0]], dim=-1)
        # if the object only occurs once in the clip, the external rectangle is himself
        single_object_idx = objs_clip_flag.sum(dim=0) == 1
        if single_object_idx.sum() > 0:
            external_box[single_object_idx] = bbox_clip[:, single_object_idx].reshape(-1, 4)[objs_clip_flag[:, single_object_idx].reshape(-1) == 1]
        overlaps_clip_ext = jaccard(external_box, decoded_priors)
        best_truth_overlap_ext, best_truth_idx_ext = overlaps_clip_ext.max(dim=0)
        # Define positive sample,  negative samples and neutral
        keep_pos_ext = best_truth_overlap_ext > pos_thresh
        keep_neg_ext = best_truth_overlap_ext < neg_thresh
        # print('pos_ext:', torch.nonzero(keep_pos_ext).reshape(1, -1))
        # print('pos_ext:', best_truth_idx_ext[keep_pos_ext].reshape(1, -1))

    if cfg.clip_prediction_with_individual_box:

        # ------ Mean of multiple boxes in the clip
        # Size [num_objects, num_priors]
        overlaps_frames = torch.zeros(n_clip_frames, n_objs, n_anchors)
        for cdx in range(n_clip_frames):
            flag = objs_clip_flag[cdx] == 1
            if cfg.use_DIoU:
                overlaps = compute_DIoU(bbox_clip[cdx][flag], decoded_priors) if not cfg.use_change_matching else change(bbox_clip[cdx][flag], decoded_priors)
                pos_thresh, neg_thresh = 0.85 * pos_thresh, 0.85 * neg_thresh

            else:
                overlaps = jaccard(bbox_clip[cdx][flag], decoded_priors) if not cfg.use_change_matching else change(bbox_clip[cdx][flag], decoded_priors)
            overlaps_frames[cdx, obj_ids_list2clip_idx[cdx]] = overlaps

        # if 1) the clip average of overlaps on an anchor is greater than pos_thresh,
        #    2) each iou of anchor and boxes on frames of the clip is greater than 0.5 * pos,
        # we assume the instance moves slowly between adjacent frames,
        # which only needs an anchor to predict the regression of bounding boxes.
        overlaps_clip_ave = overlaps_frames.sum(dim=0) / objs_clip_flag.sum(dim=0).view(-1, 1)
        best_truth_overlap_ave, best_truth_idx_ave = overlaps_clip_ave.max(0)
        best_truth_overlap_ind = overlaps_frames[:, best_truth_idx_ave, range(n_anchors)]
        keep_half_pos_ind = (best_truth_overlap_ind > (0.5 * pos_thresh)).sum(dim=0) >= objs_clip_flag.sum(dim=0)[best_truth_idx_ave]

        # We want to ensure that each gt gets used at least once so that we don't
        # waste any training data. In order to do that, find the max overlap anchor
        # with each gt, and force that anchor to use that gt.
        for _ in range(n_objs):
            # Find j, the gt with the highest overlap with a prior
            # In effect, this will loop through overlaps.size(0) in a "smart" order,
            # always choosing the highest overlap first.
            best_prior_overlap, best_prior_idx = overlaps_clip_ave.max(1)
            j = best_prior_overlap.max(0)[1]

            # Find i, the highest overlap anchor with this gt
            i = best_prior_idx[j]

            # Set all other overlaps with i to be -1 so that no other gt uses it
            overlaps_clip_ave[:, i] = -1
            # Set all other overlaps with j to be -1 so that this loop never uses j again
            overlaps_clip_ave[j, :] = -1

            # Set each box is greater than the half of pos threshold
            keep_half_pos_ind[i] = 1

            # Overwrite i's score to be 2 so it doesn't get thresholded ever
            best_truth_overlap_ave[i] = 2
            # Set the gt to be used for i to be j, overwriting whatever was there
            best_truth_idx_ave[i] = j

        keep_pos_ave = (best_truth_overlap_ave > pos_thresh) & keep_half_pos_ind
        keep_neg_ind = (best_truth_overlap_ind < neg_thresh).sum(dim=0) >= objs_clip_flag.sum(dim=0)[best_truth_idx_ave]
        keep_neg_ave = (best_truth_overlap_ave < neg_thresh) & keep_neg_ind[best_truth_idx_ave]

        # print('pos_ind:', torch.nonzero(keep_pos_ave).reshape(1, -1))
        # print('pos_ind:', best_truth_idx_ave[keep_pos_ave].reshape(1, -1))

    if cfg.clip_prediction_with_individual_box and cfg.clip_prediction_with_external_box:
        best_truth_idx = best_truth_idx_ave
        best_truth_idx[keep_pos_ext] = best_truth_idx_ext[keep_pos_ext]
        keep_pos = keep_pos_ext | keep_pos_ave
        keep_neg = keep_neg_ext & keep_neg_ave
        # print(keep_pos_ave.sum(), keep_pos.sum())
    elif cfg.clip_prediction_with_individual_box:
        best_truth_idx = best_truth_idx_ave
        keep_pos = keep_pos_ave
        keep_neg = keep_neg_ave
    else:
        best_truth_idx = best_truth_idx_ext
        keep_pos = keep_pos_ext
        keep_neg = keep_neg_ext

    matches = bbox_clip[:, best_truth_idx]                            # [n_clip, num_priors, 4], [x1, y1, x2, y2]
    loc = encode(matches.reshape(-1, 4), priors.repeat(n_clip_frames, 1), cfg.use_yolo_regressors).reshape(-1, n_anchors, 4)
    loc = loc.permute(1, 0, 2).contiguous().reshape(n_anchors, -1)    # [n_anchors, 4*n_clip_frames]
    conf = loc_data.new_ones(n_anchors).long() * -1                   # label as neutral
    conf[keep_pos] = labels_clip[best_truth_idx[keep_pos]]            # label as positive samples
    conf[keep_neg] = 0                                                # label as background

    # triple check to avoid Nan and Inf
    keep = (torch.isinf(loc).sum(-1) + torch.isnan(loc).sum(-1)) > 0
    if keep.sum() > 0:
        print('Inf or Nan occur in loc_t:', loc[keep])
        conf[keep] = -1
    # filter out those predicted boxes with inf or nan
    keep = (torch.isinf(loc_data).sum(-1) + torch.isnan(loc_data).sum(-1)) > 0
    if keep.sum() > 0:
        print('Num of Inf or Nan', len(torch.nonzero(keep)))
        conf[keep] = -1

    loc_t[idx] = loc                                                 # [num_priors, 4*clip_frames] encoded offsets to learn
    conf_t[idx] = conf                                               # [num_priors] top class label for each prior
    for cdx in range(n_clip_frames):
        obj_ids_clip2list = [obj_ids_list[cdx].tolist().index(id) if id in obj_ids_list[cdx] else -1 for id in obj_ids_unique.tolist()]
        idx_t[idx, keep_pos, cdx] = torch.tensor(obj_ids_clip2list)[best_truth_idx[keep_pos]]    # [num_priors] indices for lookup
    if obj_ids_list is not None:
        id_cur = loc_data.new_ones(n_anchors).long() * -1
        id_cur[keep_pos] = obj_ids_unique[best_truth_idx[keep_pos]]  # Only remain positive boxes for tracking
        obj_ids_t[idx] = id_cur


def match(bbox, labels, ids, crowd_boxes, priors, loc_data, loc_t, conf_t, idx_t, ids_t, idx,
          pos_thresh=0.5, neg_thresh=0.3, img_gpu=None):
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
    # some error annotatons with width = 0 or height = 0
    bbox_c = center_size(bbox)
    bbox_c[:, 2:] = torch.clamp(bbox_c[:, 2:], min=1e-7)
    bbox = point_form(bbox_c)

    # decoded_priors => [x1, y1, x2, y2]
    decoded_priors = decode(loc_data, priors, cfg.use_yolo_regressors) if cfg.use_prediction_matching else point_form(priors)
    
    # Size [num_objects, num_priors]
    if cfg.use_DIoU:
        overlaps = compute_DIoU(bbox, decoded_priors) if not cfg.use_change_matching else change(bbox, decoded_priors)
        pos_thresh, neg_thresh = 0.85 * pos_thresh, 0.85 * neg_thresh
    else:
        overlaps = jaccard(bbox, decoded_priors) if not cfg.use_change_matching else change(bbox, decoded_priors)

    # Size [num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0)

    # if a bbox is matched with two or more groundtruth boxes, this bbox will be assigned as -1
    thresh = 0.5 * (pos_thresh + neg_thresh)
    multiple_bbox = (overlaps > pos_thresh).sum(dim=0) > 1
    # print((best_truth_overlap > pos_thresh).sum(), 'multi:', multiple_bbox.sum(), 'gt:', bbox.size(0))
    best_truth_overlap[multiple_bbox] = thresh
    # temp = (best_truth_overlap > pos_thresh).reshape(-1)
    # print(temp.sum(), bbox.size(0))
    # print(best_truth_overlap[temp])

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

    matches = bbox[best_truth_idx]            # Shape: [num_priors,4]  [x1, y1, x2, y2]
    conf = labels[best_truth_idx]             # Shape: [num_priors]
    conf[~keep_pos] = -1                      # label as neutral
    conf[keep_neg] = 0                        # label as background

    if img_gpu is not None:
        pos = conf > 0
        display_pos_smaples(pos, img_gpu, decoded_priors, bbox)

    # Deal with crowd annotations for COCO
    if crowd_boxes is not None and cfg.crowd_iou_threshold < 1:
        # Size [num_priors, num_crowds]
        crowd_overlaps = jaccard(decoded_priors, crowd_boxes, iscrowd=True)
        # Size [num_priors]
        best_crowd_overlap, best_crowd_idx = crowd_overlaps.max(1)
        # Set non-positives with crowd iou of over the threshold to be neutral.
        conf[(conf <= 0) & (best_crowd_overlap > cfg.crowd_iou_threshold)] = -1

    loc = encode(matches, priors, cfg.use_yolo_regressors)  # [cx, cy, w, h]
    keep = (torch.isinf(loc).sum(-1) + torch.isnan(loc).sum(-1)) > 0
    if keep.sum() > 0:
        print('Inf or Nan occur in loc_t:', loc[keep])
        conf[keep] = -1
    # filter out those predicted boxes with inf or nan
    keep = (torch.isinf(loc_data).sum(-1) + torch.isnan(loc_data).sum(-1)) > 0
    if len(torch.nonzero(keep)) > 0:
        print('Num of Inf or Nan', len(torch.nonzero(keep)))
        conf[keep] = -1

    loc_t[idx]  = loc                                            # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf                                           # [num_priors] top class label for each prior
    idx_t[idx, keep_pos] = best_truth_idx[keep_pos].view(-1, 1)  # [num_priors] indices for lookup
    if ids is not None:
        ids_t[idx, keep_pos] = ids[best_truth_idx[keep_pos]]

@torch.jit.script
def encode(matched, priors, use_yolo_regressors:bool=False):
    """
    Encode bboxes matched with each prior into the format
    produced by the network. See decode for more details on
    this format. Note that encode(decode(x, p), p) = x.
    
    Args:
        - matched: A tensor of bboxes in point form with shape [num_priors, 4] [x,y,x2,y2]
        - priors:  The tensor of all priors with shape [num_priors, 4] [cx,cy,w,h]
    Return: A tensor with encoded relative coordinates in the format
            outputted by the network (see decode). Size: [num_priors, 4]
    """

    if use_yolo_regressors:
        # Exactly the reverse of what we did in decode
        # In fact encode(decode(x, p), p) should be x
        boxes = center_size(matched)

        loc = torch.cat((
            boxes[:, :2] - priors[:, :2],
            torch.log(boxes[:, 2:] / priors[:, 2:])
        ), 1)
    else:
        variances = [0.1, 0.2]

        # dist b/t match center and prior's center
        g_cxcy = (matched[:, :2] + matched[:, 2:])/2. - priors[:, :2]
        # encode variance
        g_cxcy /= (variances[0] * priors[:, 2:])
        # match wh / prior wh
        g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
        g_wh = torch.log(g_wh) / variances[1]
        # return target for smooth_l1_loss
        loc = torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]
        
    return loc

@torch.jit.script
def decode(loc, priors, use_yolo_regressors:bool=False):
    """
    Decode predicted bbox coordinates using the same scheme
    employed by Yolov2: https://arxiv.org/pdf/1612.08242.pdf

        b_x = (sigmoid(pred_x) - .5) / conv_w + prior_x
        b_y = (sigmoid(pred_y) - .5) / conv_h + prior_y
        b_w = prior_w * exp(loc_w)
        b_h = prior_h * exp(loc_h)
    
    Note that loc is inputed as [(s(x)-.5)/conv_w, (s(y)-.5)/conv_h, w, h]
    while priors are inputed as [x, y, w, h] where each coordinate
    is relative to size of the image (even sigmoid(x)). We do this
    in the network by dividing by the 'cell size', which is just
    the size of the convouts.
    
    Also note that prior_x and prior_y are center coordinates which
    is why we have to subtract .5 from sigmoid(pred_x and pred_y).
    
    Args:
        - loc:    The predicted bounding boxes of size [num_priors, 4]
        - priors: The priorbox coords with size [num_priors, 4]
    
    Returns: A tensor of decoded relative coordinates in point form 
             form with size [num_priors, 4]
    """

    if use_yolo_regressors:
        # Decoded boxes in center-size notation
        boxes = torch.cat((
            loc[:, :2] + priors[:, :2],
            priors[:, 2:] * torch.exp(loc[:, 2:])
        ), 1)

        boxes = point_form(boxes)
    else:
        variances = [0.1, 0.2]
        
        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

    # [x1, y1, x2, y2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1)) + x_max


@torch.jit.script
def sanitize_coordinates(_x1, _x2, img_size:int, padding:int=0, cast:bool=True):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.

    If cast is false, the result won't be cast to longs.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = torch.clamp(_x1, min=0, max=1)
    _x2 = torch.clamp(_x2, min=0, max=1)
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    if cast:
        _x1 = _x1.long()
        _x2 = _x2.long()
    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)
    x1 = torch.clamp(x1-padding, min=0)
    x2 = torch.clamp(x2+padding, max=img_size)

    return x1, x2


def sanitize_coordinates_hw(box, h, w):
    '''
    :param box: [bs, n, 4], [x1, y1, x2, y2]
    :param h: height of image
    :param w:
    :return:
    '''

    use_batch = True
    if box.dim() == 2:
        use_batch = False
        box = box[None, ...]

    x1, x2 = sanitize_coordinates(box[:, :, 0], box[:, :, 2], w, cast=False)
    y1, y2 = sanitize_coordinates(box[:, :, 1], box[:, :, 3], h, cast=False)
    box_wo_norm = torch.stack([x1, y1, x2, y2], dim=-1)

    return box_wo_norm if use_batch else box_wo_norm.squeeze(0)


# @torch.jit.script
def crop(masks, boxes, padding:int=1, return_mask=False):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, c, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form [x1,y1,x2,y2]
    """
    d_masks = len(masks.size())
    if d_masks == 3:
        masks = masks.unsqueeze(2)
    h, w, c, n = masks.size()

    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding, cast=False)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding, cast=False)

    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1, 1).expand(h, w, c, n)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1, 1).expand(h, w, c, n)
    
    masks_left  = rows >= x1.view(1, 1, 1, -1)
    masks_right = rows <  x2.view(1, 1, 1, -1)
    masks_up    = cols >= y1.view(1, 1, 1, -1)
    masks_down  = cols <  y2.view(1, 1, 1, -1)
    
    crop_mask = (masks_left * masks_right * masks_up * masks_down).float()
    cropped_masks = crop_mask * masks
    if d_masks == 3:
        crop_mask = crop_mask.squeeze(2)
        cropped_masks = cropped_masks.squeeze(2)

    return (crop_mask, cropped_masks) if return_mask else cropped_masks


def crop_sipmask(masks00, masks01, masks10, masks11, boxes, padding:int=1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).
    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = masks00.size()
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding, cast=False)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding, cast=False)
    rows = torch.arange(w, device=masks00.device, dtype=boxes.dtype).view(1, -1, 1).expand(h, w, n)
    cols = torch.arange(h, device=masks00.device, dtype=boxes.dtype).view(-1, 1, 1).expand(h, w, n)

    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    x1 = torch.clamp(x1, min=0, max=w - 1)
    y1 = torch.clamp(y1, min=0, max=h - 1)
    x2 = torch.clamp(x2, min=0, max=w - 1)
    y2 = torch.clamp(y2, min=0, max=h - 1)
    xc = torch.clamp(xc, min=0, max=w - 1)
    yc = torch.clamp(yc, min=0, max=h - 1)

    ## x1,y1,xc,yc
    crop_mask = (rows >= x1.view(1, 1, -1)) & (rows < xc.view(1, 1, -1)) & (cols >= y1.view(1, 1, -1)) & (
                cols < yc.view(1, 1, -1))
    crop_mask = crop_mask.float().detach()

    masks00 = masks00 * crop_mask

    ## xc,y1,x2,yc
    crop_mask = (rows >= xc.view(1, 1, -1)) & (rows < x2.view(1, 1, -1)) & (cols >= y1.view(1, 1, -1)) & (
                cols < yc.view(1, 1, -1))
    crop_mask = crop_mask.float().detach()
    masks01 = masks01 * crop_mask

    crop_mask = (rows >= x1.view(1, 1, -1)) & (rows < xc.view(1, 1, -1)) & (cols >= yc.view(1, 1, -1)) & (
                cols < y2.view(1, 1, -1))
    crop_mask = crop_mask.float().detach()
    masks10 = masks10 * crop_mask

    crop_mask = (rows >= xc.view(1, 1, -1)) & (rows < x2.view(1, 1, -1)) & (cols >= yc.view(1, 1, -1)) & (
                cols < y2.view(1, 1, -1))
    crop_mask = crop_mask.float().detach()
    masks11 = masks11 * crop_mask

    masks = masks00 + masks01 + masks10 + masks11

    return masks


def index2d(src, idx):
    """
    Indexes a tensor by a 2d index.

    In effect, this does
        out[i, j] = src[i, idx[i, j]]
    
    Both src and idx should have the same size.
    """

    offs = torch.arange(idx.size(0), device=idx.device)[:, None].expand_as(idx)
    idx  = idx + offs * idx.size(1)

    return src.view(-1)[idx.view(-1)].view(idx.size())


def gaussian_kl_divergence(bbox_gt, bbox_pred):
    cwh_gt = bbox_gt[:, 2:] - bbox_gt[:, :2]
    cwh_pred = bbox_pred[:, 2:] - bbox_pred[:, :2]

    mu_gt = bbox_gt[:, :2] + 0.5 * cwh_gt
    mu_pred = bbox_pred[:, :2] + 0.5 * cwh_pred
    sigma_gt = cwh_gt / 4.0
    sigma_pred = cwh_pred / 4.0

    kl_div0 = (sigma_pred / sigma_gt)**2 + (mu_pred - mu_gt)**2 / sigma_gt**2 - 1 + 2 * torch.log(sigma_gt / sigma_pred)
    kl_div1 = (sigma_gt / sigma_pred) ** 2 + (mu_gt - mu_pred) ** 2 / sigma_pred ** 2 - 1 + 2 * torch.log(
        sigma_pred / sigma_gt)
    loss = 0.25 * (kl_div0 + kl_div1).sum(-1)

    return loss


def compute_DIoU(boxes_a, boxes_b, eps: float = 1e-7):
    '''
    :param boxes_a: [n_a, 4]
    :param boxes_b: [n_b, 4]
    :return: [n_a, n_b]
    '''
    n_a, n_b = boxes_a.size(0), boxes_b.size(0)

    # calculate bbox IoUs between pos_pred_boxes and pos_gt_boxes, [n_a, n_b]
    IoU = jaccard(boxes_a, boxes_b)

    # calculate the diagonal length of the smallest enclosing box, [n_a, n_b, 4]
    x_label = torch.cat([boxes_a[:, ::2].unsqueeze(1).repeat(1, n_b, 1),
                         boxes_b[:, ::2].unsqueeze(0).repeat(n_a, 1, 1)], dim=-1)
    y_label = torch.cat([boxes_a[:, 1::2].unsqueeze(1).repeat(1, n_b, 1),
                         boxes_b[:, 1::2].unsqueeze(0).repeat(n_a, 1, 1)], dim=-1)
    c2 = (x_label.max(dim=-1)[0] - x_label.min(dim=-1)[0])**2 + (y_label.max(dim=-1)[0] - y_label.min(dim=-1)[0])**2 + eps

    # get the distance of centres between pred_bbox and gt_bbox
    boxes_a_cxy = 0.5 * (boxes_a[:, :2] + boxes_a[:, 2:])
    boxes_b_cxy = 0.5 * (boxes_b[:, :2] + boxes_b[:, 2:])
    d2 = ((boxes_a_cxy.unsqueeze(1).repeat(1, n_b, 1) - boxes_b_cxy.unsqueeze(0).repeat(n_a, 1, 1))**2).sum(dim=-1)

    # DIoU: value in [-1, 1], size is [n_a, n_b]
    # 1-DIoU: value in [0, 2]
    DIoU = IoU - d2/c2

    return DIoU




