# -*- coding: utf-8 -*-
import torch

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

@torch.jit.script
def encode(matched, priors):
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
def decode(loc, priors):
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
        - priors: The priorbox coords with size [num_priors, 4] [cx, cy, w, h]
    
    Returns: A tensor of decoded relative coordinates in point form 
             form with size [num_priors, 4]
    """

    variances = [0.1, 0.2]
    boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    # [x1, y1, x2, y2]
    return boxes

def circum_boxes(boxes):
    '''
    :param boxes: [n, 4*T], [x1^t,y1^t,x2^t,y2^t,x1^(t+1), y1^(t+1), x2^(t+1), y2^(t+1)]
    :return: [n, 4]
    '''
    cir_boxes = torch.stack([boxes[..., ::2].min(-1)[0], boxes[..., 1::2].min(-1)[0],
                             boxes[..., ::2].max(-1)[0], boxes[..., 1::2].max(-1)[0]], dim=-1)
    return cir_boxes


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
    h, w, c, _ = masks.size()
    n = boxes.size(0)

    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding, cast=True)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding, cast=True)

    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1, 1).expand(h, w, c, n)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1, 1).expand(h, w, c, n)
    
    masks_left  = rows >= x1.view(1, 1, 1, -1)
    masks_right = rows <  x2.view(1, 1, 1, -1)
    masks_up    = cols >= y1.view(1, 1, 1, -1)
    masks_down  = cols <  y2.view(1, 1, 1, -1)
    crop_mask = (masks_left * masks_right * masks_up * masks_down).float()

    if masks.size(-1) == 1 and n > 1:
        crop_mask = (crop_mask.sum(dim=-1) > 0).float()
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
    boxes_a = boxes_a.float()
    boxes_b = boxes_b.float()
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


def DIoU_loss(boxes_a, boxes_b, reduction: str = "none", eps: float = 1e-7):
    '''
    :param boxes_a: [n, 4]
    :param boxes_b: [n, 4]
    :return:
    '''
    assert boxes_a.size(0) == boxes_b.size(0), print('boxes1 and boxes2 must have same size!')
    x1, y1, x2, y2 = boxes_a.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes_b.unbind(dim=-1)

    x1, x2 = torch.min(x1, x2), torch.max(x1, x2)
    y1, y2 = torch.min(y1, y2), torch.max(y1, y2)
    x1g, x2g = torch.min(x1g, x2g), torch.max(x1g, x2g)
    y1g, y2g = torch.min(y1g, y2g), torch.max(y1g, y2g)

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    iouk = intsctk / (unionk + eps)

    # the diagonal length of smallest enclosing box
    c2 = (torch.max(x2, x2g)-torch.min(x1, x1g))**2 + (torch.max(y2, y2g)-torch.min(y1, y1g))**2 + eps
    # get the distance of centres between pred_bbox and gt_bbox
    d2 = (0.5*(x1+x2)-0.5*(x1g+x2g))**2 + (0.5*(y1+y2)-0.5*(y1g+y2g))**2

    DIoU = iouk - d2/c2

    loss = 1 - DIoU

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss




