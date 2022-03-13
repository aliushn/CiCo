# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .box_utils import crop, crop_sipmask, center_size


def generate_rel_coord_gauss(bbox, h, w, sigma_scale=2.):
    '''
    Generate relative coordinates by Gaussian, replacing relative coordinates of CondInst
    :param bbox: positive bounding boxes with shape [n, 4]: [x1, y1, x2, y2]
    :param h: height of pred_mask
    :param w: weight of pred_mask
    :return: rel_coord ==> [n, h, w, 2]
    '''

    n = bbox.size(0)
    # rescale the range form [0, 1] to [0, w] or [0, h]
    bbox_c = center_size(torch.cat([bbox[:, 0::2]*w,
                                    bbox[:, 1::2]*h], dim=-1)[:, [0, 2, 1, 3]])
    cx, cy = torch.round(bbox_c[:, 0]), torch.round(bbox_c[:, 1])
    # first create the grid with shape [h, w],
    y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w))
    # then minus the coordinates of objects central pixel (mean)
    y_rel_coord = y_grid.float().unsqueeze(0).repeat(n, 1, 1) - cy.view(-1, 1, 1)
    x_rel_coord = x_grid.float().unsqueeze(0).repeat(n, 1, 1) - cx.view(-1, 1, 1)

    # Build 2D Normal distribution: according to 3 sigma rule, F(u-3sigma <x< u+3sigma) = 99.7%，
    # the width equals to 6*sigma_x. Then the area beyond 3sigma will be zero, which is too sharp as attentions.
    # We use a large variance: sigma_x = 1.5*w. Note that normalization term is not required.
    sigma_scales = torch.ones(n, device=bbox.device)
    small_obj = bbox_c[:, 2] * bbox_c[:, 3] / h / w < 0.1
    sigma_scales[small_obj] = 0.5 * sigma_scale
    sigma_xy = torch.clamp(1.5 * bbox_c[:, 2:] / sigma_scales.reshape(-1, 1), min=1)
    rel_coord_gauss = torch.exp(-0.5 * ((x_rel_coord / sigma_xy[:, 0].reshape(-1, 1, 1))**2
                                        + (y_rel_coord / sigma_xy[:, 1].reshape(-1, 1, 1))**2))

    # You can display relative coordinates map just for double check
    # for i in range(rel_coord_gauss.size(0)):
    #     temp = rel_coord_gauss[i].cpu().numpy()
    #     plt.imshow(temp)
    #     plt.show()

    return rel_coord_gauss


def generate_rel_coord(bbox, fpn_levels, sizes_of_interest, h, w):
    '''
    Generate relative coordinates of objects, please refer to CondInst.
    :param bbox: pos bboxes with normalization ==> [x1, y1, x2, y2], whose values in [0, 1]
    :param h: height of pred_mask
    :param w: weight of pred_mask
    :return: rel_coord ==> [num_pos, h, w, 2]
    '''
    # Create the grid
    shifts_x = torch.arange(0, w,  dtype=torch.float32, device=bbox.device)
    shifts_y = torch.arange(0, h,  dtype=torch.float32, device=bbox.device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    locations = torch.stack((shift_x.reshape(-1), shift_y.reshape(-1)), dim=1) + 0.5

    instance_locations = center_size(bbox)[:, :2] * torch.tensor([w, h]).view(1, -1)
    relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
    relative_coords = relative_coords.permute(0, 2, 1).float()
    soi = sizes_of_interest[fpn_levels]
    relative_coords = relative_coords / soi.reshape(-1, 1, 1).float()

    return relative_coords.reshape(-1, 2, h, w)


def generate_mask(proto_data, mask_coeff, bbox=None, mask_dim=32, frame_wise=False, use_sipmask=False):
    '''
    Instance mask generation of Yolact or SipMask, where activation function is sigmoid.
    Support proto_data of a single frame or a clip
    :param proto_data : [h, w, 32] or [n, h, w, 32]
    :param mask_coeff: [n, 32], value in [-1, 1]
    :param bbox: [n, 4]
    :return:
    '''

    dim_proto = proto_data.dim()
    if use_sipmask:
        # SipMask uses four sets of mask coefficients to predict four parts of isntance masks:
        # including top-left, top-right, bottom-left and bottom-right boxes. https://github.com/JialeCao001/SipMask
        pred_masks00 = torch.sigmoid(generate_single_mask(proto_data,  mask_coeff[:, :mask_dim]))
        pred_masks01 = torch.sigmoid(generate_single_mask(proto_data, mask_coeff[:, mask_dim:mask_dim*2]))
        pred_masks10 = torch.sigmoid(generate_single_mask(proto_data, mask_coeff[:, mask_dim*2:mask_dim*3]))
        pred_masks11 = torch.sigmoid(generate_single_mask(proto_data, mask_coeff[:, mask_dim*3:]))
        assert bbox is not None
        pred_masks = crop_sipmask(pred_masks00, pred_masks01, pred_masks10, pred_masks11, bbox)

    else:
        # the original version of Yolact
        pred_masks = torch.sigmoid(generate_single_mask(proto_data, mask_coeff, frame_wise))
        if bbox is not None:
            pred_masks = crop(pred_masks, bbox)                                         # [h, w, T, n]

    if dim_proto == 3:
        pred_masks = pred_masks.permute(2, 0, 1).contiguous()                               # [n, h, w]
    elif dim_proto == 4:
        if pred_masks.dim() == 4:
            pred_masks = pred_masks.permute(3, 2, 0, 1).contiguous()                        # [n, T, h, w]
        else:
            pred_masks = pred_masks.permute(3, 4, 2, 0, 1).contiguous()                     # [n, 2, T, h, w]

    return pred_masks


def generate_single_mask(proto_data, mask_coeff, frame_wise=False):
    if frame_wise:
        # proto_data with shape [T, h, w, 32] and mask_coeff with shape [T, 32], where instance masks is generated
        # by linearly combining prototypes of the t-th frame and mask coefficients of the t-th frame
        pred_masks = (proto_data * mask_coeff.unsqueeze(1).unsqueeze(2)).sum(dim=-1)
        pred_masks = pred_masks.permute(1, 2, 0).contiguous()
    else:
        # Proto_data: [h,w,32] or [h,w,T,32], mask_coeff: [n,32]
        pred_masks = proto_data @ mask_coeff.t()

    return pred_masks


def mask_iou(mask_a, mask_b):
    '''
    Compute mask iou of obejcts in two frames or two clips
    :param mask_a: [c, n_a, h, w]
    :param mask_b: [c, n_b, h, w]
    :return: miou: [c, n_a, n_b]
    '''

    use_batch = True
    if mask_a.dim() == 3:
        use_batch = False
        mask_a = mask_a.unsqueeze(0)
        mask_b = mask_b.unsqueeze(0)
    c, n_a = mask_a.size()[:2]
    n_b = mask_b.size(1)
    mask_a = mask_a.view(c, n_a, 1, -1)
    mask_b = mask_b.view(c, 1, n_b, -1)
    intersection = torch.sum(mask_a * mask_b, dim=-1)   # [c, n_a, n_b]
    area_a = torch.sum(mask_a, dim=-1)                  # [c, n_a, 1  ]
    area_b = torch.sum(mask_b, dim=-1)                  # [c, 1,   n_b]
    union = (area_a + area_b) - intersection
    mask_ious = intersection.float() / torch.clamp(union.float(), min=1)
    return mask_ious if use_batch else mask_ious.squeeze(0)


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]

