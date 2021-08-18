# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .box_utils import crop, crop_sipmask, center_size
from datasets import cfg


def generate_rel_coord_gauss(det_bbox, mask_h, mask_w, sigma_scale=2.):
    '''
    :param det_box: pos bboxes ==> [x1, y1, x2, y2]
    :param mask_h: height of pred_mask
    :param mask_w: weight of pred_mask
    :return: rel_coord ==> [num_pos, mask_h, mask_w, 2]
    '''

    # generate relative coordinates
    num_pos = det_bbox.size(0)
    det_bbox_ori = torch.cat([det_bbox[:, 0::2] * mask_w, det_bbox[:, 1::2] * mask_h], dim=-1)[:, [0,2,1,3]]
    det_bbox_ori_c = center_size(det_bbox_ori)

    y_grid, x_grid = torch.meshgrid(torch.arange(mask_h), torch.arange(mask_w))
    cx, cy = torch.round(det_bbox_ori_c[:, 0]), torch.round(det_bbox_ori_c[:, 1])
    y_rel_coord = (y_grid.float().unsqueeze(0).repeat(num_pos, 1, 1) - cy.view(-1, 1, 1)) ** 2
    x_rel_coord = (x_grid.float().unsqueeze(0).repeat(num_pos, 1, 1) - cx.view(-1, 1, 1)) ** 2

    # build 2D Normal distribution
    sigma_scales = torch.ones(num_pos, device=det_bbox.device)
    small_obj = det_bbox_ori_c[:, 2] * det_bbox_ori_c[:, 3] / mask_h / mask_w < 0.1
    sigma_scales[small_obj] = 0.5 * sigma_scale
    sigma_xy = det_bbox_ori_c[:, 2:] / sigma_scales.reshape(-1, 1)
    sigma_xy = torch.clamp(sigma_xy, min=1)
    rel_coord_gauss = torch.exp(-0.5 * ((x_rel_coord / sigma_xy[:, 0].reshape(-1, 1, 1)) ** 2
                                        + (y_rel_coord / sigma_xy[:, 1].reshape(-1, 1, 1)) ** 2))

    return rel_coord_gauss


def generate_rel_coord(det_bbox, fpn_levels, sizes_of_interest, h, w):
    '''
    :param det_bbox: pos bboxes with normalization ==> [x1, y1, x2, y2], whose values in [0, 1]
    :param h: height of pred_mask
    :param w: weight of pred_mask
    :return: rel_coord ==> [num_pos, mask_h, mask_w, 2]
    '''

    shifts_x = torch.arange(0, w,  dtype=torch.float32, device=det_bbox.device)
    shifts_y = torch.arange(0, h,  dtype=torch.float32, device=det_bbox.device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + 0.5

    instance_locations = center_size(det_bbox)[:, :2] * torch.tensor([w, h]).view(1, -1)
    relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
    relative_coords = relative_coords.permute(0, 2, 1).float()
    soi = sizes_of_interest[fpn_levels]
    relative_coords = relative_coords / soi.reshape(-1, 1, 1).float()

    return relative_coords.reshape(-1, 2, h, w)


def plot_protos(protos, pred_masks, img_meta, num):
    if protos.size(1) == 9:
        protos = torch.cat([protos, protos[:, -1, :, :].unsqueeze(1)], dim=1)
    elif protos.size(1) == 8:
        protos = torch.cat([protos, pred_masks, pred_masks], dim=1)
    proto_data = protos.squeeze(0)
    num_per_row = int(proto_data.size(0) / 2)
    proto_data_list = []
    for r in range(2):
        proto_data_list.append(
            torch.cat([proto_data[i, :, :] * 5 for i in range(num_per_row * r, num_per_row * (r + 1))], dim=-1))

    img = torch.cat(proto_data_list, dim=0)
    img = img / img.max()
    plt.imshow(img.cpu().detach().numpy())
    plt.title([img_meta['video_id'], img_meta['frame_id'], 'protos'])
    plt.savefig(''.join(['results/results_0306/out_protos/',
                         str((img_meta['video_id'], img_meta['frame_id'])),
                         str(num), '.png']))


def generate_mask(proto_data, mask_coeff, bbox=None, fpn_levels=None):
    '''
    :param proto_data: [h, w, 32] or [n, h, w, 32]
    :param mask_coeff: [n, 32]
    :param bbox: [n, 4]
    :return:
    '''

    dim_proto = proto_data.dim()

    # get masks
    if cfg.use_sipmask:

        mask_coeff = cfg.mask_proto_coeff_activation(mask_coeff)
        pred_masks00 = cfg.mask_proto_mask_activation(generate_single_mask(proto_data,  mask_coeff[:, :cfg.mask_dim]))
        pred_masks01 = cfg.mask_proto_mask_activation(generate_single_mask(proto_data, mask_coeff[:, cfg.mask_dim:cfg.mask_dim*2]))
        pred_masks10 = cfg.mask_proto_mask_activation(generate_single_mask(proto_data, mask_coeff[:, cfg.mask_dim*2:cfg.mask_dim*3]))
        pred_masks11 = cfg.mask_proto_mask_activation(generate_single_mask(proto_data, mask_coeff[:, cfg.mask_dim*3:]))
        assert bbox is not None
        pred_masks = crop_sipmask(pred_masks00, pred_masks01, pred_masks10, pred_masks11, bbox)

    elif cfg.mask_proto_with_levels:
        mask_coeff = cfg.mask_proto_coeff_activation(mask_coeff)

        assert fpn_levels is not None
        n = mask_coeff.size(0)
        mask_coeff_expand = torch.zeros([n, 6, cfg.mask_dim], device=mask_coeff.device)
        mask_coeff_expand[:, fpn_levels.tolist()] = mask_coeff[:, :cfg.mask_dim]
        mask_coeff_expand[:, -1] = mask_coeff[:, -cfg.mask_dim:]
        pred_masks = generate_single_mask(proto_data, mask_coeff_expand.reshape(n, -1))
        pred_masks = cfg.mask_proto_mask_activation(pred_masks)
        if bbox is not None:
            pred_masks = crop(pred_masks, bbox)

    else:

        if not cfg.mask_proto_coeff_occlusion:
            mask_coeff = cfg.mask_proto_coeff_activation(mask_coeff)
            pred_masks = generate_single_mask(proto_data, mask_coeff)

        else:
            mask_coeff = cfg.mask_proto_coeff_activation(mask_coeff)
            # background
            pred_masks_b = generate_single_mask(proto_data, mask_coeff[:, :cfg.mask_dim])
            # target objects
            pred_masks_t = generate_single_mask(proto_data, mask_coeff[:, cfg.mask_dim:cfg.mask_dim*2])
            # parts from overlapped objects
            pred_masks_o = generate_single_mask(proto_data, mask_coeff[:, cfg.mask_dim*2:])
            pred_masks = torch.stack([pred_masks_b, pred_masks_t, pred_masks_o], dim=-2)    # [h, w, 3, n]
        pred_masks = cfg.mask_proto_mask_activation(pred_masks)
        if bbox is not None:
            pred_masks = crop(pred_masks, bbox)                                          # [h, w, n] or [h, w, 3, n]

    if dim_proto == 3:
        pred_masks = pred_masks.permute(2, 0, 1).contiguous()                               # [n, h, w]
    else:
        pred_masks = pred_masks.permute(3, 0, 1, 2).contiguous()                            # [n, h, w, 3]

    return pred_masks


def generate_single_mask(proto_data, mask_coeff):
    dim_proto = proto_data.dim()
    if dim_proto == 3:
        pred_masks = proto_data @ mask_coeff.t()
    elif dim_proto == 4:
        # to generate masks for all objects in a video clip
        pred_masks = (proto_data * mask_coeff.unsqueeze(1).unsqueeze(2)).sum(dim=-1)
        pred_masks = pred_masks.permute(1, 2, 0).contiguous()
    else:
        pred_masks = None
        print('please input the proto_data with size [h, w, c] or [n, h, w, c]')

    return pred_masks


def mask_iou(mask_a, mask_b):
    '''
    :param mask_a: [c, A, h, w]
    :param mask_b: [c, B, h, w]
    :return:
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

