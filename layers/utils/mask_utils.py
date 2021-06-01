# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .box_utils import crop, crop_sipmask
from datasets import cfg, activation_func


def generate_rel_coord(det_bbox, mask_h, mask_w, sigma_scale=2):
    '''
    :param det_box: the centers of pos bboxes ==> [cx, cy]
    :param mask_h: height of pred_mask
    :param mask_w: weight of pred_mask
    :return: rel_coord ==> [num_pos, mask_h, mask_w, 2]
    '''

    # generate relative coordinates
    num_pos = det_bbox.size(0)
    det_bbox_ori = det_bbox.new(num_pos, 4)
    det_bbox_ori[:, 0::2] = det_bbox[:, 0::2] * mask_w
    det_bbox_ori[:, 1::2] = det_bbox[:, 1::2] * mask_h
    x_range = torch.arange(mask_w)
    y_range = torch.arange(mask_h)
    y_grid, x_grid = torch.meshgrid(y_range, x_range)
    det_bbox_c = (det_bbox_ori[:, :2] + det_bbox_ori[:, 2:]) / 2
    cx, cy = torch.round(det_bbox_c[:, 0]), torch.round(det_bbox_c[:, 1])
    y_rel_coord = (y_grid.float().unsqueeze(0).repeat(num_pos, 1, 1) - cy.view(-1, 1, 1)) ** 2
    x_rel_coord = (x_grid.float().unsqueeze(0).repeat(num_pos, 1, 1) - cx.view(-1, 1, 1)) ** 2

    # build 2D Normal distribution
    det_bbox_wh = det_bbox_ori[:, 2:] - det_bbox_ori[:, :2]
    rel_coord = []
    for i in range(num_pos):
        if det_bbox_wh[i][0] * det_bbox_wh[i][1] / mask_h / mask_w < 0.1:
            sigma_scale = 0.5 * sigma_scale
        sigma_x, sigma_y = det_bbox_wh[i] / sigma_scale
        val = torch.exp(-0.5 * (x_rel_coord[i] / (sigma_x ** 2) + y_rel_coord[i] / (sigma_y ** 2)))
        rel_coord.append(val.unsqueeze(0))

    return torch.cat(rel_coord, dim=0)


def mask_head(protos, proto_coeff, num_mask_head, mask_dim=8, use_rela_coord=False, img_meta=None):
    """
    :param protos: [1, n, h, w]
    :param proto_coeff: reshape as weigths and bias
    :return: [1, 1, h, w]
    """

    # reshape proto_coef as weights and bias of filters
    if use_rela_coord:
        ch = mask_dim + 1
    else:
        ch = mask_dim
    ch2 = mask_dim * ch

    if num_mask_head == 1:
        weights1 = proto_coeff[:8].reshape(1, 8, 1, 1)
        bias1 = proto_coeff[-1].reshape(1)
        # FCN network for mask prediction
        pred_masks = F.conv2d(protos, weights1, bias1, stride=1, padding=0, dilation=1, groups=1)

    elif num_mask_head == 2:
        weights1 = proto_coeff[:ch2].reshape(mask_dim, ch, 1, 1)
        bias1 = proto_coeff[ch2:ch2 + mask_dim]
        weights2 = proto_coeff[ch2 + mask_dim:ch2 + 2 * mask_dim].reshape(1, mask_dim, 1, 1)
        bias2 = proto_coeff[-1].reshape(1)
        # FCN network for mask prediction
        protos1 = F.relu(F.conv2d(protos, weights1, bias1, stride=1, padding=0, dilation=1, groups=1))
        pred_masks = F.conv2d(protos1, weights2, bias2, stride=1, padding=0, dilation=1, groups=1)

        # plot_protos(protos, pred_masks, img_meta, num=1)
        # plot_protos(protos1, pred_masks, img_meta, num=2)

    elif num_mask_head == 3:
        weights1 = proto_coeff[:ch2].reshape(mask_dim, ch, 1, 1)
        bias1 = proto_coeff[ch2:ch2 + mask_dim]
        weights2 = proto_coeff[ch2 + mask_dim:ch2 + mask_dim + mask_dim**2].reshape(mask_dim, mask_dim, 1, 1)
        bias2 = proto_coeff[ch2 + mask_dim + mask_dim**2:ch2 + mask_dim*2 + mask_dim**2]
        weights3 = proto_coeff[ch2 + mask_dim*2 + mask_dim**2:ch2 + mask_dim*3 + mask_dim**2].reshape(1, mask_dim, 1, 1)
        bias3 = proto_coeff[-1].reshape(1)
        # FCN network for mask prediction
        protos1 = F.relu(F.conv2d(protos, weights1, bias1, stride=1, padding=0, dilation=1, groups=1))
        protos2 = F.relu(F.conv2d(protos1, weights2, bias2, stride=1, padding=0, dilation=1, groups=1))
        pred_masks = F.conv2d(protos2, weights3, bias3, stride=1, padding=0, dilation=1, groups=1)

    return pred_masks


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


def generate_mask(proto_data, mask_coeff, bbox=None):
    '''
    :param proto_data: [h, w, 32] or [n, h, w, 32]
    :param mask_coeff: [n, 32]
    :param bbox: [n, 4]
    :return:
    '''
    # get masks
    if cfg.use_sipmask:
        pred_masks00 = cfg.mask_proto_mask_activation(proto_data @ mask_coeff[:, :cfg.mask_proto_n].t())
        pred_masks01 = cfg.mask_proto_mask_activation(proto_data @ mask_coeff[:, cfg.mask_proto_n:cfg.mask_proto_n*2].t())
        pred_masks10 = cfg.mask_proto_mask_activation(proto_data @ mask_coeff[:, cfg.mask_proto_n*2:cfg.mask_proto_n*3].t())
        pred_masks11 = cfg.mask_proto_mask_activation(proto_data @ mask_coeff[:, cfg.mask_proto_n*3:].t())
        if bbox is not None:
            pred_masks = crop_sipmask(pred_masks00, pred_masks01, pred_masks10, pred_masks11, bbox)
    else:
        mask_coeff = cfg.mask_proto_coeff_activation(mask_coeff)
        if cfg.mask_proto_coeff_activation == activation_func.sigmoid:
            # hope to get mask coefficients with L0 sparsity
            mask_coeff = mask_coeff * 2 - 1
            mask_coeff = torch.clamp(mask_coeff, min=0)

        dim_proto = len(proto_data.size())
        if dim_proto == 3:
            pred_masks = proto_data @ mask_coeff.t()
        elif dim_proto == 4:
            # to generate masks for all objects in a video clip
            pred_masks = (proto_data * mask_coeff.unsqueeze(1).unsqueeze(2)).sum(dim=-1)
            pred_masks = pred_masks.permute(1, 2, 0).contiguous()
        else:
            print('please input the proto_data with size [h, w, c] or [n, h, w, c]')

        pred_masks = cfg.mask_proto_mask_activation(pred_masks)
        if bbox is not None:
            _, pred_masks = crop(pred_masks, bbox)  # [mask_h, mask_w, n]

    det_masks = pred_masks.permute(2, 0, 1).contiguous()  # [n_masks, h, w]

    return det_masks


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
