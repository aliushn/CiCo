# -*- coding: utf-8 -*-
import torch
from .box_utils import center_size, point_form, crop
import torch.distributions as dist
from datasets import cfg
import matplotlib.pyplot as plt
import os
from spatial_correlation_sampler import spatial_correlation_sample
import torch.nn.functional as F


def correlate_operator(x1, x2, patch_size=11, kernel_size=1, dilation_patch=1):
    """
    :param x1: features 1
    :param x2: features 2
    :param patch_size: the size of whole patch is used to calculate the correlation
    :return:
    """

    # Output sizes oH and oW are no longer dependant of patch size, but only of kernel size and padding
    # patch_size is now the whole patch, and not only the radii.
    # stride1 is now stride and stride2 is dilation_patch, which behave like dilated convolutions
    # equivalent max_displacement is then dilation_patch * (patch_size - 1) / 2.
    # to get the right parameters for FlowNetC, you would have
    out_corr = spatial_correlation_sample(x1,
                                          x2,
                                          kernel_size=kernel_size,
                                          patch_size=patch_size,
                                          stride=1,
                                          padding=int((kernel_size - 1)/2),
                                          dilation_patch=dilation_patch)
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph*pw, h, w) / x1.size(1)
    return F.leaky_relu_(out_corr, 0.1)


def split_bbox(bbox, idx_bbox_ori, nums_bbox_per_layer):
    num_layers = len(nums_bbox_per_layer)
    num_bboxes = len(idx_bbox_ori)
    for i in range(1, num_layers):
        nums_bbox_per_layer[i] = nums_bbox_per_layer[i-1] + nums_bbox_per_layer[i]

    split_bboxes = [[] for _ in range(num_layers)]
    split_bboxes_idx = [[] for _ in range(num_layers)]
    for i in range(num_bboxes):
        for j in range(num_layers):
            if idx_bbox_ori[i] < nums_bbox_per_layer[j]:
                split_bboxes[j].append(bbox[i].unsqueeze(0))
                split_bboxes_idx[j].append(i)
                break

    for j in range(num_layers):
        if len(split_bboxes[j]) > 0:
            split_bboxes[j] = torch.cat(split_bboxes[j])
        if j > 0:
            split_bboxes_idx[0] += split_bboxes_idx[j]

    return split_bboxes, split_bboxes_idx[0]


def generate_track_gaussian(track_data, boxes):
    h, w, c = track_data.size()
    c_boxes = center_size(boxes)
    # only use the center region to evaluate multi-vairants Gaussian distribution
    c_boxes[:, 2:] = c_boxes[:, 2:] * 0.75
    crop_mask, cropped_track_data = crop(track_data.unsqueeze(-1).repeat(1, 1, 1, boxes.size(0)),
                                         point_form(c_boxes))
    crop_mask = crop_mask.reshape(h * w, c, -1)
    cropped_track_data = cropped_track_data.reshape(h * w, c, -1)
    mu = cropped_track_data.sum(0) / crop_mask.sum(0)  # [c, n_pos]
    var = torch.sum((cropped_track_data - mu.unsqueeze(0)) ** 2, dim=0)  # [c, n_pos]

    return mu.t(), var.t()


def compute_comp_scores(match_ll, bbox_scores, bbox_ious, mask_ious, label_delta, add_bbox_dummy=False,
                        bbox_dummy_iou=0, match_coeff=None, use_FEELVOS=False):
    # compute comprehensive matching score based on matchig likelihood,
    # bbox confidence, and ious
    if add_bbox_dummy:
        bbox_iou_dummy = torch.ones(bbox_ious.size(0), 1,
                                    device=torch.cuda.current_device()) * bbox_dummy_iou
        bbox_ious = torch.cat((bbox_iou_dummy, bbox_ious), dim=1)
        mask_ious = torch.cat((bbox_iou_dummy, mask_ious), dim=1)
        label_dummy = torch.ones(bbox_ious.size(0), 1,
                                 device=torch.cuda.current_device())
        label_delta = torch.cat((label_dummy, label_delta), dim=1)

    if match_coeff is None:
        return match_ll
    else:
        # match coeff needs to be length of 4
        assert (len(match_coeff) == 4)
        if use_FEELVOS:
            match_coeff[1] = match_coeff[1] + match_coeff[2]
            match_coeff[2] = 0
        return match_ll + match_coeff[0] * bbox_scores \
               + match_coeff[1] * mask_ious \
               + match_coeff[2] * bbox_ious \
               + match_coeff[3] * label_delta


def display_association_map(embedding_tensor, mu, cov, save_dir, video_id=0, frame_id=0):
    """
    Note that the dimension should be small for visualization
    :param embedding_tensor: [h, w, c], where c is the dim of embedding vectors
    :param mu: the mean of Gaussian, [n, c], where n is the number of Gaussian distributions
    :param cov: the covariance of Gaussian ([n, c]), where diag(cov[i])
    :return:
    """
    n, c = mu.size()
    h, w, _ = embedding_tensor.size()
    embedding_tensor = embedding_tensor.reshape(-1, c)
    prob = []
    for i in range(n):
        G = dist.MultivariateNormal(mu[i], torch.diag(cov[i]))
        prob_cur = G.log_prob(embedding_tensor).exp().view(h, w, 1).repeat(h, w, 3)
        color_idx = (i * 5) % len(cfg.COLORS)
        prob.append(prob_cur * (torch.tensor(cfg.COLORS[color_idx])).view(1, 1, -1))

    prob = torch.cat(prob, dim=0)
    plt.imshow(prob.cpu().numpy())
    plt.axis('off')
    plt.title(str((video_id, frame_id)))
    save_dir = ''.join([save_dir, '/out_ass_map/'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(''.join([save_dir, str(video_id), '_', str(frame_id), '.png']))













