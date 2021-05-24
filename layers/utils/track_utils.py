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


def generate_track_gaussian(track_data, masks=None, boxes=None):
    '''
    :param track_data: [h, w, c]
    :param masks: [n, h, w]
    :param boxes:
    :return:
    '''

    h, w, c = track_data.size()
    if masks is not None:
        downsampled_masks = F.interpolate(masks.unsqueeze(1).float(), (h, w),
                                          mode='bilinear', align_corners=False).gt(0.5)
        track_data = track_data.permute(2, 0, 1).contiguous().unsqueeze(0)
        cropped_track_data = track_data * downsampled_masks
        n_pixels = downsampled_masks.sum(dim=(2, 3))
        mu = cropped_track_data.sum(dim=(2, 3)) / torch.clamp(n_pixels, min=1)  # [n_pos, c]
        var = torch.sum((cropped_track_data - mu[:, :, None, None]) ** 2, dim=(2, 3))  # [n_pos, c]

    elif boxes is not None:
        c_boxes = center_size(boxes)
        # only use the center region to evaluate multi-vairants Gaussian distribution
        c_boxes[:, 2:] = c_boxes[:, 2:] * 0.75
        crop_mask, cropped_track_data = crop(track_data.unsqueeze(-1).repeat(1, 1, 1, boxes.size(0)),
                                             point_form(c_boxes))
        crop_mask = crop_mask.reshape(h * w, c, -1)
        cropped_track_data = cropped_track_data.reshape(h * w, c, -1)
        mu = cropped_track_data.sum(0) / crop_mask.sum(0)  # [c, n_pos]
        var = torch.sum((cropped_track_data - mu.unsqueeze(0)) ** 2, dim=0)  # [c, n_pos]
        mu, var = mu.t(), var.t()

    else:
        mu, var = None, None

    return mu, var


def compute_kl_div(p_mu, p_var, q_mu, q_var):
    '''
    # kl_divergence for two Gaussian distributions, where k is the dim of variables
            # D_kl(p||q) = 0.5(log(torch.norm(Sigma_q) / torch.norm(Sigma_p)) - k
            #                  + (mu_p-mu_q)^T * torch.inverse(Sigma_q) * (mu_p-mu_q))
            #                  + torch.trace(torch.inverse(Sigma_q) * Sigma_p))
    :param p_mu: [np, c]
    :param p_var: [np, c]
    :param q_mu: [nq, c]
    :param q: [nq, c]
    :return:
    '''

    first_term = torch.sum(q_var.log(), dim=-1).unsqueeze(0) \
                 - torch.sum(p_var.log(), dim=-1).unsqueeze(1) - p_mu.size(1)  # [nq, np]
    # ([1, np, c] - [nq, 1, c]) * [1, np, c] => [nq, np] sec_kj = \sum_{i=1}^C (mu_ij-mu_ik)^2 * sigma_i^{-1}
    second_term = torch.sum((p_mu.unsqueeze(0) - q_mu.unsqueeze(1)) ** 2 / p_var.unsqueeze(0),
                            dim=-1)
    third_term = torch.mm(1. / q_var, p_var.t())  # [nq, np]
    kl_divergence = 0.5 * (first_term + second_term + third_term)  # [0, +infinite]

    return kl_divergence


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
        return 2 * match_ll + match_coeff[0] * bbox_scores \
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













