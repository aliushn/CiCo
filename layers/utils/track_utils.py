# -*- coding: utf-8 -*-
import torch
from .box_utils import center_size, point_form, crop
import torch.distributions as dist
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F


def correlate_operator(x1, x2, patch_size=11, kernel_size=1, stride=1, dilation_patch=1):
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
    from spatial_correlation_sampler import spatial_correlation_sample
    out_corr = spatial_correlation_sample(x1,
                                          x2,
                                          kernel_size=kernel_size,
                                          patch_size=patch_size,
                                          stride=stride,
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
    :param track_data: [h, w, c] or [n, h, w, c]
    :param masks: [n, h, w]
    :param boxes: [n, 4]
    :return: the means and variances of multi-vairants Gaussian distribution for tracking
    '''

    dim = len(track_data.size())
    if dim == 3:
        track_data = track_data.unsqueeze(0)
    _, h, w, c = track_data.size()

    assert masks is not None or boxes is not None

    if masks is not None:
        # use pred masks as the fired area to crop track_data
        used_masks = F.interpolate(masks.unsqueeze(1).float(), (h, w),
                                   mode='bilinear', align_corners=False).gt(0.5).squeeze(1)
        if boxes is not None:
            # For small objects, masks are too small to embed distinct Gaussian distribution
            non_small_obj = used_masks.sum(dim=(1, 2)) > h*w*0.1
            if non_small_obj.sum() > 0:
                boxes_c = center_size(boxes)
                boxes_c[non_small_obj, 2:] = boxes_c[non_small_obj, 2:] * 1.2
                boxes = point_form(boxes_c)
            boxes = torch.clamp(boxes, min=0, max=1)
            crop_masks, _ = crop(used_masks.permute(1, 2, 0).contiguous(), boxes)
            used_masks = used_masks & crop_masks.permute(2, 0, 1).contiguous().bool()
        cropped_track_data = track_data * used_masks.unsqueeze(-1)   # [n_pos, h, w, 3]

    else:
        # only use the center region of pred boxes to evaluate multi-vairants Gaussian distribution
        c_boxes = center_size(boxes)
        c_boxes[:, 2:] = c_boxes[:, 2:] * 0.75
        if dim == 3:
            track_data = track_data.repeat(c_boxes.size(0), 1, 1, 1)
        used_masks, cropped_track_data = crop(track_data.permute(1, 2, 3, 0).contiguous(),
                                              point_form(c_boxes))   # [h, w, c, n_pos]
        used_masks = used_masks.permute(3, 0, 1, 2).contiguous()
        cropped_track_data = cropped_track_data.permute(3, 0, 1, 2).contiguous()

    n_pixels = used_masks.sum(dim=(1, 2)).view(-1, 1)
    mu = cropped_track_data.sum(dim=(1, 2)) / torch.clamp(n_pixels, min=1)  # [n_pos, c]
    var = torch.sum((cropped_track_data - mu[:, None, None, :]) ** 2, dim=(1, 2))  # [n_pos, c]

    return mu, var


def compute_kl_div(p_mu, p_var, q_mu=None, q_var=None):
    '''
    # kl_divergence for two Gaussian distributions, where k is the dim of variables
            # D_kl(p||q) = 0.5(log(torch.norm(Sigma_q) / torch.norm(Sigma_p)) - k
            #                  + (mu_p-mu_q)^T * torch.inverse(Sigma_q) * (mu_p-mu_q))
            #                  + torch.trace(torch.inverse(Sigma_q) * Sigma_p))
    :param p_mu:   [bs, np, c]
    :param p_var: [bs, np, c]
    :param q_mu:  [bs, nq, c]
    :param q:    [bs, nq, c]
    :return:
    '''

    if q_mu is None:
        q_mu, q_var = p_mu, p_var

    use_batch = True
    if p_mu.dim() == 2 and q_mu.dim() == 2:
        use_batch = False
        p_mu, p_var = p_mu.unsqueeze(0), p_var.unsqueeze(0)
        q_mu, q_var = q_mu.unsqueeze(0), q_var.unsqueeze(0)
    elif p_mu.dim() == 2:
        p_mu, p_var = p_mu.unsqueeze(0), p_var.unsqueeze(0)
    elif q_mu.dim() == 2:
        q_mu, q_var = q_mu.unsqueeze(0), q_var.unsqueeze(0)

    p_var = torch.clamp(p_var, min=1e-5)
    q_var = torch.clamp(q_var, min=1e-5)

    first_term = torch.sum(q_var.log(), dim=-1).unsqueeze(2) \
                 - torch.sum(p_var.log(), dim=-1).unsqueeze(1) - p_mu.size(2)     # [bs, nq, np]
    # ([bs, 1, np, c] - [bs, nq, 1, c]) * [bs, 1, np, c] => [bs, nq, np]
    second_term = torch.sum((p_mu.unsqueeze(1) - q_mu.unsqueeze(2)) ** 2 / p_var.unsqueeze(1), dim=-1)
    third_term = torch.div(p_var.unsqueeze(1), q_var.unsqueeze(2)).sum(dim=-1)                     # [bs, nq, np]
    kl_divergence = 0.5 * (first_term + second_term + third_term) + 1e-5                 # value in [0, +infinite]

    return kl_divergence if use_batch else kl_divergence.squeeze(0)


def compute_comp_scores(match_ll, bbox_scores, bbox_ious, mask_ious, label_delta, small_objects=None,
                        add_bbox_dummy=False, bbox_dummy_iou=0, match_coeff=None):
    # compute comprehensive matching score based on matchig likelihood,
    # bbox confidence, and ious
    if add_bbox_dummy:
        bbox_iou_dummy = torch.ones(bbox_ious.size(0), 1,
                                    device=torch.cuda.current_device()) * bbox_dummy_iou
        if small_objects is not None:
            bbox_iou_dummy[small_objects] *= 0.5
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
        # color_idx = (i * 5) % len(COLORS)
        # prob.append(prob_cur * (torch.tensor(COLORS[color_idx])).view(1, 1, -1))

    prob = torch.cat(prob, dim=0)
    plt.imshow(prob.cpu().numpy())
    plt.axis('off')
    plt.title(str((video_id, frame_id)))
    save_dir = ''.join([save_dir, '/out_ass_map/'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(''.join([save_dir, str(video_id), '_', str(frame_id), '.png']))













