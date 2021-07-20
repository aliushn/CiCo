import torch
from torch.nn import functional as F
from torch import nn
from datasets import cfg
from ..utils import generate_rel_coord, generate_rel_coord_gauss


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(
        params, weight_nums + bias_nums, dim=1
    ))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits


class DynamicMaskHead(nn.Module):
    def __init__(self):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = cfg.dynamic_mask_head_layers
        self.channels = cfg.mask_dim
        self.in_channels = cfg.mask_dim
        self.disable_rel_coords = cfg.disable_rel_coords

        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1:
                weight_nums.append(self.channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        self.sizes_of_interest = torch.tensor([16, 32, 64, 128, 256])

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def __call__(self, mask_feats, mask_head_params, det_bbox, fpn_levels):
        '''
        :param mask_feats: [1, 8, h, w]
        :param mask_head_params: [n, 196]
        :return:
        '''
        n_inst, _ = mask_head_params.size()
        H, W = mask_feats.size()[-2:]
        # we hope the value of weights and bias in [-1, 1]
        mask_head_params = torch.tanh(mask_head_params)

        if not self.disable_rel_coords:
            # relative_coords = generate_rel_coord_gauss(det_bbox, H, W).unsqueeze(1).detach()
            relative_coords = generate_rel_coord(det_bbox, fpn_levels.tolist(), self.sizes_of_interest, H, W).detach()
            mask_head_inputs = torch.cat([relative_coords, mask_feats.repeat(n_inst, 1, 1, 1)], dim=1)
        else:
            mask_head_inputs = mask_feats.repeat(n_inst, 1, 1, 1)

        weights, biases = parse_dynamic_params(mask_head_params, self.channels,
                                               self.weight_nums, self.bias_nums)
        mask_logits = self.mask_heads_forward(mask_head_inputs.reshape(1, -1, H, W), weights, biases, n_inst)

        return cfg.mask_proto_mask_activation(mask_logits.squeeze(0))
