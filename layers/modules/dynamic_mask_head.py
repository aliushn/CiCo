import torch
from torch.nn import functional as F
from torch import nn
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
    def __init__(self, cfg):
        super(DynamicMaskHead, self).__init__()
        self.cfg = cfg
        self.num_layers = cfg.MODEL.MASK_HEADS.DYNAMIC_MASK_HEAD_LAYERS
        self.channels = cfg.MODEL.MASK_HEADS.MASK_DIM
        self.in_channels = cfg.MODEL.MASK_HEADS.MASK_DIM
        self.disable_rel_coords = cfg.MODEL.MASK_HEADS.DISABLE_REL_COORDS

        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 1) * self.channels)
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
        assert features.dim() == 4 or features.dim() == 5
        conv = F.conv2d if features.dim() == 4 else F.conv3d
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = conv(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def __call__(self, mask_feats, mask_head_params, det_bbox, fpn_levels):
        '''
        :param mask_feats: [t, 8, h, w]
        :param mask_head_params: [n, 169]
        :param det_bbox:[n, 4]
        :return:
        '''
        n_inst, _ = mask_head_params.size()
        H, W = mask_feats.size()[-2:]
        weights, biases = parse_dynamic_params(mask_head_params, self.channels,
                                               self.weight_nums, self.bias_nums)
        T = mask_feats.size(0)
        if T > 1:
            weights = [weight.unsqueeze(-1) for weight in weights]
            mask_feats = mask_feats.permute(1, 0, 2, 3).contiguous().unsqueeze(0).repeat(n_inst, 1, 1, 1, 1)
        else:
            mask_feats = mask_feats.repeat(n_inst, 1, 1, 1)

        if not self.disable_rel_coords:
            # TODO: Circumscribed boxes or individual boxes for frames of the clip?
            relative_coords = generate_rel_coord_gauss(det_bbox, H, W).unsqueeze(1).detach()
            if mask_feats.dim() == 5:
                relative_coords = relative_coords.unsqueeze(-3).repeat(1, 1, T, 1, 1)
            # relative_coords = generate_rel_coord(det_bbox, fpn_levels.tolist(), self.sizes_of_interest, H, W).detach()
            # print(mask_feats.min(), mask_feats.max(), relative_coords.max())
            mask_head_inputs = torch.cat([relative_coords, mask_feats], dim=1)
        else:
            mask_head_inputs = mask_feats

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W).contiguous() if mask_head_inputs.dim() == 4 \
            else mask_head_inputs.reshape(1, -1, T, H, W).contiguous()
        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst).squeeze(0)
        mask_logits = mask_logits if mask_logits.dim() == 4 else mask_logits.unsqueeze(1)

        return mask_logits.sigmoid()
