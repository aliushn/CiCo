import torch
import torch.nn as nn
from datasets import *


class NetLoss(nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.
    """

    def __init__(self, net, criterion):
        super().__init__()

        self.net = net
        self.criterion = criterion

    def forward(self, images, gt_bboxes, gt_labels, gt_masks, gt_ids, num_crowds, img_metas):
        preds = self.net(images)
        losses = self.criterion(self.net, preds, gt_bboxes, gt_labels, gt_masks, gt_ids, num_crowds)
        return losses


class CustomDataParallel(nn.DataParallel):
    """
    This is a custom version of DataParallel that works better with our training data.
    It should also be faster than the general case.
    """

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in device_ids]
        if cfg.data_type == 'coco':
            splits = prepare_data_coco(inputs[0], devices)
        elif cfg.data_type == 'vid':
            splits = prepare_data_vid(inputs[0], devices)
        else:
            splits = prepare_data_vis(inputs[0], devices)

        return [[split[device_idx] for split in splits] for device_idx in range(len(devices))], \
               [kwargs] * len(devices)

    def gather(self, outputs, output_device):
        out = {}

        for k in outputs[0]:
            out[k] = torch.stack([output[k].to(output_device) for output in outputs])

        return out