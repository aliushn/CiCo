import torch
import torch.nn as nn
from datasets import cfg
from mmcv.ops import roi_align
from layers.utils import sanitize_coordinates_hw


class TemporalNet(nn.Module):
    def __init__(self, corr_channels, mask_proto_n=32):

        super().__init__()
        self.conv1 = nn.Conv2d(corr_channels, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d((7, 7), stride=1)
        self.fc = nn.Linear(1024, 4)

        if cfg.maskshift_loss:
            if cfg.use_sipmask:
                self.fc_coeff = nn.Linear(1024, mask_proto_n * cfg.sipmask_head)
            else:
                self.fc_coeff = nn.Linear(1024, mask_proto_n)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x_reg = self.fc(x)

        if cfg.maskshift_loss:
            x_coeff = self.fc_coeff(x)

            return x_reg, x_coeff

        else:
            return x_reg


def bbox_feat_extractor(feature_maps, boxes_w_norm, h, w, pool_size):
    """
        feature_maps: size:1*C*h*w
        boxes_w_norm: Mx5 float box with (x1, y1, x2, y2) **without normalization**
    """
    # Currently only supports batch_size 1
    # from [0, 1] to [0, h or w]
    boxes = sanitize_coordinates_hw(boxes_w_norm, h, w)

    # Crop and Resize
    # Result: [num_boxes, pool_height, pool_width, channels]
    box_ind = torch.zeros(boxes.size(0))  # index of bbox in batch
    if boxes.is_cuda:
        box_ind = box_ind.cuda()

    # CropAndResizeFunction needs batch dimension
    if len(feature_maps.size()) == 3:
        feature_maps = feature_maps.unsqueeze(0)

    # make crops:
    rois = torch.cat([box_ind.unsqueeze(1), boxes], dim=1)
    pooled_features = roi_align(feature_maps, rois, pool_size)

    return pooled_features

