import torch.nn as nn
from ..utils.track_utils import correlate_operator
from mmcv.ops import DeformConv2d


class TemporalFeatureCalibration(nn.Module):
    def __init__(self, idx):

        super().__init__()
        self.corr_patch_size = [11, 11, 5, 3, 1]
        self.conv1 = nn.Conv2d(self.corr_patch_size[idx], 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 18, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_adaption = DeformConv2d(256,
                                          256,
                                          kernel_size=3,
                                          padding=1,
                                          deform_groups=1)

    def forward(self, x_ref, x_next, idx):
        corr = correlate_operator(x_next, x_ref, patch_size=self.corr_patch_size[idx], kernel_size=3)
        x = self.relu(self.conv1(corr))
        x = self.relu(self.conv2(x))
        offset = self.relu(self.conv3(x))  # [H, W, 18]
        x_ref_calibrated = self.relu(self.conv_adaption(x_ref, offset))

        return 0.5 * (x_next + x_ref_calibrated)


