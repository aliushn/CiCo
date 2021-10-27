import torch
import torch.nn as nn
import torch.nn.functional as F


class InterclipsClass(nn.Module):
    def __init__(self, dim_cla_fea, num_clip, num_clip_frames, num_classes):
        super().__init__()

        in_channels = dim_cla_fea * num_clip * num_clip_frames
        self.fc1 = nn.Linear(in_channels, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, cla_fea):
        x = F.relu(torch.flatten(cla_fea, start_dim=1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x