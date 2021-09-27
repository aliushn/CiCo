import torch.nn as nn
import torch.nn.functional as F


class NonLocalBlack(nn.Module):
    '''
    NonLocalBlack aims to simultaneously explore spatial and temporal information from features.
    Please read more from https://arxiv.org/pdf/1711.07971.pdf
    '''
    def __init__(self, in_channels):
        super(NonLocalBlack, self).__init__()

        self.theta = nn.Conv3d(in_channels, in_channels//2, kernel_size=(1,1,1))
        self.phi = nn.Conv3d(in_channels, in_channels//2, kernel_size=(1,1,1))
        self.g = nn.Conv3d(in_channels, in_channels//2, kernel_size=(1,1,1))
        self.k = nn.Conv3d(in_channels//2, in_channels, kernel_size=(1,1,1))

    def forward(self, x):
        bs, C_in, T, H, W = x.size()
        v1 = self.theta(x).reshape(bs, -1, T*H*W, 1)
        v2 = self.phi(x).reshape(bs, -1, 1, T*H*W)
        weights = F.softmax(v1 * v2, dim=-1)
        y = (self.g(x).reshape(bs, -1, 1, T*H*W) * weights).sum(-1)
        z = x + self.k(y.reshape(bs, -1, T, H, W))

        return z


