# https://github.com/sevakon/efficientdet/blob/master/model/bifpn.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFPN(nn.Module):
    """
    BiFPN block.
    Depending on its order, it either accepts
    seven feature maps (if this block is the first block in FPN) or
    otherwise five feature maps from the output of the previous BiFPN block
    """

    EPS: float = 1e-04
    REDUCTION_RATIO: int = 2

    def __init__(self, n_channels):
        super(BiFPN, self).__init__()

        self.conv_4_td = DWSConv(n_channels, n_channels, relu=False)
        self.conv_5_td = DWSConv(n_channels, n_channels, relu=False)
        self.conv_6_td = DWSConv(n_channels, n_channels, relu=False)

        self.weights_4_td = nn.Parameter(torch.ones(2))
        self.weights_5_td = nn.Parameter(torch.ones(2))
        self.weights_6_td = nn.Parameter(torch.ones(2))

        self.conv_3_out = DWSConv(n_channels, n_channels, relu=False)
        self.conv_4_out = DWSConv(n_channels, n_channels, relu=False)
        self.conv_5_out = DWSConv(n_channels, n_channels, relu=False)
        self.conv_6_out = DWSConv(n_channels, n_channels, relu=False)
        self.conv_7_out = DWSConv(n_channels, n_channels, relu=False)

        self.weights_3_out = nn.Parameter(torch.ones(2))
        self.weights_4_out = nn.Parameter(torch.ones(3))
        self.weights_5_out = nn.Parameter(torch.ones(3))
        self.weights_6_out = nn.Parameter(torch.ones(3))
        self.weights_7_out = nn.Parameter(torch.ones(2))

        self.upsample = lambda x: F.interpolate(x, scale_factor=self.REDUCTION_RATIO)
        self.downsample = MaxPool2dSamePad(self.REDUCTION_RATIO + 1, self.REDUCTION_RATIO)

        self.act = MemoryEfficientSwish()

    def forward(self, features):
        if len(features) == 5:
            p_3, p_4, p_5, p_6, p_7 = features
            p_4_2, p_5_2 = None, None
        else:
            p_3, p_4, p_4_2, p_5, p_5_2, p_6, p_7 = features

        # Top Down Path
        p_6_td = self.conv_6_td(
            self._fuse_features(
                weights=self.weights_6_td,
                features=[p_6, self.upsample(p_7)]
            )
        )
        p_5_td = self.conv_5_td(
            self._fuse_features(
                weights=self.weights_5_td,
                features=[p_5, self.upsample(p_6_td)]
            )
        )
        p_4_td = self.conv_4_td(
            self._fuse_features(
                weights=self.weights_4_td,
                features=[p_4, self.upsample(p_5_td)]
            )
        )

        p_4_in = p_4 if p_4_2 is None else p_4_2
        p_5_in = p_5 if p_5_2 is None else p_5_2

        # Out
        p_3_out = self.conv_3_out(
            self._fuse_features(
                weights=self.weights_3_out,
                features=[p_3, self.upsample(p_4_td)]
            )
        )
        p_4_out = self.conv_4_out(
            self._fuse_features(
                weights=self.weights_4_out,
                features=[p_4_in, p_4_td, self.downsample(p_3_out)]
            )
        )
        p_5_out = self.conv_5_out(
            self._fuse_features(
                weights=self.weights_5_out,
                features=[p_5_in, p_5_td, self.downsample(p_4_out)]
            )
        )
        p_6_out = self.conv_6_out(
            self._fuse_features(
                weights=self.weights_6_out,
                features=[p_6, p_6_td, self.downsample(p_5_out)]
            )
        )
        p_7_out = self.conv_7_out(
            self._fuse_features(
                weights=self.weights_7_out,
                features=[p_7, self.downsample(p_6_out)]
            )
        )

        return [p_3_out, p_4_out, p_5_out, p_6_out, p_7_out]

    def _fuse_features(self, weights, features):
        weights = F.relu(weights)
        num = sum([w * f for w, f in zip(weights, features)])
        det = sum(weights) + self.EPS
        x = self.act(num / det)
        return x


class ConvModule(nn.Module):
    """ Regular Convolution with BatchNorm """
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class DWSConv(nn.Module):
    """ DepthWise Separable Convolution with BatchNorm and ReLU activation """
    def __init__(self, in_channels, out_channels, bath_norm=True, relu=True, bias=False):
        super(DWSConv, self).__init__()
        self.conv_dw = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                 padding=1, groups=in_channels, bias=False)
        self.conv_pw = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                 padding=0, bias=bias)

        self.bn = None if not bath_norm else \
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        self.act = None if not relu else Swish()

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class MaxPool2dSamePad(nn.MaxPool2d):
    """ TensorFlow-like 2D Max Pooling with same padding """

    PAD_VALUE: float = -float('inf')

    def __init__(self, kernel_size: int, stride=1, padding=0,
                 dilation=1, ceil_mode=False, count_include_pad=True):
        assert padding == 0, 'Padding in MaxPool2d Same Padding should be zero'

        kernel_size = (kernel_size, kernel_size)
        stride = (stride, stride)
        padding = (padding, padding)
        dilation = (dilation, dilation)

        super(MaxPool2dSamePad, self).__init__(kernel_size, stride, padding,
                                               dilation, ceil_mode, count_include_pad)

    def forward(self, x):
        h, w = x.size()[-2:]

        pad_h = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] + \
                (self.kernel_size[0] - 1) * self.dilation[0] + 1 - h
        pad_w = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] + \
                (self.kernel_size[1] - 1) * self.dilation[1] + 1 - w

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                          pad_h - pad_h // 2], value=self.PAD_VALUE)

        x = F.max_pool2d(x, self.kernel_size, self.stride,
                         self.padding, self.dilation, self.ceil_mode)
        return x


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

