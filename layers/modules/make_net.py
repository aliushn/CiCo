import torch.nn as nn
from layers.utils.interpolate import InterpolateModule


def make_net(in_channels, conf, use_3D=False, norm_type=None, include_last_relu=True):
    """
    A helper function to take a config setting and turn it into a network.
    Used by protonet and extrahead. Returns (network, out_channels)
    """

    def make_layer(layer_cfg, use_3D=False):
        nonlocal in_channels

        # Possible patterns:
        # ( 256, 3, padding) -> conv
        # ( 256,-2, padding) -> deconv
        # (None,-2, {}) -> bilinear interpolate
        # ('cat',[],{}) -> concat the subnetworks in the list
        #
        # You know it would have probably been simpler just to adopt a 'c' 'd' 'u' naming scheme.
        # Whatever, it's too late now.
        if isinstance(layer_cfg[0], str):
            layer_name = layer_cfg[0]

            if layer_name == 'cat':
                nets = [make_net(in_channels, x) for x in layer_cfg[1]]
                layer = [Concat([net[0] for net in nets], layer_cfg[2])]
                num_channels = sum([net[1] for net in nets])
        else:
            num_channels = layer_cfg[0]
            kernel_size = layer_cfg[1]

            if isinstance(kernel_size, tuple) or (isinstance(kernel_size, int) and kernel_size > 0):
                conv = nn.Conv3d if use_3D else nn.Conv2d
                layer1 = conv(in_channels, num_channels, kernel_size, padding=layer_cfg[2])
                if norm_type is None:
                    layer = [layer1]
                else:
                    if norm_type == 'batch_norm':
                        Norm = nn.BatchNorm3d(num_channels) if use_3D else nn.BatchNorm2d(num_channels)
                    elif norm_type == 'layer_norm':
                        Norm = nn.LayerNorm(num_channels)
                    layer = [layer1, Norm]

            else:
                if num_channels is None:
                    up_mode = 'trilinear' if use_3D else 'bilinear'
                    kernel_size = (1, -kernel_size, -kernel_size) if use_3D else (-kernel_size, -kernel_size)
                    layer = [InterpolateModule(scale_factor=kernel_size, mode=up_mode, align_corners=True)]
                else:
                    layer = [nn.ConvTranspose2d(in_channels, num_channels, -kernel_size, padding=layer_cfg[2])]

        in_channels = num_channels if num_channels is not None else in_channels

        # Don't return a ReLU layer if we're doing an upsample. This probably doesn't affect anything
        # output-wise, but there's no need to go through a ReLU here.
        # Commented out for backwards compatibility with previous models
        # if num_channels is None:
        #     return [layer]
        # else:
        return layer + [nn.ReLU(inplace=True)]

    # Use sum to concat together all the component layer lists
    net = sum([make_layer(x, use_3D) for x in conf], [])
    if not include_last_relu:
        if norm_type is not None:
            net = net[:-2] if conf[-1][1] > 0 else net[:-1]
        else:
            net = net[:-1]

    return nn.Sequential(*net), in_channels


