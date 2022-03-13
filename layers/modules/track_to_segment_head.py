import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import roi_align
from layers.utils import center_size, point_form, decode, sanitize_coordinates_hw
from ..utils.visualization_utils import display_correlation_map_patch, display_box_shift


class TemporalFusionNet(nn.Module):
    '''
    Temporal fusion net of STMask https://github.com/MinghanLi/STMask
    The input is the concatenated features of the reference frame, the target frame
    and their correlation maps.
    The outputs includes the offsets of bounding boxes and mask coefficients between
    the reference frame and the target frame.
    '''
    def __init__(self, args, in_channels=256, n_mask_protos=32):

        super().__init__()
        self.maskshift_loss = args.TRAIN_MASKSHIFT
        self.pooling_size = args.POOLING_SIZE
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(1024, 4)
        self.pool = nn.AvgPool2d((self.pooling_size, self.pooling_size), stride=1)

        if self.maskshift_loss:
            self.fc_coeff = nn.Linear(1024, n_mask_protos)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x_reg = self.fc(x)

        if self.maskshift_loss:
            x_coeff = self.fc_coeff(x)
            return x_reg, torch.tanh(x_coeff)
        else:
            return x_reg, None


def correlate_operator(x1, x2, patch_size=11, kernel_size=1, stride=1, dilation_patch=1):
    """
    compute the correlation maps between two frames
    :param x1: features of the reference frame
    :param x2: features of the target frame
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


def bbox_feat_extractor(feature_maps, boxes_w_norm, h, w, pool_size):
    """
    based on roi_align operation to extract features in bounding boxes
    Args:
        - feature_maps: size:1*C*h*w
        - boxes_w_norm: Mx5 float box with (x1, y1, x2, y2) **without normalization**
    """
    # Currently only supports batch_size 1, rescale from [0, 1] to [0, h or w]
    # TODO: Double check [0, 1] or [0, h/w] in different mmcv versions
    boxes = sanitize_coordinates_hw(boxes_w_norm, h, w)

    # Crop and Resize
    # Result: [num_boxes, pool_height, pool_width, channels]
    box_ind = torch.zeros(boxes.size(0))  # index of bbox in batch
    if boxes.is_cuda:
        box_ind = box_ind.cuda()

    # CropAndResizeFunction needs batch dimension
    if feature_maps.dim() == 3:
        feature_maps = feature_maps.unsqueeze(0)

    # make crops:
    rois = torch.cat([box_ind.unsqueeze(1), boxes], dim=1)
    pooled_features = roi_align(feature_maps, rois, pool_size)

    return pooled_features


class T2S_Head(nn.Module):
    '''
    track-to-segment module of STMask https://github.com/MinghanLi/STMask
    '''
    def __init__(self, args, in_channels=256, n_mask_protos=32):
        super(T2S_Head, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self.n_mask_protos = n_mask_protos
        self.TemporalFusionNet = TemporalFusionNet(args, in_channels, n_mask_protos)

    def forward(self, feat_ref, feat_tar, boxes_ref):
        assert feat_ref.dim() == feat_tar.dim()
        feat_ref = feat_ref.unsqueeze(0) if feat_ref.dim() == 3 else feat_ref
        feat_tar = feat_tar.unsqueeze(0) if feat_tar.dim() == 3 else feat_tar
        feat_h, feat_w = feat_ref.size()[-2:]

        # Compute the features correlation between two adjacent frames, inspired by Optical Flow
        corr = correlate_operator(feat_ref, feat_tar,
                                  patch_size=self.args.CORRELATION_PATCH_SIZE,
                                  kernel_size=1)
        concat_feat = torch.cat((feat_ref, corr, feat_tar), dim=1)

        # Extract features on the predicted bbox, and align cropped features of bounding boxes as 7*7
        cur_gt_boxes_ref_c = center_size(boxes_ref)
        cur_gt_boxes_ref_c[:, 2:] *= 1.2
        cur_gt_boxes_ref = torch.clamp(point_form(cur_gt_boxes_ref_c), min=0, max=1)
        boxes_feats = bbox_feat_extractor(concat_feat, cur_gt_boxes_ref, feat_h, feat_w, 7)

        # To obtain the offsets of bounding boxes and mask coefficients from the reference frame to the target frame
        boxes_off, mask_coeff_tar = self.TemporalFusionNet(boxes_feats)

        display = False
        if display:
            # display_correlation_map(corr)
            display_correlation_map_patch(boxes_feats[:, 256:377])
            display_box_shift(boxes_ref, decode(boxes_off, center_size(boxes_ref)))

        return boxes_off, mask_coeff_tar


