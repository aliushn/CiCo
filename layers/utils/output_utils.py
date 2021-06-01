""" Contains functions used to sanitize and prepare the output of Yolact. """
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import pycocotools.mask as mask_util
from matplotlib.patches import Polygon

from datasets import cfg, MEANS, STD, activation_func
from utils.augmentations import Resize
from utils import timer
import os
from .box_utils import crop, sanitize_coordinates, center_size
from .mask_utils import generate_mask


def postprocess(det_output, w, h, batch_idx=0, interpolation_mode='bilinear',
                visualize_lincomb=False, crop_masks=True, score_threshold=0):
    """
    Postprocesses the output of Yolact on testing mode into a format that makes sense,
    accounting for all the possible configuration settings.

    Args:
        - det_output: The lost of dicts that Detect outputs.
        - w: The real with of the image.
        - h: The real height of the image.
        - batch_idx: If you have multiple images for this batch, the image's index in the batch.
        - interpolation_mode: Can be 'nearest' | 'area' | 'bilinear' (see torch.nn.functional.interpolate)

    Returns 4 torch Tensors (in the following order):
        - classes [num_det]: The class idx for each detection.
        - scores  [num_det]: The confidence score for each detection.
        - boxes   [num_det, 4]: The bounding box for each detection in absolute point form.
        - masks   [num_det, h, w]: Full image masks for each detection.
    """

    dets = det_output[batch_idx]
    net = dets['net']
    dets = dets['detection']

    if dets is None:
        return [torch.Tensor()] * 4  # Warning, this is 4 copies of the same thing

    if score_threshold > 0:
        keep = dets['score'] > score_threshold

        for k in dets:
            if k != 'proto':
                dets[k] = dets[k][keep]

        if dets['score'].size(0) == 0:
            return [torch.Tensor()] * 4

    if dets['box'].size(0) == 0:
        return [torch.Tensor()] * 4

    # Actually extract everything from dets now
    classes = dets['class']
    boxes = dets['box']
    scores = dets['score']
    masks_coeff = dets['mask_coeff']

    if cfg.eval_mask_branch:
        # At this points masks is only the coefficients
        proto_data = dets['proto']

        # Test flag, do not upvote
        if cfg.mask_proto_debug:
            np.save('scripts/proto.npy', proto_data.cpu().numpy())

        if visualize_lincomb:
            display_lincomb(proto_data, masks_coeff)

        masks = generate_mask(proto_data, masks_coeff, boxes.view(-1, 4))
        # Scale masks up to the full image
        masks = F.interpolate(masks.unsqueeze(0), (h, w), mode=interpolation_mode, align_corners=False).squeeze(0)

        # Binarize the masks
        masks.gt_(0.5)

    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, cast=False)
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, cast=False)
    boxes = boxes.long()

    return classes, scores, boxes, masks


def postprocess_ytbvis(detection, pad_h, pad_w, img_meta, interpolation_mode='bilinear',
                       display_mask=False, visualize_lincomb=False, crop_masks=True, score_threshold=0,
                       img_ids=None, mask_det_file=None):
    """
    Postprocesses the output of Yolact on testing mode into a format that makes sense,
    accounting for all the possible configuration settings.

    Args:
        - det_output: The lost of dicts that Detect outputs.
        - w: The real with of the image.
        - h: The real height of the image.
        - batch_idx: If you have multiple images for this batch, the image's index in the batch.
        - interpolation_mode: Can be 'nearest' | 'area' | 'bilinear' (see torch.nn.functional.interpolate)

    Returns 4 torch Tensors (in the following order):
        - classes [num_det]: The class idx for each detection.
        - scores  [num_det]: The confidence score for each detection.
        - boxes   [num_det, 4]: The bounding box for each detection in absolute point form.
        - masks   [num_det, h, w]: Full image masks for each detection.
    """

    dets = {}
    for k, v in detection.items():
        dets[k] = v.clone()

    ori_h, ori_w = img_meta['ori_shape'][:2]
    img_h, img_w = img_meta['img_shape'][:2]
    s_w, s_h = (img_w / pad_w, img_h / pad_h)

    if dets['box'].nelement() == 0:
        dets['segm'] = torch.Tensor()
        return dets

    # double check
    if score_threshold > 0:
        keep = dets['score'].view(-1) > score_threshold

        for k, v in dets.items():
            if k not in {'proto'}:
                dets[k] = v[keep]

    # Undo the padding introduced with preserve_aspect_ratio
    if cfg.preserve_aspect_ratio and dets['score'].nelement() != 0:
        # Get rid of any detections whose centers are outside the image
        boxes = dets['box']
        boxes = center_size(boxes)
        not_outside = ((boxes[:, 0] > s_w) + (boxes[:, 1] > s_h)) < 1  # not (a or b)
        for k, v in dets.items():
            if k not in {'proto'}:
                dets[k] = v[not_outside]

    if dets['score'].size(0) == 0:
        dets['segm'] = torch.Tensor()
        return dets

    # Actually extract everything from dets now
    boxes = dets['box']
    mask_coeff = dets['mask_coeff']
    masks = dets['mask']

    if visualize_lincomb:
        display_lincomb(dets['proto'], mask_coeff, img_ids, mask_det_file)

    # Undo padding for masks
    masks = masks[:, :int(s_h*masks.size(1)), :int(s_w*masks.size(2))]
    # Scale masks up to the full image
    if cfg.preserve_aspect_ratio:
        masks = F.interpolate(masks.unsqueeze(0), (ori_h, ori_w), mode=interpolation_mode,
                              align_corners=False).squeeze(0)
    else:
        masks = F.interpolate(masks.unsqueeze(0), (img_h, img_w), mode=interpolation_mode,
                              align_corners=False).squeeze(0)
    # Binarize the masks
    masks.gt_(0.5)

    if display_mask:
        dets['segm'] = masks
    else:
        # segm annotation: png2rle
        masks_output_json = []
        for i in range(masks.size(0)):
            cur_mask = mask_util.encode(np.array(masks[i].cpu(), order='F', dtype='uint8'))
            # masks[i, :, :] = torch.from_numpy(mask_util.decode(cur_mask)).cuda()
            masks_output_json.append(cur_mask)
        dets['segm'] = masks_output_json

    # Undo padding for bboxes
    boxes[:, 0::2] = boxes[:, 0::2] / s_w
    boxes[:, 1::2] = boxes[:, 1::2] / s_h

    if cfg.preserve_aspect_ratio:
        out_w = ori_w
        out_h = ori_h
    else:
        out_w = img_w
        out_h = img_h

    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], out_w, cast=False)
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], out_h, cast=False)

    boxes = boxes.long()
    dets['box'] = boxes

    return dets


def undo_image_transformation(img, img_meta, pad_h, pad_w, interpolation_mode='bilinear'):
    """
    Takes a transformed image tensor and returns a numpy ndarray that is untransformed.
    Arguments w and h are the original height and width of the image.
    """
    ori_h, ori_w = img_meta['ori_shape'][0:2]
    img_h, img_w = img_meta['img_shape'][0:2]
    s_w, s_h = (img_w / pad_w, img_h / pad_h)

    # Undo padding
    img = img[:, :int(s_h * img.size(1)), :int(s_w * img.size(2))]
    if cfg.preserve_aspect_ratio:
        img = F.interpolate(img.unsqueeze(0), (ori_h, ori_w), mode=interpolation_mode,
                            align_corners=False).squeeze(0)
    else:
        img = F.interpolate(img.unsqueeze(0), (img_h, img_w), mode=interpolation_mode,
                            align_corners=False).squeeze(0)

    img_numpy = img.permute(1, 2, 0).cpu().numpy()
    img_numpy = img_numpy[:, :, (2, 1, 0)]  # To BRG

    if cfg.backbone.transform.normalize:
        img_numpy = (img_numpy * np.array(STD) + np.array(MEANS)) / 255.0
    elif cfg.backbone.transform.subtract_means:
        img_numpy = (img_numpy / 255.0 + np.array(MEANS) / 255.0).astype(np.float32)

    img_numpy = img_numpy[:, :, (2, 1, 0)]  # To RGB
    img_numpy = np.clip(img_numpy, 0, 1)

    return img_numpy


def display_lincomb(proto_data, masks, img_ids=None, mask_det_file=None):
    proto_data = proto_data.squeeze()
    out_masks = torch.matmul(proto_data, masks.t())
    out_masks = cfg.mask_proto_mask_activation(out_masks)

    for kdx in range(1):
        jdx = kdx + 0
        import matplotlib.pyplot as plt
        coeffs = masks[jdx, :].cpu().numpy()
        idx = np.argsort(-np.abs(coeffs))
        # plt.bar(list(range(idx.shape[0])), coeffs[idx])
        # plt.show()

        coeffs_sort = coeffs[idx]
        arr_h, arr_w = (2, 4)
        proto_h, proto_w, _ = proto_data.size()
        arr_img = np.zeros([proto_h * arr_h, proto_w * arr_w])
        arr_run = np.zeros([proto_h * arr_h, proto_w * arr_w])
        test = torch.sum(proto_data, -1).cpu().numpy()

        for y in range(arr_h):
            for x in range(arr_w):
                i = arr_w * y + x
                proto_data_cur = proto_data[:, :, idx[i]]

                if i == 0:
                    running_total = proto_data_cur.cpu().numpy()   # * coeffs_sort[i]
                else:
                    running_total += proto_data_cur.cpu().numpy()   # * coeffs_sort[i]

                running_total_nonlin = running_total
                if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                    running_total_nonlin = (1 / (1 + np.exp(-running_total_nonlin)))

                arr_img[y * proto_h:(y + 1) * proto_h, x * proto_w:(x + 1) * proto_w] = \
                    ((proto_data_cur - torch.min(proto_data_cur)) / torch.max(proto_data_cur)).cpu().numpy()   # * coeffs_sort[i]
                arr_run[y * proto_h:(y + 1) * proto_h, x * proto_w:(x + 1) * proto_w] = (
                            running_total_nonlin > 0.5).astype(np.float)
        plt.imshow(arr_img*10)
        plt.axis('off')
        root_dir = ''.join([mask_det_file[:-12], 'out_proto/', str(img_ids[0]), '/'])
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        if img_ids is not None:
            plt.title(str(img_ids))
            plt.savefig(''.join([root_dir, str(img_ids[1]), '_protos.png']))
        # plt.show()
        # plt.imshow(arr_run)
        # plt.show()
        # plt.imshow(test)
        # plt.show()

    for jdx in range(out_masks.size(2)):
        plt.imshow(out_masks[:, :, jdx].cpu().numpy())
        if img_ids is not None:
            plt.title(str(img_ids))
            plt.savefig(''.join([root_dir, str(img_ids[1]), '_', str(jdx), '_mask.png']))
        # plt.show()


def display_fpn_outs(outs, img_ids=None, mask_det_file=None):

    for batch_idx in range(outs[0].size(0)):
        for idx in range(len(outs)):
            cur_out = outs[idx][batch_idx]
            import matplotlib.pyplot as plt
            arr_h, arr_w = (4, 4)
            _, h, w = cur_out.size()
            arr_img = np.zeros([h * arr_h, w * arr_w])

            for y in range(arr_h):
                for x in range(arr_w):
                    i = arr_w * y + x
                    arr_img[y * h:(y + 1) * h, x * w:(x + 1) * w] = cur_out[i, :, :].cpu().numpy()

            plt.imshow(arr_img)
            if img_ids is not None:
                plt.title(str(img_ids))
                plt.savefig(''.join([mask_det_file, str(img_ids), 'outs', str(batch_idx), str(idx), '.png']))
            plt.show()
