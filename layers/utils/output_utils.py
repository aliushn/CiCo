""" Contains functions used to sanitize and prepare the output of Yolact. """
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import pycocotools.mask as mask_util
from matplotlib.patches import Polygon

from datasets import cfg, MEANS, STD, activation_func
import os
from ..utils import sanitize_coordinates, center_size, generate_single_mask, jaccard


def postprocess(det_output, ori_h, ori_w, img_h, img_w, img_id, batch_idx=0, interpolation_mode='bilinear',
                visualize_lincomb=False, score_threshold=0, output_file=None):
    """
    Postprocesses the output of Yolact on testing mode into a format that makes sense,
    accounting for all the possible configuration settings.

    Args:
        - det_output: The lost of dicts that Detect outputs.
        - ori_w: The real width of the image.
        - ori_h: The real height of the image.
        - img_w: The input width of the images
        - img_h: The input height of the images
        - batch_idx: If you have multiple images for this batch, the image's index in the batch.
        - interpolation_mode: Can be 'nearest' | 'area' | 'bilinear' (see torch.nn.functional.interpolate)

    Returns 4 torch Tensors (in the following order):
        - classes [num_det]: The class idx for each detection.
        - scores  [num_det]: The confidence score for each detection.
        - boxes   [num_det, 4]: The bounding box for each detection in absolute point form.
        - masks   [num_det, h, w]: Full image masks for each detection.
    """

    dets = det_output[batch_idx]

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
    pad_h, pad_w = dets['mask'].size()[-2:]
    classes = dets['class']
    boxes = dets['box']
    scores = dets['score']
    masks_coeff = dets['mask_coeff']

    coeffs_norm = F.normalize(cfg.mask_proto_coeff_activation(masks_coeff), dim=1)
    cos_sim = torch.mm(coeffs_norm, coeffs_norm.t())
    # Rescale to be between 0 and 1
    cos_sim = (cos_sim + 1) / 2

    instance_t = classes.view(-1)  # juuuust to make sure
    # inst_eq = (instance_t[:, None].expand_as(cos_sim) == instance_t[None, :].expand_as(cos_sim))
    # box_iou = jaccard(boxes, boxes)

    if cfg.eval_mask_branch:
        # At this points masks is only the coefficients
        proto_data = dets['proto'][:int(dets['proto'].size(0)/pad_h * img_h), :int(dets['proto'].size(1)/pad_w * img_w)]

        # Test flag, do not upvote
        if cfg.mask_proto_debug:
            np.save('scripts/proto.npy', proto_data.cpu().numpy())

        # First undo padding area and then scale masks up to the full image
        masks = dets['mask'][:, :img_h, :img_w]

        if visualize_lincomb:
            display_lincomb(proto_data, masks, masks_coeff, img_ids=[img_id], output_file=output_file)

        masks = F.interpolate(masks.unsqueeze(0), (ori_h, ori_w), mode=interpolation_mode, align_corners=False).squeeze(0)

        # Binarize the masks
        masks.gt_(0.5)

    # Undo padding for bboxes
    boxes[:, 0::2] *= pad_w / img_w
    boxes[:, 1::2] *= pad_h / img_h
    # scale boxes upto the original width and height
    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], ori_w, cast=False)
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], ori_h, cast=False)
    boxes = boxes.long()

    return classes, scores, boxes, masks


def postprocess_ytbvis(detection, img_meta, interpolation_mode='bilinear',
                       display_mask=False, visualize_lincomb=False, crop_masks=True, score_threshold=0,
                       img_ids=None, output_file=None):
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

    if dets['box'].nelement() == 0:
        dets['segm'] = torch.Tensor()
        return dets

    ori_h, ori_w = img_meta['ori_shape'][:2]
    img_h, img_w = img_meta['img_shape'][:2]
    pad_h, pad_w = img_meta['pad_shape'][:2]
    s_w, s_h = (img_w / pad_w, img_h / pad_h)

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

    if dets['score'].nelement() == 0:
        dets['segm'] = torch.Tensor()
        return dets

    # Actually extract everything from dets now
    boxes = dets['box']
    mask_coeff = dets['mask_coeff']
    masks = dets['mask']
    if cfg.mask_proto_coeff_occlusion:
        masks_non_target = dets['mask_non_target']
        masks = torch.clamp(masks - masks_non_target, min=0)
    else:
        masks_non_target = None

    if visualize_lincomb:
        display_lincomb(dets['proto'], masks, mask_coeff, img_ids, output_file, masks_non_target)

    # Undo padding for masks
    masks = masks[:, :int(s_h*masks.size(1)), :int(s_w*masks.size(2))]
    # Scale masks up to the full image
    masks = F.interpolate(masks.unsqueeze(0), (ori_h, ori_w), mode=interpolation_mode,
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
    boxes[:, 0::2] /= s_w
    boxes[:, 1::2] /= s_h

    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], ori_w, cast=False)
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], ori_h, cast=False)

    boxes = boxes.long()
    dets['box'] = boxes

    return dets


def undo_image_transformation(img, ori_h, ori_w, img_h=None, img_w=None, interpolation_mode='bilinear'):
    """
    Takes a transformed image tensor and returns a numpy ndarray that is untransformed.
    Arguments w and h are the original height and width of the image.
    """

    pad_h, pad_w = img.size()[-2:]
    if img_h is None or img_w is None:
        img_w, img_h = pad_w, pad_h

    # Undo padding
    img = img[:, :img_h, :img_w]
    img = F.interpolate(img.unsqueeze(0), (ori_h, ori_w), mode=interpolation_mode,
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


def display_lincomb(protos_data, out_masks, masks_coeff=None, img_ids=None, output_file=None, masks_non_target=None):

    bs = 1 if protos_data.dim() == 3 else protos_data.size(0)

    masks_coeff = cfg.mask_proto_coeff_activation(masks_coeff)
    out_masks_pos = generate_single_mask(protos_data, torch.clamp(masks_coeff, min=0))
    out_masks_neg = generate_single_mask(protos_data, torch.clamp(masks_coeff, max=0))

    out_masks_temp = cfg.mask_proto_mask_activation(out_masks_pos + out_masks_neg)
    out_masks_pos = cfg.mask_proto_mask_activation(out_masks_pos)
    out_masks_neg = cfg.mask_proto_mask_activation(-out_masks_neg)

    for kdx in range(bs):
        import matplotlib.pyplot as plt
        proto_data = protos_data if protos_data.dim() == 3 else protos_data[kdx]

        arr_h, arr_w = (4, 8)
        proto_h, proto_w, _ = proto_data.size()
        arr_img = np.zeros([proto_h * arr_h, proto_w * arr_w])
        arr_run = np.zeros([proto_h * arr_h, proto_w * arr_w])

        for y in range(arr_h):
            for x in range(arr_w):
                i = arr_w * y + x
                proto_data_cur = proto_data[:, :, i]

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
        plt.imshow(arr_img)
        plt.axis('off')
        root_dir = ''.join([output_file, 'out/'])
        if len(img_ids) > 1:
            root_dir = ''.join([root_dir, str(img_ids[0]), '/'])

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        plt.title(str(img_ids[-1]))
        plt.savefig(''.join([root_dir, str(img_ids[-1]), '_protos.png']))
        plt.clf()
        # plt.show()
        # plt.imshow(arr_run)
        # plt.show()
        # plt.imshow(test)
        # plt.show()

    for jdx in range(out_masks.size(0)):
        # plot mask coeffs
        y = masks_coeff[jdx].cpu().numpy()
        x = torch.arange(len(y)).cpu().numpy()
        y0 = torch.zeros(len(y)).cpu().numpy()
        plt.plot(x, y, 'r+--', x, y0)
        plt.ylim(-1, 1)
        plt.title(str(img_ids[-1]))
        # plt.savefig(''.join([root_dir, str(img_ids[-1]), '_', str(jdx), '_mask_coeff.png']))
        plt.clf()

        # plot single mask
        plt.imshow(out_masks[jdx].cpu().numpy())
        plt.axis('off')
        plt.title(str(img_ids[-1]))
        # plt.savefig(''.join([root_dir, str(img_ids[-1]), '_', str(jdx), '_mask.png']))
        plt.clf()

        # plt masks with non-target objects
        if masks_non_target is not None:
            plt.imshow(masks_non_target[jdx].cpu().numpy())
            if img_ids is not None:
                plt.title(str(img_ids))
                plt.savefig(''.join([root_dir, str(img_ids[1]), '_', str(jdx), '_mask_non_target.png']))
                plt.clf()


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


def display_conf_outs(tensor, img_id=0, mask_det_file=None):
    mask_det_file = 'weights/COCO/weights_r50_m32_yolact_DIoU_012_640_768_randomclip_c5/'
    tensor = torch.tanh(tensor)

    for batch_idx in range(tensor.size(0)):
        import matplotlib.pyplot as plt
        h, w = tensor.size()[-2:]
        arr_img = np.zeros([3 * h, 3 * w])
        for idx in range(3):
            for jdx in range(3):
                cur_conf = tensor[batch_idx][24 * ((idx * 3) + jdx)]
                arr_img[jdx * h:(jdx + 1) * h, idx * w:(idx + 1) * w] = ((cur_conf - cur_conf.min()) / cur_conf.max()).cpu().numpy()

        plt.imshow(arr_img)
        plt.axis('off')
        plt.savefig(''.join([mask_det_file, 'out/', str(img_id), '_proto', '.png']))
        plt.clf()



