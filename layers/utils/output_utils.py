""" Contains functions used to sanitize and prepare the output of CiCo. """
import torch
import cv2
import torch.nn.functional as F
import numpy as np

from datasets import MEANS, STD
import os
from ..utils import sanitize_coordinates, center_size, generate_mask, jaccard, point_form
import matplotlib.pyplot as plt


def postprocess(dets, ori_h, ori_w, s_h, s_w, img_id, train_masks=True, interpolation_mode='bilinear',
                visualize_lincomb=False, score_threshold=0, output_file=None):
    """
    On CoCo, Postprocesses the output of Yolact on testing mode into a format that makes sense,
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

    if dets is None:
        return [torch.Tensor()] * 4  # Warning, this is 4 copies of the same thing

    proto_data = dets['proto']
    if score_threshold > 0:
        keep = dets['score'] > score_threshold

        dets = {k: v[keep] for k, v in dets.items() if k != 'proto'}
        if dets['score'].size(0) == 0:
            return [torch.Tensor()] * 4

    if dets['box'].size(0) == 0:
        return [torch.Tensor()] * 4

    # Actually extract everything from dets now
    masks = dets['mask'].squeeze(1) if dets['mask'].dim() == 4 else dets['mask']
    classes = dets['class']
    boxes = dets['box'].reshape(-1, 4)
    scores = dets['score']
    masks_coeff = dets['mask_coeff']

    if train_masks:
        # At this points masks is only the coefficients
        proto_data = proto_data[:int(proto_data.size(0)*s_h), :int(proto_data.size(1)*s_w)]

        # First undo padding area and then scale masks up to the full image
        masks = masks[:, :int(s_h*masks.size(1)), :int(s_w*masks.size(2))]

        if visualize_lincomb:
            display_lincomb(proto_data, masks_coeff, img_ids=[img_id], output_file=output_file)

        masks = F.interpolate(masks.unsqueeze(0), (ori_h, ori_w), mode=interpolation_mode, align_corners=False).squeeze(0)
        # Binarize the masks
        masks.gt_(0.5)

    # Undo padding for bboxes
    boxes[:, 0::2] /= s_w
    boxes[:, 1::2] /= s_h
    # scale boxes upto the original width and height
    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], ori_w, cast=False)
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], ori_h, cast=False)
    boxes = boxes.long()

    return classes, scores, boxes, masks


def postprocess_ytbvis(dets_output, img_meta, train_masks=True, interpolation_mode='bilinear',
                       visualize_lincomb=False, output_file=None):
    """
    For VIS, Postprocesses the output of CiCo on testing mode into a format that makes sense,
    accounting for all the possible configuration settings.

    Args:
        - dets_output: The lost of dicts that Detect outputs.
        - img_meta: a list of image shape information.
        - batch_idx: If you have multiple images for this batch, the image's index in the batch.
        - interpolation_mode: Can be 'nearest' | 'area' | 'bilinear' (see torch.nn.functional.interpolate)
        - visualize_lincomb: display prototypes and mask coefficients on output_file.

    Returns 4 torch Tensors (in the following order):
        - classes [num_det]: The class idx for each detection.
        - scores  [num_det]: The confidence score for each detection.
        - boxes   [num_det, 4]: The bounding box for each detection in absolute point form.
        - masks   [num_det, h, w]: Full image masks for each detection.
    """
    # TODO: merge postprocess and postprocess_ytbvis functions
    dets = dict()
    for k, v in dets_output.items():
        if k is not None:
            dets[k] = v.clone()

    if dets['box'].nelement() == 0:
        return dets

    img_meta = img_meta[0] if isinstance(img_meta, list) else img_meta
    ori_h, ori_w = img_meta['ori_shape'][:2]
    img_h, img_w = img_meta['img_shape'][:2]
    pad_h, pad_w = img_meta['pad_shape'][:2]
    s_w, s_h = img_w / pad_w, img_h / pad_h

    n_objs = dets['box'].size(0)
    # Undo padding for bboxes
    boxes = dets['box'].reshape(-1, 4)
    boxes[:, 0::2] /= s_w
    boxes[:, 1::2] /= s_h
    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], ori_w, cast=False)
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], ori_h, cast=False)
    dets['box'] = boxes.long().reshape(n_objs, -1, 4)

    if 'box_cir' in dets.keys():
        boxes_cir = dets['box_cir']
        boxes_cir[:, 0::2] /= s_w
        boxes_cir[:, 1::2] /= s_h
        boxes_cir[:, 0], boxes_cir[:, 2] = sanitize_coordinates(boxes_cir[:, 0], boxes_cir[:, 2], ori_w, cast=False)
        boxes_cir[:, 1], boxes_cir[:, 3] = sanitize_coordinates(boxes_cir[:, 1], boxes_cir[:, 3], ori_h, cast=False)
        dets['box_cir'] = boxes_cir.long()

    if 'priors' in dets.keys():
        priors = point_form(dets['priors'])
        priors[:, 0::2] /= s_w
        priors[:, 1::2] /= s_h
        priors[:, 0], priors[:, 2] = sanitize_coordinates(priors[:, 0], priors[:, 2], ori_w, cast=False)
        priors[:, 1], priors[:, 3] = sanitize_coordinates(priors[:, 1], priors[:, 3], ori_h, cast=False)
        dets['priors'] = priors.long()

    # Actually extract everything from dets now
    if train_masks:
        masks = dets['mask']
        mask_coeff = dets['mask_coeff']

        proto = dets['proto'][:int(s_h*masks.size(-2)), :int(s_w*masks.size(-1))]
        if visualize_lincomb:
            img_ids = (img_meta['video_id'], img_meta['frame_id'])
            display_lincomb(proto, mask_coeff, img_ids, output_file)

        # Scale masks up to the full image
        # proto = F.interpolate(proto.permute(2,0,1).unsqueeze(1), (ori_h, ori_w), mode=interpolation_mode,
        #                       align_corners=False).squeeze(1)
        # masks = (proto.permute(1,2,0).contiguous() @ mask_coeff.t()).permute(2,0,1).contiguous()
        mask_dim = masks.dim()
        masks = masks.unsqueeze(1) if mask_dim == 3 else masks
        # Undo padding for masks
        masks = masks[..., :int(s_h*masks.size(-2)), :int(s_w*masks.size(-1))]
        masks = F.interpolate(masks, (ori_h, ori_w), mode=interpolation_mode,
                              align_corners=False)

        # Binarize the masks
        masks.gt_(0.5)
        dets['mask'] = masks.squeeze(1) if mask_dim == 3 else masks

    return dets


def undo_image_transformation(img, ori_h, ori_w, img_h=None, img_w=None, interpolation_mode='bilinear'):
    """
    Takes a transformed image tensor and returns a numpy ndarray that is untransformed.
    Arguments ori_w and ori_h are the original height and width of the image,
    while img_w and img_h are the reshaped height and width of the image without padding.
    """

    pad_h, pad_w = img.size()[-2:]
    if img_h is None or img_w is None:
        img_w, img_h = pad_w, pad_h

    # Undo padding
    img = img[..., :img_h, :img_w]
    img_dim = img.dim()
    img = img.unsqueeze(0) if img_dim == 3 else img
    img = F.interpolate(img, (ori_h, ori_w), mode=interpolation_mode,align_corners=False)
    img = img.permute(0, 2, 3, 1).contiguous()
    img = img[..., (2, 1, 0)]  # To BRG

    # TODO: If the backbone is not Resnet, please double check the image transformation according to backbone.py
    # if cfg.backbone.transform.normalize:
    img = (img * torch.tensor(STD) + torch.tensor(MEANS)) / 255.0
    # elif cfg.backbone.transform.subtract_means:
    #     img_numpy = (img_numpy / 255.0 + np.array(MEANS) / 255.0).astype(np.float32)

    img = img[..., (2, 1, 0)]  # To RGB
    img_numpy = torch.clamp(img, 0, 1).cpu().numpy()

    return img_numpy.squeeze(0) if img_dim == 3 else img_numpy


def display_lincomb(protos_data, masks_coeff=None, img_ids=None, output_file=None,
                    display_coeff=False, display_mask=False):
    '''
    Display prototypes and mask coefficients into the output_file.
    '''
    if protos_data.dim() == 3:
        protos_data = protos_data.unsqueeze(-2)
        out_masks = generate_mask(protos_data, masks_coeff)
    else:
        out_masks = generate_mask(protos_data, masks_coeff)
    protos_data = protos_data.permute(2, 0, 1, 3).contiguous()
    root_dir = os.path.join(output_file, 'protos/')
    if len(img_ids) > 1:
        root_dir = ''.join([root_dir, str(img_ids[0]), '/'])
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for kdx in range(protos_data.size(0)):
        proto_data = protos_data[kdx]

        # We set the number of prototypes is 32 as default, if you use others, please adaptively change it.
        arr_h, arr_w = (4, 8)
        proto_h, proto_w, _ = proto_data.size()
        arr_img = np.zeros([proto_h * arr_h, proto_w * arr_w])
        arr_run = np.zeros([proto_h * arr_h, proto_w * arr_w])

        # plot prototypes
        for y in range(arr_h):
            for x in range(arr_w):
                i = arr_w * y + x
                proto_data_cur = proto_data[:, :, i]

                if i == 0:
                    running_total = proto_data_cur.cpu().numpy()   # * coeffs_sort[i]
                else:
                    running_total += proto_data_cur.cpu().numpy()   # * coeffs_sort[i]

                running_total_nonlin = running_total
                running_total_nonlin = (1 / (1 + np.exp(-running_total_nonlin)))
                arr_img[y * proto_h:(y + 1) * proto_h, x * proto_w:(x + 1) * proto_w] = \
                    ((proto_data_cur - torch.min(proto_data_cur)) / torch.max(proto_data_cur)+1e-5).cpu().numpy()   # * coeffs_sort[i]
                arr_run[y * proto_h:(y + 1) * proto_h, x * proto_w:(x + 1) * proto_w] = (
                            running_total_nonlin > 0.5).astype(np.float)
        plt.imshow(arr_img)
        plt.axis('off')
        plt.title(str(img_ids[-1]))
        plt.savefig(''.join([root_dir, str(img_ids[-1]), '_protos.png']))
        plt.clf()

        # Plot mask coeff
        if display_coeff:
            x = torch.arange(len(masks_coeff[0])).cpu().numpy()
            plt.subplots(figsize=(30, 5))
            for jdx in range(masks_coeff.size(0)):
                y = masks_coeff[jdx].cpu().numpy()
                plt.bar(x, y, width=1, edgecolor='blue')
                for x1, y1 in zip(x, y):
                    if y1 > 0:
                        plt.text(x1-0.35, y1+0.05, '%.2f'%y1, fontsize=18)
                    else:
                        plt.text(x1-0.45, y1-0.18, '%.2f'%y1, fontsize=18)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.ylim(-1.2, 1.2)
                plt.xlim(-0.5, 31.5)
                plt.savefig(''.join([root_dir, str(img_ids[-1]), str(jdx), '_mask_coeff.png']))
                plt.clf()

                # plot single mask
                if display_mask:
                    plt.imshow(out_masks[jdx, kdx].cpu().numpy())
                    plt.axis('off')
                    plt.title(str(img_ids[-1]))
                    plt.savefig(''.join([root_dir, str(img_ids[-1]), '_', str(jdx), '_mask.png']))
                    plt.clf()


def draw_dotted_rectangle(img, x0, y0, x1, y1, color, thickness=1, gap=20):
    '''
    Draw a dotted rectangle, where x0, y0, x1, y1 are the coordinates of the top-left pixel and the bottom-right pixel.
    gap is the interval of dotted lines.
    '''
    draw_dotted_line(img, (x0, y0), (x0, y1), color, thickness, gap, vertical=True)
    draw_dotted_line(img, (x0, y0), (x1, y0), color, thickness, gap)
    draw_dotted_line(img, (x0, y1), (x1, y1), color, thickness, gap)
    draw_dotted_line(img, (x1, y0), (x1, y1), color, thickness, gap, vertical=True)


def draw_dotted_line(img, pt1, pt2, color, thickness=1, gap=20, vertical=False):
    '''
    Draw dotted a line, where pt1, pt2 are the locations of the beginning and end points.
    '''
    dist = ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**.5
    pts = []
    for i in range(0, int(dist), gap):
        r = i/dist
        x = int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y = int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x, y)
        pts.append(p)

    for p in pts:
        if vertical:
            cv2.line(img, p, (p[0], p[1] + int(gap//3)), color, thickness)
        else:
            cv2.line(img, p, (p[0] + int(gap//3), p[1]), color, thickness)


# for making bounding boxes pretty when display in eval.py
COLORS = ((244,  67,  54),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          (  0, 150, 136),
          ( 76, 175,  80),
          (139, 195,  74),
          (205, 220,  57),
          (255, 235,  59),
          (255, 193,   7),
          (255, 152,   0),
          (255,  87,  34),
          (121,  85,  72),
          (158, 158, 158),
          ( 96, 125, 139))


# Quick and dirty lambda for selecting the color for a particular index
# Also keeps track of a per-gpu color cache for maximum speed
def get_color(j, color_type=None, on_gpu=None, undo_transform=True):
    if color_type is None:
        color_idx = j * 5 % len(COLORS)
    else:
        color_idx = color_type[j] * 5 % len(COLORS)

    color = COLORS[color_idx]
    if not undo_transform:
        # The image might come in as RGB or BRG, depending
        color = (color[2], color[1], color[0])
    if on_gpu is not None:
        color = torch.Tensor(color).to(on_gpu).float() / 255.

    return color




