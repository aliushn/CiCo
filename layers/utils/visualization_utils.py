# Visualization correlation of STMask: https://github.com/MinghanLi/STMask
import torch
import numpy as np
import os
import cv2
from datasets import MEANS, STD
from math import sqrt
import matplotlib.pyplot as plt
import mmcv
from .output_utils import get_color, draw_dotted_rectangle


def display_box_shift(boxes, box_shift=None, img_meta=None, img_gpu=None, conf=None):
    '''
    display the offsets of bounding boxes between two adjacent frames,
    which are predicted by the temporal fusion module of STMask: https://github.com/MinghanLi/STMask
    :param boxes: bounding boxes from the previous frame
    :param box_shift: the predicted boxes from the previous frame to the current frame
    :param img_meta: inlcuding video_id and frame_id
    :param img_gpu:
    :param conf: class confidence
    :return:
    '''
    save_dir = 'outputs/box_shift'
    if img_meta is not None:
        save_dir = os.path.join(save_dir, str(img_meta['video_id']))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path = '/'.join([save_dir, str(img_meta['frame_id'])+'box_shift.png'])

    if img_gpu is None:
        # Make empty black image
        h, w = 384, 640
        img_numpy = np.ones((h, w, 3), np.uint8) * 255
    else:
        # transformed images
        h, w = img_gpu.size()[1:]
        img_gpu = img_gpu.squeeze(0).permute(1, 2, 0).contiguous()
        img_gpu = img_gpu[:, :, (2, 1, 0)]  # To BRG
        img_gpu = img_gpu * torch.tensor(STD) + torch.tensor(MEANS)
        img_gpu = img_gpu[:, :, (2, 1, 0)]  # To RGB
        img_numpy = torch.clamp(img_gpu, 0, 255).byte().cpu().numpy()

    # plot predicted bounding box form the previous frame and shifted boxes with dotted_rectangle
    color_type = range(boxes.size(0))
    for i in reversed(range(boxes.size(0))):
        color = get_color(i, color_type)
        box = boxes[i].reshape(-1, 4)
        for j in range(box.size(0)):
            cv2.rectangle(img_numpy, (box[j, 0]*w, box[j, 1]*h), (box[j, 2]*w, box[j, 3]*h), color, 2)
        if box_shift is not None:
            draw_dotted_rectangle(img_numpy, box_shift[i, 0] * w, box_shift[i, 1] * h,
                                  box_shift[i, 2] * w, box_shift[i, 3] * h, color, 2, gap=10)

        if conf is not None:
            scores, classes = conf[:, 1:].max(dim=1)
            text_str = '%s: %.2f' % (classes[i].item()+1, scores[i])

            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.5
            font_thickness = 1
            text_pt = (box_shift[i, 0]*w, box_shift[i, 1]*h - 3)
            text_color = [255, 255, 255]
            cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    plt.axis('off')
    plt.imshow(img_numpy)
    plt.savefig(path)
    plt.clf()


def display_correlation_map_patch(x_corr, img_meta=None):
    '''
    display correlation map on patches (actually similarity)
    :param x_corr: correlation
    :param img_meta:
    :return:
    '''
    video_id, frame_id = (img_meta['video_id'], img_meta['frame_id']) if img_meta is not None else (0, 0)
    save_dir = 'outputs/corr_map/'+str(video_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    bs, ch, h, w = x_corr.size()

    r = int(sqrt(ch))
    for i in range(bs):
        x_corr_cur = x_corr[i]
        x_corr_cur = (x_corr_cur - x_corr_cur.min()) / x_corr_cur.max()
        x_show = x_corr_cur.view(r, r, h, w)
        x_show = x_show.permute(0, 2, 1, 3).contiguous().view(h*r, r*w)
        x_numpy = x_show.detach().cpu().numpy()

        path = '/'.join([save_dir, str(frame_id)+'_'+str(i)+'_corr_patch.png'])
        plt.imshow(x_numpy)
        plt.axis('off')
        plt.savefig(path)
        plt.clf()


def display_correlation_map(x_corr, imgs=None, img_meta=None, idx=0):
    '''
    display correlation map
    :param x_corr: [bs, patch-h*patch_w, h, w]
    :param imgs: [bs, frames, 3, h, w]
    :param img_meta:
    :param idx:
    :return:
    '''
    if img_meta is not None:
        video_id, frame_id = img_meta['video_id'], img_meta['frame_id']
        s_h = img_meta['img_shape'][0] / float(img_meta['pad_shape'][0])
        s_w = img_meta['img_shape'][1] / float(img_meta['pad_shape'][1])
    else:
        video_id, frame_id = 0, 0
        s_h, s_w = 1, 1

    save_dir = 'outputs/corr_map/'
    save_dir = os.path.join(save_dir, str(video_id))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    bs, ch, h, w = x_corr.size()
    crop_h, crop_w = int(h*s_h), int(w*s_w)
    x_corr = x_corr[:, :, :crop_h, :crop_w]
    x_corr /= torch.max(x_corr, dim=1)[0].unsqueeze(1)

    r = int(sqrt(ch))
    for i in range(bs):
        # display correlation map
        x_show = x_corr[i].view(r, r, crop_h, crop_w)
        x_show = x_show.permute(0, 2, 1, 3).contiguous().view(crop_h*r, r*crop_w)
        x_numpy = x_show.detach().cpu().numpy()

        path = ''.join([save_dir, '/', str(frame_id), '_', str(i), '_corr', str(idx), '.png'])
        plt.imshow(x_numpy)
        plt.axis('off')
        plt.savefig(path)
        plt.clf()


def display_shifted_masks(shifted_masks, img_meta=None):
    '''
    display shifted masks generated from the previous frame to the current frame
    :param shifted_masks:
    :param img_meta:
    :return:
    '''
    n, h, w = shifted_masks.size()
    for i in range(n):
        if img_meta is not None:
            path = '/'.join(['outputs', str(img_meta['video_id'])+'_'+
                            str(img_meta['frame_id'])+'_'+str(i)+'_shifted_masks.png'])

        else:
            path = 'outputs/shifted_mask.png'
        shifted_masks = shifted_masks.gt(0.3).float()
        shifted_masks_numpy = shifted_masks[i].cpu().numpy()
        plt.axis('off')
        plt.pcolormesh(mmcv.imflip(shifted_masks_numpy*10, direction='vertical'))
        plt.savefig(path)
        plt.clf()