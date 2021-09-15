import torch
import numpy as np
import os
import cv2
from datasets import cfg, MEANS, STD
import random
from math import sqrt
import matplotlib.pyplot as plt
import mmcv
import torch.nn.functional as F
from matplotlib.colors import hsv_to_rgb


# Quick and dirty lambda for selecting the color for a particular index
# Also keeps track of a per-gpu color cache for maximum speed
def get_color(j, color_type=None, on_gpu=None, undo_transform=True):
    if color_type is None:
        color_idx = j * 5 % len(cfg.COLORS)
    else:
        color_idx = color_type[j] * 5 % len(cfg.COLORS)

    color = cfg.COLORS[color_idx]
    if not undo_transform:
        # The image might come in as RGB or BRG, depending
        color = (color[2], color[1], color[0])
    if on_gpu is not None:
        color = torch.Tensor(color).to(on_gpu).float() / 255.

    return color


def display_pos_smaples(pos, img_gpu, decoded_priors, bbox):
    h, w = 384, 640
    img_numpy = img_gpu.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_numpy = img_numpy[:, :, (2, 1, 0)]  # To BRG
    img_numpy = (img_numpy * np.array(STD) + np.array(MEANS)) / 255.0
    img_numpy = np.clip(img_numpy, 0, 1)
    img_gpu = torch.Tensor(img_numpy).cuda()
    image = (img_gpu * 255).byte().cpu().numpy()

    pos_decoded_priors = decoded_priors[pos]
    # Create a named colour
    green = [0, 255, 0]
    blue = [255, 0, 0]
    epsilon = [0, 0, 1]

    def list_add(a, b):
        c = []
        w = torch.randint(255, (1,)).tolist()[0]
        for i in range(len(a)):
            c.append(a[i] + b[i] * w)
        return c

    # plot anchors of positive samples
    for i in range(pos.sum()):
        cv2.rectangle(image, (pos_decoded_priors[i, 0] * w, pos_decoded_priors[i, 1] * h),
                      (pos_decoded_priors[i, 2] * w, pos_decoded_priors[i, 3] * h), list_add(blue, epsilon), 1)

    # plot GT boxes of positive samples
    for i in range(bbox.size(0)):
        cv2.rectangle(image, (bbox[i, 0] * w, bbox[i, 1] * h),
                      (bbox[i, 2] * w, bbox[i, 3] * h), green, 2)
    path = ''.join(['weights/pos_samples_OVIS/', str(torch.randint(1000, (1,)).tolist()[0]), '.png'])
    cv2.imwrite(path, image)


def display_box_shift(box, box_shift, img_meta, img_gpu=None, conf=None):
    save_dir = 'weights/OVIS/weights_r152_m32_yolact_dice_DIoU_012_768_960_randomclip_c5/box_shift/'
    save_dir = os.path.join(save_dir, str(img_meta['video_id']))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path = ''.join([save_dir, '/', str(img_meta['frame_id']), '_c10.png'])

    # Make empty black image
    if img_gpu is None:
        h, w = 384, 640
        image = np.ones((h, w, 3), np.uint8) * 255
    else:
        h, w = img_gpu.size()[1:]
        img_numpy = img_gpu.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_numpy = img_numpy[:, :, (2, 1, 0)]  # To BRG
        img_numpy = (img_numpy * np.array(STD) + np.array(MEANS)) / 255.0
        # img_numpy = img_numpy[:, :, (2, 1, 0)]  # To RGB
        img_numpy = np.clip(img_numpy, 0, 1)
        img_gpu = torch.Tensor(img_numpy).cuda()
        image = (img_gpu * 255).byte().cpu().numpy()

    if conf is not None:
        scores, classes = conf[:, 1:].max(dim=1)

    # plot pred bbox
    color_type = range(box.size(0))
    for i in range(box.size(0)):
        color = get_color(i, color_type)
        cv2.rectangle(image, (box[i, 0]*w, box[i, 1]*h), (box[i, 2]*w, box[i, 3]*h), color, 2)

        draw_dotted_rectangle(image, box_shift[i, 0] * w, box_shift[i, 1] * h,
                              box_shift[i, 2] * w, box_shift[i, 3] * h, color, 2, gap=10)

        if conf is not None:
            text_str = '%s: %.2f' % (classes[i].item()+1, scores[i])

            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.5
            font_thickness = 1
            text_pt = (box_shift[i, 0]*w, box_shift[i, 1]*h - 3)
            text_color = [255, 255, 255]
            cv2.putText(image, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.imwrite(path, image)


def display_feature_align_dcn(detection, offset, loc_data, img_gpu=None, img_meta=None, use_yolo_regressors=False):
    h, w = 384, 640
    # Make empty black image
    if img_gpu is None:
        image = np.ones((h, w, 3), np.uint8) * 255
    else:
        img_numpy = img_gpu.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_numpy = img_numpy[:, :, (2, 1, 0)]  # To BRG
        img_numpy = (img_numpy * np.array(STD) + np.array(MEANS)) / 255.0
        # img_numpy = img_numpy[:, :, (2, 1, 0)]  # To RGB
        img_numpy = np.clip(img_numpy, 0, 1)
        img_gpu = torch.Tensor(img_numpy).cuda()
        image = (img_gpu * 255).byte().cpu().numpy()

    n_dets = detection['box'].size(0)
    n = 0
    p = detection['priors'][n]
    decoded_loc = detection['box'][n]
    id = detection['bbox_idx'][n]
    loc = loc_data[0, id, :]
    pixel_id = id // 3
    prior_id = id % 3
    if prior_id == 0:
        o = offset[0, :18, pixel_id]
        ks_h, ks_w = 3, 3
        grid_w = torch.tensor([-1, 0, 1] * ks_h)
        grid_h = torch.tensor([[-1], [0], [1]]).repeat(1, ks_w).view(-1)
    elif prior_id == 1:
        o = offset[0, 18:48, pixel_id]
        ks_h, ks_w = 3, 5
        grid_w = torch.tensor([-2, -1, 0, 1, 2] * ks_h)
        grid_h = torch.tensor([[-1], [0], [1]]).repeat(1, ks_w).view(-1)
    else:
        o = offset[0, 48:, pixel_id]
        ks_h, ks_w = 5, 3
        grid_w = torch.tensor([-1, 0, 1] * ks_h)
        grid_h = torch.tensor([[-2], [-1], [0], [1], [2]]).repeat(1, ks_w).view(-1)

    # thransfer the rectange to 9 points
    cx1, cy1, w1, h1 = p[0], p[1], p[2], p[3]
    dw1 = grid_w * w1 / (ks_w-1) + cx1
    dh1 = grid_h * h1 / (ks_h-1) + cy1

    dwh = p[2:] * ((loc.detach()[2:] * 0.2).exp() - 1)
    # regressed bounding boxes
    new_dh1 = dh1 + loc[1] * p[3] * 0.1 + dwh[1] / ks_h * grid_h
    new_dw1 = dw1 + loc[0] * p[2] * 0.1 + dwh[0] / ks_w * grid_w
    # points after the offsets of dcn
    new_dh2 = dh1 + o[::2].view(-1) * 0.5 * p[3]
    new_dw2 = dw1 + o[1::2].view(-1) * 0.5 * p[2]

    # Create a named colour
    blue = [255, 0, 0]  # bgr
    purple = [128, 0, 128]
    red = [0, 0, 255]

    # plot pred bbox
    cv2.rectangle(image, (decoded_loc[0] * w, decoded_loc[1] * h), (decoded_loc[2] * w, decoded_loc[3] * h),
                  blue, 2, lineType=8)

    # plot priors
    pxy1 = p[:2] - p[2:] / 2
    pxy2 = p[:2] + p[2:] / 2
    cv2.rectangle(image, (pxy1[0] * w, pxy1[1] * h), (pxy2[0] * w, pxy2[1] * h),
                  purple, 2, lineType=8)
    for i in range(len(dw1)):
        cv2.circle(image, (new_dw2[i] * w, new_dh2[i] * h), radius=0, color=blue, thickness=10)
        cv2.circle(image, (new_dw1[i]*w, new_dh1[i]*h), radius=0, color=blue, thickness=6)
        cv2.circle(image, (dw1[i] * w, dh1[i] * h), radius=0, color=purple, thickness=6)

    if img_meta is not None:
        path = ''.join(['results/results_1024_2/FCB/', str(img_meta[0]['video_id']), '_',
                        str(img_meta[0]['frame_id']), '.png'])
    else:
        path = 'results/results_1024_2/FCB/0.png'
    cv2.imwrite(path, image)


def display_correlation_map_patch(x_corr, img_meta=None):
    if img_meta is not None:
        video_id, frame_id = img_meta['video_id'], img_meta['frame_id']
    else:
        video_id, frame_id = 0, 0

    save_dir = 'weights/OVIS/weights_r152_plus_m32_yolact_bce_012_960_clip_track_loss/box_shift/'
    save_dir = os.path.join(save_dir, str(video_id))
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

        #         path_max = ''.join([save_dir, '/', str(frame_id), '_', str(i), '_max_corr_patch.png'])
        #         max_corr = x_corr_cur.max(dim=0)[0].detach().cpu().numpy()
        #         plt.imshow(max_corr)
        #         plt.savefig(path_max)

        path = ''.join([save_dir, '/', str(frame_id), '_', str(i), '_corr_patch.png'])
        plt.imshow(x_numpy)
        plt.axis('off')
        plt.savefig(path)
        plt.clf()


def display_correlation_map(x_corr, imgs=None, img_meta=None, idx=0):
    '''
    :param x_corr: [bs, patch-h*patch_w, h, w]
    :param imgs: [bs, frames, 3, h, w]
    :param img_meta:
    :param idx:
    :return:
    '''
    if img_meta is not None:
        video_id, frame_id = img_meta['video_id'], img_meta['frame_id']
        s_h, s_w = img_meta['img_shape'][0] / float(img_meta['pad_shape'][0]), img_meta['img_shape'][1] / float(img_meta['pad_shape'][1])
    else:
        video_id, frame_id = 0, 0
        s_h, s_w = 1, 1

    save_dir = 'weights/OVIS/weights_plus_r50_m32_yolact_clip_prediction_w_corr_a6_768_ext_box_focal/out/'
    save_dir = os.path.join(save_dir, str(video_id))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    bs, ch, h, w = x_corr.size()
    crop_h, crop_w = int(h*s_h), int(w*s_w)
    x_corr = x_corr[:, :, :crop_h, :crop_w]
    x_corr /= torch.max(x_corr, dim=1)[0].unsqueeze(1)

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    display_sparse_flow = False
    display_dense_flow = False
    from .utils import undo_image_transformation

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

        if display_sparse_flow or display_dense_flow:
            max_corr_val, max_corr_idx = x_corr[i].max(0)
            flow_x = max_corr_idx // r - 5
            flow_y = max_corr_idx % r - 5
            x = torch.tensor(range(crop_w))
            y = torch.tensor(range(crop_h))
            grid_y, grid_x = torch.meshgrid(y, x)
            grid_y, grid_x = grid_y.reshape(-1).tolist(), grid_x.reshape(-1).tolist()
            img = []
            for cdx in range(imgs.size(1)):
                img.append(undo_image_transformation(imgs[i, cdx], h, w, img_h=None, img_w=None,
                                                     interpolation_mode='bilinear'))
            # img = F.interpolate(imgs[i], (h, w))
            img = np.stack(img, axis=0).transpose((0, 2, 1, 3))
            img = img[:, :crop_h, :crop_w]

            if display_sparse_flow:
                # display sparse optical flow map: draw the tracks
                flow_x, flow_y = flow_x.reshape(-1).tolist(), flow_y.reshape(-1).tolist()
                temp = (img[0]*255).astype('uint8')
                for j in range(crop_h*crop_w):
                    if flow_x[j] > 2 or flow_y[j] > 2:
                        j_temp = j % 100
                        cv2.arrowedLine(temp, (grid_x[j], grid_y[j]),
                                        (max(grid_x[j]+flow_x[j], 0), max(grid_y[j]+flow_y[j], 0)),
                                        [255, 255, 255], 2)
                path = ''.join([save_dir, '/', str(frame_id), '_', str(i), '_corr', str(idx), '_sparse_flow.png'])
                plt.imshow(temp)
                plt.axis('off')
                plt.savefig(path)
                plt.clf()

            if display_dense_flow:
                # display dense optical flow map
                hsv = np.zeros_like(img[0])
                hsv[..., 1] = 255
                flow_x = flow_x.cpu().numpy().astype(np.float32)
                flow_y = flow_y.cpu().numpy().astype(np.float32)
                mag, ang = cv2.cartToPolar(flow_x, flow_y)
                # prvs = cv2.cvtColor(img[0].cpu().numpy(), cv2.COLOR_BGR2GRAY)
                # next = cv2.cvtColor(img[1].cpu().numpy(), cv2.COLOR_BGR2GRAY)
                # flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                # mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb_show = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                # cv2.imshow('dense_flow', rgb_show)
                path = ''.join([save_dir, '/', str(frame_id), '_', str(i), '_corr', str(idx), '_dense_flow.png'])
                cv2.imwrite(path, rgb_show)
                # plt.imshow(rgb_show)
                # plt.axis('off')
                # plt.savefig(path)
                # plt.clf()


def display_embedding_map(matching_map_all, idx, img_meta=None):
    if img_meta is not None:
        path = ''.join(['results/results_1227_1/embedding_map/', str(img_meta['video_id']), '_',
                        str(img_meta['frame_id']), '_', str(idx), '.png'])
        path2 = ''.join(['results/results_1227_1/embedding_map/', str(img_meta['video_id']), '_',
                         str(img_meta['frame_id']), '_', str(idx), '_m.png'])

    else:
        path = 'results/results_1227_1/embedding_map/0.png'
        path2 = 'results/results_1227_1/embedding_map/0_m.png'

    matching_map_all = matching_map_all.squeeze(0)
    r, r, h, w = matching_map_all.size()
    # matching_map_mean = matching_map_all.view(r**2, h, w).mean(0)  # / (r**2)
    matching_map, _ = matching_map_all.view(r ** 2, h, w).max(0)  # / (r**2)
    x_show = matching_map_all.permute(0, 2, 1, 3).contiguous()
    x_show = x_show.view(h * r, r * w)
    x_numpy = (x_show[h*2:h*10, w*2:w*10]).cpu().numpy()

    plt.axis('off')
    plt.pcolormesh(mmcv.imflip(x_numpy, direction='vertical'))
    plt.savefig(path)
    plt.clf()

    matching_map_numpy = matching_map.squeeze(0).cpu().numpy()
    plt.axis('off')
    plt.imshow(matching_map_numpy)
    plt.savefig(path2)
    plt.clf()


def display_shifted_masks(shifted_masks, img_meta=None):
    n, h, w = shifted_masks.size()

    for i in range(n):
        if img_meta is not None:
            path = ''.join(['results/results_1227_1/embedding_map/', str(img_meta['video_id']), '_',
                            str(img_meta['frame_id']), '_', str(i), '_shifted_masks.png'])

        else:
            path = 'results/results_1227_1/fea_ref/0_shifted_mask.png'
        shifted_masks = shifted_masks.gt(0.3).float()
        shifted_masks_numpy = shifted_masks[i].cpu().numpy()
        plt.axis('off')
        plt.pcolormesh(mmcv.imflip(shifted_masks_numpy*10, direction='vertical'))
        plt.savefig(path)
        plt.clf()


def draw_dotted_rectangle(img, x0, y0, x1, y1, color, thickness=1, gap=20):

    draw_dotted_line(img, (x0, y0), (x0, y1), color, thickness, gap, vertical=True)
    draw_dotted_line(img, (x0, y0), (x1, y0), color, thickness, gap)
    draw_dotted_line(img, (x0, y1), (x1, y1), color, thickness, gap)
    draw_dotted_line(img, (x1, y0), (x1, y1), color, thickness, gap, vertical=True)


def draw_dotted_line(img, pt1, pt2, color, thickness=1, gap=20, vertical=False):
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


