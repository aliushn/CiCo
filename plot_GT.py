from datasets import *
import pandas as pd
import numpy as np
import csv
import os
import cv2
import torch
import json
from datasets import get_dataset, prepare_data_vis
from configs.load_config import load_config
import torch.utils.data as data
import argparse
import torch.nn.functional as F
import pycocotools.mask as mask_util
from utils.functions import ProgressBar

from collections import defaultdict
import matplotlib.pyplot as plt
from layers.utils import undo_image_transformation, center_size, jaccard


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Yolact Training Script')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--data_type', default='valid',
                    help='The config object to use.')
parser.add_argument('--config', default='configs/VIS/r50_base_YTVIS2019_cubic_c3_indbox_1X.py',
                    help='The config object to use.')
parser.add_argument('--save_path', default='results/eval_mask_detections.json', type=str,
                    help='The output file for coco mask results if --coco_results is set.')
parser.add_argument('--display_bboxes', default=False, type=str2bool,
                    help='Whether or not to display bboxes around masks')
parser.add_argument('--display_masks', default=True, type=str2bool,
                    help='Whether or not to display bboxes around masks')
parser.add_argument('--display_single_mask', default=False, type=str2bool,
                    help='Whether or not to display bboxes around masks')

args = parser.parse_args()
color_cache = defaultdict(lambda: {})


def count_temporal_BIoU(cfg):
    type = 0
    if type == 0:
        train_dataset = get_dataset(cfg.DATASETS.TYPE, cfg.DATASETS.TRAIN, cfg.INPUT,
                                    cfg.SOLVER.NUM_CLIP_FRAMES, cfg.SOLVER.NUM_CLIPS)
        data_loader = data.DataLoader(train_dataset, 1,
                                      num_workers=cfg.SOLVER.NUM_WORKERS,
                                      collate_fn=detection_collate_vis,
                                      shuffle=False,
                                      pin_memory=True)
        biou_objs = [[], [], []]
        for i, data_batch in enumerate(data_loader):
            # if i > 20:
            #     break
            if i % 1000 == 0:
                print('processing %6d images' % (i))
            images, gt_bboxes, gt_labels, gt_masks, gt_ids, num_crowds, img_metas = prepare_data_vis(data_batch,
                                                                                                     devices=[torch.cuda.current_device()])
            boxes = gt_bboxes[0][0]
            for j in range(boxes.size(0)):
                CT = boxes.size(1) // 2
                ccbox = torch.prod(center_size(boxes[j].reshape(-1, 4))[:, 2:], dim=-1)
                if ccbox[CT] > 0:
                    if ccbox[0] > 0 or ccbox[-1] > 0:
                        biou = 0
                        if ccbox[0] > 0:
                            biou = jaccard(boxes[j, CT].reshape(1, 4), boxes[j, 0].reshape(1, 4))
                        if ccbox[-1] > 0:
                            biou += jaccard(boxes[j, CT].reshape(1, 4), boxes[j, -1].reshape(1, 4))
                        biou_objs[2] += [float(biou/((ccbox[0] > 0).float()+(ccbox[-1] > 0).float()))]
                    if ccbox[1] > 0 or ccbox[-2] > 0:
                        biou = 0
                        if ccbox[1] > 0:
                            biou = jaccard(boxes[j, CT].reshape(1, 4), boxes[j, 1].reshape(1, 4))
                        if ccbox[-2] > 0:
                            biou += jaccard(boxes[j, CT].reshape(1, 4), boxes[j, -2].reshape(1, 4))
                        biou_objs[1] += [float(biou/((ccbox[1] > 0).float()+(ccbox[-2] > 0).float()))]

                    if ccbox[2] > 0 or ccbox[-3] > 0:
                        biou = 0
                        if ccbox[2] > 0:
                            biou = jaccard(boxes[j, CT].reshape(1, 4), boxes[j, 2].reshape(1, 4))
                        if ccbox[-3] > 0:
                            biou += jaccard(boxes[j, CT].reshape(1, 4), boxes[j, -3].reshape(1, 4))
                        biou_objs[0] += [float(biou/((ccbox[2] > 0).float()+(ccbox[-3] > 0).float()))]
        print('Finish all images!')

        for i in range(3):
            data_numpy = np.asarray(biou_objs[i])
            print('Save counts as %2d.csv file!'%(2*(i+8)+1))
            x_df = pd.DataFrame(data_numpy)
            x_df.to_csv('../datasets/YouTube_VOS2019/temporal_BIoU_c%1d_train.csv'%(2*(i+8)+1))
            print('Plot histogram!')
            bins = np.arange(0, 1.05, 0.05)   # fixed bin size
            plt.hist(data_numpy, bins=bins, alpha=0.5, facecolor='blue')
            plt.xlabel('BIoU of temporal boxes (bin size = 0.02)')
            plt.ylabel('count')
            plt.savefig('../datasets/YouTube_VOS2019/temporal_BIoU_c%1d_train_hist.png'%(2*(i+8)+1))
            plt.clf()
    else:
        x, y = [], []
        for i in range(3, 16, 4):
            if i < 10:
                file = open('../datasets/YouTube_VOS2019/temporal_BIoU_c%1d_train.csv'%(i))
            else:
                file = open('../datasets/YouTube_VOS2019/temporal_BIoU_c%2d_train.csv'%(i))
            csvreader = csv.reader(file)
            data_numpy = np.asarray([float(row[1]) for row in csvreader])
            file.close()

            bins = np.arange(0, 1.05, 0.05)   # fixed bin size
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.hist(data_numpy, bins=bins, alpha=0.5, facecolor='gray')
            x1, y1 = [], []
            for rect in ax.patches:
                x1 += [rect.get_x()+rect.get_width()/2]
                y1 += [float(rect.get_height())/1000.]
            x += [x1]
            y += [y1]
            plt.clf()

        line_types = ['-', '--', '-.', ':', '*-', '*--', '*-.',  '*:']
        fig, ax = plt.subplots(figsize=(6, 5))
        print('Plot figures!')
        for i, interval in enumerate(range(3, 16, 4)):
            precent = str(round(sum(y[i][15:])/sum(y[i])*100, 1))+'%'
            if i < 10:
                ax.plot(x[i], y[i], line_types[i], label=r'$\delta$=%1d, '%(interval//2)+precent, LineWidth=2)
                # ax.text(x[i][-3], y[i][-2], str(round(sum(y[i][15:])/sum(y[i])*100, 1))+'%')
            else:
                ax.plot(x[i], y[i], line_types[i], label=r'$\delta$=%2d, '%(interval//2)+precent, LineWidth=2)
        legend = ax.legend(loc='upper left', shadow=True, fontsize='medium')
        # Put a nicer background color on the legend.
        legend.get_frame().set_facecolor('white')
        plt.xlabel('Temporal Box IoU')
        plt.ylabel('Count (K)')
        plt.savefig('../datasets/YouTube_VOS2019/temporal_BIoU_train_lines.png')
        print('Finish All!')


def plt_masks_from_json(dataset, type='vis', anns=None, mask_alpha=0.45):
    # for datum in data_loader:
    n_frames = 1
    for vdx, vid in enumerate(dataset.vid_ids):
        if type == 'vis':
            len_vid = dataset.vid_infos[vdx]['length']
        else:
            len_vid = len(dataset.vid_infos[vdx])

        len_clips = (len_vid + n_frames-1) // n_frames
        progress_bar_clip = ProgressBar(len_clips, len_clips)
        for cdx in range(len_clips):
            progress_clip = (cdx + 1) / len_clips * 100
            progress_bar_clip.set_val(cdx+1)
            print('\rProcessing Clips of Video %s  %6d / %6d (%5.2f%%)     '
                  % (repr(progress_bar_clip), cdx+1, len_clips, progress_clip), end='')

            clip_frame_ids = range(cdx*n_frames, min((cdx+1)*n_frames, len_vid))
            imgs, imgs_meta, targets = dataset.pull_clip_from_video(vid, clip_frame_ids)

            batch_size = imgs.size(0)
            for i in range(batch_size):
                ori_h, ori_w = imgs_meta[i]['ori_shape'][:2]
                img_cur = imgs[i]
                gt_bboxes, gt_ids, gt_labels, gt_masks = [], [], [], []
                frame_id = clip_frame_ids[i]
                id = -1
                for ann in anns:
                    if ann['video_id'] == vid:
                        mask_rle = ann['segmentations'][frame_id]
                        if mask_rle is None:
                            mask_binary = torch.zeros(ori_h, ori_w)
                        else:
                            mask_binary = torch.tensor(mask_util.decode(mask_rle)).float()

                        gt_masks.append(mask_binary)
                        gt_labels.append(ann['category_id'])
                        id += 1
                        gt_ids.append(id)
                gt_ids_cur = gt_ids
                gt_masks_cur = torch.stack(gt_masks, dim=0).cuda()
                labels_cur = gt_labels

                n_cur = gt_masks_cur.size(0)
                if args.display_single_mask:
                    for j in range(n_cur):
                        colors = get_color(j, gt_ids_cur, on_gpu=imgs.device.index).view(1, 1, 3)
                        gt_masks_cur_color = gt_masks_cur[j].unsqueeze(-1) * colors * mask_alpha

                        if args.display_ori_shape:
                            gt_masks_cur_color = gt_masks_cur_color.permute(2, 0, 1).contiguous()
                            gt_masks_cur_color = F.interpolate(gt_masks_cur_color.unsqueeze(0), (ori_h, ori_w),
                                                               mode='bilinear',
                                                               align_corners=False).squeeze(0)
                            gt_masks_cur_color = gt_masks_cur_color.permute(1, 2, 0).contiguous()
                        img_numpy = gt_masks_cur_color.cpu().numpy()

                        plt.imshow(img_numpy)
                        plt.axis('off')
                        root_dir = os.path.join(args.save_path, 'out', str(vid))
                        if not os.path.exists(root_dir):
                            os.makedirs(root_dir)
                        plt.savefig(''.join([root_dir, '/', str(frame_id), '_', str(j), '_mask.png']))
                        plt.clf()

                else:
                    img_numpy = undo_image_transformation(img_cur, imgs_meta[i])
                    img_cur = torch.Tensor(img_numpy).cuda()  # [360, 640, 3]

                    gt_masks_cur = gt_masks_cur.unsqueeze(-1).repeat(1, 1, 1, 3)
                    # This is 1 everywhere except for 1-mask_alpha where the mask is
                    inv_alph_masks = gt_masks_cur.sum(0) * (-mask_alpha) + 1
                    gt_masks_cur_color = []

                    if args.display_masks:
                        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
                        for j in range(n_cur):
                            colors = get_color(j, gt_ids_cur, on_gpu=imgs.device.index).view(1, 1, 3)
                            gt_masks_cur_color.append(gt_masks_cur[j] * colors * mask_alpha)
                        gt_masks_cur_color = torch.stack(gt_masks_cur_color, dim=0).sum(0)

                        img_color = (img_cur * inv_alph_masks + gt_masks_cur_color)
                    else:
                        img_color = img_cur
                    img_numpy = img_color.cpu().numpy()

                    if args.display_bboxes:
                        bboxes_cur = gt_bboxes[i]
                        bboxes_cur = torch.clamp(bboxes_cur, min=1e-3, max=1-1e-3)
                        bboxes_cur[:, ::2] = bboxes_cur[:, ::2] * ori_w
                        bboxes_cur[:, 1::2] = bboxes_cur[:, 1::2] * ori_h
                        for j in reversed(range(n_cur)):
                            x1, y1, x2, y2 = bboxes_cur[j, :]
                            color = get_color(j, gt_ids_cur)
                            color = [c / 255. for c in color]
                            cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 2)

                            _class = cfg.classes[labels_cur[j] - 1]
                            text_str = '%s' % _class
                            font_face = cv2.FONT_HERSHEY_DUPLEX
                            font_scale = int(max(ori_h, ori_w) / 360.) * 0.5
                            font_thickness = 2
                            text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                            text_pt = (max(x1, 10), max(y1 - 3, 10))
                            text_color = [1, 1, 1]
                            cv2.rectangle(img_numpy, (max(x1, 10), max(y1, 10)), (x1 + text_w, y1 - text_h - 4), color, -1)
                            cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                                        cv2.LINE_AA)

                plt.imshow(img_numpy)
                plt.axis('off')
                plt.title(str(frame_id))
                root_dir = os.path.join(args.save_path, 'out', str(vid))
                if not os.path.exists(root_dir):
                    os.makedirs(root_dir)
                plt.savefig(''.join([root_dir, '/', str(frame_id), '.png']))
                plt.clf()


# Quick and dirty lambda for selecting the color for a particular index
# Also keeps track of a per-gpu color cache for maximum speed
def get_color(j, color_type, on_gpu=None, undo_transform=True):
    global color_cache
    color_idx = color_type[j] * 5 % len(cfg.COLORS)

    if on_gpu is not None and color_idx in color_cache[on_gpu]:
        return color_cache[on_gpu][color_idx]
    else:
        color = cfg.COLORS[color_idx]
        if not undo_transform:
            # The image might come in as RGB or BRG, depending
            color = (color[2], color[1], color[0])
        if on_gpu is not None:
            color = torch.Tensor(color).to(on_gpu).float() / 255.
            color_cache[on_gpu][color_idx] = color
        return color


# display motion path of instance in the video
def plot_path(mask_alpha=0.45):
    root_dir = os.path.join(args.save_path, 'out_path')
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    cfg.valid_sub_dataset.test_mode = False
    cfg.valid_sub_dataset.flip_ratio = 0
    train_dataset = get_dataset(cfg.valid_sub_dataset)
    data_loader = data.DataLoader(train_dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False,
                                  collate_fn=detection_collate_vis,
                                  pin_memory=True)

    print()
    video_cxy = None
    key_img = None
    n_max_inst = 0
    # try-except so you can use ctrl+c to save early and stop training
    # for datum in data_loader:
    for data_batch in data_loader:
        imgs, gt_bboxes, gt_labels, gt_masks, gt_ids, imgs_meta = prepare_data_vis(data_batch,
                                                                                   devices=torch.cuda.current_device())
        batch_size = imgs.size(0)
        for i in range(0, batch_size, 2):
            n_cur = gt_masks[i].size(0)
            gt_masks_cur = gt_masks[i]
            img_cur = imgs[i]
            gt_ids_cur = gt_ids[i]
            bboxes_cur = gt_bboxes[i]
            labels_cur = gt_labels[i]
            img_meta = imgs_meta[i]
            ori_h, ori_w = img_meta['ori_shape'][:2]
            img_h, img_w = img_meta['img_shape'][:2]
            pad_h, pad_w = img_meta['pad_shape'][:2]
            s_h, s_w = pad_h / img_h, pad_w / img_w

            if img_meta['is_first']:
                min_id = gt_ids_cur[0]
                if video_cxy is not None:
                    color_type = range(len(video_cxy))
                    # draw path of instances one-by-one
                    for k, cxy_list in video_cxy.items():
                        color = get_color(int(k), color_type)
                        color = [c / 255. for c in color]
                        # the first center should be a point, indicating the begining location
                        cv2.circle(key_img, (cxy_list[0][0], cxy_list[0][1]), 1, color, 4)
                        # link the locations from the last frame to the current frame by line with arrow
                        for cxy_idx, cxy in enumerate(cxy_list[1:]):
                            start_point = (int(cxy_list[cxy_idx][0].item()), int(cxy_list[cxy_idx][1].item()))
                            end_point = (int(cxy[0].item()), int(cxy[1].item()))
                            cv2.arrowedLine(key_img, start_point, end_point, color, 5, 8, 0, 0.2)

                    # plot the image with path of instances
                    video_id = img_meta['video_id'] - 1
                    plt.imshow(key_img)
                    plt.axis('off')
                    plt.title(str(video_id))
                    plt.savefig(''.join([root_dir, '/', str(video_id), '.png']))
                    plt.clf()

                # assign empty for the next video
                video_cxy = dict()
                key_img = None
                n_max_inst = 0
            gt_ids_cur = gt_ids_cur - min_id

            bboxes_cur[:, ::2] = bboxes_cur[:, ::2] * s_w
            bboxes_cur[:, 1::2] = bboxes_cur[:, 1::2] * s_h
            bboxes_cur = torch.clamp(bboxes_cur, min=0, max=1)
            bboxes_cur[:, ::2] = bboxes_cur[:, ::2] * ori_w
            bboxes_cur[:, 1::2] = bboxes_cur[:, 1::2] * ori_h
            cxy_bboxes_cur = center_size(bboxes_cur)[:, :2]
            for idx, id in enumerate(gt_ids_cur):
                k = id.item()
                if k not in video_cxy.keys():
                    video_cxy[k] = [cxy_bboxes_cur[idx]]
                else:
                    video_cxy[k].append(cxy_bboxes_cur[idx])

            if n_cur > n_max_inst:
                n_max_inst = n_cur
                img_numpy = undo_image_transformation(img_cur, img_meta)
                img_cur = torch.Tensor(img_numpy).cuda()  # [360, 640, 3]

                # Undo padding for masks
                gt_masks_cur = gt_masks_cur[:, :img_h, :img_w].float()
                gt_masks_cur = gt_masks_cur.unsqueeze(-1).repeat(1, 1, 1, 3)
                # This is 1 everywhere except for 1-mask_alpha where the mask is
                inv_alph_masks = gt_masks_cur.sum(0) * (-mask_alpha) + 1
                gt_masks_cur_color = []

                # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
                for j in range(n_cur):
                    colors = get_color(j, gt_ids_cur, on_gpu=imgs.device.index).view(1, 1, 3)
                    gt_masks_cur_color.append(gt_masks_cur[j] * colors * mask_alpha)
                gt_masks_cur_color = torch.stack(gt_masks_cur_color, dim=0).sum(0)

                img_color = (img_cur * inv_alph_masks + gt_masks_cur_color).permute(2, 0, 1).contiguous()
                img_color = F.interpolate(img_color.unsqueeze(0), (ori_h, ori_w), mode='bilinear',
                                          align_corners=False).squeeze(0)
                img_numpy = img_color.permute(1, 2, 0).contiguous().cpu().numpy()

                if args.display_bboxes:
                    for j in range(n_cur):
                        x1, y1, x2, y2 = bboxes_cur[j, :]
                        color = get_color(j, gt_ids_cur)
                        color = [c / 255. for c in color]
                        cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 2)

                        _class = cfg.classes[labels_cur[j] - 1]
                        text_str = '%s' % _class
                        font_face = cv2.FONT_HERSHEY_DUPLEX
                        font_scale = int(max(ori_h, ori_w) / 360.) * 0.5
                        font_thickness = 2
                        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                        text_pt = (max(x1, 10), max(y1 - 3, 10))
                        text_color = [1, 1, 1]
                        cv2.rectangle(img_numpy, (max(x1, 10), max(y1, 10)), (x1 + text_w, y1 - text_h - 4), color, -1)
                        cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                                    cv2.LINE_AA)

                key_img = img_numpy


if __name__ == '__main__':
    cfg = load_config(args.config)
    type = 0
    if type == 0:
        count_temporal_BIoU(cfg)
    else:
        if args.data_type == 'valid_sub':
            dataset = get_dataset(cfg.DATASETS.TYPE, cfg.DATASETS.VALID_SUB, cfg.INPUT, cfg.SOLVER.NUM_CLIP_FRAMES,
                                  inference=True)
        else:
            # display predicted masks on valid data
            dataset = get_dataset(cfg.DATASETS.TYPE, cfg.DATASETS.VALID, cfg.INPUT, cfg.SOLVER.NUM_CLIP_FRAMES,
                                  inference=True)

        # path of the .json file that stores your predicted masks
        ann_file = '/data/VIS/VisTR_OVIS/weights/r50/results.json'
        anns = json.load(open(ann_file, 'r'))
        plt_masks_from_json(dataset, anns=anns)
