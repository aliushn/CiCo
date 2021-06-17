from datasets import *
import os
import cv2
import torch
import json
from datasets import get_dataset, prepare_data
import torch.utils.data as data
import argparse
import torch.nn.functional as F
import pycocotools.mask as mask_util

from collections import defaultdict
import matplotlib.pyplot as plt
from layers.utils import undo_image_transformation, center_size


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Yolact Training Script')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--display_ori_shape', default=True, type=str2bool,
                    help='display original shape of images')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--config', default='STMask_plus_base_config',
                    help='The config object to use.')
parser.add_argument('--save_path', default='results/eval_mask_detections.json', type=str,
                    help='The output file for coco mask results if --coco_results is set.')
parser.add_argument('--display_bboxes', default=True, type=str2bool,
                    help='Whether or not to display bboxes around masks')

parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True)
args = parser.parse_args()

if args.config is not None:
    set_cfg(args.config)

color_cache = defaultdict(lambda: {})


def plt_masks_from_json(data_loader, type='gt', anns=None, mask_alpha=0.45):
    # for datum in data_loader:
    for data_batch in data_loader:

        if type == 'gt':
            imgs, gt_bboxes, gt_labels, gt_masks, gt_ids, imgs_meta = prepare_data(data_batch,
                                                                                   devices=torch.cuda.current_device())
        else:
            assert anns is not None
            imgs, imgs_meta = prepare_data(data_batch, train_mode=False, devices=torch.cuda.current_device())

        batch_size = imgs.size(0)
        for i in range(0, batch_size, 2):
            img_meta = imgs_meta[i]
            ori_h, ori_w = img_meta['ori_shape'][:2]
            img_h, img_w = img_meta['img_shape'][:2]

            img_cur = imgs[i]
            if type == 'gt':
                gt_masks_cur = gt_masks[i]
                gt_ids_cur = gt_ids[i]
                labels_cur = gt_labels[i]
                if img_meta['is_first']:
                    min_id = gt_ids_cur[0]
                gt_ids_cur = gt_ids_cur - min_id

            else:

                gt_bboxes, gt_ids, gt_labels, gt_masks = [], [], [], []
                frame_id = img_meta['frame_id']
                id = -1
                for ann in anns:
                    if ann['video_id'] - 1 == img_meta['video_id']:
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
                gt_masks_cur = F.interpolate(gt_masks_cur.unsqueeze(0), (img_h, img_w)).squeeze(0)
                labels_cur = gt_labels

            n_cur = gt_masks_cur.size(0)
            img_numpy = undo_image_transformation(img_cur, img_meta)
            img_cur = torch.Tensor(img_numpy).cuda()  # [360, 640, 3]

            # Undo padding for masks
            # gt_masks_cur = gt_masks_cur[:, :img_h, :img_w].float()
            gt_masks_cur = gt_masks_cur.unsqueeze(-1).repeat(1, 1, 1, 3)
            # This is 1 everywhere except for 1-mask_alpha where the mask is
            inv_alph_masks = gt_masks_cur.sum(0) * (-mask_alpha) + 1
            gt_masks_cur_color = []

            # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
            for j in range(n_cur):
                colors = get_color(j, gt_ids_cur, on_gpu=imgs.device.index).view(1, 1, 3)
                gt_masks_cur_color.append(gt_masks_cur[j] * colors * mask_alpha)
            gt_masks_cur_color = torch.stack(gt_masks_cur_color, dim=0).sum(0)

            img_color = (img_cur * inv_alph_masks + gt_masks_cur_color)
            if args.display_ori_shape:
                img_color = img_color.permute(2, 0, 1).contiguous()
                img_color = F.interpolate(img_color.unsqueeze(0), (ori_h, ori_w), mode='bilinear',
                                          align_corners=False).squeeze(0)
                img_color = img_color.permute(1, 2, 0).contiguous()
            img_numpy = img_color.cpu().numpy()

            if type == 'gt' and args.display_bboxes:
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

            video_id, frame_id = imgs_meta[i]['video_id'], imgs_meta[i]['frame_id']
            plt.imshow(img_numpy)
            plt.axis('off')
            plt.title(str(frame_id))

            root_dir = os.path.join(args.save_path, 'out', str(video_id))
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
        imgs, gt_bboxes, gt_labels, gt_masks, gt_ids, imgs_meta = prepare_data(data_batch,
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
    type = 'pred'

    if type == 'gt':
        # display GT masks on valid_sub data
        cfg.valid_sub_dataset.test_mode = True
        cfg.valid_sub_dataset.flip_ratio = 0
        dataset = get_dataset(cfg.valid_sub_dataset)
        data_loader = data.DataLoader(dataset, args.batch_size,
                                      shuffle=False,
                                      collate_fn=detection_collate_vis,
                                      pin_memory=True)

        plt_masks_from_json(data_loader, type)

    else:
        # display predicted masks on valid data
        cfg.valid_dataset.test_mode = True
        cfg.valid_dataset.flip_ratio = 0
        dataset = get_dataset(cfg.valid_dataset)
        data_loader = data.DataLoader(dataset, args.batch_size,
                                      shuffle=False,
                                      collate_fn=detection_collate_vis,
                                      pin_memory=True)

        # path of the .json file that stores your predicted masks
        ann_file = '../STMask/weights/YTVIS2019/weights_r50_new/results.json'
        anns = json.load(open(ann_file, 'r'))
        plt_masks_from_json(data_loader, type, anns=anns)
