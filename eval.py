from datasets import *
from STMask import STMask
from utils.functions import MovingAverage, ProgressBar
from utils import timer
from utils.functions import SavePath
from layers.utils.output_utils import postprocess_ytbvis, undo_image_transformation
from layers.visualization_temporal import draw_dotted_rectangle, get_color

from datasets import prepare_data_vis, prepare_data_coco
import mmcv
import math
import torch
import torch.backends.cudnn as cudnn
import argparse
import random
import numpy as np
import os
import json
from layers.utils.eval_utils import bbox2result_video, calc_metrics

import matplotlib.pyplot as plt
import cv2
from torch.utils.tensorboard import SummaryWriter


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--epoch', default=0, type=int)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for training')
    parser.add_argument('--trained_model',
                        default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--eval_data', default='valid', type=str, help='data type')
    parser.add_argument('--overlap_frames', default=0, type=int, help='the overlapped frames between two video clips')
    parser.add_argument('--top_k', default=100, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--display_single_mask', default=False, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_single_mask_occlusion', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display_fpn_outs', default=True, type=str2bool,
                        help='Whether or not to display outputs after fpn')
    parser.add_argument('--display', default=False, type=str2bool,
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--save_folder', default='results/', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--eval_dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
    parser.add_argument('--eval_types', default=['segm'], type=str, nargs='+', choices=['bbox', 'segm'], help='eval types')
    parser.set_defaults(display=False, resume=False, detect=False)

    global args
    args = parser.parse_args(argv)


def prep_display(dets_out, img, img_meta=None, undo_transform=True, mask_alpha=0.45):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    -- display_model: 'train', 'test', 'None' means groundtruth results
    """
    ori_h, ori_w, _ = img_meta['ori_shape']
    img_h, img_w, _ = img_meta['img_shape']
    if undo_transform:
        img_numpy = undo_image_transformation(img, ori_h, ori_w, img_h, img_w)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0

    with timer.env('Postprocess'):
        dets_out = postprocess_ytbvis(dets_out, img_meta,
                                      visualize_lincomb=args.display_lincomb,
                                      output_file=args.save_folder)
        torch.cuda.synchronize()

    scores = dets_out['score'][:args.top_k].view(-1).detach().cpu().numpy()
    boxes = dets_out['box'][:args.top_k].detach().cpu().numpy()
    classes = dets_out['class'][:args.top_k].view(-1).detach().cpu().numpy()
    num_dets_to_consider = min(args.top_k, classes.shape[0])
    color_type = dets_out['box_ids'].view(-1)
    masks = dets_out['mask'][:args.top_k]
    centerness = dets_out['centerness'][:args.top_k].view(-1).detach().cpu().numpy() if 'centerness' in dets_out.keys() else None
    num_tracked_mask = dets_out['tracked_mask'] if 'tracked_mask' in dets_out.keys() else None

    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break

    if num_dets_to_consider == 0:
        # No detections found so just output the original image
        return (img_gpu * 255).byte().cpu().numpy()

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if args.display_masks and cfg.eval_mask_branch:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat(
            [get_color(j, color_type, on_gpu=img_gpu.device.index, undo_transform=undo_transform).view(1, 1, 1, 3)
             for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)
        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if args.display_text or args.display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            # get the bbox_idx to know box's layers (after FPN): p3-p7
            # box_idx = dets_out['bbox_idx'][j]
            # p_nums = [34560, 43200, 45360, 45900, 46035]
            # p_nums = [11520, 14400, 15120, 15300, 15345]
            # p = 0
            # for i in range(len(p_nums)):
            #     if box_idx < p_nums[i]:
            #         p = i + 3
            #         break

            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j, color_type)
            score = scores[j]
            num_tracked_mask_j = num_tracked_mask[j] if num_tracked_mask is not None else None

            if args.display_bboxes:
                if num_tracked_mask_j == 0 or num_tracked_mask_j is None:
                    cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 3)
                else:
                    draw_dotted_rectangle(img_numpy, x1, y1, x2, y2, color, 3, gap=10)

            if args.display_text:
                if classes[j] - 1 < 0:
                    _class = 'None'
                else:
                    _class = cfg.classes[classes[j] - 1]

                if score is not None:
                    # if cfg.use_maskiou and not cfg.rescore_bbox:
                    if centerness is not None:
                        rescore = centerness[j] * score
                        text_str = '%s: %.2f: %.2f: %s' % (_class, score, rescore, str(color_type[j].cpu().numpy())) \
                            if args.display_scores else _class
                    else:

                        text_str = '%s: %.2f: %s' % (
                            _class, score, str(color_type[j].cpu().numpy())) if args.display_scores else _class
                else:
                    text_str = '%s' % _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 2
                font_thickness = 2

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (max(x1, 10), max(y1 - 3, 10))
                text_color = [255, 255, 255]
                cv2.rectangle(img_numpy, (max(x1, 10), max(y1, 10)), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                            cv2.LINE_AA)

    return img_numpy


def prep_display_single(dets_out, img, img_meta=None, undo_transform=True, mask_alpha=0.45,
                        fps_str='', display_mode=None):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    -- display_model: 'train', 'test', 'None' means groundtruth results
    """
    ori_h, ori_w, _ = img_meta['ori_shape']
    img_h, img_w, _ = img_meta['img_shape']

    if undo_transform:
        img_numpy = undo_image_transformation(img, ori_h, ori_w, img_h, img_w)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0

    with timer.env('Postprocess'):
        cfg.mask_proto_debug = args.mask_proto_debug
        cfg.preserve_aspect_ratio = False
        dets_out = postprocess_ytbvis(dets_out, img_meta,
                                      visualize_lincomb=args.display_lincomb,
                                      crop_masks=args.crop,
                                      output_file=args.mask_det_file[:-12])
        torch.cuda.synchronize()
        scores = dets_out['score'][:args.top_k].detach().cpu().numpy()
        boxes = dets_out['box'][:args.top_k].detach().cpu().numpy()
        masks = dets_out['mask'][:args.top_k]


    classes = dets_out['class'][:args.top_k].detach().cpu().numpy()

    num_dets_to_consider = min(args.top_k, classes.shape[0])
    color_type = dets_out['box_ids']
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break

    if num_dets_to_consider == 0:
        # No detections found so just output the original image
        return (img_gpu * 255).byte().cpu().numpy()

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if args.display_masks and cfg.eval_mask_branch:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat(
            [get_color(j, color_type, on_gpu=img_gpu.device.index, undo_transform=undo_transform).view(1, 1, 1, 3)
             for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)
        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    if args.display_fps:
        # Draw the box for the fps on the GPU
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

        img_gpu[0:text_h + 8, 0:text_w + 8] *= 0.6  # 1 - Box alpha

    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if args.display_fps:
        # Draw the text on the CPU
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    if args.display_text or args.display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j, color_type)
            # plot priors
            priors = dets_out['priors'].detach().cpu().numpy()
            if j < dets_out['priors'].size(0):
                cpx, cpy, pw, ph = priors[j, :] * [img_w, img_h, img_w, img_h]
                px1, py1 = cpx - pw / 2.0, cpy - ph / 2.0
                px2, py2 = cpx + pw / 2.0, cpy + ph / 2.0
                px1, py1, px2, py2 = int(px1), int(py1), int(px2), int(py2)
                pcolor = [255, 0, 255]

            # plot the range of features for classification and regression
            pred_scales = [24, 48, 96, 192, 384]
            x = torch.clamp(torch.tensor([x1, x2]), min=2, max=638).tolist(),
            y = torch.clamp(torch.tensor([y1, y2]), min=2, max=358).tolist(),
            x, y = x[0], y[0]

            if display_mode is not None:
                score = scores[j]

            font_thickness = 0.5*max(min(ori_h, ori_w)//360, 1)

            if args.display_bboxes:
                cv2.rectangle(img_numpy, (x[0], y[0]), (x[1], y[1]), color, 1)
                if j < dets_out['priors'].size(0):
                    cv2.rectangle(img_numpy, (px1, py1), (px2, py2), pcolor, font_thickness, lineType=8)
                # cv2.rectangle(img_numpy, (x[4], y[4]), (x[5], y[5]), fcolor, 2)

            if args.display_text:
                if classes[j] - 1 < 0:
                    _class = 'None'
                else:
                    _class = cfg.classes[classes[j] - 1]

                if display_mode == 'test':
                    # if cfg.use_maskiou and not cfg.rescore_bbox:
                    train_centerness = False
                    if train_centerness:
                        rescore = dets_out['DIoU_score'][j] * score
                        text_str = '%s: %.2f: %.2f: %s' % (_class, score, rescore, str(color_type[j].cpu().numpy())) \
                            if args.display_scores else _class
                    else:

                        text_str = '%s: %.2f: %s' % (
                            _class, score, str(color_type[j].cpu().numpy())) if args.display_scores else _class
                else:
                    text_str = '%s' % _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.5

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]
                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                            cv2.LINE_AA)

    return img_numpy


def evaluate(net: STMask, dataset, ann_file=None, epoch=-1):
    eval_clip_frames = cfg.eval_clip_frames
    n_overlapped_frames = args.overlap_frames
    n_newly_frames = eval_clip_frames - n_overlapped_frames

    dataset_size = len(dataset.vid_ids)
    progress_bar = ProgressBar(dataset_size, dataset_size)

    print()

    json_results = []

    timer.reset()
    for vdx, vid in enumerate(dataset.vid_ids):
        results_video = {}

        len_vid = dataset.vid_infos[vdx]['length']
        if eval_clip_frames == 1:
            len_clips = (len_vid + args.batch_size//2) // args.batch_size
        else:
            len_clips = (len_vid + n_overlapped_frames) // n_newly_frames
        for cdx in range(len_clips):
            with timer.env('Load Data'):
                if eval_clip_frames > 1:
                    left = cdx * n_newly_frames
                    clip_frame_ids = range(left, min(left+eval_clip_frames, len_vid))
                else:
                    clip_frame_ids = range(cdx*args.batch_size, min((cdx+1)*args.batch_size, len_vid))

                print('Process video id: ', vid, 'frames: ', clip_frame_ids)
                images, images_meta, targets = dataset.pull_clip_from_video(vid, clip_frame_ids)
                images = torch.stack([torch.from_numpy(img).permute(2, 0, 1) for img in images], dim=0).cuda()

            with timer.env('Network Extra'):
                preds = net(images, img_meta=images_meta)

            # Remove overlapped frames
            if eval_clip_frames > 1 and cdx > 0:
                clip_frame_ids = clip_frame_ids[n_overlapped_frames:]
                images = images[n_overlapped_frames:]
                images_meta = images_meta[n_overlapped_frames:]
                preds = preds[n_overlapped_frames:]

            for batch_id, pred in enumerate(preds):

                if args.display:
                    img_id = (vid, clip_frame_ids[batch_id])
                    root_dir = os.path.join(args.save_folder, 'out', str(vid))
                    if not os.path.exists(root_dir):
                        os.makedirs(root_dir)
                    if not args.display_single_mask:
                        img_numpy = prep_display(pred, images[batch_id],
                                                 img_meta=images_meta[batch_id])
                        plt.imshow(img_numpy)
                        plt.axis('off')
                        plt.title(str(img_id))
                        plt.savefig(''.join([root_dir, '/', str(img_id[1]), '.png']))
                        plt.clf()

                    else:
                        for p in range(pred['box'].size(0)):
                            pred_single = {'proto': pred['proto']}
                            for k, v in pred.items():
                                if k not in {'proto'}:
                                    pred_single[k] = v[p].unsqueeze(0)

                            img_numpy = prep_display(pred_single, images[batch_id],
                                                     img_meta=images_meta[batch_id])
                            plt.imshow(img_numpy)
                            plt.axis('off')
                            plt.savefig(''.join([root_dir, '/', str(img_id[1]), '_', str(p), '.png']))
                            plt.clf()

                else:
                    if pred is not None:
                        pred = postprocess_ytbvis(pred, images_meta[batch_id],
                                                  output_file=args.save_folder)
                    bbox2result_video(results_video, pred, clip_frame_ids[batch_id], types=args.eval_types)

        if not args.display:
            for obj_id, result_obj in results_video.items():
                result_obj['video_id'] = vid
                result_obj['score'] = np.array(result_obj['score']).mean().item()
                result_obj['category_id'] = np.bincount(result_obj['category_id']).argmax().item()
                json_results.append(result_obj)

        progress = (vdx + 1) / dataset_size * 100
        progress_bar.set_val(vdx + 1)
        print('\rProcessing Images  %s %6d / %6d (%5.2f%%)      '
              % (repr(progress_bar), vdx+1, dataset_size, progress), end='')

    if not args.display:
        if epoch >= 0:
            json_path = os.path.join(args.save_folder, 'results_' + str(epoch) + '.json')
        else:
            json_path = os.path.join(args.save_folder, 'results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f)
        if ann_file is not None:
            metric_path = os.path.join(args.save_folder, str(epoch) + '.txt')
            calc_metrics(ann_file, json_path, output_file=metric_path)

    print('Finish inference.')


def evaluate_clip(net: STMask, dataset, ann_file=None, epoch=-1):
    clip_frames = cfg.train_dataset.clip_frames
    n_newly_frames = clip_frames - args.overlap_frames
    dataset_size = len(dataset.vid_ids)
    # progress_bar = ProgressBar(dataset_size, dataset_size)

    print()
    json_results = []

    # Main eval loop
    timer.reset()
    for vdx, vid in enumerate(dataset.vid_ids):
        progress = (vdx + 1) / dataset_size * 100
        # progress_bar.set_val(vdx+1)
        print()
        print('Processing Videos:  %2d / %2d (%5.2f%%) ' % (vdx+1, dataset_size, progress))

        vid_objs = {}
        len_vid = dataset.vid_infos[vdx]['length']
        len_clips = (len_vid + n_newly_frames//2) // n_newly_frames
        progress_bar_clip = ProgressBar(len_clips, len_clips)
        for cdx in range(len_clips):
            progress_clip = (cdx + 1) / len_clips * 100
            progress_bar_clip.set_val(cdx+1)
            print('\rProcessing Clips of Video %s  %6d  %6d / %6d (%5.2f%%)     '
              % (repr(progress_bar_clip), vid, cdx+1, len_clips, progress_clip), end='')

            with timer.env('Load Data'):
                left = cdx * n_newly_frames
                clip_frame_ids = range(left, min(left+clip_frames, len_vid))
                # clip_frame_ids = range(cdx*clip_frames, min((cdx+1)*clip_frames, len_vid))
                images, images_meta, targets = dataset.pull_clip_from_video(vid, clip_frame_ids)
                images = torch.stack([torch.from_numpy(img).permute(2, 0, 1) for img in images], dim=0).cuda()
                if images.size(0) < clip_frames:
                    images = torch.cat([images, images[:clip_frames-images.size(0)]], dim=0)
                    images_meta += images_meta[:clip_frames-images.size(0)]

            with timer.env('Network Extra'):
                preds_clip = net(images, img_meta=images_meta)
                pred_clip = preds_clip[0]

            pred_frame = dict()
            for k, v in pred_clip.items():
                if k in {'score', 'class', 'mask_coeff', 'box_ids'}:
                    pred_frame[k] = v

            for batch_id, frame_id in enumerate(clip_frame_ids[:n_newly_frames]):
                img_id = (vid, frame_id)
                pred_frame['proto'] = pred_clip['proto'][batch_id]
                if pred_clip['box'].size(0) == 0:
                    pred_frame['mask'] = torch.Tensor()
                    pred_frame['box'] = torch.Tensor()
                else:
                    pred_frame['mask'] = pred_clip['mask'][..., batch_id]
                    pred_frame['box'] = pred_clip['box'][:, batch_id * 4:(batch_id + 1) * 4]

                if args.display:
                    root_dir = os.path.join(args.save_folder, 'out', str(vid))
                    if not os.path.exists(root_dir):
                        os.makedirs(root_dir)
                    img_numpy = prep_display(pred_frame, images[batch_id],
                                             img_meta=images_meta[batch_id])
                    plt.imshow(img_numpy)
                    plt.axis('off')
                    plt.title(str(img_id))
                    plt.savefig(''.join([root_dir, '/', str(frame_id), '.png']))
                    plt.clf()

                else:
                    preds_cur = postprocess_ytbvis(pred_frame, images_meta[batch_id],
                                                   output_file=args.save_folder)
                    bbox2result_video(vid_objs, preds_cur, frame_id, types=args.eval_types)

        if not args.display:
            for obj_id, vid_obj in vid_objs.items():
                vid_obj['video_id'] = vid
                vid_obj['score'] = np.array(vid_obj['score']).mean().item()
                vid_obj['category_id'] = np.bincount(vid_obj['category_id']).argmax().item()
                json_results.append(vid_obj)

    if not args.display:
        json_path = os.path.join(args.save_folder, 'results_'+str(epoch)+'.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f)
        if ann_file is not None:
            metric_path = os.path.join(args.save_folder, str(epoch)+'.txt')
            calc_metrics(ann_file, json_path, output_file=metric_path)

    print('Finish inference.')


def evaluate_single(net: STMask, im_path=None, save_path=None, idx=None):
    im = mmcv.imread(im_path)
    ori_shape = im.shape
    im, w_scale, h_scale = mmcv.imresize(im, (640, 360), return_scale=True)
    img_shape = im.shape

    if cfg.backbone.transform.normalize:
        im = (im - MEANS) / STD
    elif cfg.backbone.transform.subtract_means:
        im = (im - MEANS)
    elif cfg.backbone.transform.to_float:
        im = im / 255.
    im = mmcv.impad_to_multiple(im, 32)
    pad_shape = im.shape
    im = torch.tensor(im).permute(2, 0, 1).contiguous().unsqueeze(0).float().cuda()
    pad_h, pad_w = im.size()[2:4]
    img_meta = {'ori_shape': ori_shape, 'img_shape': img_shape, 'pad_shape': pad_shape}
    if idx is not None:
        img_meta['frame_id'] = idx
    if idx is None or idx == 0:
        img_meta['is_first'] = True
    else:
        img_meta['is_first'] = False

    preds = net(im, img_meta=[img_meta])
    preds[0]['detection']['box_ids'] = torch.arange(preds[0]['detection']['box'].size(0))
    cfg.preserve_aspect_ratio = True
    img_numpy = prep_display(preds[0], im[0], pad_h, pad_w, img_meta=img_meta)
    if save_path is None:
        plt.imshow(img_numpy)
        plt.axis('off')
        plt.show()
    else:
        cv2.imwrite(save_path, img_numpy)


def evalimages(net: STMask, input_folder: str, output_folder: str):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    print()
    path_list = os.listdir(input_folder)
    path_list.sort(key=lambda x:int(x[:-4]))
    for idx, p in enumerate(path_list):
        path = str(p)
        name = os.path.basename(path)
        name = '.'.join(name.split('.')[:-1]) + '.png'
        out_path = os.path.join(output_folder, name)
        in_path = os.path.join(input_folder, path)

        evaluate_single(net, in_path, out_path, idx)
        print(path + ' -> ' + out_path)
    print('Done.')


def evalvideo(net: STMask, input_folder: str, output_folder: str):
    return


if __name__ == '__main__':
    parse_args()
    ann_file = None

    if args.config is not None:
        set_cfg(args.config)

    if args.trained_model == 'interrupt':
        args.trained_model = SavePath.get_interrupt('weights/')
    elif args.trained_model == 'latest':
        args.trained_model = SavePath.get_latest('weights/', cfg.name)

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    if args.detect:
        cfg.eval_mask_branch = False

    if args.image is None and args.images is None:
        if args.eval_dataset is not None:
            set_dataset(args.eval_dataset, 'eval')

        if args.eval_data == 'train':
            print('load train_sub dataset')
            cfg.train_dataset.has_gt = False
            val_dataset = get_dataset('vis', cfg.train_dataset, cfg.backbone.transform)
            ann_file = cfg.train_dataset.ann_file
        elif args.eval_data == 'valid_sub':
            print('load valid_sub dataset')
            cfg.valid_sub_dataset.has_gt = False
            val_dataset = get_dataset('vis', cfg.valid_sub_dataset, cfg.backbone.transform)
            ann_file = cfg.valid_sub_dataset.ann_file

        elif args.eval_data == 'test':
            print('load test dataset')
            cfg.test_dataset.has_gt = False
            val_dataset = get_dataset('vis', cfg.test_dataset, cfg.backbone.transform)
        elif args.eval_data == 'valid':
            print('load valid dataset')
            val_dataset = get_dataset('vis', cfg.valid_dataset, cfg.backbone.transform)
    else:
        val_dataset = None

    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        if args.cuda:
            cudnn.benchmark = True
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        print('Loading model...', end='')
        net = STMask()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')

        if args.cuda:
            net = net.cuda()

        if args.image is not None:
            save_path = os.path.join(os.path.split(args.image[0]), 'out')
            evaluate_single(net, args.image, save_path)

        elif args.images is not None:
            if ':' in args.images:
                inp, out = args.images.split(':')
                evalimages(net, inp, out)
            else:
                out = args.images + '_out'
                evalimages(net, args.images, out)

        elif args.video is not None:
            if ':' in args.video:
                inp, out = args.video.split(':')
                evalvideo(net, inp, out)
            else:
                evalvideo(net, args.video)

        else:
            if args.eval_data == 'metric':
                print('calculate evaluation metrics ...')
                ann_file = cfg.valid_sub_dataset.ann_file
                dt_file = args.save_folder + 'results.json'
                print('det_file:', dt_file)
                metrics = calc_metrics(ann_file, dt_file)
                metrics_name = ['mAP', 'AP50', 'AP75', 'small', 'medium', 'large',
                                'AR1', 'AR10', 'AR100', 'AR100_small', 'AR100_medium', 'AR100_large']
                log_dir = 'weights/temp/train_log'
                writer = SummaryWriter(log_dir=log_dir, comment='_scalars', filename_suffix='VIS')
                for i_m in range(len(metrics_name)):
                    writer.add_scalar('valid_metrics/' + metrics_name[i_m], metrics[i_m], 1)
            else:
                if cfg.train_track and cfg.clip_prediction_module:
                    evaluate_clip(net, val_dataset, ann_file=ann_file, epoch=args.epoch)
                else:
                    evaluate(net, val_dataset, ann_file=ann_file, epoch=args.epoch)


