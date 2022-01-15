from datasets import *
from CoreNet import CoreNet
from utils.functions import MovingAverage, ProgressBar
from utils import timer
from utils.functions import SavePath
from layers.utils.output_utils import postprocess_ytbvis, undo_image_transformation
from layers.visualization_temporal import draw_dotted_rectangle, get_color
from configs.load_config import load_config

import time
import mmcv
import torch
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import os
import json
from layers.utils.eval_utils import bbox2result_video, calc_metrics
from configs._base_.datasets import get_dataset_config

import matplotlib.pyplot as plt
import cv2
import random


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='STMask Evaluation in video domain')
    parser.add_argument('--epoch', default=0, type=int)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for training')
    parser.add_argument('--trained_model',
                        default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--eval_data', default='valid', type=str, help='data type')
    parser.add_argument('--eval', default=True, type=str2bool,
                        help='False, only calculate metrics between ground truth annotations and prediction json file.')
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
    parser.add_argument('--display_bboxes_cir', default=False, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display_fpn_outs', default=True, type=str2bool,
                        help='Whether or not to display outputs after fpn')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--save_folder', default='weights/prototypes/', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--display_lincomb', dest='display_lincomb', action='store_true',
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--eval_types', default=['segm'], type=str, nargs='+', choices=['bbox', 'segm'], help='eval types')
    parser.set_defaults(display=False, resume=False, detect=False)

    global args
    args = parser.parse_args(argv)


def prep_display(dets_out, img, img_meta=None, undo_transform=True, mask_alpha=0.55):
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

    num_dets_to_consider = min(args.top_k, dets_out['box'].size(0))
    scores = dets_out['score'][:args.top_k].view(-1).detach().cpu().numpy()
    boxes = dets_out['box'][:args.top_k].detach().cpu().numpy()
    color_type = dets_out['box_ids'].view(-1)
    num_tracked_mask = dets_out['tracked_mask'] if 'tracked_mask' in dets_out.keys() else None
    boxes_cir = dets_out['box_cir'] if 'box_cir' in dets_out.keys() else None
    if 'mask' in dets_out.keys():
        masks = dets_out['mask'][:args.top_k]

    if num_dets_to_consider == 0:
        # No detections found so just output the original image
        return (img_gpu * 255).byte().cpu().numpy()

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if args.display_masks:
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
            color = get_color(j, color_type)
            # get the bbox_idx to know box's layers (after FPN): p3-p7
            # if 'priors' in dets_out.keys():
            #     px1, py1, px2, py2 = dets_out['priors'][j, :]
            #     draw_dotted_rectangle(img_numpy, px1, py1, px2, py2, color, 3, gap=10)

            if args.display_bboxes_cir and boxes_cir is not None:
                x1, y1, x2, y2 = boxes_cir[j, :]
            else:
                x1, y1, x2, y2 = boxes[j, :]
            score = scores[j]
            num_tracked_mask_j = num_tracked_mask[j] if num_tracked_mask is not None else None

            if args.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 3)
                if num_tracked_mask_j == 0 or num_tracked_mask_j is None:
                    cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 3)
                else:
                    draw_dotted_rectangle(img_numpy, x1, y1, x2, y2, color, 3, gap=10)

            if args.display_text:
                classes = dets_out['class'][:args.top_k].view(-1).detach().cpu().numpy()
                if classes[j] - 1 < 0:
                    _class = 'None'
                else:
                    _class = class_names[classes[j] - 1]

                if score is not None:
                    if 'score_global' in dets_out.keys():
                        rescore = dets_out['score_global'][:args.top_k].view(-1).detach().cpu().numpy()[j]
                        text_str = '%s: %.2f: %.2f: %s' % (_class, score, rescore, str(color_type[j].cpu().numpy())) \
                            if args.display_scores else _class
                    # if 'centerness' in dets_out.keys():
                    #     rescore = dets_out['centerness'][:args.top_k].view(-1).detach().cpu().numpy()[j]
                    #     text_str = '%s: %.2f: %.2f: %s' % (_class, score, rescore, str(color_type[j].cpu().numpy())) \
                    #         if args.display_scores else _class
                    else:
                        text_str = '%s: %.2f: %s' % (
                            _class, score, str(color_type[j].cpu().numpy())) if args.display_scores else _class
                else:
                    text_str = '%s' % _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.8 * max((max(img_numpy.shape) // 640), 1)
                font_thickness = max((max(img_numpy.shape) // 640), 1)

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (max(x1, 0), max(y1 - 3, 25))
                text_color = [255, 255, 255]
                x1, y1 = max(x1, 0), max(y1, 30)
                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w+2, y1 - text_h - 4), color, -1)
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
        dets_out = postprocess_ytbvis(dets_out, img_meta,
                                      visualize_lincomb=args.display_lincomb,
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
    if args.display_masks:
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
                    _class = class_names[classes[j] - 1]

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


def evaluate(net: CoreNet, dataset, data_type='vis', pad_h=None, pad_w=None, eval_clip_frames=1, output_dir=None):
    '''
    :param net:
    :param dataset:
    :param data_type: 'vis' or 'vid'
    :param eval_clip_frames:
    :param output_dir:
    :return:
    '''
    n_overlapped_frames = args.overlap_frames
    n_newly_frames = eval_clip_frames - n_overlapped_frames
    dataset_size = len(dataset.vid_ids)

    print()
    json_results = []
    frame_times = MovingAverage()
    for vdx, vid in enumerate(dataset.vid_ids):
        progress = (vdx + 1) / dataset_size * 100
        print()
        print('Processing Videos:  %2d / %2d (%5.2f%%) ' % (vdx+1, dataset_size, progress))
        results_video = {}

        if data_type == 'vis':
            len_vid = dataset.vid_infos[vdx]['length']
            train_masks, use_vid_metric = True, False
        else:
            len_vid = len(dataset.vid_infos[vdx])
            train_masks, use_vid_metric = False, True
        if eval_clip_frames == 1:
            len_clips = (len_vid + args.batch_size-1) // args.batch_size
        else:
            len_clips = (len_vid + n_newly_frames-1) // n_newly_frames
        progress_bar_clip = ProgressBar(len_clips, len_clips)
        for cdx in range(len_clips):
            progress_clip = (cdx + 1) / len_clips * 100
            progress_bar_clip.set_val(cdx+1)

            timer.reset()
            with timer.env('Load Data'):
                if eval_clip_frames == 1:
                    clip_frame_ids = range(cdx*args.batch_size, min((cdx+1)*args.batch_size, len_vid))
                else:
                    left = cdx * n_newly_frames
                    clip_frame_ids = range(left, min(left+eval_clip_frames, len_vid))
                if cdx == len_clips-1:
                    assert clip_frame_ids[-1] == len_vid-1, \
                        'The last clip should include the last frame! Please double check!'
                images, images_meta, targets = dataset.pull_clip_from_video(vid, clip_frame_ids)
                images = [torch.from_numpy(img).permute(2, 0, 1) for img in images]
                images = ImageList_from_tensors(images, size_divisibility=32, pad_h=pad_h, pad_w=pad_w).cuda()
                pad_shape = {'pad_shape': (images.size(-2), images.size(-1), 3)}
                for k in range(len(images_meta)):
                    images_meta[k].update(pad_shape)

            preds = net(images, img_meta=images_meta)
            frame_times.add(timer.total_time())
            progress_bar_clip.set_val(cdx+1)
            avg_fps = 1. / (frame_times.get_avg() / args.batch_size) if vdx > 0 or cdx > 0 else 0
            print('\rProcessing Clips of Video %s  %6d / %6d (%5.2f%%)   %5.2f FPS  '
                  % (repr(progress_bar_clip), cdx+1, len_clips, progress_clip, avg_fps), end='')

            # Remove overlapped frames
            if eval_clip_frames > 1 and cdx > 0:
                clip_frame_ids = clip_frame_ids[n_overlapped_frames:]
                images = images[n_overlapped_frames:]
                images_meta = images_meta[n_overlapped_frames:]
                preds = preds[n_overlapped_frames:]

            for batch_id, pred in enumerate(preds):
                if args.display:
                    img_id = (vid, clip_frame_ids[batch_id])
                    root_dir = os.path.join(output_dir, 'out', str(vid))
                    if not os.path.exists(root_dir):
                        os.makedirs(root_dir)
                    if not args.display_single_mask:
                        img_numpy = prep_display(pred, images[batch_id],
                                                 img_meta=images_meta[batch_id])
                        plt.imshow(img_numpy)
                        plt.axis('off')
                        plt.title(str(img_id))
                        plt.savefig(''.join([root_dir, '/', str(img_id[1]), '.jpg']))
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
                            plt.savefig(''.join([root_dir, '/', str(img_id[1]), '_', str(p), '.jpg']))
                            plt.clf()

                else:
                    if pred is not None:
                        pred = postprocess_ytbvis(pred, images_meta[batch_id], train_masks=train_masks,
                                                  output_file=output_dir)

                    if data_type == 'vid' and use_vid_metric:
                        for k, v in pred.items():
                            pred[k] = v.tolist()
                        pred['video_id'] = vid
                        pred['frame_id'] = clip_frame_ids[batch_id]
                        json_results.append(pred)
                    else:
                        bbox2result_video(results_video, pred, clip_frame_ids[batch_id], types=args.eval_types)

        if not args.display:
            for obj_id, result_obj in results_video.items():
                result_obj['video_id'] = vid
                result_obj['score'] = np.array(result_obj['score']).mean().item()
                result_obj['category_id'] = np.bincount(result_obj['category_id']).argmax().item()
                json_results.append(result_obj)

    timer.print_stats()
    return json_results


def evaluate_clip(net: CoreNet, dataset, data_type='vis', pad_h=None, pad_w=None, clip_frames=1, n_clips=1,
                  TRAIN_INTERCLIPS_CLASS=False, output_dir=None):
    '''
    :param net:
    :param dataset:
    :param data_type: 'vis' or 'vid'
    :param clip_frames:
    :param output_dir:
    :return:
    '''
    n_newly_frames = clip_frames - args.overlap_frames
    dataset_size = len(dataset.vid_ids)

    print()
    json_results = []
    # Main eval loop, only support a single clip
    frame_times = MovingAverage()
    for vdx, vid in enumerate(dataset.vid_ids):
        progress = (vdx + 1) / dataset_size * 100
        print()
        print('Processing Videos:  %2d / %2d (%5.2f%%) ' % (vdx+1, dataset_size, progress))

        # Inverse direction
        if args.display:
            vdx = -vdx - 1
            vid = dataset.vid_ids[vdx]

        vid_objs = {}
        if data_type == 'vis':
            len_vid = dataset.vid_infos[vdx]['length']
            train_masks, use_vid_metric = True, False
        else:
            len_vid = len(dataset.vid_infos[vdx])
            train_masks, use_vid_metric = False, True

        if TRAIN_INTERCLIPS_CLASS:
            conf_feats_video = dict()
            pred_frames_video, images_video, images_meta_video = [], [], []

        len_clips = (len_vid + n_newly_frames-1) // n_newly_frames
        progress_bar_clip = ProgressBar(len_clips, len_clips)
        for cdx in range(len_clips):
            timer.reset()
            progress_clip = (cdx + 1) / len_clips * 100

            with timer.env('Load Data'):
                left = cdx * n_newly_frames
                if left+clip_frames > len_vid:
                    clip_frame_ids = list(range(len_vid-clip_frames, len_vid))
                else:
                    clip_frame_ids = list(range(left, left+clip_frames))
                # Just mask sure all frames have been processed
                if cdx == len_clips-1:
                    assert clip_frame_ids[-1] == len_vid-1, \
                        'The last clip should include the last frame! Please double check!'
                images, images_meta, targets = dataset.pull_clip_from_video(vid, clip_frame_ids)
                images = [torch.from_numpy(img).permute(2, 0, 1).contiguous() for img in images]
                images = ImageList_from_tensors(images, size_divisibility=32, pad_h=pad_h, pad_w=pad_w).cuda()
                pad_shape = {'pad_shape': (images.size(-2), images.size(-1), 3)}
                for k in range(len(images_meta)):
                    images_meta[k].update(pad_shape)
                if images.size(0) < clip_frames:
                    images = images.repeat(clip_frames, 1, 1, 1)[:clip_frames]
                    images_meta = (images_meta*clip_frames)[:clip_frames]
                    clip_frame_ids = (clip_frame_ids * clip_frames)[:clip_frames]

            # Interesting tests inputs with different orders
            # order_idx = list(range(clip_frames))[::-1]
            # images = torch.flip(images, [0])
            order_idx = range(clip_frames)
            t = time.time()
            preds_clip = net(images, img_meta=[images_meta])
            avg_fps_wodata = 1. / ((time.time()-t)/clip_frames)
            pred_clip = preds_clip[0]
            frame_times.add(timer.total_time())
            progress_bar_clip.set_val(cdx+1)
            avg_fps = 0  # 1. / (frame_times.get_avg() / clip_frames) if vdx > 0 or cdx > 0 else 0
            print('\rProcessing Clips of Video %s  %6d / %6d (%5.2f%%)  %5.2f fps  %5.2f fps '
                  % (repr(progress_bar_clip), cdx+1, len_clips, progress_clip, avg_fps, avg_fps_wodata), end='')

            # Here store all classification features for later inter-clips classification
            if TRAIN_INTERCLIPS_CLASS and pred_clip['box_ids'].nelement() > 0:
                box_ids = pred_clip['box_ids']
                for obj_idx, obj_id in enumerate(box_ids):
                    if int(obj_id) not in conf_feats_video:
                        conf_feats_video[int(obj_id)] = [pred_clip['conf_feat'][obj_idx]]
                    else:
                        conf_feats_video[int(obj_id)] += [pred_clip['conf_feat'][obj_idx]]

            temporal_dependent_keys = ['proto', 'fpn_feat', 'fpn_feat_temp', 'sem_seg', 'mask']
            pred_frame = dict()
            for k, v in pred_clip.items():
                if k not in temporal_dependent_keys:
                    pred_frame[k] = v
            if 'priors' in pred_clip.keys():
                pred_frame['priors'] = pred_clip['priors'][:, :4]

            if cdx+1 == len_clips:
                newly_clip_frame_ids = clip_frame_ids[left-len_vid:]
                newly_idx = range(clip_frames)[left-len_vid:]
            else:
                newly_clip_frame_ids = clip_frame_ids[:n_newly_frames]
                newly_idx = range(clip_frames)[:n_newly_frames]

            for idx, frame_id in zip(newly_idx, newly_clip_frame_ids):
                batch_id = order_idx[idx]
                if pred_clip['box'].size(0) == 0:
                    if train_masks:
                        pred_frame['mask'] = torch.Tensor()
                    pred_frame['box'] = torch.Tensor()
                else:
                    if pred_clip['box'].size(-1) != 4:
                        pred_frame['box'] = pred_clip['box'][:, batch_id*4:(batch_id+1)*4]

                    for k, v in pred_clip.items():
                        if k in temporal_dependent_keys:
                            pred_frame[k] = v[:, batch_id] if k == 'mask' else v[..., batch_id, :]

                if args.display:
                    if not TRAIN_INTERCLIPS_CLASS:
                        root_dir = os.path.join(output_dir, 'out', str(vid))
                        if not os.path.exists(root_dir):
                            os.makedirs(root_dir)
                        img_numpy = prep_display(pred_frame, images[batch_id],
                                                 img_meta=images_meta[idx])
                        plt.imshow(img_numpy)
                        plt.axis('off')
                        plt.savefig(''.join([root_dir, '/', str(frame_id), '.jpg']), bbox_inches='tight', pad_inches=0)
                        plt.clf()
                    else:
                        pred_frame_single = dict()
                        for k, v in pred_frame.items():
                            pred_frame_single[k] = v.clone()
                        pred_frames_video += [pred_frame_single]
                        images_video += [images[batch_id]]
                        images_meta_video += [images_meta[idx]]

                else:
                    preds_cur = postprocess_ytbvis(pred_frame, images_meta[idx], train_masks=train_masks,
                                                   output_file=output_dir)
                    if data_type == 'vid' and use_vid_metric:
                        for k, v in preds_cur.items():
                            preds_cur[k] = v.tolist()
                        preds_cur['video_id'] = vid
                        preds_cur['frame_id'] = frame_id
                        json_results.append(preds_cur)
                    else:
                        bbox2result_video(vid_objs, preds_cur, frame_id, types=args.eval_types)

        # Predict interclips classification or global classification for each objects
        if TRAIN_INTERCLIPS_CLASS:
            category_ids_vid = dict()
            for obj_id, conf_feats_obj in conf_feats_video.items():
                conf_feats_obj = torch.stack(conf_feats_obj, dim=0)
                iters = (len(conf_feats_obj)-1)//n_clips+1

                conf_all, category_id_obj_all = [], []
                for _ in range(1):
                    selected_idx = torch.randint(len(conf_feats_obj), (iters, n_clips))
                    conf_data = net.InterclipsClass(conf_feats_obj[selected_idx.reshape(-1)].reshape(iters, -1))
                    # conf_all, category_id_obj_all = torch.softmax(conf_data, dim=-1).max(-1)
                    conf, category_id_obj = torch.softmax(conf_data, dim=-1).mean(0).max(0)
                    if int(category_id_obj+1) not in category_id_obj_all:
                        conf_all.append(conf)
                        category_id_obj_all.append(int(category_id_obj+1))
                    else:
                        match_idx = category_id_obj_all.index(int(category_id_obj+1))
                        conf_all[match_idx] = max(conf_all[match_idx], conf)

                category_ids_vid[obj_id] = {'category_id': category_id_obj_all, 'score': conf_all}

            if args.display:
                root_dir = os.path.join(output_dir, 'out', str(vid))
                if not os.path.exists(root_dir):
                    os.makedirs(root_dir)
                for pred_frame, image, image_meta in zip(pred_frames_video, images_video, images_meta_video):
                    frame_id = image_meta['frame_id']
                    pred_frame['class'] = torch.tensor([category_ids_vid[int(id)]['category_id'][0] for id in pred_frame['box_ids']])
                    pred_frame['score_global'] = torch.tensor([category_ids_vid[int(id)]['score'][0] for id in pred_frame['box_ids']])
                    img_numpy = prep_display(pred_frame, image, img_meta=image_meta)
                    plt.imshow(img_numpy)
                    plt.axis('off')
                    plt.title(str((vid, frame_id)))
                    plt.savefig(''.join([root_dir, '/', str(frame_id), '.jpg']))
                    plt.clf()

        if not args.display and data_type == 'vis':
            for obj_id, vid_obj in vid_objs.items():
                vid_obj['video_id'] = vid
                stuff_score = np.array(vid_obj['score']).mean().item()
                if TRAIN_INTERCLIPS_CLASS:
                    for cat_id, score in zip(category_ids_vid[obj_id]['category_id'], category_ids_vid[obj_id]['score']):
                        vid_obj['category_id'] = cat_id
                        vid_obj['score'] = stuff_score * score.cpu().numpy().item()
                        assert len(vid_obj['segmentations']) == len_vid
                        json_results.append(vid_obj)
                else:
                    vid_obj['category_id'] = np.bincount(vid_obj['category_id']).argmax().item()
                    vid_obj['score'] = stuff_score
                    assert len(vid_obj['segmentations']) == len_vid
                    json_results.append(vid_obj)

    timer.print_stats()
    print('Total numbers of instances:', len(json_results))
    return json_results


def evaluate_single(net: CoreNet, im_path=None, save_path=None, idx=None):
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


def evalimages(net: CoreNet, input_folder: str, output_folder: str):
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


def evalvideo(net: CoreNet, input_folder: str, output_folder: str):
    return


def evaldatasets(net: CoreNet, val_dataset, data_type, output_dir, eval_clip_frames=1, n_clips=1,
                 cubic_mode=False, cfg_input=None, TRAIN_INTERCLIPS_CLASS=False):
    args.save_folder = output_dir
    if cfg_input is not None:
        pad_h, pad_w = cfg_input.MIN_SIZE_TEST, cfg_input.MAX_SIZE_TEST
    else:
        pad_h, pad_w = None, None
    json_path = os.path.join(output_dir, 'results_' + str(args.epoch) + '.json')
    if args.eval:
        print('Begin Inference!')
        if cubic_mode:
            results = evaluate_clip(net, val_dataset, data_type=data_type, pad_h=pad_h, pad_w=pad_w,
                                    clip_frames=eval_clip_frames, n_clips=n_clips,
                                    TRAIN_INTERCLIPS_CLASS=TRAIN_INTERCLIPS_CLASS, output_dir=output_dir)
        else:
            results = evaluate(net, val_dataset, data_type=data_type, pad_h=pad_h, pad_w=pad_w,
                               eval_clip_frames=eval_clip_frames, output_dir=output_dir)

        if not args.display:
            print()
            print('Save predictions into {} :'.format(json_path))
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f)
            print('Finish save!')

    if val_dataset.has_gt:
        print('Begin calculate metrics!')
        ann_file = val_dataset.ann_file if data_type == 'vis' else \
            val_dataset.img_prefix + '/cache/' + val_dataset.img_index.split('/')[-1].replace('.txt', '_anno.json')
        use_vid_metric = False if data_type == 'vis' else True
        metric_path = json_path.replace('.json', '.txt')
        iouType = 'segm' if data_type == 'vis' else 'bbox'
        if data_type == 'vid' and not use_vid_metric:
            ann_file = ann_file[:-4] + '_eval' + ann_file[-4:]
        calc_metrics(ann_file, json_path, output_file=metric_path, iouType=iouType, data_type=data_type,
                     use_vid_metric=use_vid_metric)

    print('Finish!')


if __name__ == '__main__':
    parse_args()

    if args.config is not None:
        cfg = load_config(args.config)
    else:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        cfg = load_config(args.config)
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.NAME)
    global class_names
    class_names = get_dataset_config(cfg.DATASETS.TRAIN, cfg.DATASETS.TYPE)['class_names']

    if args.image is None and args.images is None:
        print('Load dataset:', cfg.DATASETS, args.eval_data)
        if args.eval_data == 'train':
            val_dataset = get_dataset(cfg.DATASETS.TYPE, cfg.DATASETS.TRAIN, cfg.INPUT, cfg.SOLVER.NUM_CLIP_FRAMES, 
                                      inference=True)

        elif args.eval_data == 'valid_sub':
            val_dataset = get_dataset(cfg.DATASETS.TYPE, cfg.DATASETS.VALID_SUB, cfg.INPUT, cfg.SOLVER.NUM_CLIP_FRAMES, 
                                      inference=True)

        elif args.eval_data == 'valid':
            val_dataset = get_dataset(cfg.DATASETS.TYPE, cfg.DATASETS.VALID, cfg.INPUT, cfg.SOLVER.NUM_CLIP_FRAMES,
                                      inference=True)
        else:
            val_dataset = get_dataset(cfg.DATASETS.TYPE, cfg.DATASETS.TEST, cfg.INPUT, cfg.SOLVER.NUM_CLIP_FRAMES,
                                      inference=True)
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

        print('Loading model from {}'.format(args.trained_model))
        net = CoreNet(cfg, args.display)
        net.load_weights(args.trained_model)
        net.eval()
        print('Loading model Done.')

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
            evaldatasets(net, val_dataset, cfg.DATASETS.TYPE, cfg.OUTPUT_DIR, cfg.TEST.NUM_CLIP_FRAMES,
                         cfg.SOLVER.NUM_CLIPS, cfg.MODEL.PREDICTION_HEADS.CUBIC_MODE, cfg.INPUT,
                         TRAIN_INTERCLIPS_CLASS=cfg.MODEL.CLASS_HEADS.TRAIN_INTERCLIPS_CLASS)



