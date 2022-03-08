from datasets import *
from CoreNet import CoreNet
from utils.functions import MovingAverage
from utils import timer
from utils.functions import SavePath
from layers.utils.output_utils import postprocess_ytbvis, undo_image_transformation
from layers.visualization_temporal import draw_dotted_rectangle, get_color
from configs.load_config import load_config

import time
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
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--epoch', default=-1, type=int)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for training')
    parser.add_argument('--trained_model',
                        default='path to your trained models', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--NUM_CLIP_FRAMES', default=3, type=int,
                        help='number of frames in a clip')
    parser.add_argument('--top_k', default=100, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--eval_data', default='valid', type=str, help='data type')
    parser.add_argument('--overlap_frames', default=0, type=int, help='the overlapped frames between two video clips')
    parser.add_argument('--display_single_mask', default=False, type=str2bool,
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
    parser.add_argument('--display_bg', default=True, type=str2bool,
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--save_folder', default='outputs/', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--display_lincomb', dest='display_lincomb', action='store_true',
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--eval_types', default=['segm'], type=str, nargs='+', choices=['bbox', 'segm'], help='eval types')
    parser.set_defaults(display=False, resume=False, detect=False)

    global args
    args = parser.parse_args(argv)


def prep_display(dets_out, imgs, img_meta=None, undo_transform=True, mask_alpha=0.55):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    -- display_model: 'train', 'test', 'None' means groundtruth results
    """
    img_meta_temp = img_meta[0] if isinstance(img_meta, list) else img_meta
    ori_h, ori_w, _ = img_meta_temp['ori_shape']
    img_h, img_w, _ = img_meta_temp['img_shape']
    root_dir = os.path.join(args.save_folder, 'out', str(img_meta_temp['video_id']))
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    if args.display_bg:
        if undo_transform:
            imgs_numpy = undo_image_transformation(imgs, ori_h, ori_w, img_h, img_w)
            imgs_gpu = torch.Tensor(imgs_numpy).cuda()
        else:
            imgs_gpu = imgs / 255.0
    else:
        imgs_gpu = torch.ones(imgs.size(0), ori_h, ori_w, 3)

    # torch.cuda.synchronize()
    if dets_out['box'].nelement() == 0:
        # No detections found so just output the original image
        for t in range(imgs_gpu.size(0)):
            img_gpu = imgs_gpu[t]
            img_numpy = (img_gpu * 255).byte().cpu().numpy()
            plt.imshow(img_numpy)
            plt.axis('off')
            plt.savefig(''.join([root_dir, '/', str(img_meta[t]['frame_id']), '.jpg']), bbox_inches='tight',
                        pad_inches=0)
            plt.clf()
    else:
        num_dets_to_consider = min(args.top_k, dets_out['box'].size(0))
        scores = dets_out['score'][:args.top_k].view(-1).detach().cpu().numpy()
        color_type = dets_out['box_ids'].view(-1)
        num_tracked_mask = dets_out['tracked_mask'] if 'tracked_mask' in dets_out.keys() else None
        boxes_cir = dets_out['box_cir'] if 'box_cir' in dets_out.keys() else None

        for t in range(imgs_gpu.size(0)):
            img_gpu = imgs_gpu[t]
            boxes = dets_out['box'][:args.top_k, t].detach().cpu().numpy()
            if args.display_masks:
                masks = dets_out['mask'][:args.top_k, t]
                # After this, mask is of size [num_dets, h, w, 1]
                masks = masks[:num_dets_to_consider, :, :, None]

            if args.display_single_mask:
                for j in range(num_dets_to_consider):
                    # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
                    color = get_color(j, color_type, on_gpu=img_gpu.device.index,
                                      undo_transform=undo_transform).view(1, 1, 3)

                    mask_color = masks[j].repeat(1, 1, 3) * color * mask_alpha

                    # This is 1 everywhere except for 1-mask_alpha where the mask is
                    inv_alph_masks = masks[j] * (-mask_alpha) + 1
                    img_gpu_single = img_gpu * inv_alph_masks + mask_color

                    # Then draw the stuff that needs to be done on the cpu
                    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
                    img_numpy_single = (img_gpu_single * 255).byte().cpu().numpy()

                    plt.imshow(img_numpy_single)
                    plt.axis('off')
                    plt.savefig('/'.join([root_dir, str(img_meta[t]['frame_id'])+'_'+str(j)+'.jpg']),
                                bbox_inches='tight', pad_inches=0)
                    plt.clf()

            # First, draw the masks on the GPU where we can do it really fast
            # Beware: very fast but possibly unintelligible mask-drawing code ahead
            # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
            if args.display_masks:
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
                for j in range(num_dets_to_consider):
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

            plt.imshow(img_numpy)
            plt.axis('off')
            plt.savefig(''.join([root_dir, '/', str(img_meta[t]['frame_id'])+str(j), '.jpg']),
                        bbox_inches='tight', pad_inches=0)
            plt.clf()


def evaluate_clip(net: CoreNet, dataset, data_type='vis', pad_h=None, pad_w=None, clip_frames=1, use_cico=False):
    '''
    :param net:
    :param dataset:
    :param data_type: 'vis' or 'vid'
    :param clip_frames:
    :param output_dir:
    :return:
    '''
    n_newly_frames = (clip_frames - args.overlap_frames) * args.batch_size
    n_frames = clip_frames * args.batch_size
    dataset_size = len(dataset.vid_ids)

    print()
    json_results = []
    # Main eval loop, only support a single clip
    frame_times = MovingAverage()
    for vdx, vid in enumerate(dataset.vid_ids):
        progress_videos = (vdx + 1) / dataset_size * 100
        # Inverse direction to process videos
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

        len_clips = (len_vid + n_newly_frames-1) // n_newly_frames
        for cdx in range(len_clips):
            timer.reset()

            t = time.time()
            with timer.env('Load Data'):
                left = cdx * n_newly_frames
                clip_frame_ids = range(left, min(left+n_frames, len_vid))
                # Just mask sure all frames have been processed
                if cdx == len_clips-1:
                    assert clip_frame_ids[-1] == len_vid-1, \
                        'The last clip should include the last frame!! Please double check!'

                images, imgs_meta, targets = dataset.pull_clip_from_video(vid, clip_frame_ids)
                images = [torch.from_numpy(img).permute(2, 0, 1).contiguous() for img in images]
                images = ImageList_from_tensors(images, size_divisibility=32, pad_h=pad_h, pad_w=pad_w).cuda()
                pad_shape = {'pad_shape': (images.size(-2), images.size(-1), 3)}
                for k in range(len(imgs_meta)):
                    imgs_meta[k].update(pad_shape)
                if use_cico and images.size(0) < n_frames:
                    images = images.repeat(n_frames, 1, 1, 1)[:n_frames]
                    imgs_meta = (imgs_meta*n_frames)[:n_frames]
                imgs_meta = [imgs_meta] if use_cico else imgs_meta

            # Interesting tests inputs with inverse orders
            # order_idx = list(range(clip_frames))[::-1]
            # images = torch.flip(images, [0])
            preds = net(images, img_meta=imgs_meta)
            avg_fps_wodata = 1. / ((time.time()-t) / n_frames)
            frame_times.add(timer.total_time())
            avg_fps = 1. / max((frame_times.get_avg() / clip_frames, 1e-2))
            print('\rProcessing videos:  %2d / %2d (%5.2f %%), clips:  %2d / %2d;  '
                  'FPS w /wo load_data: %5.2f / %5.2f fps '
                  % (vdx+1, dataset_size, progress_videos, cdx+1, len_clips, avg_fps, avg_fps_wodata), end='')

            for tdx, pred_clip in enumerate(preds):
                interval = n_newly_frames if use_cico else clip_frames
                left, right = tdx * interval, (tdx+1) * interval
                clip_frame_ids_cur = clip_frame_ids[left:min(right, len_vid - left)]
                images_cur = images[left:min(right, len_vid - left)]
                if use_cico:
                    imgs_meta_cur = imgs_meta[tdx]
                    for k, v in pred_clip.items():
                        if k in {'box', 'mask'} and v.nelement() > 0:
                            pred_clip[k] = v[:, left:min(right, len_vid - left)]
                        elif k in {'img_ids'}:
                            pred_clip[k] = v[left:min(right, len_vid - left)]
                else:
                    imgs_meta_cur = imgs_meta[left:min(right, len_vid - left)]
                pred_clip_pp = postprocess_ytbvis(pred_clip, imgs_meta_cur, train_masks=train_masks,
                                                  visualize_lincomb=args.display_lincomb,
                                                  output_file=args.save_folder)

                if args.display:
                    prep_display(pred_clip_pp, images_cur, img_meta=imgs_meta_cur)

                else:
                    if data_type == 'vid' and use_vid_metric:
                        for k, v in pred_clip_pp.items():
                            pred_clip_pp[k] = v.tolist()
                        pred_clip_pp['video_id'] = vid
                        json_results.append(pred_clip_pp)
                    else:
                        bbox2result_video(vid_objs, pred_clip_pp, clip_frame_ids_cur, types=args.eval_types)

        if not args.display and data_type == 'vis':
            for obj_id, vid_obj in vid_objs.items():
                vid_obj['video_id'] = vid
                vid_obj['category_id'] = np.bincount(vid_obj['category_id']).argmax().item()
                vid_obj['score'] = np.array(vid_obj['score']).mean().item()
                assert len(vid_obj['segmentations']) == len_vid, \
                    'Different lengths between output masks and input images, please double check!!'
                json_results.append(vid_obj)

    timer.print_stats()
    print('Total numbers of instances:', len(json_results))
    return json_results


def evaldatasets(net: CoreNet, val_dataset, data_type, cfg_input=None, use_cico=False):
    pad_h, pad_w = (cfg_input.MIN_SIZE_TEST, cfg_input.MAX_SIZE_TEST) if cfg_input is not None else (None, None)
    json_path = '/'.join([args.save_folder, 'results_'+str(args.epoch)+'.json']) if args.epoch >= 0 \
        else '/'.join([args.save_folder, 'results.json'])

    print('Begin Inference!')
    results = evaluate_clip(net, val_dataset, data_type=data_type, pad_h=pad_h, pad_w=pad_w,
                            clip_frames=args.NUM_CLIP_FRAMES, use_cico=use_cico)

    if not args.display:
        print()
        print('Save predicted results into {} for evaluation on Codelab:'.format(json_path))
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

# Update training parameters from the config if necessary
def replace(name, dict_cfg):
    if getattr(args, name) is not None: setattr(dict_cfg, name, getattr(args, name))


if __name__ == '__main__':
    parse_args()

    model_path = SavePath.from_str(args.trained_model)
    # TODO: Bad practice? Probably want to do a name lookup instead.
    args.config = '/'.join(['configs/CiCo', '_'.join(model_path.model_name.split('_')[:4])]) \
        if args.config is None else args.config
    print('Parsed %s from the file name.\n' % args.config)
    cfg = load_config(args.config)
    replace('NUM_CLIP_FRAMES', cfg.SOLVER)
    replace('NUM_CLIP_FRAMES', cfg.TEST)

    args.save_folder = '/'.join(args.trained_model.split('/')[:-1])
    cfg.display = args.display
    global class_names
    class_names = get_dataset_config(cfg.DATASETS.TRAIN, cfg.DATASETS.TYPE)['class_names']

    print('Load dataset:', cfg.DATASETS, args.eval_data)
    if args.eval_data == 'train':
        data_cfg = cfg.DATASETS.TRAIN
    elif args.eval_data == 'valid_sub':
        data_cfg = cfg.DATASETS.VALID_SUB
    elif args.eval_data == 'valid':
        data_cfg = cfg.DATASETS.VALID
    else:
        data_cfg = cfg.DATASETS.TEST
    val_dataset = get_dataset(cfg.DATASETS.TYPE, data_cfg, cfg.INPUT, args.NUM_CLIP_FRAMES, inference=True)

    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        cudnn.benchmark = True
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print('Loading model from {}'.format(args.trained_model))
        net = CoreNet(cfg)
        net.load_weights(args.trained_model)
        net.eval()
        print('Loading model Done.')

        net = net.cuda()
        evaldatasets(net, val_dataset, cfg.DATASETS.TYPE, cfg.INPUT, cfg.CiCo.ENGINE)



