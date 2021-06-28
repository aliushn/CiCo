from datasets import *
from utils import timer
import argparse
from layers.functions import Track_TF_Clip_json
import numpy as np
import json
import mmcv
import torch

import pycocotools.mask as mask_util
from utils.functions import MovingAverage, ProgressBar


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
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for training')
    parser.add_argument('--remove_duplicated', default=False, type=bool,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--top_k', default=100, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--out_file', default='weights/OVIS/VisTR/results.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--iou_threshold', default=0.5, type=float,
                        help='NMS with a score higher this threshold will be viewed as decuplicated masks and remove it.')
    parser.add_argument('--overlap', default=9, type=int,
                        help='Further restrict the number of predictions to parse')

    global args
    args = parser.parse_args(argv)


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def evaluate():

    json_results = []

    # path of the .json file that stores your predicted masks
    results_file = 'weights/OVIS/VisTR/video_internal_15_overlap.json'
    candidates = json.load(open(results_file, 'r'))
    Track_TF_Clip_vistr = Track_TF_Clip_json(remove_duplicated=args.remove_duplicated,
                                             iou_threshold=args.iou_threshold,
                                             overlap=args.overlap)

    video_size = len(candidates)
    progress_bar = ProgressBar(30, video_size)
    frame_times = MovingAverage()

    # Main eval loop
    for it, candidate in enumerate(candidates):
        if it > 5:
            break
        timer.reset()
        # tracking between clips using MIoI in overlapped frames of two clips
        results_single_video = Track_TF_Clip_vistr(candidate)
        vid_id = candidate['video_id']
        json_results += clip_results2json_videoseg(vid_id, results_single_video, args.overlap)

        # First couple of images take longer because we're constructing the graph.
        # Since that's technically initialization, don't include those in the FPS calculations.
        if it > 1:
            frame_times.add(timer.total_time())

        if it > 1 and frame_times.get_avg() > 0:
            fps = 1 / frame_times.get_avg()
        else:
            fps = 0
        progress = (it + 1) / video_size * 100
        progress_bar.set_val(it + 1)
        print('\rProcessing Images  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
              % (repr(progress_bar), it + 1, video_size, progress, fps), end='')

    mmcv.dump(json_results, args.out_file)
    print('Done')


def clip_results2json_videoseg(vid_id, results, overlap):
    '''
    results: [clip1, clip2, ...]
    '''
    json_results = []
    vid_objs = dict()
    max_frame_id = -1
    # store results of  the current video
    for clip_idx, result in enumerate(results):

        if len(result) == 0:
            if clip_idx == 0:
                n_frames = 18
            else:
                n_frames = 18-overlap
            cur_frame_ids = range(max_frame_id + 1, max_frame_id + 1 + n_frames)

        else:
            n_frames = len(result[0]['segmentations'])
            cur_frame_ids = range(max_frame_id + 1, max_frame_id + 1 + n_frames)
            for obj in result:
                obj_id = obj['obj_ids']
                if obj_id >= 0:
                    if obj_id not in vid_objs:
                        vid_objs[obj_id] = {'score': [], 'category_id': [], 'segmentations': {}}
                    vid_objs[obj_id]['score'].append(obj['score'])
                    vid_objs[obj_id]['category_id'].append(obj['category_id'])

                    for frame_idx, frame_id in enumerate(cur_frame_ids):
                        vid_objs[obj_id]['segmentations'][frame_id] = obj['segmentations'][frame_idx]

        if max(cur_frame_ids) > max_frame_id:
            max_frame_id = max(cur_frame_ids)

    for obj_id, obj in vid_objs.items():
        data = dict()

        data['video_id'] = vid_id
        data['score'] = np.array(obj['score']).mean().item()
        # majority voting of those frames with top k highest scores for sequence catgory
        # scores_sorted_idx = np.argsort(-1 * np.stack(obj['score'], axis=0))
        # cats_with_highest_scores = np.stack(obj['cats'], axis=0)[scores_sorted_idx[:20]]
        data['category_id'] = np.bincount(obj['category_id']).argmax().item()

        # majority voting for sequence category
        # data['category_id'] = np.bincount(np.array(obj['cats'])).argmax().item()
        vid_seg = []
        for fid in range(max_frame_id + 1):
            if fid in obj['segmentations']:
                vid_seg.append(obj['segmentations'][fid])
            else:
                vid_seg.append(None)
        data['segmentations'] = vid_seg
        json_results.append(data)

    return json_results


if __name__ == '__main__':
    parse_args()

    if args.config is not None:
        set_cfg(args.config)

    evaluate()


