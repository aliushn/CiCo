# -*- coding: utf-8 -*-
import numpy as np
import mmcv
import os
import json
from cocoapi.PythonAPI.pycocotools.ytvos import YTVOS
from cocoapi.PythonAPI.pycocotools.ytvoseval import YTVOSeval
from cocoapi.PythonAPI.pycocotools.vid import VID
from cocoapi.PythonAPI.pycocotools.videval import VIDeval
import pycocotools.mask as mask_util
from datasets.vid_eval import do_vid_evaluation


def bbox2result_with_id(preds, img_meta):
    """Convert detection results to a list of numpy arrays.

    Args:
        preds (Tensor): shape (n, 5)
        img_meta (Tensor): shape (n, )

    Returns:
        list(ndarray): bbox results of each class
    """

    video_id, frame_id = img_meta['video_id'], img_meta['frame_id']
    results = {'video_id': video_id, 'frame_id': frame_id}
    if preds['box'].shape[0] == 0:
        return results
    else:
        bboxes = preds['box'].cpu().numpy()
        if preds['class'] is not None:
            labels = preds['class'].view(-1).cpu().numpy()
        else:
            labels = None
        scores = preds['score'].view(-1).cpu().numpy()
        segms = preds['segm']
        obj_ids = preds['box_ids'].view(-1).cpu().numpy()
        if labels is not None:
            for bbox, label, score, segm, obj_id in zip(bboxes, labels, scores, segms, obj_ids):
                if obj_id >= 0:
                    results[obj_id] = {'bbox': bbox, 'label': label, 'score': score, 'segm': segm}
        else:
            for bbox, score, segm, obj_id in zip(bboxes, scores, segms, obj_ids):
                if obj_id >= 0:
                    results[obj_id] = {'bbox': bbox, 'score': score, 'segm': segm}

        return results


def results2json_videoseg(results, out_file):
    json_results = []
    vid_objs = {}
    size = len(results)

    for idx in range(size):
        # assume results is ordered

        vid_id, frame_id = results[idx]['video_id'], results[idx]['frame_id']
        if idx == size - 1:
            is_last = True
        else:
            vid_id_next, frame_id_next = results[idx+1]['video_id'], results[idx+1]['frame_id']
            is_last = vid_id_next != vid_id

        det = results[idx]
        for obj_id in det:
            if obj_id not in {'video_id', 'frame_id'}:
                bbox = det[obj_id]['bbox']
                score = det[obj_id]['score']
                segm = det[obj_id]['segm']
                label = det[obj_id]['label']
                # label_all = det[obj_id]['label_all']
                if obj_id not in vid_objs:
                    vid_objs[obj_id] = {'scores': [], 'cats': [], 'segms': {}}
                vid_objs[obj_id]['scores'].append(score)
                vid_objs[obj_id]['cats'].append(label)
                segm['counts'] = segm['counts'].decode()
                vid_objs[obj_id]['segms'][frame_id] = segm
        if is_last:
            # store results of  the current video
            for obj_id, obj in vid_objs.items():
                data = dict()

                data['video_id'] = vid_id
                data['score'] = np.array(obj['scores']).mean().item()
                # majority voting of those frames with top k highest scores for sequence catgory
                scores_sorted_idx = np.argsort(-1 * np.stack(obj['scores'], axis=0))
                cats_with_highest_scores = np.stack(obj['cats'], axis=0)[scores_sorted_idx[:20]]
                data['category_id'] = np.bincount(cats_with_highest_scores).argmax().item()

                # majority voting for sequence category
                # data['category_id'] = np.bincount(np.array(obj['cats'])).argmax().item()
                vid_seg = []
                for fid in range(frame_id + 1):
                    if fid in obj['segms']:
                        vid_seg.append(obj['segms'][fid])
                    else:
                        vid_seg.append(None)
                data['segmentations'] = vid_seg
                json_results.append(data)

            vid_objs = {}
    if not os.path.exists(out_file[:-13]):
        os.makedirs(out_file[:-13])

    mmcv.dump(json_results, out_file)
    print('Done')


def bbox2result_video(results, preds, frame_idx, types=None):
    """Convert detection results to a list of numpy arrays.

    Args:
        preds (Tensor): shape (n, 5)
        classes (list): class category, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    types = ['segm'] if types is None else types

    if len(results) > 0:
        for type in types:
            for obj_id in results:
                if type == 'segm':
                    results[obj_id]['segmentations'] += [None]
                elif type == 'bbox':
                    results[obj_id]['bbox'] += [None]

    if preds is None or (preds is not None and preds['box'].shape[0] == 0):
        return results
    else:
        bboxes = preds['box'].cpu().numpy()
        labels = preds['class'].view(-1).cpu().numpy() if 'class' in preds.keys() else None
        scores = preds['score'].view(-1).cpu().numpy()
        obj_ids = preds['box_ids'].view(-1).cpu().numpy()
        for type in types:
            for idx, score, bbox, obj_id in zip(range(len(obj_ids)), scores, bboxes, obj_ids):
                if type == 'segm':
                    segm = preds['mask'][idx]
                    # segm annotation: png2rle
                    segm = mask_util.encode(np.array(segm.cpu(), order='F', dtype='uint8'))
                    # .json file can not deal with var with 'bytes'
                    segm['counts'] = segm['counts'].decode()

                if obj_id not in results:
                    results[obj_id] = {'score': [score]}
                    if labels is not None:
                        results[obj_id]['category_id'] = [labels[idx]]
                    if type == 'segm':
                        results[obj_id]['segmentations'] = [None]*frame_idx + [segm]
                    elif type == 'bbox':
                        results[obj_id]['bbox'] = [None]*frame_idx + [bbox.tolist()]
                else:
                    results[obj_id]['score'] += [score]
                    if labels is not None:
                        results[obj_id]['category_id'] += [labels[idx]]
                    if type == 'segm':
                        results[obj_id]['segmentations'][-1] = segm
                    elif type == 'bbox':
                        results[obj_id]['bbox'][-1] = bbox.tolist()

        return results


def calc_metrics(anno_file, dt_file, output_file=None, iouType='segm', data_type='vis', use_vid_metric=True):
    # iouType is 'segm' or 'bbox'
    if data_type == 'vis':
        ytvosGt = YTVOS(anno_file)
        ytvosDt = ytvosGt.loadRes(dt_file)

        E = YTVOSeval(ytvosGt, ytvosDt, iouType=iouType, output_file=output_file)
        E.evaluate()
        E.accumulate()
        E.summarize()
        return E.stats
    elif data_type == 'vid':
        if use_vid_metric:
            gt_annos = json.load(open(anno_file, 'r'))
            dt_annos = json.load(open(dt_file, 'r'))
            do_vid_evaluation(gt_annos, dt_annos, output_file)
        else:
            vidGt = VID(anno_file)
            vidDt = vidGt.loadRes(dt_file)

            E = VIDeval(vidGt, vidDt, iouType=iouType, output_file=output_file)
            E.evaluate()
            E.accumulate()
            E.summarize()
            print('finish validation')

            return E.stats


def ytvos_eval(result_file, result_types, ytvos, max_dets=(100, 300, 1000), save_path_valid_metrics=None):
    if mmcv.is_str(ytvos):
        ytvos = YTVOS(ytvos)
    assert isinstance(ytvos, YTVOS)

    if len(ytvos.anns) == 0:
        print("Annotations does not exist")
        return
    assert result_file.endswith('.json')
    ytvos_dets = ytvos.loadRes(result_file)

    vid_ids = ytvos.getVidIds()
    for res_type in result_types:
        iou_type = res_type
        ytvosEval = YTVOSeval(ytvos, ytvos_dets, iou_type, output_file=save_path_valid_metrics)
        ytvosEval.params.vidIds = vid_ids
        if res_type == 'proposal':
            ytvosEval.params.useCats = 0
            ytvosEval.params.maxDets = list(max_dets)
        ytvosEval.evaluate()
        ytvosEval.accumulate()
        ytvosEval.summarize()
