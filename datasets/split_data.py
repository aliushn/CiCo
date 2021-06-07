import json
import mmcv
import random
import os


def split_data(annotation_file):
    dataset_train, dataset_valid = dict(), dict()

    dataset = json.load(open(annotation_file, 'r'))
    for k, v in dataset.items():
        if k not in {'videos', 'annotations'}:
            dataset_train[k] = v
            dataset_valid[k] = v

    dataset_valid['videos'], dataset_valid['annotations'] = [], []
    dataset_train['videos'], dataset_train['annotations'] = [], []

    len_vid = len(dataset['videos'])
    num_valid_videos = int(len_vid * 0.1)
    valid_videos_idx = range(len_vid-num_valid_videos, len_vid)
    for idx, video in enumerate(dataset['videos']):
        if idx in valid_videos_idx:
            dataset_valid['videos'].append(video)
        else:
            dataset_train['videos'].append(video)

    for idx, ann in enumerate(dataset['annotations']):
        if ann['video_id']-1 in valid_videos_idx:
            dataset_valid['annotations'].append(ann)
        else:
            dataset_train['annotations'].append(ann)

    dir = os.path.split(annotation_file)[:-1][0]
    valid_path = ''.join([dir, '/annotations_valid_sub.json'])
    train_path = ''.join([dir, '/annotations_train_sub.json'])
    mmcv.dump(dataset_valid, valid_path)
    mmcv.dump(dataset_train, train_path)
    print('Done')


if __name__ == '__main__':
    annotations_file = '/home/lmh/Downloads/VIS/code/datasets/OVIS/annotations_train.json'
    split_data(annotations_file)