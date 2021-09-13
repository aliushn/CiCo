import copy

YouTube_VOS_CLASSES = ('person', 'giant_panda', 'lizard', 'parrot', 'skateboard',
                       'sedan', 'ape', 'dog', 'snake', 'monkey',
                       'hand', 'rabbit', 'duck', 'cat', 'cow',
                       'fish', 'train', 'horse', 'turtle', 'bear',
                       'motorbike', 'giraffe', 'leopard', 'fox', 'deer',
                       'owl', 'surfboard', 'airplane', 'truck', 'zebra',
                       'tiger', 'elephant', 'snowboard', 'boat', 'shark',
                       'mouse', 'frog', 'eagle', 'earless seal', 'tennis_racket')

YouTube_VOS_LABEL_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                         9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16,
                         17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24,
                         25: 25, 26: 26, 27: 27, 28: 28, 29: 29, 30: 30, 31: 31, 32: 32,
                         33: 33, 34: 34, 35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 40: 40}

YouTube_VOS2021_CLASSES = ('airplane', 'bear', 'bird', 'boat', 'car',
                           'cat', 'cow', 'deer', 'dog', 'duck',
                           'earless_seal', 'elephant', 'fish', 'flying_disc', 'fox',
                           'frog', 'giant_panda', 'giraffe', 'horse', 'leopard',
                           'lizard', 'monkey', 'motorbike', 'mouse', 'parrot',
                           'person', 'rabbit', 'shark', 'skateboard', 'snake',
                           'snowboard', 'squirrel', 'surfboard', 'tennis_racket', 'tiger',
                           'train', 'truck', 'turtle', 'whale', 'zebra')

OVIS_CLASSES = ('person', 'bird', 'cat', 'dog', 'horse',
                'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'poultry', 'giant panda', 'lizard', 'parrot',
                'monkey', 'rabbit', 'tiger', 'fish', 'turtle',
                'bicycle', 'motorcycle', 'airplane', 'boat', 'vehicle')

OVIS_LABEL_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                  9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16,
                  17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24,
                  25: 25}


class DatasetCatalog_vis(object):
    DATA_DIR = '../datasets'
    dataset_base_vis = {
        'name': 'Base VIS Dataset',
        # Training images and annotations
        'img_prefix': 'path_to_images_file',
        'ann_file': 'path_to_annotation_file',

        # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
        'has_gt': True,
        'with_crowd': False,
        'class_names': YouTube_VOS_CLASSES,
    }

    # YouTube 2019
    train_YouTube_VOS2019_dataset = copy.deepcopy(dataset_base_vis)
    train_YouTube_VOS2019_dataset['name'] = 'YTVIS2019_train'
    train_YouTube_VOS2019_dataset['img_prefix'] = 'YouTube_VOS2019/train/JPEGImages'
    train_YouTube_VOS2019_dataset['ann_file'] = 'YouTube_VOS2019/annotations_instances/valid_sub.json'
    train_YouTube_VOS2019_dataset['class_names'] = YouTube_VOS_CLASSES

    valid_sub_YouTube_VOS2019_dataset = copy.deepcopy(dataset_base_vis)
    valid_sub_YouTube_VOS2019_dataset['name'] = 'YTVIS2019_valid_sub'
    valid_sub_YouTube_VOS2019_dataset['img_prefix'] = 'YouTube_VOS2019/train/JPEGImages'
    valid_sub_YouTube_VOS2019_dataset['ann_file'] = 'YouTube_VOS2019/annotations_instances/valid_sub.json'
    valid_sub_YouTube_VOS2019_dataset['class_names'] = YouTube_VOS_CLASSES

    valid_YouTube_VOS2019_dataset = copy.deepcopy(dataset_base_vis)
    valid_YouTube_VOS2019_dataset['name'] = 'YTVIS2019_valid'
    valid_YouTube_VOS2019_dataset['img_prefix'] = 'YouTube_VOS2019/valid/JPEGImages'
    valid_YouTube_VOS2019_dataset['ann_file'] = 'YouTube_VOS2019/annotations_instances/valid.json'
    valid_YouTube_VOS2019_dataset['has_gt'] = False
    valid_YouTube_VOS2019_dataset['class_names'] = YouTube_VOS_CLASSES

    test_YouTube_VOS2019_dataset = copy.deepcopy(dataset_base_vis)
    test_YouTube_VOS2019_dataset['name'] = 'YTVIS2019_test'
    test_YouTube_VOS2019_dataset['img_prefix'] = 'YouTube_VOS2019/test/JPEGImages'
    test_YouTube_VOS2019_dataset['ann_file'] = 'YouTube_VOS2019/annotations_instances/test.json'
    test_YouTube_VOS2019_dataset['has_gt'] = False
    test_YouTube_VOS2019_dataset['class_names'] = YouTube_VOS_CLASSES

    # YouTube 2021
    train_YouTube_VOS2021_dataset = copy.deepcopy(dataset_base_vis)
    train_YouTube_VOS2021_dataset['name'] = 'YTVIS2021_train'
    train_YouTube_VOS2021_dataset['img_prefix'] = 'YouTube_VOS2021/train/JPEGImages'
    train_YouTube_VOS2021_dataset['ann_file'] = 'YouTube_VOS2021/train/instances.json'
    train_YouTube_VOS2021_dataset['class_names'] = YouTube_VOS2021_CLASSES

    valid_sub_YouTube_VOS2021_dataset = copy.deepcopy(dataset_base_vis)
    valid_sub_YouTube_VOS2021_dataset['name'] = 'YTVIS2021_valid_sub'
    valid_sub_YouTube_VOS2021_dataset['img_prefix'] = 'YouTube_VOS2021/train/JPEGImages'
    valid_sub_YouTube_VOS2021_dataset['ann_file'] = 'YouTube_VOS2021/train/valid_sub_150.json'
    valid_sub_YouTube_VOS2021_dataset['class_names'] = YouTube_VOS2021_CLASSES

    valid_YouTube_VOS2021_dataset = copy.deepcopy(dataset_base_vis)
    valid_YouTube_VOS2021_dataset['name'] = 'YTVIS2021_valid'
    valid_YouTube_VOS2021_dataset['img_prefix'] = 'YouTube_VOS2021/valid/JPEGImages'
    valid_YouTube_VOS2021_dataset['ann_file'] = 'YouTube_VOS2021/valid/instances.json'
    valid_YouTube_VOS2021_dataset['has_gt'] = False
    valid_YouTube_VOS2021_dataset['class_names'] = YouTube_VOS2021_CLASSES

    test_YouTube_VOS2021_dataset = copy.deepcopy(dataset_base_vis)
    test_YouTube_VOS2021_dataset['name'] = 'YTVIS2021_test'
    test_YouTube_VOS2021_dataset['img_prefix'] = 'YouTube_VOS2021/test/JPEGImages'
    test_YouTube_VOS2021_dataset['ann_file'] = 'YouTube_VOS2021/test/instances.json'
    test_YouTube_VOS2021_dataset['has_gt'] = False
    test_YouTube_VOS2021_dataset['class_names'] = YouTube_VOS2021_CLASSES

    # OVIS 2021
    train_OVIS_dataset = copy.deepcopy(dataset_base_vis)
    train_OVIS_dataset['name'] = 'OVIS_train'
    train_OVIS_dataset['img_prefix'] = 'OVIS/train'
    train_OVIS_dataset['ann_file'] = 'OVIS/annotations_train.json'
    train_OVIS_dataset['class_names'] = OVIS_CLASSES

    valid_sub_OVIS_dataset = copy.deepcopy(dataset_base_vis)
    valid_sub_OVIS_dataset['name'] = 'OVIS_valid_sub'
    valid_sub_OVIS_dataset['img_prefix'] = 'OVIS/train'
    valid_sub_OVIS_dataset['ann_file'] = 'OVIS/annotations_valid_sub.json'
    valid_sub_OVIS_dataset['class_names'] = OVIS_CLASSES

    valid_OVIS_dataset = copy.deepcopy(dataset_base_vis)
    valid_OVIS_dataset['name'] = 'OVIS_valid'
    valid_OVIS_dataset['img_prefix'] = 'OVIS/valid'
    valid_OVIS_dataset['ann_file'] = 'OVIS/annotations_valid.json'
    valid_OVIS_dataset['has_gt'] = False
    valid_OVIS_dataset['class_names'] = OVIS_CLASSES

    test_OVIS_dataset = copy.deepcopy(dataset_base_vis)
    test_OVIS_dataset['name'] = 'OVIS_test'
    test_OVIS_dataset['img_prefix'] = 'OVIS/test'
    test_OVIS_dataset['ann_file'] = 'OVIS/annotations_test.json'
    test_OVIS_dataset['has_gt'] = False
    test_OVIS_dataset['class_names'] = OVIS_CLASSES

    DATASETS = {
        'train_YouTube_VOS2019_dataset': train_YouTube_VOS2019_dataset,
        'valid_sub_YouTube_VOS2019_dataset': valid_sub_YouTube_VOS2019_dataset,
        'valid_YouTube_VOS2019_dataset': valid_YouTube_VOS2019_dataset,
        'test_YouTube_VOS2019_dataset': test_YouTube_VOS2019_dataset,

        'train_YouTube_VOS2021_dataset': train_YouTube_VOS2021_dataset,
        'valid_sub_YouTube_VOS2021_dataset': valid_sub_YouTube_VOS2021_dataset,
        'valid_YouTube_VOS2021_dataset': valid_YouTube_VOS2021_dataset,
        'test_YouTube_VOS2021_dataset': test_YouTube_VOS2021_dataset,

        'train_OVIS_dataset': train_OVIS_dataset,
        'valid_sub_OVIS_dataset': valid_sub_OVIS_dataset,
        'valid_OVIS_dataset': valid_OVIS_dataset,
        'test_OVIS_dataset': test_OVIS_dataset,


    }