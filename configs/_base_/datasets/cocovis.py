import copy

COCO_VIS_CLASSES = ('person', 'car', 'motorcycle', 'airplane',
                    'train', 'truck', 'boat', 'bird', 'cat',
                    'dog', 'horse', 'cow', 'elephant', 'bear',
                    'zebra', 'giraffe', 'snowboard', 'skateboard', 'surfboard',
                    'tennis racket', 'mouse')

COCO_VIS_LABEL_MAP = {1: 26,   3: 5,   4: 23,  5: 1,
                      7: 36,   8: 37,  9: 4,  16: 3,  17: 6,
                      18: 9,  19: 19, 21: 7,  22: 12, 23: 2,
                      24: 40, 25: 18, 36: 31, 41: 29, 42: 33,
                      43: 34, 74: 24}


class DatasetCatalog_cocovis(object):
    DATA_DIR = '../datasets'
    dataset_base_coco = {
        'name': 'Base COCO Dataset',
        # Training images and annotations
        'img_prefix': 'coco/train2017/',
        'ann_file': 'path_to_annotation_file',

        # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
        'has_gt': True,

        # A list of names for each of you classes.
        'class_names': COCO_VIS_CLASSES,

        # COCO class ids aren't sequential, so this is a bandage fix. If your ids aren't sequential,
        # provide a map from category_id -> index in class_names + 1 (the +1 is there because it's 1-indexed).
        # If not specified, this just assumes category ids start at 1 and increase sequentially.
        'label_map': COCO_VIS_LABEL_MAP
    }

    coco2017_train_dataset = copy.deepcopy(dataset_base_coco)
    coco2017_train_dataset['name'] = 'COCO_2017_train'
    coco2017_train_dataset['img_prefix'] = 'coco/train2017/'
    coco2017_train_dataset['ann_file'] = 'coco/annotations/instances_train2017.json'

    coco2017_valid_dataset = copy.deepcopy(dataset_base_coco)
    coco2017_valid_dataset['name'] = 'COCO_2017_valid'
    coco2017_valid_dataset['img_prefix'] = 'coco/val2017/'
    coco2017_valid_dataset['ann_file'] = 'coco/annotations/instances_val2017.json'

    coco2017_testdev_dataset = copy.deepcopy(dataset_base_coco)
    coco2017_testdev_dataset['name'] = 'COCO_2017_testdev'
    coco2017_testdev_dataset['img_prefix'] = 'coco/test2017/'
    coco2017_testdev_dataset['ann_file'] = 'coco/annotations/image_info_test-dev2017.json'
    coco2017_testdev_dataset['has_gt'] = False

    DATASETS = {
                'coco2017_train_dataset': coco2017_train_dataset,
                'coco2017_valid_dataset': coco2017_valid_dataset,
                'coco2017_testdev_dataset': coco2017_testdev_dataset
                }