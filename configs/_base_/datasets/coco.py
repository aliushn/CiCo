import copy

COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock',
                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

COCO_LABEL_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                  9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}


class DatasetCatalog_coco(object):
    DATA_DIR = '../datasets'
    dataset_base_coco = {
        'name': 'Base COCO Dataset',
        # Training images and annotations
        'img_prefix': 'coco/train2017/',
        'ann_file': 'path_to_annotation_file',

        # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
        'has_gt': True,

        # A list of names for each of you classes.
        'class_names': COCO_CLASSES,

        # COCO class ids aren't sequential, so this is a bandage fix. If your ids aren't sequential,
        # provide a map from category_id -> index in class_names + 1 (the +1 is there because it's 1-indexed).
        # If not specified, this just assumes category ids start at 1 and increase sequentially.
        'label_map': COCO_LABEL_MAP
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