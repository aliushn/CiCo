
class DatasetCatalog_vid(object):
    DATA_DIR = '../datasets'

    train_VID_dataset = {
        'name': 'VID_train',
        'img_prefix': 'ILSVRC2015/Data/VID',
        'ann_file': 'ILSVRC2015/Annotations/VID',
        'img_index': 'ILSVRC2015/ImageSets/VID_train_every10frames.txt',
        'has_gt': True,
        'with_crowd': False,
    }

    valid_sub_VID_dataset = {
        'name': 'VID_valid_sub',
        'img_prefix': 'ILSVRC2015/Data/VID',
        'ann_file': 'ILSVRC2015/Annotations/VID',
        'img_index': 'ILSVRC2015/ImageSets/VID_val_videos_every10frames.txt',
        'has_gt': True,
        'with_crowd': False,
    }

    valid_VID_dataset = {
        'name': 'VID_valid',
        'img_prefix': 'ILSVRC2015/Data/VID',
        'ann_file': 'ILSVRC2015/Annotations/VID',
        'img_index': 'ILSVRC2015/ImageSets/VID_val_videos.txt',
        'has_gt': False,
        'with_crowd': False,
    }

    train_DET_dataset = {
        'name': 'DET_train',
        'img_prefix': 'ILSVRC2015/Data/DET',
        'ann_file': 'ILSVRC2015/Annotations/DET',
        'img_index': 'ILSVRC2015/ImageSets/DET_train_30classes.txt',
        'has_gt': True,
        'with_crowd': False,
    }
    DATASETS = {
        'train_VID_dataset': train_VID_dataset,
        'valid_sub_VID_dataset': valid_sub_VID_dataset,
        'valid_VID_dataset': valid_VID_dataset,
        'train_DET_dataset': train_DET_dataset
    }