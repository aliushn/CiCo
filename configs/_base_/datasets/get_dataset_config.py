from .coco import DatasetCatalog_coco
from .cocovis import DatasetCatalog_cocovis
from .vid import DatasetCatalog_vid
from .vis import DatasetCatalog_vis


def get_dataset_config(name, data_type):
    if data_type == 'coco':
        attrs = DatasetCatalog_coco.DATASETS[name]
        attrs['data_dir'] = DatasetCatalog_coco.DATA_DIR
    elif data_type == 'cocovis':
        attrs = DatasetCatalog_cocovis.DATASETS[name]
        attrs['data_dir'] = DatasetCatalog_cocovis.DATA_DIR
    elif data_type == 'vis':
        attrs = DatasetCatalog_vis.DATASETS[name]
        attrs['data_dir'] = DatasetCatalog_vis.DATA_DIR
    elif data_type in {'vid', 'det'}:
        attrs = DatasetCatalog_vid.DATASETS[name]
        attrs['data_dir'] = DatasetCatalog_vid.DATA_DIR
    else:
        RuntimeError("Dataset not available: {}".format(name))

    return attrs