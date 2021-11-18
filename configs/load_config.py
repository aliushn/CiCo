import os
import sys
from importlib import import_module
from configs._base_.models import cfg
from yacs.config import CfgNode as CN


def load_config(config_file):
    # Update configs
    def load_cfg_from_py_file(file):
        config_dir, config_name = os.path.split(file)
        sys.path.insert(0, config_dir)
        mod = import_module(os.path.splitext(config_name)[0])
        sys.path.pop(0)
        cfg_base, cfg_dict = [], dict()
        for name, value in mod.__dict__.items():
            if not name.startswith('_'):
                cfg_dict[name] = value
            elif name.startswith('_base_'):
                cfg_base += value
        return cfg_base, cfg_dict

    cfg_base, cfg_dict = load_cfg_from_py_file(config_file)
    for cfg_file in cfg_base:
        if os.path.splitext(cfg_file)[1] in ['.yaml']:
            # print(os.getcwd())
            os.chdir('/home/lmh/Downloads/VIS/code/OSTMask/')
            cfg.merge_from_file(cfg_file)
        elif os.path.splitext(cfg_file)[1] in ['.py']:
            _, cfg_dict_cur = load_cfg_from_py_file(cfg_file)
            cfg.merge_from_other_cfg(CN(cfg_dict_cur))
    cfg.merge_from_other_cfg(CN(cfg_dict))

    return cfg