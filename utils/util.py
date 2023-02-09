import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import argparse
import numpy as np
import random
import os

def set_global_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # WARNING: if cudnn.enabled=False => greatly reduces training/inference speed.
    torch.backends.cudnn.enabled = True

def concat_jsons(json1, json2):
    # jsons are concatenated (merged) up to 2 levels in depth.
    for key in json1:
        if key in json2:
            for subkey in json1[key]:
                json2[key][subkey] = json1[key][subkey]
        else:
            json2[key] = json1[key]
    return json2

def update_config_with_arguments(config_dict, args, ARGS_TYPES, ARGS_CONFIGPATH):
    # double check
    for k in ARGS_CONFIGPATH:
        assert k in ARGS_TYPES, f"'{k}' not in ARGS_TYPES"
    for k in ARGS_TYPES:
        assert k in ARGS_CONFIGPATH, f"'{k}' not in ARGS_CONFIGPATH"

    for key in vars(args):
        new_value = getattr(args, key)
        if new_value is not None and key in ARGS_TYPES:
            # we find the place inside the nested dict where the new value needs to be put
            config_tree = config_dict
            for subtree_key in ARGS_CONFIGPATH[key][:-1]:
                config_tree = config_tree[subtree_key]

            key_tobereplaced = ARGS_CONFIGPATH[key][-1]
            config_tree[key_tobereplaced] = new_value

def add_dict_to_argparser(parser, default_dict):
    for k, v_type in default_dict.items():
        parser.add_argument(f"--{k}", type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def contains(self, key):
        return key in self._data.index

    def add_new(self, key):
        self._data.loc[key] = 0

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count