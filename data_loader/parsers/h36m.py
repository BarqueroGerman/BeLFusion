# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from data_loader.h36m import H36MDataLoader


# python -m data_loader.parsers.h36m --gpu
# it will pre-process the Human36M dataset
if __name__ == '__main__':
    # run this script so that mean/var are precomputed from all training samples
    np.random.seed(0)
    
    training = ["S1", "S5", "S6", "S7", "S8"]
    test = ["S9", "S11"]
    actions = "all" 

    batch_size = 128
    annotations_folder = "./datasets/Human36M/"
    precomputed_folder = "./auxiliar/datasets/Human36M"
    obs_length = 25
    pred_length = 100
    stride = 10
    augmentation = 0

    # first time we call the data_loader, it will preprocess all subjects + generate statistics
    # with the subjects that we want to use for training
    data_loader = H36MDataLoader(batch_size, annotations_folder, precomputed_folder, 
                obs_length, pred_length, drop_root=True, 
                subjects=training, actions=actions, drop_last=False,
                stride=stride, shuffle=True, augmentation=0, normalize_data=False,
                dtype="float32")