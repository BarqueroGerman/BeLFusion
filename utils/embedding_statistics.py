import argparse
import torch
from utils.torch import *
from tqdm import tqdm
import data_loader as module_data
import models as module_arch
import os
import pandas
from utils import read_json, write_json
from parse_config import ConfigParser
import numpy as np
import urllib
from utils.visualization.generic import AnimationRenderer
from metrics.evaluation import apd, ade, fde, mmade, mmfde, get_multimodal_gt
import csv
import time 


def get_prediction(x, model):
    assert hasattr(model, "encode"), f"Model needs to have the method 'encode' implemented to extract an embedding."
    # right now we predict 'sample_num' times with our deterministic model
    y = model.encode(x)

    if isinstance(y, tuple) or isinstance(y, list):
        y = y[0] # always a tuple where first element is the prediction that needs to match "target"

    return y.to('cpu')


def prepare_model(config, data_loader_name, shuffle=False, drop_last=True, num_workers=None, batch_size=None):
    torch.set_grad_enabled(False)
    config[data_loader_name]["args"]["shuffle"] = shuffle
    config[data_loader_name]["args"]["drop_last"] = drop_last
    if batch_size is not None:
        config[data_loader_name]["args"]["batch_size"] = batch_size
    if num_workers is not None:
        config[data_loader_name]["args"]["num_workers"] = 0
    data_loader = config.init_obj(data_loader_name, module_data)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    #print(model)

    for i in range(torch.cuda.device_count()):
        print(f"> GPU {i} ready: {torch.cuda.get_device_name(i)}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading checkpoint: {} ...'.format(config.resume))
    if ".pth" not in config.resume: # support for models stored in ".p" format
        print("Loading from a '.p' checkpoint. Only evaluation is supported. Only model weights will be loaded.")
        import pickle
        state_dict = pickle.load(open(config.resume, "rb"))['model_dict']
    else: # ".pth" format
        checkpoint = torch.load(config.resume, map_location=device)
        state_dict = checkpoint['state_dict']

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()
    return model, data_loader, device, checkpoint


def compute_statistics(config, dataset_split, batch_size=None):
    #output_dir = os.path.join('./out/%s' % dataset_name, "%s_%s" % (exp_name, exp_id), dataset_split)
    data_loader_name = f"data_loader_{dataset_split}"

    model, data_loader, device, checkpoint = prepare_model(config, data_loader_name, shuffle=True, drop_last=False, num_workers=0, batch_size=batch_size)

    preds = []

    for batch_idx, batch in tqdm(enumerate(data_loader)):
        data, target, extra = batch
        data, target = data.to(device), target.to(device)
        
        pred = get_prediction(data, model) # target == data for autoencoders
        
        preds.append(pred)


    preds = torch.cat(preds, axis=0)

    checkpoint["statistics"] = {
        "min": preds.min(axis=0).values,
        "max": preds.max(axis=0).values,
        "mean": preds.mean(axis=0),
        "std": preds.std(axis=0),
        "var": preds.var(axis=0),
    }
    #print(checkpoint["statistics"])
    
    store_path =  config.resume#.replace(".pth", "_stats.pth")
    if dataset_split == "training":
        torch.save(checkpoint, store_path)
        print(f"Statistics successfully stored inside checkpoint: '{store_path}'.")
    else:
        # statistics must be computed with training set always
        print(f"Statistics must be computed with training set always. Skipping storing of statistics for '{dataset_split}' set.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', required=True)
    #parser.add_argument('-t', '--type', required=True, help="'obs' or 'pred' to feed into the model")
    parser.add_argument('-d', '--data', default='training')
    parser.add_argument('-b', '--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    #assert args.type in ["obs", "pred"]
    #assert "config.json" in args.cfg, "The argument 'cfg' must point to a 'config.json' file."

    config_path = os.path.join(os.path.dirname(args.cfg), "config.json")
    checkpoint_path = args.cfg

    if not os.path.exists(checkpoint_path):
        raise Exception(f"Checkpoint not found in: %s" % checkpoint_path)
        
        
    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    exp_folder = "/".join(checkpoint_path.split("/")[:-1])
    config_path = os.path.join(exp_folder, "config.json")
    exp_name, exp_id, checkpoint = checkpoint_path.split("/")[-3:]
    dataset = checkpoint_path.split("/")[-5]

    print(f"> Dataset: '{dataset}'")
    print(f"> Exp name: '{exp_name}'")
    print(f"> Exp ID: '{exp_id}'")
    print(f"> Checkpoint: '{checkpoint}'")
    
    config = read_json(config_path)
    configparser = ConfigParser(config, resume=os.path.join(exp_folder, checkpoint), save=False)


    t0 = time.time()
    compute_statistics(configparser, args.data, batch_size=args.batch_size)
    tim = int(time.time() - t0)


