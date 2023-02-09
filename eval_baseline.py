import argparse
import torch
from utils.torch import *
from utils.util import AverageMeter
import data_loader as module_data
import models as module_arch
import os
from utils import read_json, set_global_seed
from parse_config import ConfigParser
import numpy as np
from utils.visualization.generic import AnimationRenderer
from metrics.evaluation import apd, ade, fde, mmade, mmfde, cmd, get_multimodal_gt
import time 
import json
from metrics.fid import fid
import pandas as pd

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

def get_stats_funcs(mode):
    if "no_mm" == mode.lower():
        stats_func = { 'APD': apd, 'ADE': ade, 'FDE': fde }
    elif "all" == mode.lower():
        stats_func = {  'APD': apd, 'ADE': ade, 'FDE': fde, 
                        'MMADE': mmade, 'MMFDE': mmfde,
                    }
    else :
        raise NotImplementedError(f"stats_mode not implemented: {mode}")
    return stats_func

# BASELINES for VISUAL comparison
BASELINES = {
                "context": lambda x, y: np.concatenate([x, y], axis=0), # compulsory, as it freezes once it starts the prediction
                "gt": lambda x, y: np.concatenate([x, y], axis=0),
            } # show baselines as comparison


def get_prediction(x, model, sample_num, sample_idces=None, device='cpu', extra=None):
    try:
        ys = model(x, None, sample_num) # the model already predicts 'sample_num' samples in its forward function
    except:
        ys = model(x, None, 50) # for DLow-like models. this is shitty programming, but I don't have time to fix it
        if sample_num != 50: # DLow models predict 50 samples, but we want to sample only 'sample_num' samples
            # randomly sample_num samples from dim=1
            ys = ys[:, torch.randperm(ys.shape[1])[:sample_num], ...]

    return ys

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

    for i in range(torch.cuda.device_count()):
        print(f"> GPU {i} ready: {torch.cuda.get_device_name(i)}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prepare model for testing
    model = model.to(device)
    model.eval()
    return model, data_loader, device

def visualize(config, dataset_split, output_dir, samples=5, ncols=0, type='2d', batch_size=None, store=False):
    data_loader_name = f"data_loader_{dataset_split}"

     # IMPORTANT num_workers=0: problems with multiprocessing + generators
    model, data_loader, device = prepare_model(config, data_loader_name, shuffle=True, num_workers=0, batch_size=batch_size)
    print(f'Model loaded to "{device}"! Running inference with batch_size={config[data_loader_name]["args"]["batch_size"]} for data_loader={dataset_split}.')

    # for visualization of custom samples
    def pose_generator():
        while True:
            for batch_idx, batch in enumerate(data_loader):
                data, target, extra = batch
                idces = extra["sample_idx"]
                data, target = data.to(device), target.to(device)
                
                predictions = get_prediction(data, model, sample_num=samples, sample_idces=idces, device=device, extra={
                    "obs": data,
                    "target": target,
                })

                # gt
                all_x_gt = data_loader.dataset.recover_landmarks(data.cpu().numpy(), rrr=True, fill_root=True)
                all_y_gt = data_loader.dataset.recover_landmarks(target.cpu().numpy(), rrr=True, fill_root=True)

                for sample_idx, sample_name in enumerate(idces):
                    x_gt, y_gt = all_x_gt[sample_idx], all_y_gt[sample_idx]
                    y_pred = predictions[sample_idx]

                    poses = {}
                    for baseline in BASELINES:
                        poses[baseline] = BASELINES[baseline](x_gt, y_gt)

                    y_pred = data_loader.dataset.recover_landmarks(y_pred.cpu().numpy(), rrr=True, fill_root=True)
                    for i in range(samples):
                        poses[f'{"algorithm"}_{i}'] = np.concatenate([x_gt, y_pred[i]], axis=0)

                    yield poses, sample_name.item()
            if store:
                break

    pose_gen = pose_generator()
    baselines = list(BASELINES.keys())
    ncols = ncols if ncols != 0 else len(baselines) + samples

    if store is not None:
        data_loader.dataset.augmentation = 0
        data_loader.dataset.da_mirroring = 0.0
        data_loader.dataset.da_rotations = 0.0
        data_loader.shuffle = False
        # sample 'batch_size' idces from the dataset with seed '0' (samples choice will ALWAYS be the same, for comparison)
        np.random.seed(0)
        idces = np.random.choice(len(data_loader.dataset), batch_size, replace=False)
        data_loader.samples_to_track = idces
        data_loader = data_loader.get_tracked_sampler()
        AnimationRenderer(data_loader.dataset.skeleton, pose_gen, ["algorithm", ], config.config["obs_length"], config.config["pred_length"], 
                            ncol=ncols, size=4, output_dir=output_dir, baselines=baselines, type=type).store_all(type=store)
    else:
        AnimationRenderer(data_loader.dataset.skeleton, pose_gen, ["algorithm", ], config.config["obs_length"], config.config["pred_length"], 
                            ncol=ncols, output_dir=output_dir, baselines=baselines, type=type).run_animation()


def compute_stats(config, dataset_split, multimodal_threshold, samples=5, batch_size=None, 
                    store_folder=None, stats_mode="no_mm", metrics_at_cpu=False):
    data_loader_name = f"data_loader_{dataset_split}"

    model, data_loader, device = prepare_model(config, data_loader_name, shuffle=False, drop_last=False, num_workers=0, batch_size=batch_size)
    print(f'Model loaded to "{device}"! Running inference with batch_size={config[data_loader_name]["args"]["batch_size"]} for data_loader={dataset_split}.')

    metrics_device = device if not metrics_at_cpu else "cpu"
    IDX_TO_CLASS = data_loader.dataset.idx_to_class
    CLASS_TO_IDX = data_loader.dataset.class_to_idx
    try:
        classifier_for_fid = data_loader.dataset.get_classifier(metrics_device)
        print("Loading classifier for FID and Accuracy metrics...")
        print(f"Classifier correctly loaded!")
    except:
        classifier_for_fid = None

    # for APDE computation
    mmapd_gt_path = os.path.join(data_loader.dataset.precomputed_folder, "mmapd_GT.csv")
    assert os.path.exists(mmapd_gt_path), f"Cannot find mmapd_GT.csv in {data_loader.dataset.precomputed_folder}"
    mmapd_gt = pd.read_csv(mmapd_gt_path, index_col=0)["gt_APD"]
    mmapd_gt = mmapd_gt.replace(0, np.NaN)

    stats_func = get_stats_funcs(stats_mode)
    if "MMADE" in stats_func or "MMFDE" in stats_func:
        print(f"Computing multimodal GT...")
        traj_gt_arr = get_multimodal_gt(data_loader, multimodal_threshold, device=metrics_device)

        # store apd to pandas
        print(f"Done! Starting evaluation...")
    else:
        traj_gt_arr = None


    stats_names = list(stats_func.keys())
    stats_meter = {x: AverageMeter() for x in stats_names}
    histogram_data = []

    all_gt_activations = [] # for FID. We need to compute the activations of the GT
    all_pred_activations = [] # for FID. We need to compute the activations of the predictions
    all_obs_classes = []

    all_results = np.zeros((len(data_loader.dataset), 2 + len(stats_names))) 
    column_names = ['id', 'class_gt']
    for n in stats_names:
        column_names.append(f"{n}")

    counter = 0
    batch_size = data_loader.batch_size
    for nbatch, batch in enumerate(data_loader):
        
        data, target, extra = batch
        data, target = data.to(device), target.to(device)
        idces = extra["sample_idx"]

        f, t = nbatch * batch_size, min(all_results.shape[0], (nbatch + 1) * batch_size)
        all_results[f:t, 0] = idces.numpy()
        
        pred = get_prediction(data, model, sample_num=samples, sample_idces=idces, device=metrics_device, extra={
                    "obs": data,
                    "target": target,
                }) # [batch_size, n_samples, seq_length, num_part, num_joints, features]

        # predictions -> [BS, nSamples, STEPS, Seq_length, Partic, Joints, Feat]
        pred = data_loader.dataset.recover_landmarks(pred, rrr=False) # do not recover root, only denormalize if needed

        # gt
        target = data_loader.dataset.recover_landmarks(target, rrr=False)
        # all_gt -> [batch_size, seq_length, num_part, num_joints, features]

        
        if metrics_at_cpu:
            print("Moving data to CPU... And computing metrics")
            pred, target = pred.cpu(), target.cpu()

        for k, stats in enumerate(stats_names):
            # pred has shape (batch_size, num_samples, diff_steps, seq_length, num_joints, features)
            mm_traj = traj_gt_arr[counter: counter + target.shape[0]] if traj_gt_arr is not None else None
            values = stats_func[stats](target, pred, mm_traj).cpu().numpy()
            for i in range(values.shape[0]):
                stats_meter[stats].update(values[i]) # individual update for each batch element

                all_results[f + i, k+2] = values[i] if not isinstance(values[i], np.ndarray) else values[i].mean()


        classes = [CLASS_TO_IDX[c] for c in extra["metadata"][data_loader.dataset.metadata_class_idx]]
        all_obs_classes.append(classes)
        if classifier_for_fid is not None:
            # ----------------------------- Computing features for FID -----------------------------
            # pred -> [batch_size, samples, seq_length, n_part, n_joints, n_features])
            pred_ = pred.reshape(list(pred.shape[:-2]) + [-1, ])[..., 0, :] # [batch_size, samples, seq_length, n_features])
            pred_ = pred_.reshape([-1, ] + list(pred_.shape[-2:])) # [batch_size * samples, seq_length, n_features])
            pred_ = pred_.permute(0, 2, 1) # [batch_size * samples, n_features, seq_length])

            # same for target: but no need for step + no need to join batch_size + samples
            target_ = target.reshape(list(target.shape[:-2]) + [-1, ])[..., 0, :] # [batch_size, samples, seq_length, n_features])
            target_ = target_.permute(0, 2, 1) # [batch_size * samples, n_features, seq_length])

            pred_activations = classifier_for_fid.get_fid_features(motion_sequence=pred_.float()).cpu().data.numpy()
            gt_activations = classifier_for_fid.get_fid_features(motion_sequence=target_.float()).cpu().data.numpy()

            all_gt_activations.append(gt_activations)
            all_pred_activations.append(pred_activations)


        motion_pred = pred[:, :, :, 0]
        motion = (torch.linalg.norm(motion_pred[:, :, 1:] - motion_pred[:, :, :-1], axis=-1)).mean(axis=-1).mean(axis=1)
        # this is for autoencoders evaluation
        histogram_data.append(motion.cpu().detach().numpy())


        counter += target.shape[0]

        print('-' * 80)
        for stats in stats_names:
            s = stats_meter[stats]
            if not isinstance(s.val, np.ndarray):
                print(f'{counter-batch_size}-{counter:04d} {stats}: {s.val:.4f}({s.avg:.4f})')
            else:
                print(f'{counter-batch_size}-{counter:04d} {stats}: {s.val.mean():.4f}({s.avg.mean():.4f})')
        #break

    results = {}
    
    # ----------------------------- Computing FID -----------------------------
    all_obs_classes = np.concatenate(all_obs_classes, axis=0)
    if classifier_for_fid is not None:
        # fid computation
        results["FID"] = fid(np.concatenate(all_gt_activations, axis=0), np.concatenate(all_pred_activations, axis=0))

    # ----------------------------- Computing CMD -----------------------------
    try:
        motion_data = np.concatenate(histogram_data, axis=0)

        motion_data_mean = motion_data.mean(axis=0)

        # CMD weighed by class
        results[f"CMD"] = 0
        for i, (name, class_val_ref) in enumerate(zip(IDX_TO_CLASS, data_loader.dataset.mean_motion_per_class)):
            mask = all_obs_classes == i
            if mask.sum() == 0:
                continue
            motion_data_mean = motion_data[mask].mean(axis=0)
            results["CMD"] += cmd(motion_data_mean, class_val_ref) * (mask.sum() / all_obs_classes.shape[0])
            
    except Exception as e:
        print(f"Error computing motion: {e}")
        print("Motion computation failed. Probably due to missing motion mean in dataset class.")

    # ----------------------------- Storing sequent-wise results + APDE -----------------------------
    all_results = all_results.reshape(all_results.shape[0], -1).astype(np.float32)
    df = pd.DataFrame(all_results, columns=column_names)
    sw_path = os.path.join(store_folder, f"results_{samples}.csv")
    assert len(mmapd_gt) == len(df), f"mmapd_gt and df have different length: {len(mmapd_gt)} vs {len(df)}"

    df["APDE"] = abs(df["APD"] - mmapd_gt)
    results["APDE"] = np.mean(df["APDE"])

    df.to_csv(sw_path, index=False)

    # ----------------------------- Averaging scores -----------------------------
    for stats in stats_meter:
        if not isinstance(stats_meter[stats].val, np.ndarray):
            results[stats] = stats_meter[stats].avg
        else:
            results[stats] = [float(val) for val in stats_meter[stats].avg] # to json serializable 
            results[stats + "_avg"] = float(stats_meter[stats].avg.mean())
    
    # ----------------------------- Printing results -----------------------------
    print('=' * 80)
    for stats in results:
        print(f'Total {stats}: {results[stats]:.4f}')
    print('=' * 80)


    # ----------------------------- Storing overall results -----------------------------
    ov_path = os.path.join(store_folder, f'results_{samples}.json')
    with open(ov_path, 'w') as f:
        json.dump(results, f)

    print(f"Sequence-wise results saved to {sw_path}")
    print(f"Overall results saved to {ov_path}")
    print('=' * 80)

    return results


# python
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', required=True)
    parser.add_argument('-m', '--mode', default='vis', type=str, help='vis: visualize results\ngen: generate and store all visualizations for a single batch\nstats: launch numeric evaluation')
    parser.add_argument('-stats_mode', '--stats_mode', type=str, default="no_mm")
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    parser.add_argument('-cpu', '--cpu', action='store_true')
    parser.add_argument('-s', '--samples', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-o', '--output_folder', type=str, default="./out")

    parser.add_argument('--ncols', type=int, default=0)
    parser.add_argument('-d', '--data', default='test')
    parser.add_argument('-t', '--type', default='3d') # 2d or 3d
    args = parser.parse_args()
    
    
    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_global_seed(args.seed)
    
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # build the config/checkpoint path
    exp_folder = args.cfg
    # remove trailing slash
    if exp_folder[-1] == '/':
        exp_folder = exp_folder[:-1]

    config_path = os.path.join(exp_folder, "config.json")
    dataset, exp_name = exp_folder.split("/")[-3:-1]

    print(f"> Dataset: '{dataset}'")
    print(f"> Exp name: '{exp_name}'")

    # read config json
    config = read_json(config_path)
    configparser = ConfigParser(config, save=False)


    if args.mode == 'vis' or args.mode == 'gen':
        store = 'gif' if args.mode == 'gen' else None # --> generate and store random generated sequences of a single batch.
        if store:
            print("Generating random sequences and storing them as 'gif'...")
        num_samples = args.samples if args.samples != -1 else 5
        output_dir = os.path.join('./out/%s' % dataset, "%s" % (exp_name), args.data)
        os.makedirs(output_dir, exist_ok=True)
        visualize(configparser, args.data, output_dir, samples=num_samples, 
                    ncols=args.ncols, type=args.type, batch_size=args.batch_size,
                    store=store)

    elif args.mode == 'stats':
        print(f"[WARNING] Remember: batch_size has an effect over the randomness of results. Keep batch_size fixed for comparisons.")
        num_samples = args.samples if args.samples != -1 else 50
        store_folder = os.path.join(exp_folder, "eval")
        os.makedirs(store_folder, exist_ok=True)
        t0 = time.time()
        compute_stats(configparser, args.data, args.multimodal_threshold, 
                    samples=num_samples, batch_size=args.batch_size, store_folder=store_folder,
                    stats_mode=args.stats_mode, metrics_at_cpu=args.cpu)
        tim = int(time.time() - t0)
        print(f"[INFO] Evaluation took {tim // 60}min, {tim % 60}s.")

    else:
        raise NotImplementedError()


