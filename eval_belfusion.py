
import matplotlib.pyplot as plt

import argparse
from lib2to3.pgen2.token import RARROW
from termios import N_SLIP
import torch
from utils.torch import *
from utils.util import AverageMeter

import models.diffusion as module_diffusion
import data_loader as module_data
import models as module_arch
import os
import pandas as pd
from utils import read_json, set_global_seed
from parse_config import ConfigParser
import numpy as np
from utils.visualization.generic import AnimationRenderer
from metrics.evaluation import lat_apd, get_multimodal_gt, cmd
import time
import json
from eval_baseline import BASELINES, get_stats_funcs
from metrics.fid import fid
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

def get_prediction(obs, pred, model, diffusion, sample_num, pred_length, steps=None, sampler_name="ddpm", silent=False):
    """
    If idces and to_store_folder != None => prediction will be loaded/stored to avoid generating it again.
    """

    # right now we predict 'sample_num' times with our deterministic model
    bs, obs_length, p, j, f = obs.shape
    diffusion_steps = diffusion.num_timesteps
    num_steps = diffusion_steps if steps is None else len(steps) # if unspecified, all denoising steps are stored

    ys = torch.zeros((bs, sample_num, num_steps, pred_length, p, j, f), device=obs.device)
    all_enc = torch.zeros((bs, sample_num, num_steps, 128), device=obs.device)

    toenumerate = range(sample_num) if silent else tqdm(range(sample_num))
    for i in toenumerate:
        model_args = {
            "obs": obs # for conditioning generation
        }
        shape = (bs, pred_length, p, j, f) # shape -> [N, Seq_length, Partic, Joints, Feat]
        
        sampler = getattr(diffusion, SAMPLERS[sampler_name])

        step_counter = 0
        for s, out in enumerate(sampler(model, shape, progress=False, model_kwargs=model_args, pred=pred)):
            if steps is None or s+1 in steps:
                ys[:, i, step_counter, :] = out["pred_xstart"]
                all_enc[:, i, step_counter] = out["pred_xstart_enc"]
                step_counter += 1
            

    return ys, all_enc

def prepare_model(config, data_loader_name, shuffle=False, augmentation=0, da_mirroring=0, da_rotations=0, drop_last=True, num_workers=None, batch_size=None, silent=False):
    for i in range(torch.cuda.device_count()):
        if not silent:
            print(f"> GPU {i} ready: {torch.cuda.get_device_name(i)}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.set_grad_enabled(False)
    #config["arch"]["args"]["beta_schedule"] = "linear"
    config[data_loader_name]["args"]["shuffle"] = shuffle

    config[data_loader_name]["args"]["da_mirroring"] = da_mirroring
    config[data_loader_name]["args"]["da_rotations"] = da_rotations
    config[data_loader_name]["args"]["augmentation"] = augmentation
    config[data_loader_name]["args"]["drop_last"] = drop_last
    if batch_size is not None:
        config[data_loader_name]["args"]["batch_size"] = batch_size
    if num_workers is not None:
        config[data_loader_name]["args"]["num_workers"] = 0
    data_loader = config.init_obj(data_loader_name, module_data)


    # build model architecture
    model = config.init_obj('arch', module_arch)
    diffusion = config.init_obj('diffusion', module_diffusion)
    #print(model)

    if not silent:
        print('Loading checkpoint: {} ...'.format(config.resume))
    if ".pth" not in config.resume: # support for models stored in ".p" format
        if not silent:
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
    
    return model, diffusion, data_loader, device

def visualize(config, dataset_split, output_dir, samples=5, ncols=0, type='2d', 
                batch_size=None, store=False, store_idx=-1,
                diffusion_stride=1, sampler_name="ddpm"):
    data_loader_name = f"data_loader_{dataset_split}"

     # IMPORTANT num_workers=0: problems with multiprocessing + generators
    model, diffusion, data_loader, device = prepare_model(config, data_loader_name, shuffle=True, num_workers=0, batch_size=batch_size)
    diff_steps = diffusion.num_timesteps
    print(f'Model loaded to "{device}"! Running inference with batch_size={config[data_loader_name]["args"]["batch_size"]} for data_loader={dataset_split}.')

    assert store_idx > 0 or store_idx == -1, "store_idx must be -1 (last denoising step) or > 0"
    if store_idx == -1 and not store: # if we don't want any specific step, we show all steps
        steps_to_show = list(range(1, diff_steps+1, diffusion_stride))
        if diff_steps not in steps_to_show:
            steps_to_show += [diff_steps, ] # last step always evaluated, no matter what stride used
    elif store_idx == -1 and store:
        steps_to_show = [diff_steps, ] # last step only needed
    else:
        steps_to_show = [store_idx, ]

    # print info about the steps shown
    if len(steps_to_show) == 1:
        print(f"Loading predictions at denoising step {steps_to_show[0]}")
    else:
        print(f"Loading predictions at denoising steps '{steps_to_show}'. Press keys from 0 to {len(steps_to_show)-1} to change the step shown.")

    def pose_generator():
        while True:
            for batch_idx, batch in enumerate(data_loader):
                data, target, extra = batch
                idces = extra["sample_idx"]
                data, target = data.to(device), target.to(device)
                
                pred_length = target.shape[1]
                predictions, _ = get_prediction(data, target, model, diffusion, samples, pred_length, steps=steps_to_show, sampler_name=sampler_name)
                predictions = predictions.cpu().numpy()
                # predictions -> [BS, nSamples, STEPS, Seq_length, Partic, Joints, Feat]

                # gt
                all_x_gt = data_loader.dataset.recover_landmarks(data.cpu().numpy(), rrr=True, fill_root=True)
                all_y_gt = data_loader.dataset.recover_landmarks(target.cpu().numpy(), rrr=True, fill_root=True)

                for sample_idx, sample_name in enumerate(idces):
                    x_gt, y_gt = all_x_gt[sample_idx], all_y_gt[sample_idx]

                    y_pred = predictions[sample_idx]

                    poses = {}

                    for baseline in BASELINES:
                        poses[baseline] = BASELINES[baseline](x_gt, y_gt)

                    y_pred = data_loader.dataset.recover_landmarks(y_pred, rrr=True, fill_root=True)
                    for i in range(samples):
                        for diff_step in range(len(steps_to_show)):
                            poses[f"diff{steps_to_show[diff_step]:04d}_{i}"] = np.concatenate([x_gt, y_pred[i, diff_step]], axis=0)

                    yield poses, sample_name.item()
            if store:
                break

    pose_gen = pose_generator()
    baselines = list(BASELINES.keys())
    ncols = ncols if ncols != 0 else len(baselines) + samples + 1
 
    algos = []
    for diff_step in range(len(steps_to_show)):
        algos.append(f"diff{steps_to_show[diff_step]:04d}")

    if store is not None:
        # disable data augmentation here, to compare across models
        data_loader.dataset.augmentation = 0
        data_loader.dataset.da_mirroring = 0.0
        data_loader.dataset.da_rotations = 0.0
        data_loader.shuffle = False
        # sample 'batch_size' idces from the dataset with seed '0' (samples choice will ALWAYS be the same)
        np.random.seed(0)
        data_loader.samples_to_track = np.random.choice(len(data_loader.dataset), 64, replace=False)
        data_loader = data_loader.get_tracked_sampler()
        AnimationRenderer(data_loader.dataset.skeleton, pose_gen, algos, config.config["obs_length"], config.config["pred_length"], 
                            ncol=ncols, size=4,
                            output_dir=output_dir, baselines=baselines, type=type).store_all(type=store, idx=-1)
    else:
        AnimationRenderer(data_loader.dataset.skeleton, pose_gen, algos, config.config["obs_length"], config.config["pred_length"], 
                            ncol=ncols, size=4,
                            output_dir=output_dir, baselines=baselines, type=type).run_animation()

def compute_stats(config, dataset_split, multimodal_threshold, samples=5, batch_size=None, #num_seeds=1, 
                silent=False, diffusion_stride=25, find_best_batch_size = False, 
                store_folder=None, sampler_name="ddpm", stats_mode="no_mm", metrics_at_cpu=False):
                
    data_loader_name = f"data_loader_{dataset_split}"
    if store_folder is not None:
        os.makedirs(store_folder, exist_ok=True)

    model, diffusion, data_loader, device = prepare_model(config, data_loader_name, shuffle=False, drop_last=False, num_workers=0, batch_size=batch_size, silent=silent)
    diffusion_steps = diffusion.num_timesteps
    metrics_device = device if not metrics_at_cpu else "cpu"
    
    stats_func = get_stats_funcs(stats_mode)
    stats_func["latAPD"] = lat_apd
    if "MMADE" in stats_func or "MMFDE" in stats_func:
        if not silent:
            print(f'Model loaded to "{device}"! Running inference with batch_size={config[data_loader_name]["args"]["batch_size"]} for data_loader={dataset_split}.')

            print(f"Computing multimodal GT...")
        traj_gt_arr = get_multimodal_gt(data_loader, multimodal_threshold, device=metrics_device, split=dataset_split)
    else:
        traj_gt_arr = None

    IDX_TO_CLASS = data_loader.dataset.idx_to_class
    CLASS_TO_IDX = data_loader.dataset.class_to_idx
    try:
        classifier_for_fid = data_loader.dataset.get_classifier(metrics_device)
        if not silent:
            print(f"Classifier correctly loaded!")
    except:
        classifier_for_fid = None

    # for APDE computation
    mmapd_gt_path = os.path.join(data_loader.dataset.precomputed_folder, "mmapd_GT.csv")
    assert os.path.exists(mmapd_gt_path), f"Cannot find mmapd_GT.csv in {data_loader.dataset.precomputed_folder}"
    mmapd_gt = pd.read_csv(mmapd_gt_path, index_col=0)["gt_APD"]
    mmapd_gt = mmapd_gt.replace(0, np.NaN)
    
    if find_best_batch_size: # we find the biggest batch size that works with the given GPU resources
        print("Checking how big the batch can be...")
        initial_batch_size = 4096 # all test set fits to this size
        found = False
        while not found:
            config[data_loader_name]["args"]["batch_size"] = initial_batch_size
            data_loader = config.init_obj(data_loader_name, module_data)
            data, target, extra = data_loader.__iter__().__next__()
            idces = extra["sample_idx"]
            
            try:
                data, target = data.to(device), target.to(device)
                bs, obs_length, p, j, f = data.shape
                pred_length = target.shape[1]
                shape = [bs, pred_length, p, j, f]
                arr = torch.randn(*([diffusion_steps // diffusion_stride + 1, ] + shape), device=device, dtype=torch.float32)
                for i in diffusion.p_sample_loop_progressive(model, shape, progress=False, model_kwargs={ "obs": data }):
                    break # only one forward step needed to check if it fits
                found = True
            except:
                initial_batch_size //= 2
                #print(f"[FAILED] Not enough memory in GPU. Trying with batch_size={initial_batch_size}...")
                
        if not found:
            raise Exception(f"GPU memory does not fit any batch size. Check what is happening!")
        del data
        del target
        del arr
        initial_batch_size = len(idces) // 4 # just to make sure that we have enough memory for metrics computation
        config[data_loader_name]["args"]["batch_size"] = initial_batch_size
        data_loader = config.init_obj(data_loader_name, module_data)
        print(f"Found! ----> batch_size={initial_batch_size}...")

    batch_size = data_loader.batch_size

    if not silent:
        print(f"Done! Starting evaluation...")

    if diffusion_stride != -1:
        steps_to_evaluate = list(range(1, diffusion_steps+1, diffusion_stride))
        if diffusion_steps not in steps_to_evaluate:
            steps_to_evaluate += [diffusion_steps, ] # last step always evaluated, no matter what stride used
    else:
        steps_to_evaluate = [diffusion_steps, ] # only final result

    stats_names = list(stats_func.keys())
    stats_meter = {x: 
                    {
                        str(i): AverageMeter() for i in steps_to_evaluate
                    } 
                    for x in stats_names}
    histogram_data = {str(i): [] for i in steps_to_evaluate}

    all_gt_activations = {str(i): [] for i in steps_to_evaluate} # for FID. We need to compute the activations of the GT
    all_pred_activations = {str(i): [] for i in steps_to_evaluate} # for FID. We need to compute the activations of the predictions
    all_pred_classes = {str(i): [] for i in steps_to_evaluate}
    all_gt_classes = {str(i): [] for i in steps_to_evaluate} 
    all_obs_classes = {str(i): [] for i in steps_to_evaluate}
    
    all_results = np.zeros((len(data_loader.dataset), 2 + len(stats_names), len(steps_to_evaluate))) 
    column_names = ['id', 'class_gt']
    for n in stats_names:
        for i in range(len(steps_to_evaluate)):
            column_names.append(f"{n}_{steps_to_evaluate[i]}")

    counter = 0
    batches_toenumerate = enumerate(data_loader) if not silent else enumerate(tqdm(data_loader))
    for nbatch, batch in batches_toenumerate:
        data, target, extra = batch
        idces = extra["sample_idx"]
        data, target = data.to(device), target.to(device)
        pred_length = target.shape[1]

        f, t = nbatch * batch_size, min(all_results.shape[0], (nbatch + 1) * batch_size)
        all_results[f:t, 0] = idces.numpy()[..., None]
        
        #for i in range(num_seeds):
        if not silent:
            print(f"Generating {samples} samples for batch {nbatch+1}/{len(data_loader)} (batch_size={batch_size})")
        pred, lat_pred = get_prediction(data, target, model, diffusion, samples, pred_length, silent=silent,
                                steps=steps_to_evaluate, sampler_name=sampler_name) # [batch_size, n_samples, seq_length, num_part, num_joints, features]

        lat_pred = lat_pred.to('cpu')

        # predictions -> [BS, nSamples, STEPS, Seq_length, Partic, Joints, Feat]
        pred = data_loader.dataset.recover_landmarks(pred, rrr=False) # do not recover root, only denormalize if needed
        pred_flat = pred[:, :, :, :, 0] # we squeeze the participants' axis

        # gt
        target = data_loader.dataset.recover_landmarks(target, rrr=False)
        # all_gt -> [batch_size, seq_length, num_part, num_joints, features]

        if metrics_at_cpu:
            if not silent:
                print(f"Moving data to CPU... And computing metrics for {len(steps_to_evaluate)} denoising steps.")
            data, pred, target = data.cpu(), pred.cpu(), target.cpu()
        elif not silent:
            print(f"Computing metrics for {len(steps_to_evaluate)} denoising steps at GPU...")

        steps_toenumerate = tqdm(range(len(steps_to_evaluate))) if not silent else range(len(steps_to_evaluate))
        for step in steps_toenumerate:
            for k, stats in enumerate(stats_names):
                # pred has shape (batch_size, num_samples, diff_steps, seq_length, num_joints, features)
                mm_traj = traj_gt_arr[counter: counter + target.shape[0]] if traj_gt_arr is not None else None
                values = stats_func[stats](target, pred[:, :, step], mm_traj, lat_pred[:, :, step]).cpu().numpy()
                for i in range(values.shape[0]):
                    stats_meter[stats][str(steps_to_evaluate[step])].update(values[i]) # individual update for each batch element
                    
                    all_results[nbatch * batch_size + i, k+2, step] = values[i] if not isinstance(values[i], np.ndarray) else values[i].mean()

            # pred_flat -> [batch_size, samples, steps, seq_length, n_features])
            # we append the motion (L2 distance between pose at t-1 and t) of the predictions for the CMD computation
            motion = (torch.linalg.norm(pred_flat[:, :, step, 1:] - pred_flat[:, :, step, :-1], axis=-1)).mean(axis=1).mean(axis=-1)
            histogram_data[str(steps_to_evaluate[step])].append(motion.cpu().detach().numpy())

            classes = np.array([CLASS_TO_IDX[c] for c in extra["metadata"][data_loader.dataset.metadata_class_idx]])
            all_obs_classes[str(steps_to_evaluate[step])].append(classes)
            if classifier_for_fid is not None:
                # ----------------------------- Computing features for FID -----------------------------
                # pred -> [batch_size, samples, steps, seq_length, n_part, n_joints, n_features])
                pred_step_ = pred.reshape(list(pred.shape[:-2]) + [-1, ])[:, :, step, :, 0, :] # [batch_size, samples, seq_length, n_features])
                pred_step_ = pred_step_.reshape([-1, ] + list(pred_step_.shape[-2:])) # [batch_size * samples, seq_length, n_features])
                pred_step_ = pred_step_.permute(0, 2, 1) # [batch_size * samples, n_features, seq_length])

                # same for target: but no need for step + no need to join batch_size + samples
                target_step_ = target.reshape(list(target.shape[:-2]) + [-1, ])[..., 0, :] # [batch_size, samples, seq_length, n_features])
                target_step_ = target_step_.permute(0, 2, 1) # [batch_size * samples, n_features, seq_length])

                pred_activations = classifier_for_fid.get_fid_features(motion_sequence=pred_step_).cpu().data.numpy()
                gt_activations = classifier_for_fid.get_fid_features(motion_sequence=target_step_).cpu().data.numpy()

                all_gt_activations[str(steps_to_evaluate[step])].append(gt_activations)
                all_pred_activations[str(steps_to_evaluate[step])].append(pred_activations)

                pred_classes = classifier_for_fid(motion_sequence=pred_step_.float()).cpu().data.numpy().argmax(axis=1)
                # recover the batch size and samples dimension
                pred_classes = pred_classes.reshape([pred.shape[0], samples])
                gt_classes = classifier_for_fid(motion_sequence=target_step_.float()).cpu().data.numpy().argmax(axis=1)
                # append to the list
                all_pred_classes[str(steps_to_evaluate[step])].append(pred_classes)
                all_gt_classes[str(steps_to_evaluate[step])].append(gt_classes)

        counter += target.shape[0]
        if not silent:
            print('-' * 80)
            for stats in stats_meter:
                s = stats_meter[stats][str(diffusion_steps)]
                if not isinstance(s.val, np.ndarray):
                    print(f'{counter-batch_size}-{counter:04d} {stats}: {s.val:.4f}({s.avg:.4f})')
                else:
                    print(f'{counter-batch_size}-{counter:04d} {stats}: {s.val.mean():.4f}({s.avg.mean():.4f})')
        #break
        
    results = {}
    results["steps"] = steps_to_evaluate

    # ----------------------------- Computing FID -----------------------------
    for step in steps_to_evaluate:
        if classifier_for_fid is not None:
            step = str(step)
            if "FID" not in results:
                results["FID"] = []
            results["FID"].append(fid(np.concatenate(all_gt_activations[step], axis=0), np.concatenate(all_pred_activations[step], axis=0)))
            step_obs_classes = np.concatenate(all_obs_classes[step], axis=0)

    # ----------------------------- Computing CMD -----------------------------
    try:
        results[f"CMD"] = [0] * len(steps_to_evaluate)
        motion_datas = {}
        for step_idx, step in enumerate(steps_to_evaluate):
            step = str(step)
            step_obs_classes = np.concatenate(all_obs_classes[step], axis=0)
            motion_data = np.concatenate(histogram_data[step], axis=0)
            motion_data_mean = motion_data.mean(axis=0)
            motion_datas[step] = motion_data_mean

            motion_per_class = np.zeros((len(IDX_TO_CLASS), motion_data.shape[1]))
            # CMD weighted by class
            for i, (name, class_val_ref) in enumerate(zip(IDX_TO_CLASS, data_loader.dataset.mean_motion_per_class)):
                mask = step_obs_classes == i
                if mask.sum() == 0:
                    continue
                motion_data_mean = motion_data[mask].mean(axis=0)
                motion_per_class[i] = motion_data_mean
                results["CMD"][step_idx] += cmd(motion_data_mean, class_val_ref) * (mask.sum() / step_obs_classes.shape[0])
    except Exception as e:
        print(f"Error computing motion: {e}")
        print("Motion computation failed. Probably due to missing motion mean in dataset class.")

    # ----------------------------- Averaging scores for each step evaluated -----------------------------
    for stats in stats_meter:
        if not isinstance(stats_meter[stats][str(steps_to_evaluate[0])].val, np.ndarray):
            results[stats] = [stats_meter[stats][str(step)].avg for step in steps_to_evaluate]
        else:
            results[stats] = [[float(val) for val in stats_meter[stats][str(step)].avg] for step in steps_to_evaluate] # to json serializable 
            results[stats + "_avg"] = [float(stats_meter[stats][str(step)].avg.mean()) for step in steps_to_evaluate]

    # ----------------------------- Storing sequent-wise results + APDE -----------------------------
    all_results = all_results.reshape(all_results.shape[0], -1)
    idces = [0, len(steps_to_evaluate)] + [i for i in range(2 * len(steps_to_evaluate), all_results.shape[1])]
    all_results = all_results[:,idces].astype(np.float32) # we remove duplicate idces
    df = pd.DataFrame(all_results, columns=column_names)
    sw_path = os.path.join(store_folder, f"results_{samples}_{diffusion_stride}.csv")
    assert len(mmapd_gt) == len(df), f"mmapd_gt and df have different length: {len(mmapd_gt)} vs {len(df)}"

    # APDE computation
    all_apdes = []
    for i in range(len(steps_to_evaluate)):
        apd_n = f"APD_{steps_to_evaluate[i]}"
        apde_n = f"APDE_{steps_to_evaluate[i]}"
        df[apde_n] = abs(df[apd_n] - mmapd_gt)
        all_apdes.append(np.mean(df[apde_n]))
    results["APDE"] = all_apdes

    # store sequent-wise results
    df.to_csv(sw_path, index=False)

    # ----------------------------- Printing results -----------------------------
    print('=' * 80)
    for stats in results:
        print(f'Total {stats}: {results[stats][-1]:.4f}')
    print('=' * 80)

    # ----------------------------- Storing overall results -----------------------------
    # x-axis -> diffusion steps, y-axis -> stat
    if len(steps_to_evaluate) > 1:
        # plot results
        steps = results["steps"]
        for stat in results:
            if stat.lower() == "steps":
                continue
            stat_name = f"test_{samples}_{stat}"
            if isinstance(results[stat][0], list):
                continue # skip non-scalar stats
            ys = [results[stat][i] for i, step in enumerate(steps)]
            plt.plot(steps, ys, label=stat_name)
            plt.title(stat_name)
            plt.savefig(os.path.join(store_folder, f"{stat_name}.png"))
            plt.clf()

    # write results as json in plots folder
    ov_path = os.path.join(store_folder, f"results_{samples}.json")
    with open(ov_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Sequence-wise results saved to {sw_path}")
    print(f"Overall results saved to {ov_path}")
    print('=' * 80)
    return results


SAMPLERS = {
    "ddpm": "p_sample_loop_progressive",
    "ddim": "ddim_sample_loop_progressive"
}

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

    parser.add_argument('--dstride', type=int, default=-1)
    parser.add_argument('-i', '--iter', type=int, default=None)
    parser.add_argument('-e', '--ema', action='store_true')
    parser.add_argument('-sampler', '--sampler', default='ddim', help=f"options={list(SAMPLERS.keys())}")
    parser.add_argument('-store_idx', '--store_idx', type=int, default=-1) # index of diffusion step to be stored
    parser.add_argument('--silent', action='store_true')

    args = parser.parse_args()
    assert args.sampler in SAMPLERS, f"options for sampling={list(SAMPLERS.keys())}"

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_global_seed(args.seed)

    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    
    # build the config/checkpoint path
    ext = ".pth"
    if args.ema:
        ext = "_ema" + ext

    iter = "best" if args.iter is None else args.iter
    config_path = args.cfg
    checkpoint_path = os.path.join(args.cfg, f"model_best{ext}" if iter == "best" else f"checkpoint-epoch{iter}{ext}")
    checkpoint_folder = os.path.join(args.cfg, str(iter))
    if not os.path.exists(checkpoint_path):
        raise Exception(f"Checkpoint not found in: %s" % checkpoint_path)
        
    exp_folder = "/".join(checkpoint_path.split("/")[:-1])
    config_path = os.path.join(exp_folder, "config.json")
    exp_name, exp_id, checkpoint = checkpoint_path.split("/")[-3:]
    dataset_name = checkpoint_path.split("/")[-4]
    dataset_split = args.data

    print(f"> Dataset: '{dataset_name}'")
    print(f"> Exp name: '{exp_name}'")
    print(f"> Exp ID: '{exp_id}'")
    print(f"> Checkpoint: '{checkpoint}'")
    
    # read config json
    config = read_json(config_path)
    configparser = ConfigParser(config, resume=os.path.join(exp_folder, checkpoint), save=False)


    if args.mode == 'vis' or args.mode == 'gen':
        store = 'gif' if args.mode == 'gen' else None # --> generate and store random generated sequences of a single batch.
        if store:
            print("Generating random sequences and storing them as 'gif'...")
        num_samples = args.samples if args.samples != -1 else 5
        output_folder = os.path.join(output_folder, '%s' % dataset_name, "%s_%s" % (exp_name, exp_id), str(iter), f"{dataset_split}_{'ema_' if args.ema else ''}{args.sampler}")
        os.makedirs(output_folder, exist_ok=True)
        visualize(configparser, args.data, output_folder,
                    samples=num_samples, ncols=args.ncols, type=args.type, diffusion_stride=args.dstride,
                    batch_size=args.batch_size, store=store, store_idx=args.store_idx, sampler_name=args.sampler)
    elif args.mode == 'stats':
        print(f"[WARNING] Remember: batch_size has an effect over the randomness of results. Keep batch_size fixed for comparisons, or implement several runs with different seeds to reduce stochasticity.")
        num_samples = args.samples if args.samples != -1 else 50
        stats_folder = os.path.join(exp_folder, str(iter), f"eval_{'ema_' if args.ema else ''}{args.sampler}")
        os.makedirs(stats_folder, exist_ok=True)
        t0 = time.time()
        compute_stats(configparser, args.data, args.multimodal_threshold, samples=num_samples, batch_size=args.batch_size,
                        diffusion_stride=args.dstride, silent=args.silent, find_best_batch_size=False, store_folder=stats_folder, sampler_name=args.sampler,
                        stats_mode=args.stats_mode, metrics_at_cpu=args.cpu)
        tim = int(time.time() - t0)
        print(f"[INFO] Evaluation took {tim // 60}min, {tim % 60}s.")

    else:
        raise NotImplementedError()


