import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch
import pickle
import os
import json
from scipy import linalg
from scipy.stats import gaussian_kde
from tqdm import tqdm

def time_slice(array, t0, t, axis):
    if t == -1:
        return torch.index_select(array, axis, torch.arange(t0, array.shape[axis], device=array.device, dtype=torch.int32))
    else:
        return torch.index_select(array, axis, torch.arange(t0, t, device=array.device, dtype=torch.int32))
    
def cmd(val_per_frame, val_ref):
    # val_per_frame (array of floats) -> M_t, val_ref (float) -> M
    T = len(val_per_frame) + 1
    return np.sum([(T - t) * np.abs(val_per_frame[t-1] - val_ref) for t in range(1, T)])

def apd(target, pred, *args, t0=0, t=-1):
    pred = time_slice(pred, t0, t, 2)
    # (batch_size, num_samples, seq_length, num_parts, num_joints, features) to (num_samples, all_others)
    batch_size, n_samples = pred.shape[:2]
    if n_samples == 1: # only one sample => no APD possible
        return torch.tensor([0] * batch_size, device=pred.device)

    arr = pred.reshape(batch_size, n_samples, -1) # (batch_size, num_samples, others)
    dist = torch.cdist(arr, arr)
    dist_shape = dist.shape[-1]

    iu = np.triu_indices(dist_shape, 1) # symmetric matrix, only keep the upper ones, diagonal not considered
    pdist_shape = iu[0].shape[0]
    bs_indices = np.expand_dims(np.array([i for i in range(batch_size) for j in range(pdist_shape)]), 0) # we expand it to all batch_size
    values_mask = np.tile(iu, batch_size)
    final_mask = np.concatenate((bs_indices, values_mask), axis=0)

    # we filter only upper triangular values
    results = dist[final_mask].reshape((batch_size, pdist_shape)).mean(axis=-1)

    return results


def ade(target, pred, *args, t0=0, t=-1):
    pred, target = time_slice(pred, t0, t, 2), time_slice(target, t0, t, 1)
    batch_size, n_samples, seq_length = pred.shape[:3]
    # from (batch_size, num_samples, seq_length, num_parts, num_joints, features) to (batch_size, num_samples, seq_length, num_parts * num_joints * features)
    pred = pred.reshape((batch_size, n_samples, seq_length, -1))
    # from (batch_size, seq_length, num_parts, num_joints, features) to (batch_size, seq_length, num_parts * num_joints * features)
    target = target.reshape((batch_size, 1, seq_length, -1))

    diff = pred - target
    dist = torch.linalg.norm(diff, axis=-1).mean(axis=-1)
    return dist.min(axis=-1).values


def fde(target, pred, *args, t0=0, t=-1):
    pred, target = time_slice(pred, t0, t, 2), time_slice(target, t0, t, 1)
    batch_size, n_samples, seq_length = pred.shape[:3]
    # from (batch_size, num_samples, seq_length, num_parts, num_joints, features) to (batch_size, num_samples, seq_length, num_parts * num_joints * features)
    pred = pred.reshape((batch_size, n_samples, seq_length, -1))
    # from (batch_size, seq_length, num_parts, num_joints, features) to (batch_size, seq_length, num_parts * num_joints * features)
    target = target.reshape((batch_size, 1, seq_length, -1))
    
    diff = pred - target
    dist = torch.linalg.norm(diff, axis=-1)[..., -1]
    return dist.min(axis=-1).values


def mmade(target, pred, gt_multi, *args, t0=0, t=-1): # memory efficient version
    pred, target = time_slice(pred, t0, t, 2), time_slice(target, t0, t, 1)
    batch_size, n_samples, seq_length = pred.shape[:3]
    results = torch.zeros((batch_size, ))
    for i in range(batch_size):
        n_gts = gt_multi[i].shape[0]

        p = pred[i].reshape((n_samples, seq_length, -1)).unsqueeze(0)
        gt = time_slice(gt_multi[i], t0, t, 1).reshape((n_gts, seq_length, -1)).unsqueeze(1)

        diff = p - gt
        dist = torch.linalg.norm(diff, axis=-1).mean(axis=-1)
        results[i] = dist.min(axis=-1).values.mean()

    return results

def mmfde(target, pred, gt_multi, *args, t0=0, t=-1):
    pred, target = time_slice(pred, t0, t, 2), time_slice(target, t0, t, 1)
    batch_size, n_samples, seq_length = pred.shape[:3]
    results = torch.zeros((batch_size, ))
    for i in range(batch_size):
        n_gts = gt_multi[i].shape[0]

        p = pred[i].reshape((n_samples, seq_length, -1)).unsqueeze(0)
        gt = time_slice(gt_multi[i], t0, t, 1).reshape((n_gts, seq_length, -1)).unsqueeze(1)

        diff = p - gt
        dist = torch.linalg.norm(diff, axis=-1)[..., -1]
        results[i] = dist.min(axis=-1).values.mean()

    return results

def lat_apd(target, pred, gt_multi, lat_pred, *args):
    sample_num = pred.shape[1]
    mask = torch.tril(torch.ones([sample_num, sample_num], device='cpu')) == 0 # we only keep values from a single triangle of the matrix
    pdist = torch.cdist(lat_pred, lat_pred, p=1)[:, mask]
    return pdist.mean(axis=-1)

def get_multimodal_gt(data_loader, multimodal_threshold, device='cpu', split='test'):
    # this is not needed for now. the dataset is not big enough to use this
    """
    path = os.path.join(data_loader.dataset.precomputed_folder, f'multimodal_gt_{split}.pkl')
    if os.path.exists(path):
        print('Loading multimodal gt from precomputed folder')
        with open(path, 'rb') as f:
            multimodal_gt = pickle.load(f)
            return multimodal_gt
    """
    avg, cnt = 0, 0

    all_preds, all_last_obs, all_idces = [], [], []
    for batch_data in data_loader:
        data, target, extra = batch_data
        idx = extra["sample_idx"]
        seq_length, n_parts, n_joints, n_features = target.shape[1:]
        all_last_obs.append(data[:, -1].reshape(data.shape[0], -1))
        all_preds.append(target.reshape(target.shape[0], target.shape[1], -1))
        all_idces.append(idx)
    
    all_last_obs = np.concatenate(all_last_obs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_idces = np.concatenate(all_idces, axis=0)
    pd = squareform(pdist(all_last_obs))
    traj_gt_arr = []
    gt_idces_dict = {}
    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < multimodal_threshold)
        # from (n_similar, seq_length, num_parts * num_joints * features) to (n_similar, seq_length, num_parts, num_joints, features)
        # otherwise, the fde/ade computation will break
        c = all_preds[ind]
        traj_gt_arr.append(torch.tensor(c.reshape((c.shape[0], seq_length, n_parts, n_joints, n_features)), device=device))
        gt_idces_dict[str(all_idces[i])] = [int(i) for i in all_idces[ind]]
        avg += len(gt_idces_dict[str(all_idces[i])])
        cnt += 1
    avg /= cnt
    print(f'Average number of similar trajectories: {avg}')


    return traj_gt_arr
