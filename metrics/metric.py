import torch
import torch.nn.functional as F
import numpy as np

# IMPORTANT: all of them receive (batch_size, pred_length, num_agents, num_landmarks, num_features)
# It means that we are not using metrics that account for multiple samples, for now.

# ----------------- basic metrics which are called from other metrics -----------------

def mse(target, output, init=None, end=None, participant=None, joints=None, n_dims=3):
    # output[0]/target := (batch_size, pred_length, num_agents, num_landmarks, num_features)

    output = output[0] # only coordinates
    with torch.no_grad():
        if joints is not None: # indices to consider for evaluation
            output, target = [arr[..., joints, :] for arr in [output, target]]
        if participant is not None: # indices to consider for evaluation
            output, target = [arr[:, :, participant:participant+1] for arr in [output, target]]

        if init is not None and end is not None:
            output, target = [arr[:, init:end, ..., :n_dims] for arr in [output, target]]
        elif init is not None:
            output, target = [arr[:, init:, ..., :n_dims] for arr in [output, target]]
        elif end is not None:
            output, target = [arr[:, :end, ..., :n_dims] for arr in [output, target]]

        value = F.mse_loss(output, target, reduction="mean")
        return value

def div(target, output):
    # output[0]/target := (batch_size, pred_length, num_agents, num_landmarks, num_features)
    output = output[0] # only coordinates
    with torch.no_grad():
        return torch.norm(output[:, :, 1:].ravel() - output[:, :, :-1].ravel())

# -------------------------------------------------------------------------------------
