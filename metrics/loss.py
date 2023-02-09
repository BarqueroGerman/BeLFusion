from lib2to3 import refactor
import torch.nn.functional as F
import torch
import numpy as np
import random

IDCES_2d = (0,1)
IDCES_3d = (0,1,2)

def _mse_loss(target, output, init=None, end=None, landmarks_idces=None, dimensions_idces=IDCES_3d):
    # idces -> landmarks to consider
    # output/target := (batch_size, pred_length, num_agents, num_landmarks, num_features)
    
    if landmarks_idces is not None: # indices to consider for evaluation
        target, output = output[:, :, :, landmarks_idces], target[:, :, :, landmarks_idces]

    if init is not None and end is not None:
        output, target = [arr[:, init:end, ..., dimensions_idces] for arr in [output, target]]
    elif init is not None:
        output, target = [arr[:, init:, ..., dimensions_idces] for arr in [output, target]]
    elif end is not None:
        output, target = [arr[:, :end, ..., dimensions_idces] for arr in [output, target]]

    return F.mse_loss(output, target, reduction="mean")

def mse_loss(obs, target, output, init=None, end=None, n_dims=3):
    output = output[0] # we ignore mu, logvar for VAE-like archs
    assert n_dims in (2, 3)
    assert target.shape[-1] >= n_dims and output.shape[-1] >= n_dims
    loss_mse = _mse_loss(target, output, init=init, end=end, 
                            dimensions_idces=IDCES_2d if n_dims == 2 else IDCES_3d
                        )
    return loss_mse, np.array([loss_mse.item(), ]), ["loss", ]

def BehaviorNet_loss(obs, target, output, aux_output, beta=0.5, delta=0.5, n_dims=3):
    assert len(output) == 5, "'prediction', 'b', 'mu', 'logvar' and 'pre' are needed for this loss."
    pred, b, mu, logvar, pre = output
    # pred/target := (batch_size, pred_length, num_agents, num_landmarks, num_features)

    batch_size = target.shape[0]

    MSE = (pred - target).pow(2).sum() / batch_size
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    
    aux_MSE = (aux_output - target).pow(2).sum() / batch_size
    Full_MSE(obs, target, aux_output)[0]
    aux_loss_name = "loss_mse_aux"

    loss_r = MSE + beta * KLD - delta * aux_MSE

    return loss_r, [loss_r.item(), MSE.item(), KLD.item(), aux_MSE.item()], ["loss", "loss_mse", "loss_KLD", aux_loss_name]


def Full_MSE(obs, target, output, n_dims=3):
    # loss for auxiliar decoder in behaviornet
    pred = output
    # target/pred := (batch_size, pred_length, num_agents, num_landmarks, num_features)
    assert len(pred.shape) == 5

    batch_size = target.shape[0]

    MSE = (pred - target).pow(2).sum() / batch_size
    return MSE, [MSE.item(), ], ["loss_mse", ]
