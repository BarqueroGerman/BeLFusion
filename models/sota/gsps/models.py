import torch
from torch.nn import Module, Sequential, ModuleList, ModuleDict, Linear, GELU, Tanh, BatchNorm1d
import numpy as np

from .GCN import GCNParts
from ..diverse_sampling.dct import get_dct_matrix, dct_transform_torch, reverse_dct_torch
from torch import nn
import pickle


class GSPSWrapper(Module):
    def __init__(self, n_features, n_landmarks, obs_length, pred_length,
                        model_path, parts, num_stage=4, is_bn=True,
                        node_n=16, hidden_dim=256, n_pre = 10, 
                        nz=64, p_dropout=0,
                        act_f=nn.Tanh()):
        super(GSPSWrapper, self).__init__()

        self.model = GCNParts(n_pre * 3 + nz, hidden_dim,
                        parts=parts,
                        p_dropout=p_dropout, num_stage=num_stage, node_n=node_n, is_bn=is_bn,
                        act_f=act_f
                        )
        #assert n_landmarks * n_features == node_n, 'n_landmarks should be equal to node_n'
        self.z_dim = nz
        self.n_features = n_features
        self.n_landmarks = n_landmarks
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.t_length = obs_length + pred_length

        self.parts = parts

        ## dct
        self.dct_n = n_pre
        self.dct_m, self.i_dct_m = get_dct_matrix(self.t_length)
        #if self.cfg.device != "cpu":
        self.dct_m = torch.from_numpy(self.dct_m)#.float()#.cuda()
        self.i_dct_m = torch.from_numpy(self.i_dct_m)#.float()#.cuda()        

        # we load the model
        self.load_model(model_path)

    def load_model(self, model_path):
        model_cp = pickle.load(open(model_path, "rb"))['model_dict']
        self.model.load_state_dict(model_cp)

    def to(self, device):
        self.model = self.model.to(device)
        self.dct_m = self.dct_m.to(device)
        self.i_dct_m = self.i_dct_m.to(device)
        return self

    def eval(self):
        self.model.eval()

    def forward(self, x, y, nk):
        '''
        Args:
            x: batch_size, obs_length, participants, n_landmarks, n_features
            y: batch_size, pred_length, participants, n_landmarks, n_features
        Returns:
        '''
        # preprocess the data with our project's data format
        batch_size = x.shape[0]
        participants = x.shape[2]
        device = x.device
        x = x.reshape((batch_size, self.obs_length, -1)).permute(0, 2, 1)  # b, n_landmarks * n_features, obs_length

        # DCT transformations
        with torch.no_grad():
            padded_inputs = x[:, :, list(range(self.obs_length)) + [self.obs_length - 1] * self.pred_length]
            padded_inputs_dct = dct_transform_torch(padded_inputs, self.dct_m, dct_n=self.dct_n)  # b, 48, 10
            padded_inputs_dct = padded_inputs_dct.view(batch_size, -1, 3 * self.dct_n)  # # b, 16, 3*10
        
        all_z = torch.randn(batch_size*nk, len(self.parts), self.z_dim, device=device)

        all_outs_dct = self.model(torch.repeat_interleave(padded_inputs_dct, repeats=nk, dim=0), z=all_z)

        all_outs_dct = all_outs_dct.reshape(batch_size * nk, -1, self.dct_n)  # b*h, 48, 10
        outputs = reverse_dct_torch(all_outs_dct, self.i_dct_m, self.t_length)  # b*h, 48, 125
        outputs = outputs.view(batch_size, nk, -1, self.t_length)  # b, nk, 48, 125

        # back to the original shape
        outputs = outputs.permute(0, 1, 3, 2).reshape(batch_size, nk, self.t_length, participants, self.n_landmarks, self.n_features)  # b, nk, 125, 16, 3

        return outputs[:, :, -self.pred_length:]


if __name__ == '__main__':
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cuda:0')
    dtype = torch.float32
    n_landmarks, n_features = 16, 3
    nz = 64
    obs, horizon = 25, 100
    batch_size = 64
    participants = 1

    model_path = "./results/models/baselines/h36m/GSPS/exp/vae_0500.p"
        

    parts = [ [ 0, 1, 2, 3, 4, 5 ], [ 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ] ]
    m = GSPSWrapper(n_features, n_landmarks, obs, horizon,
                            model_path, parts).to(device)
    #print(f"{sum(p.numel() for p in m.parameters()) / 1e6}")

    # dumb input
    x = torch.tensor(np.zeros((batch_size, obs, participants, n_landmarks, n_features), dtype=np.float64), device=device, dtype=dtype).contiguous()
    y = torch.tensor(np.zeros((batch_size, horizon, participants, n_landmarks, n_features), dtype=np.float64), device=device, dtype=dtype).contiguous()
    print("\n> INPUT:\nx", x.shape, "\ny", y.shape)

    out = m.forward(x, y, 50)[0]
    print("\n> OUTPUT:\n", out.shape)