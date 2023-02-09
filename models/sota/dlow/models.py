import torch
from torch.nn import Module
import numpy as np
import pickle
from base.base_model import BaseModel
from models.basics import RNN, MLP
import torch.nn as nn

class VAE(BaseModel): # It looks like it is the implementation of ThePoseKnows
    def __init__(self, n_features, n_landmarks, nz, obs_length, pred_length, nh_mlp=[300, 200], 
                nh_rnn=128, use_drnn_mlp=False, rnn_type='lstm'):
        super(VAE, self).__init__(n_features, n_landmarks, obs_length, pred_length)
        # nz -> dimension of the sampling space
        self.nx = n_features * n_landmarks
        self.ny = self.nx
        self.nz = nz
        self.horizon = pred_length
        self.rnn_type = rnn_type
        self.use_drnn_mlp = use_drnn_mlp
        self.nh_rnn = nh_rnn
        self.nh_mlp = nh_mlp
        # encode
        self.x_rnn = RNN(self.nx, nh_rnn, cell_type=rnn_type)
        self.e_rnn = RNN(self.ny, nh_rnn, cell_type=rnn_type)
        self.e_mlp = MLP(2*nh_rnn, nh_mlp)
        self.e_mu = nn.Linear(self.e_mlp.out_dim, nz)
        self.e_logvar = nn.Linear(self.e_mlp.out_dim, nz)
        # decode
        if self.use_drnn_mlp:
            self.drnn_mlp = MLP(nh_rnn, nh_mlp + [nh_rnn], activation='tanh')
        self.d_rnn = RNN(self.ny + nz + nh_rnn, nh_rnn, cell_type=rnn_type)
        self.d_mlp = MLP(nh_rnn, nh_mlp)
        self.d_out = nn.Linear(self.d_mlp.out_dim, self.ny)
        self.d_rnn.set_mode('step')


    def encode_x(self, x):
        h_x = self.x_rnn(x)[-1]
        return h_x

    def encode_y(self, y):
        return self.e_rnn(y)[-1]

    def encode(self, x, y):
        h_x = self.encode_x(x)
        h_y = self.encode_y(y)
        h = torch.cat((h_x, h_y), dim=1)
        h = self.e_mlp(h)
        return self.e_mu(h), self.e_logvar(h)

    def decode(self, x, z):
        h_x = self.encode_x(x)
        if self.use_drnn_mlp:
            # MLP used to initialize decoder RNN from encoded X
            h_d = self.drnn_mlp(h_x)
            self.d_rnn.initialize(batch_size=z.shape[0], hx=h_d)
        else:
            self.d_rnn.initialize(batch_size=z.shape[0])
        y = []
        for i in range(self.horizon):
            y_p = x[-1] if i == 0 else y_i
            rnn_in = torch.cat([h_x, z, y_p], dim=1)
            h = self.d_rnn(rnn_in)
            h = self.d_mlp(h)
            y_i = self.d_out(h)
            y.append(y_i)
        y = torch.stack(y)
        
        return y



class DLow(BaseModel): # originally, NFDiag
    def __init__(self, n_features, n_landmarks, obs_length, pred_length, nk, nz=128, nh_mlp=[300, 200], nh_rnn=128, rnn_type='gru'):
        super(DLow, self).__init__(n_features, n_landmarks, obs_length, pred_length)
        self.nx = n_features * n_landmarks
        self.ny = nz
        self.nk = nk

        self.nh = nh_mlp
        self.nh_rnn = nh_rnn
        self.rnn_type = rnn_type
        
        self.nac = nac = nk
        self.x_rnn = RNN(self.nx, nh_rnn, cell_type=rnn_type)
        self.mlp = MLP(nh_rnn, nh_mlp)
        self.head_A = nn.Linear(nh_mlp[-1], self.ny * nac)
        self.head_b = nn.Linear(nh_mlp[-1], self.ny * nac)

    def encode_x(self, x):
        return self.x_rnn(x)[-1]

    def encode(self, x, y):
        h_x = self.encode_x(x)
        h = self.mlp(h_x)
        a = self.head_A(h).view(-1, self.nk, self.ny)[:, 0, :]
        b = self.head_b(h).view(-1, self.nk, self.ny)[:, 0, :]
        z = (y - b) / a
        return z

    def forward(self, x, z=None):

        # x -> [seq_length, batch_size, features (landmarks * 3)]
        h_x = self.encode_x(x)
        if z is None:
            z = torch.randn((h_x.shape[0], self.ny), device=x.device)
        z = z.repeat_interleave(self.nk, dim=0)
        h = self.mlp(h_x)
        
        a = self.head_A(h).view(-1, self.ny)
        b = self.head_b(h).view(-1, self.ny)
        y = a * z + b
        return y, a, b

    def sample(self, x, z=None):
        return self.forward(x, z)[0]

    def get_kl(self, a, b):
        var = a ** 2
        KLD = -0.5 * torch.sum(1 + var.log() - b.pow(2) - var)
        return 

class DLowWrapper(Module):
    def __init__(self, n_features, n_landmarks, obs_length, pred_length,
                        model_path_vae, model_path_dlow,
                        # vae
                        nh_mlp=[300, 200], nh_rnn=128, use_drnn_mlp=False, rnn_type='lstm', nz=128,
                        # dlow
                        nh_mlp_dlow=[300, 200], nk=10, nh_rnn_dlow=128, rnn_type_dlow='gru',
                        # control
                        only_vae=False):
        super(DLowWrapper, self).__init__()

        self.model_vae = VAE(n_features, n_landmarks, nz, obs_length, pred_length, nh_mlp=nh_mlp, 
                                nh_rnn=nh_rnn, use_drnn_mlp=use_drnn_mlp, rnn_type=rnn_type)
        self.model_dlow = DLow(n_features, n_landmarks, obs_length, pred_length, nk, nz=nh_rnn,
                                nh_mlp=nh_mlp_dlow, nh_rnn=nh_rnn_dlow, rnn_type=rnn_type_dlow)
        #print(self.model)
        #print(self.model_t1)
        self.only_vae = only_vae
        self.z_dim = nz

        self.n_features = n_features
        self.n_landmarks = n_landmarks
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.t_length = obs_length + pred_length

        # we load the model
        self.load_model(model_path_vae, model_path_dlow)

    def load_model(self, model_path_vae, model_path_dlow):
        self.model_vae.load_state_dict(pickle.load(open(model_path_vae, "rb"))['model_dict'])
        if not self.only_vae:
            self.model_dlow.load_state_dict(pickle.load(open(model_path_dlow, "rb"))['model_dict'])

    def to(self, device):
        self.model_vae = self.model_vae.to(device)
        self.model_dlow = self.model_dlow.to(device)
        return self

    def eval(self):
        self.model_vae.eval()
        self.model_dlow.eval()

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
        x = x.reshape((batch_size, self.obs_length, -1)).permute(1, 0, 2)  # b, n_landmarks * n_features, obs_length

        if not self.only_vae:
            Z_g = self.model_dlow.sample(x)
        else:
            Z_g = torch.randn(batch_size*nk, self.z_dim, device=device)

        x = x.repeat_interleave(nk, dim=1)
        X = x # start with the original context, but replace it in a new variable
        Y = self.model_vae.decode(X, Z_g)

        # back to the original shape
        return Y.permute(1, 0, 2).reshape((batch_size, nk, self.pred_length, participants, self.n_landmarks, self.n_features))


if __name__ == '__main__':
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda:0')
    dtype = torch.float64
    n_landmarks, n_features = 16, 3
    nz = 64
    obs, horizon = 25, 100
    batch_size = 64
    participants = 1

    model_path_vae = "results/models/baselines/h36m/DLow/exp/vae_0500.p"
    model_path_dlow = "results/models/baselines/h36m/DLow/exp/dlow_0500.p"


    m = DLowWrapper(n_features, n_landmarks, obs, horizon,
                            model_path_vae, model_path_dlow,
                            rnn_type='gru', nz=128, recurrent=False, residual=False, nh_mlp=[300, 200], nh_rnn=128, use_drnn_mlp=True,
                            nk=50, nh_mlp_dlow=[1024, 512]
                            ).to(device)
    #print(f"{sum(p.numel() for p in m.parameters()) / 1e6}")

    # dumb input
    x = torch.tensor(np.zeros((batch_size, obs, participants, n_landmarks, n_features), dtype=np.float64), device=device, dtype=dtype).contiguous()
    y = torch.tensor(np.zeros((batch_size, horizon, participants, n_landmarks, n_features), dtype=np.float64), device=device, dtype=dtype).contiguous()
    print("\n> INPUT:\nx", x.shape, "\ny", y.shape)

    out = m.forward(x, y, 50)[0]
    print("\n> OUTPUT:\n", out.shape)