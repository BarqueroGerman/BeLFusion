import torch
import numpy as np
from torch import nn
import torch.nn as nn
from einops import rearrange
from base import BaseAutoencoder
from models.basics import rc, rc_recurrent, RNN, MLP


# imported from VAEs of DLow, but it is really a seq2seq model
class Seq2Seq_Auto(BaseAutoencoder):
    # Fixed version of Seq2Seq_Auto. It encodes from 1 to 'pred_length'
    def __init__(self, n_features, n_landmarks, obs_length, pred_length, nz=None, nh_mlp=[300, 200], 
                nh_rnn=128, use_drnn_mlp=False, rnn_type='lstm', encoding_length=None, dropout=0, noise_augmentation=0,
                recurrent=True, residual=True):
        super(Seq2Seq_Auto, self).__init__(n_features, n_landmarks, obs_length, pred_length)
        # nz -> dimension of the sampling space
        assert obs_length == pred_length, "For autoencoders, obs_length and pred_length must be equal"
        self.encoding_length = encoding_length
        self.nx = n_features * n_landmarks
        self.ny = self.nx
        self.seq_length = pred_length - 1
        self.rnn_type = rnn_type
        self.use_drnn_mlp = use_drnn_mlp
        self.nh_rnn = nh_rnn
        self.nh_mlp = nh_mlp
        # new params for recurrent + residual encoding of predictions
        self.recurrent = recurrent
        self.residual = residual
        # encode
        self.x_rnn = RNN(self.nx, nh_rnn, cell_type=rnn_type)
        # decode
        if self.use_drnn_mlp:
            self.drnn_mlp = MLP(nh_rnn, nh_mlp + [nh_rnn], activation='tanh')
        self.d_rnn = RNN(self.ny + nh_rnn, nh_rnn, cell_type=rnn_type)
        self.d_mlp = MLP(nh_rnn, nh_mlp)
        self.d_out = nn.Linear(self.d_mlp.out_dim, self.ny)
        self.d_rnn.set_mode('step')

        self.dropout = nn.Dropout(dropout)
        self.noise_augmentation = noise_augmentation

    def _encode(self, x):
        assert x.shape[0] == self.seq_length # otherwise it does not make sense
        return self.x_rnn(x)[-1]

    def encode(self, x):
        # seq length must be equal to self.seq_length
        tf = 'b s p l f  -> s (b p) (l f)' # TODO here we need to apply FOR EACH PARTICIPANT
        x = rearrange(x, tf )
        return self.x_rnn(x)[-1]

    def _decode(self, x_start, h_y):
        h_y = self.dropout(h_y)
        if self.noise_augmentation != 0:
            h_y = h_y + torch.randn_like(h_y) * self.noise_augmentation
            
        if self.use_drnn_mlp:
            # MLP used to initialize decoder RNN from encoded X
            h_d = self.drnn_mlp(h_y)
            self.d_rnn.initialize(batch_size=x_start.shape[0], hx=h_d)
        else:
            self.d_rnn.initialize(batch_size=x_start.shape[0])
        y = []
        for i in range(self.seq_length):
            y_p = x_start if i == 0 else y_i
            rnn_in = torch.cat([h_y, y_p], dim=1)
            h = self.d_rnn(rnn_in)
            h = self.d_mlp(h)
            y_i = self.d_out(h)
            y.append(y_i)
        y = torch.stack(y)

        if self.residual and self.recurrent:
            y = rc_recurrent(x_start, y, batch_first=False) # prediction of offsets w.r.t. previous obs/prediction
        elif self.residual:
            y = rc(x_start, y, batch_first=False) # prediction of offsets w.r.t. last obs (residual connection)

        return y

    def decode(self, obs, h_y): # from observation and hidden state of prediction
        x_start = obs[:, -1]
        # seq length must be equal to self.seq_length
        n_agents, n_landmarks, n_features = x_start.shape[-3:]
        tf = 'b p l f  -> (b p) (l f)' # TODO here we need to apply FOR EACH PARTICIPANT
        x_start = rearrange(x_start, tf )

        y = self._decode(x_start, h_y)
        y = rearrange(y, "s (b p) (n f) -> b s p n f", n=n_landmarks, f=n_features, p=n_agents)
        return y

    def autoencode(self, x):
        x_start = x[0]
        h_x = self._encode(x[1:])
        return torch.cat([x[0][None, ...], self._decode(x_start, h_x)], axis=0)

    def forward(self, x, y):
        # we need to treat all agents independently for now
        n_agents, n_landmarks, n_features = x.shape[-3:]
        # x, y are of shape [batch_size, seq_length, participants, n_landmarks, n_features]

        # TO DLOW-FRIENDLY SHAPES
        # x, y are converted to shape -> [seq_length, batch_size, features (landmarks * 3)]
        tf = 'b s p l f  -> s (b p) (l f)' # TODO here we need to apply 'decode' FOR EACH PARTICIPANT
        x = rearrange(x, tf )

        # original code from DLow repository
        Y_r = self.autoencode(x)

        # BACK TO PROJECT-FRIENDLY SHAPES
        # Y_r -> [seq_length, batch_size, features] ---> [batch_size, seq_length, participants, n_landmarks, n_features]
        Y_r = rearrange(Y_r, "s (b p) (n f) -> b s p n f", n=n_landmarks, f=n_features, p=n_agents)

        return Y_r




if __name__ == '__main__':
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    # need to run 'export PYTHONPATH=$PWD' from root folder
    #rnn = RNN(12, 24, 'gru', bi_dir=True)
    #input = zeros(5, 3, 12)
    #out = rnn(input)
    #print(out.shape)
    device = torch.device('cpu')
    dtype = torch.float64
    n_landmarks, n_features = 16, 3
    nz = 128
    obs, horizon = 100, 100
    batch_size = 64
    participants = 1
    #nx = ny = n_features * n_landmarks
    auto = Seq2Seq_Auto(n_features, n_landmarks, obs, horizon, recurrent=True, residual=True, rnn_type='gru').to(device)

    # dumb input
    x = torch.tensor(np.zeros((batch_size, obs, participants, n_landmarks, n_features), dtype=np.float64), device=device, dtype=dtype).contiguous()
    y = torch.tensor(np.zeros((batch_size, horizon, participants, n_landmarks, n_features), dtype=np.float64), device=device, dtype=dtype).contiguous()
    print("\n> INPUT:\nx", x.shape, "\ny", y.shape)

    out = auto(x, y)
    #out = vae.forward_test(x, y)
    print("\n> OUTPUT:\noutput", out.shape, "\nmu")#, mu.shape, "\nlogvar", logvar.shape)