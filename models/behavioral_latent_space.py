import torch
from torch import nn
from torch.nn.utils import weight_norm
from base import BaseModel
from models.basics import rc, rc_recurrent, reparametrize_logstd, RNN, MLP


class NormConv2d(nn.Module):
    """
    Convolutional layer with l2 weight normalization and learned scaling parameters
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0
    ):
        super().__init__()
        self.beta = nn.Parameter(
            torch.zeros([1, out_channels, 1, 1], dtype=torch.float32)
        )
        self.gamma = nn.Parameter(
            torch.ones([1, out_channels, 1, 1], dtype=torch.float32)
        )
        self.conv = weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            name="weight",
        )

    def forward(self, x):
        # weight normalization
        # self.conv.weight = normalize(self.conv.weight., dim=[0, 2, 3])
        out = self.conv(x)
        out = self.gamma * out + self.beta
        return out


class BEncoder(nn.Module):
    def __init__(
        self,
        n_in,
        n_layers,
        dim_hidden,
        use_linear,
        dim_linear
    ):
        super(BEncoder, self).__init__()


        self.rnn = nn.LSTM(
            input_size=n_in,
            hidden_size=dim_hidden,
            num_layers=n_layers,
            batch_first=True,
        )

        self.n_layer = n_layers
        self.dim_hidden = dim_hidden
        self.hidden = self.init_hidden(bs=1, device="cpu")

        self.use_linear = use_linear
        self.linear = None
        if self.use_linear:
            self.linear = nn.Linear(self.dim_hidden, dim_linear)

        # functions to map tp mu and sigma
        self.mu_fn = NormConv2d(self.dim_hidden, self.dim_hidden, 1)
        self.std_fn = NormConv2d(self.dim_hidden, self.dim_hidden, 1)

    def init_hidden(self, bs, device):
        # num_layers x bs x dim_hidden
        self.hidden = (
            torch.zeros((self.n_layer, bs, self.dim_hidden), device=device),
            torch.zeros((self.n_layer, bs, self.dim_hidden), device=device),
        )

    def set_hidden(self, bs, device, hidden=None, cell=None):
        if hidden is None and cell is None:
            self.init_hidden(bs, device)
        elif hidden is None:
            self.hidden = (torch.zeros_like(cell), cell)
        elif cell is None:
            self.hidden = (hidden, torch.zeros_like(hidden))
        else:
            self.hidden = (hidden, cell)

    def forward(self, x, sample=False):
        out, self.hidden = self.rnn(x, self.hidden)

        # if self.use_linear:
        #     out = self.linear(out[:, -1].squeeze(1))
        # else:
        #     out = out.squeeze(dim=1)
        pre = self.hidden[0][-1]
        mu = (
            self.mu_fn(pre.unsqueeze(dim=-1).unsqueeze(dim=-1))
            .squeeze(dim=-1)
            .squeeze(dim=-1)
        )
        logstd = (
            self.std_fn(pre.unsqueeze(dim=-1).unsqueeze(dim=-1))
            .squeeze(dim=-1)
            .squeeze(dim=-1)
        )
        if sample:
            out = sample(mu)
        else:
            out = reparametrize_logstd(mu, logstd)
        return (out, mu, logstd, pre)



class ResidualRNNDecoder(nn.Module):
    def __init__(
        self, n_in, n_out, n_hidden, rnn_type="lstm", use_nin=False
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.rnn_type = rnn_type
        self.use_nin = use_nin

        if self.rnn_type == "gru":
            self.rnn = nn.GRUCell(self.n_in, self.n_hidden)
            self.n_out = nn.Linear(self.n_hidden, self.n_out)
        else:
            self.rnn = nn.LSTMCell(self.n_in, self.n_hidden)
            self.n_out = nn.Linear(self.n_hidden, self.n_out)

        if self.use_nin:
            self.n_in = nn.Linear(self.n_in, self.n_in)



        self.init_hidden(bs=1, device="cpu")

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.squeeze(dim=1)
        elif len(x.shape) != 2:
            raise TypeError("invalid shape of tensor.")

        res = x
        if self.use_nin:
            x = self.n_in(x)

        if self.rnn_type == "lstm":
            self.hidden = self.rnn(x, self.hidden)
            out_rnn = self.hidden[0]
        else:
            self.hidden = self.rnn(x, self.hidden)
            out_rnn = self.hidden

        out = self.n_out(out_rnn)

        return out + res, res

    def forward_noresidual(self, x):
        if len(x.shape) == 3:
            x = x.squeeze(dim=1)
        elif len(x.shape) != 2:
            raise TypeError("invalid shape of tensor.")

        if self.use_nin:
            x = self.n_in(x)

        if self.rnn_type == "lstm":
            self.hidden = self.rnn(x, self.hidden)
            out_rnn = self.hidden[0]
        else:
            self.hidden = self.rnn(x, self.hidden)
            out_rnn = self.hidden

        return self.n_out(out_rnn)

    def init_hidden(self, bs, device):
        # num_layers x bs x dim_hidden

        if self.rnn_type == "lstm":
            self.hidden = (
                torch.zeros((bs, self.n_hidden), device=device),
                torch.zeros((bs, self.n_hidden), device=device),
            )
        elif self.rnn_type == "gru":
            self.hidden = torch.zeros((bs, self.n_hidden), device=device)


    def set_hidden(self, bs, device, hidden=None, cell=None):
        if self.rnn_type == "lstm":
            if hidden is None and cell is None:
                self.init_hidden(bs, device)
            elif hidden is None:
                self.hidden = (torch.zeros_like(cell), cell)
            elif cell is None:
                self.hidden = (hidden, torch.zeros_like(hidden))
            else:
                self.hidden = (hidden, cell)
        elif self.rnn_type == "gru":
            if hidden is None:
                self.init_hidden(bs, device)
            else:
                self.hidden = hidden



class ResidualBehaviorNet(BaseModel):
    def __init__(self, n_features, n_landmarks, obs_length, pred_length, 
                dim_hidden_b=128, dim_decoder_state=128, context_length=3, decoder_arch='lstm', linear_nin_decoder=False,
                dim_linear_enc=128, residual=False, recurrent=False):
        super().__init__(n_features, n_landmarks, obs_length, pred_length)
        assert obs_length == pred_length, "obs_length != pred_length. Autoencoder can't be used for this case."
        n_kps = n_features * n_landmarks
        self.context_length = context_length
        self.residual = residual
        self.recurrent = recurrent

        self.dec_type = decoder_arch
        self.use_nin_dec = linear_nin_decoder
        self.dim_hidden_b = dim_hidden_b
        # (n_in, n_layers, dim_hidden, use_linear, dim_linear)

        self.b_enc = BEncoder(
            n_kps,
            1,
            self.dim_hidden_b,
            use_linear=False,
            dim_linear=1,
        )

        self.decoder = ResidualRNNDecoder(
            n_in = dim_hidden_b + dim_linear_enc,
            n_out= n_features * n_landmarks,
            n_hidden=dim_decoder_state,
            rnn_type=self.dec_type,
            use_nin=self.use_nin_dec,
        )

        self.context_encoder = nn.Linear(context_length * n_features * n_landmarks, dim_linear_enc)

    def forward(self, x, y):
        # [BS, length, participants, landmarks, features]
        batch_size, seq_len, num_agents, num_landmarks, num_features = x.shape
        # TODO: for simplicty we consider participants = 1
        x = torch.flatten(x, start_dim=3)[:, :, 0] # [bs, len, nfeat]
        
        b, mu, logstd, pre = self.infer_b(x)
        xs, _ = self.generate_seq(b, x_context=x[:, :self.context_length])
        
        xs = xs.reshape((batch_size, self.obs_length, num_agents, num_landmarks, num_features))
        return xs, b, mu, logstd, pre

    def encode(self, x):
        # [BS, length, participants, landmarks, features]
        batch_size, seq_len, num_agents, num_landmarks, num_features = x.shape
        # TODO: for simplicty we consider participants = 1
        x = torch.flatten(x, start_dim=3)[:, :, 0] # [bs, len, nfeat]
        
        b, mu, logstd, pre = self.infer_b(x)
        return b

    def infer_b(self, s):
        """
        :param s: The input sequence from which b is inferred
        :return:
        """
        bs = s.shape[0]
        self.b_enc.init_hidden(bs, device=s.device)
        outs = self.b_enc(s)
        return outs

    def decode(self, obs, b):
        batch_size = obs.shape[0]
        obs = torch.flatten(obs, start_dim=3)[:, :, 0] # [bs, len, nfeat]
        xs, _ = self.generate_seq(b, x_context=obs[:, -self.context_length:])
        
        xs = xs.reshape((batch_size, self.obs_length, 1, self.n_landmarks, self.n_features))[:, self.context_length:]
        return xs

    def generate_seq(self, b, x_context):
        # b -> [batch_size, b_size]
        # x_context -> [batch_size, context_length, n_landmarks * n_features]
        batch_size = b.shape[0]
        xs = torch.zeros([batch_size, self.obs_length, x_context.shape[-1]], device=b.device)

        #self.decoder.set_hidden(batch_size, device=b.device, hidden=b, cell=b) # OLD
        self.decoder.init_hidden(batch_size, device=b.device)
        
        context_length = x_context.shape[1]
        xs[:, :context_length] = x_context[:, :context_length]

        context_enc = self.context_encoder(x_context.view(batch_size, -1))
        hs = torch.cat([b, context_enc], dim=1)
        for i in range(context_length, self.obs_length): # we now decode the other part of the sequence (not context)

            x = self.decoder.forward_noresidual(hs)
            xs[:, i] = x

        last_obs = x_context[:, -1]
        if self.residual and self.recurrent:
            xs[:, context_length:] = rc_recurrent(last_obs, xs[:, context_length:], batch_first=True) # prediction of offsets w.r.t. previous obs/prediction
        elif self.residual:
            xs[:, context_length:] = rc(last_obs, xs[:, context_length:], batch_first=True) # prediction of offsets w.r.t. last obs (residual connection)
        return xs, b



class DecoderBehaviorNet(BaseModel):
    def __init__(self, n_features, n_landmarks, obs_length, pred_length, 
                dim_hidden_b=1024, dim_hidden_state=128, decoder_arch='lstm', linear_nin_decoder=False):
        super().__init__(n_features, n_landmarks, obs_length, pred_length)
        assert obs_length == pred_length, "obs_length != pred_length. Autoencoder can't be used for this case."
        n_kps = n_features * n_landmarks

        self.dec_type = decoder_arch
        self.use_nin_dec = linear_nin_decoder
        self.dim_hidden_b = dim_hidden_b
        self.dim_hidden_state = dim_hidden_state
        # (n_in, n_layers, dim_hidden, use_linear, dim_linear)

        self.decoder = ResidualRNNDecoder(
            n_in = self.dim_hidden_b,
            n_out= n_kps,
            n_hidden=self.dim_hidden_state,
            rnn_type=self.dec_type,
            use_nin=self.use_nin_dec,
        )

    def forward(self, hs):
        batch_size = hs.shape[0]
        xs = torch.zeros([batch_size, self.obs_length, self.n_features*self.n_landmarks], device=hs.device)

        self.decoder.init_hidden(batch_size, device=hs.device)

        for i in range(0, self.obs_length):
            xs[:, i] = self.decoder.forward_noresidual(hs)

        xs = xs.reshape((batch_size, self.obs_length, 1, self.n_landmarks, self.n_features))
        return xs


if __name__=="__main__":
    import numpy as np
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cuda:0')

    n_landmarks, n_features = 16, 3
    obs_len, pred_len = 13, 13
    batch_size = 64
    participants = 1
    hidden_length = 1024
    context_length = 3
    residual, recurrent = True, False

    model = ResidualBehaviorNet
    
    # dumb input
    b = torch.tensor(np.zeros((batch_size, hidden_length), dtype=np.float64), device=device, dtype=dtype).contiguous()
    y = torch.tensor(np.zeros((batch_size, pred_len, participants, n_landmarks, n_features), dtype=np.float64), device=device, dtype=dtype).contiguous()
    obs = torch.tensor(np.zeros((batch_size, obs_len, participants, n_landmarks, n_features), dtype=np.float64), device=device, dtype=dtype).contiguous()

    model = model(n_features, n_landmarks, obs_len, pred_len, dim_hidden_b=hidden_length, context_length=context_length,
                    residual=residual, recurrent=recurrent).to(device)
    print("-"*20, "model", "-"*20)
    print(model)
    print([v.shape for v in model(obs, None)])
    print("-"*20, "aux_model", "-"*20)
    print(aux_model)
    print(aux_model(b).shape)
    