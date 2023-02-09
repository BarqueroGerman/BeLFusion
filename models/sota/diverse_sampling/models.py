import torch
from torch.nn import Module, Sequential, ModuleList, ModuleDict, Linear, GELU, Tanh, BatchNorm1d
import numpy as np

from .gcn_layers import GraphConv, GraphConvBlock, ResGCB
from .dct import get_dct_matrix, dct_transform_torch, reverse_dct_torch
from .cvae import CVAE


def _sample_weight_gumbel_softmax(logits, temperature=1, eps=1e-20):
    # b*h, 1, 10
    assert temperature > 0, "temperature must be greater than 0 !"

    U = torch.rand(logits.shape, device=logits.device)
    g = -torch.log(-torch.log(U + eps) + eps)

    y = logits + g
    y = y / temperature
    y = torch.softmax(y, dim=-1)
    return y

class DiverseSampling(Module):
    def __init__(self, node_n=16, hidden_dim=256, base_dim = 64, z_dim=64, dct_n=10, base_num_p1=10, dropout_rate=0):
        super(DiverseSampling, self).__init__()
        self.z_dim = z_dim
        self.base_dim = base_dim
        self.base_num_p1 = base_num_p1

        self.condition_enc = Sequential(
            GraphConvBlock(in_len=3*dct_n, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=False),
            ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=True),
            ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=True),
        )
        self.bases_p1 = Sequential(
            Linear(node_n*hidden_dim, self.base_num_p1 * self.base_dim), # 1024 -> M * 64
            BatchNorm1d(self.base_num_p1 * self.base_dim),
            Tanh()
        )

        self.mean_p1 = Sequential(
            Linear(self.base_dim, 64),  # 64 -> 64
            BatchNorm1d(64),
            Tanh(),
            Linear(64, self.z_dim) # 64 -> 64
        )
        self.logvar_p1 = Sequential(
            Linear(self.base_dim, 64),  # 64 -> 64
            BatchNorm1d(64),
            Tanh(),
            Linear(64, self.z_dim) # 64 -> 64
        )

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


    def forward(self, condition, repeated_eps=None, many_weights=None, multi_modal_head=10):
        '''
        Args:
            condition: [b, 48, 25] / [b, 16, 3*10]
            repeated_eps: b*50, 64
        Returns:
        '''
        b, v, ct = condition.shape
        condition_enced = self.condition_enc(condition)  # b, 16, 256

        bases = self.bases_p1(condition_enced.view(b, -1)).view(b, self.base_num_p1, self.base_dim)  # b, 10, 64

        repeat_many_bases = torch.repeat_interleave(bases, repeats=multi_modal_head, dim=0)  # b*h, 10, 64
        many_bases_blending = torch.matmul(many_weights, repeat_many_bases).squeeze(dim=1).view(-1, self.base_dim)  # b*h, 64

        all_mean = self.mean_p1(many_bases_blending)
        all_logvar = self.logvar_p1(many_bases_blending)

        all_z = torch.exp(0.5 * all_logvar) * repeated_eps + all_mean

        return all_z, all_mean, all_logvar


class DiverseSamplingWrapper(Module):
    def __init__(self, n_features, n_landmarks, obs_length, pred_length,
                        model_path_t1, model_path_t2,
                        node_n=16, hidden_dim=256, base_dim = 128, 
                        z_dim=64, dct_n=10, 
                        base_num_p1=40, temperature_p1=0.85, 
                        dropout_rate=0, only_vae=False):
        super(DiverseSamplingWrapper, self).__init__()

        self.model_t1 = CVAE(node_n=node_n, hidden_dim=hidden_dim, z_dim=z_dim,
                                     dct_n=dct_n, dropout_rate=dropout_rate)
        self.model = DiverseSampling(node_n=node_n, hidden_dim=hidden_dim, base_dim=base_dim, 
                                    z_dim=z_dim, dct_n=dct_n, base_num_p1=base_num_p1, 
                                    dropout_rate=dropout_rate)
        #print(self.model)
        #print(self.model_t1)
        self.only_vae = only_vae

        assert n_landmarks == node_n, 'n_landmarks should be equal to node_n'
        self.z_dim = z_dim
        self.n_features = n_features
        self.n_landmarks = n_landmarks
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.t_length = obs_length + pred_length

        ## dct
        self.dct_n = dct_n
        self.dct_m, self.i_dct_m = get_dct_matrix(self.t_length)
        #if self.cfg.device != "cpu":
        self.dct_m = torch.from_numpy(self.dct_m)#.float()#.cuda()
        self.i_dct_m = torch.from_numpy(self.i_dct_m)#.float()#.cuda()        

        # others
        self.temperature_p1 = temperature_p1
        self.base_num_p1 = base_num_p1

        # we load the model
        self.load_model(model_path_t1, model_path_t2)

    def load_model(self, model_path_t1, model_path_t2):
        self.model_t1.load_state_dict(torch.load(model_path_t1)["model"])
        if not self.only_vae:
            self.model.load_state_dict(torch.load(model_path_t2)["model"])

    def to(self, device):
        self.model = self.model.to(device)
        self.model_t1 = self.model_t1.to(device)
        self.dct_m = self.dct_m.to(device)
        self.i_dct_m = self.i_dct_m.to(device)
        return self

    def eval(self):
        self.model.eval()
        self.model_t1.eval()

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
        
        # >>> many bases
        logtics = torch.ones((batch_size * nk, 1, self.base_num_p1), device=device) / self.base_num_p1  # b*h, 1, 10
        many_weights = _sample_weight_gumbel_softmax(logtics, temperature=self.temperature_p1)  # b*h, 1, 10

        if not self.only_vae:
            # we compute all_z with the model proposed
            eps = torch.randn((batch_size, self.z_dim), device=device) # source of noise
            repeated_eps = torch.repeat_interleave(eps, repeats=nk, dim=0) # this is transparent at this stage
            all_z, all_mean, all_logvar = self.model(padded_inputs_dct, repeated_eps, many_weights, nk)
        else:
            all_z = torch.randn(batch_size*nk, self.z_dim, device=device)

        all_outs_dct = self.model_t1.inference(
                condition=torch.repeat_interleave(padded_inputs_dct, repeats=nk, dim=0),
                z=all_z)

        all_outs_dct = all_outs_dct.reshape(batch_size * nk, -1, self.dct_n)  # b*h, 48, 10
        outputs = reverse_dct_torch(all_outs_dct, self.i_dct_m, self.t_length)  # b*h, 48, 125
        outputs = outputs.view(batch_size, nk, -1, self.t_length)  # b, nk, 48, 125

        # back to the original shape
        outputs = outputs.permute(0, 1, 3, 2).reshape(batch_size, nk, self.t_length, participants, self.n_landmarks, self.n_features)  # b, nk, 125, 16, 3

        return outputs[:, :, -self.pred_length:]

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

    model_path_t1 = "./auxiliar/models/diverse_sampling/h36m_t1.pth"
    model_path_t2 = "./auxiliar/models/diverse_sampling/h36m_t2.pth"
        

    m = DiverseSamplingWrapper(n_features, n_landmarks, obs, horizon,
                            model_path_t1, model_path_t2,
                            node_n=n_landmarks, hidden_dim=256, base_dim = 128, z_dim=nz, dct_n=10, base_num_p1=40, dropout_rate=0).to(device)
    #print(f"{sum(p.numel() for p in m.parameters()) / 1e6}")

    # dumb input
    x = torch.tensor(np.zeros((batch_size, obs, participants, n_landmarks, n_features), dtype=np.float64), device=device, dtype=dtype).contiguous()
    y = torch.tensor(np.zeros((batch_size, horizon, participants, n_landmarks, n_features), dtype=np.float64), device=device, dtype=dtype).contiguous()
    print("\n> INPUT:\nx", x.shape, "\ny", y.shape)

    out = m.forward(x, y, 50)[0]
    print("\n> OUTPUT:\n", out.shape)