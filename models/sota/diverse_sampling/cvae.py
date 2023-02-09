import torch
from torch.nn import Module, Sequential, Linear

from .gcn_layers import GraphConv, GraphConvBlock, ResGCB
import numpy as np

class CVAE(Module):
    def __init__(self, node_n=16, hidden_dim=256, z_dim=64, dct_n=10, dropout_rate=0):
        super(CVAE, self).__init__()

        self.node_n = node_n
        self.dct_n = dct_n
        self.z_dim = z_dim

        self.enc = Sequential(
            GraphConvBlock(in_len=3 * dct_n * 2, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n,
                           dropout_rate=dropout_rate, bias=True, residual=False),
            ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate,
                   bias=True, residual=True),
            ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate,
                   bias=True, residual=True),
            ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate,
                   bias=True, residual=True),
            ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate,
                   bias=True, residual=True),
        )

        self.mean = Linear(hidden_dim * node_n, z_dim)
        self.logvar = Linear(hidden_dim * node_n, z_dim)

        self.dec = Sequential(
            GraphConvBlock(in_len=3*dct_n+z_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=False),
            ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=True),
            ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=True),
            ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=True),
            ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=True),
            GraphConv(in_len=hidden_dim, out_len=3*dct_n, in_node_n=node_n, out_node_n=node_n, bias=True)
        )

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def _sample(self, mean, logvar):
        """Returns a sample from a Gaussian with diagonal covariance."""
        return torch.exp(0.5*logvar) * torch.randn_like(logvar) + mean

    def forward(self, condition, data):
        '''
        Args:
            condition: [b, 48, 25] / [b, 16, 30]
            data: [b, 48, 100] / [b, 16, 30]
        Returns:
        '''
        b, v, ct = condition.shape
        feature = self.enc(torch.cat((condition, data), dim=-1)) # b, 16, 60 -> b, 16, 256

        mean = self.mean(feature.view(b, -1))
        logvar = self.logvar(feature.view(b, -1))
        z = self._sample(mean, logvar)

        out = self.dec(torch.cat((condition, z.unsqueeze(dim=1).repeat([1, self.node_n, 1])), dim=-1))  # b, 16, 30+64
        out = out + condition
        return out, mean, logvar


    def inference(self, condition, z):
        '''
        Args:
            condition: [b, 48, 25] / [b, 16, 30]
            z: b, 64
        Returns:
        '''
        out = self.dec(torch.cat((condition, z.unsqueeze(dim=1).repeat([1, self.node_n, 1])), dim=-1))  # b, 16, 30+64
        out = out + condition
        return out