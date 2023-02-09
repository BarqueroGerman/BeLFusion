
import torch.nn as nn
import torch
import numpy as np
from abc import abstractmethod
from einops import rearrange, repeat


nl = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "softplus": nn.Softplus,
    "softsign": nn.Softsign,
    "leaky_relu": nn.LeakyReLU,
    "none": lambda x: x,
}

rnn = {
    "lstm": nn.LSTM,
    "gru": nn.GRU
}

rnn_default_initial_states = {
    "lstm": lambda h_size, bs, dev: (torch.zeros((1, bs, h_size)).to(dev), torch.zeros((1, bs, h_size)).to(dev)),
    "gru": lambda h_size, bs, dev: torch.zeros((1, bs, h_size)).to(dev)
}


class BaseModel(nn.Module):
    
    def __init__(self, n_features, n_landmarks, obs_length, pred_length):
        super(BaseModel, self).__init__()
        self.n_features = n_features
        self.n_landmarks = n_landmarks
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.total_length = obs_length + pred_length

    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError
        
    def get_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params


    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class BaseAutoencoder(BaseModel):
    def __init__(self, n_features, n_landmarks, obs_length, pred_length):
        super(BaseAutoencoder, self).__init__(n_features, n_landmarks, obs_length, pred_length)

    @abstractmethod
    def encode(self, x):
        raise NotImplementedError

    @abstractmethod
    def decode(self, x_start, h_y):
        raise NotImplementedError
        


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode="fan_out")
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Linear):
        # print("weights ", module)
        for name, param in module.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_uniform_(param)
    elif (
        isinstance(module, nn.LSTM)
        or isinstance(module, nn.RNN)
        or isinstance(module, nn.LSTMCell)
        or isinstance(module, nn.RNNCell)
        or isinstance(module, nn.GRU)
        or isinstance(module, nn.GRUCell)
    ):
        # https://www.cse.iitd.ac.in/~mausam/courses/col772/spring2018/lectures/12-tricks.pdf
        # • It can take a while for a RNN to learn to remember information
        # • Initialize biases for LSTM’s forget gate to 1 to remember more by default.
        # • Similarly, initialize biases for GRU’s reset gate to -1.
        DIV = 3 if isinstance(module, nn.GRU) or isinstance(module, nn.GRUCell) else 4
        for name, param in module.named_parameters():
            if "bias" in name:
                #print(name)
                nn.init.constant_(
                    param, 0.0
                )  
                if isinstance(module, nn.LSTMCell) \
                    or isinstance(module, nn.LSTM):
                    n = param.size(0)
                    # LSTM: (W_ii|W_if|W_ig|W_io), W_if (forget gate) => bias 1
                    start, end = n // DIV, n // 2
                    param.data[start:end].fill_(1.) # to remember more by default
                elif isinstance(module, nn.GRU) \
                    or isinstance(module, nn.GRUCell):
                    # GRU: (W_ir|W_iz|W_in), W_ir (reset gate) => bias -1
                    end = param.size(0) // DIV
                    param.data[:end].fill_(-1.) # to remember more by default
            elif "weight" in name:
                nn.init.xavier_normal_(param)
                if isinstance(module, nn.LSTMCell) \
                    or isinstance(module, nn.LSTM) \
                    or isinstance(module, nn.GRU) \
                    or isinstance(module, nn.GRUCell):
                    if 'weight_ih' in name: # input -> hidden weights
                        mul = param.shape[0] // DIV
                        for idx in range(DIV):
                            nn.init.xavier_uniform_(param[idx * mul:(idx + 1) * mul])
                    elif 'weight_hh' in name: # hidden -> hidden weights (recurrent)
                        mul = param.shape[0] // DIV
                        for idx in range(DIV):
                            nn.init.orthogonal_(param[idx * mul:(idx + 1) * mul]) # orthogonal initialization https://arxiv.org/pdf/1702.00071.pdf
    else:
        print(f"[WARNING] Module not initialized: {module}")