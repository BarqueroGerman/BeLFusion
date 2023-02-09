import torch
import torch.nn as nn


class ClassifierForFID(nn.Module):
    def __init__(self, input_size=48, hidden_size=128, hidden_layer=2, output_size=15, device="", use_noise=None):
        super(ClassifierForFID, self).__init__()
        self.device = device

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.use_noise = use_noise

        self.recurrent = nn.GRU(input_size, hidden_size, hidden_layer)
        self.linear1 = nn.Linear(hidden_size, 30)
        self.linear2 = nn.Linear(30, output_size)

    
    def forward(self, motion_sequence, hidden_unit=None):
        '''
        motion_sequence: b, 48, 100
        hidden_unit:
        '''
        motion_sequence = motion_sequence.permute(2, 0, 1).contiguous() # [100, b, 48]
        # dim (motion_length, num_samples, hidden_size)
        if hidden_unit is None:
            hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer) # [2, b, 128]

        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit) # [100, b, 48]
        # dim (num_samples, 30)
        lin1 = self.linear1(gru_o[-1, :, :]) # [b, 48]
        lin1 = torch.tanh(lin1)
        # dim (num_samples, output_size)
        lin2 = self.linear2(lin1)
        return lin2

    def get_fid_features(self, motion_sequence, hidden_unit=None):
        '''
        motion_sequence: b, 48, 100
        hidden_unit:
        '''
        motion_sequence = motion_sequence.permute(2, 0, 1).contiguous()  # [100, b, 48]

        # dim (motion_length, num_samples, hidden_size)
        if hidden_unit is None:
            hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer)

        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)
        # dim (num_samples, 30)
        lin1 = self.linear1(gru_o[-1, :, :])
        lin1 = torch.tanh(lin1)
        return lin1


    def initHidden(self, num_samples, layer):
        return torch.randn(layer, num_samples, self.hidden_size, device=self.device, requires_grad=False)
