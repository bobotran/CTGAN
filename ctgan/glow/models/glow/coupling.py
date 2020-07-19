import torch
import torch.nn as nn
import torch.nn.functional as F

from ctgan.glow.models.glow.act_norm import ActNorm


class Coupling(nn.Module):
    """Affine coupling layer originally used in Real NVP and described by Glow.

    Note: The official Glow implementation (https://github.com/openai/glow)
    uses a different affine coupling formulation than described in the paper.
    This implementation follows the paper and Real NVP.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate activation
            in NN.
    """
    def __init__(self, in_channels, hidden_layers):
        super(Coupling, self).__init__()
        self.nn = NN(in_channels // 2, hidden_layers, in_channels)
        self.scale = nn.Parameter(torch.ones(in_channels // 2))

    def forward(self, x, ldj, reverse=False):
        x_id, x_change = x.squeeze().chunk(2, dim=1)

        st = self.nn(x_id)
        s, t = st[:, 0::2], st[:, 1::2]
        s = self.scale * torch.tanh(s)

        # Scale and translate
        if reverse:
            x_change = x_change * s.mul(-1).exp() - t
            ldj = ldj - s.sum(-1)
        else:
            x_change = (x_change + t) * s.exp()
            ldj = ldj + s.sum(-1)

        x = torch.cat((x_id, x_change), dim=1).unsqueeze(-1).unsqueeze(-1)

        return x, ldj

class NN(nn.Module):
    def __init__(self, in_channels, hidden_layers, out_channels, use_act_norm=False):
        super(NN, self).__init__()
        norm_fn = ActNorm if use_act_norm else nn.BatchNorm1d

        modules = []
        modules.append(norm_fn(in_channels))

        linear_layer = nn.Linear(in_channels, hidden_layers[0])
        nn.init.normal_(linear_layer.weight, 0., 0.05)
        modules.append(linear_layer)
        modules.append(norm_fn(hidden_layers[0]))
        modules.append(nn.ReLU())
        
        layers_to_add = hidden_layers
        while len(layers_to_add) >= 2:
            linear_layer = nn.Linear(layers_to_add[0], layers_to_add[1])
            nn.init.normal_(linear_layer.weight, 0., 0.05)
            modules.append(linear_layer)
            modules.append(norm_fn(layers_to_add[1]))
            modules.append(nn.ReLU())

        linear_layer = nn.Linear(hidden_layers[-1], out_channels)
        nn.init.zeros_(linear_layer.weight)
        nn.init.zeros_(linear_layer.bias)
        modules.append(linear_layer)

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)
