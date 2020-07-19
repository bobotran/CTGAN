import torch
import torch.nn as nn
import torch.nn.functional as F

from ctgan.glow.models.glow.act_norm import ActNorm
from ctgan.glow.models.glow.coupling import Coupling
from ctgan.glow.models.glow.inv_conv import InvConv


class Glow(nn.Module):
    """Glow Model

    Based on the paper:
    "Glow: Generative Flow with Invertible 1x1 Convolutions"
    by Diederik P. Kingma, Prafulla Dhariwal
    (https://arxiv.org/abs/1807.03039).

    Args:
        hidden_layers (list(int)): List of hidden layer sizes in fc nets estimating scale-shift.
        num_levels (int): Number of levels in the entire model.
        num_steps (int): Number of steps of flow for each level.
    """
    def __init__(self, dim, hidden_layers, num_levels, num_steps):
        super(Glow, self).__init__()

        # Use bounds to rescale images before converting to logits, not learned
        self.register_buffer('bounds', torch.tensor([0.9], dtype=torch.float32))
        self.flows = _Glow(dim=dim,  # Feed rows as (N, 157, 1, 1) "images"
                           hidden_layers=hidden_layers,
                           num_levels=num_levels,
                           num_steps=num_steps)

    def forward(self, x, reverse=False):
        assert len(x.size()) == 2, 'Expects (N, D) inputs'
        x = x.unsqueeze(-1).unsqueeze(-1)
#        if reverse:
        sldj = torch.zeros(x.size(0), device=x.device)
#        else:
#            # Expect inputs in [0, 1]
#            if x.min() < 0 or x.max() > 1:
#                raise ValueError('Expected x in [0, 1], got min/max {}/{}'
#                                 .format(x.min(), x.max()))

            # De-quantize and convert to logits
#            x, sldj = self._pre_process(x)

        x, sldj = self.flows(x, sldj, reverse)

        return x.squeeze(), sldj

    def _pre_process(self, x):
        """Dequantize the input image `x` and convert to logits.

        See Also:
            - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
            - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1

        Args:
            x (torch.Tensor): Input image.

        Returns:
            y (torch.Tensor): Dequantized logits of `x`.
        """
        y = (x * 255. + torch.rand_like(x)) / 256.
        y = (2 * y - 1) * self.bounds
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) \
            - F.softplus((1. - self.bounds).log() - self.bounds.log())
        sldj = ldj.flatten(1).sum(-1)

        return y, sldj


class _Glow(nn.Module):
    """Recursive constructor for a Glow model. Each call creates a single level.

    Args:
        dim (int): Number of channels in the input.
        hidden_layers (int): Number of channels in hidden layers of each step.
        num_levels (int): Number of levels to construct. Counter for recursion.
        num_steps (int): Number of steps of flow for each level.
    """
    def __init__(self, dim, hidden_layers, num_levels, num_steps):
        super(_Glow, self).__init__()
        self.steps = nn.ModuleList([_FlowStep(dim=dim,
                                              hidden_layers=hidden_layers)
                                    for _ in range(num_steps)])

        if num_levels > 1:
            self.next = _Glow(dim=dim // 2,
                              hidden_layers=hidden_layers,
                              num_levels=num_levels - 1,
                              num_steps=num_steps)
        else:
            self.next = None

    def forward(self, x, sldj, reverse=False):
        if not reverse:
            for step in self.steps:
                x, sldj = step(x, sldj, reverse)

        if self.next is not None:
            x, x_split = x.chunk(2, dim=1)
            x, sldj = self.next(x, sldj, reverse)
            x = torch.cat((x, x_split), dim=1)

        if reverse:
            for step in reversed(self.steps):
                x, sldj = step(x, sldj, reverse)

        return x, sldj


class _FlowStep(nn.Module):
    def __init__(self, dim, hidden_layers):
        super(_FlowStep, self).__init__()

        # Activation normalization, invertible 1x1 convolution, affine coupling
        self.norm = ActNorm(dim, return_ldj=True)
        self.conv = InvConv(dim)
        self.coup = Coupling(dim, hidden_layers)

    def forward(self, x, sldj=None, reverse=False):
        if reverse:
            x, sldj = self.coup(x, sldj, reverse)
            x, sldj = self.conv(x, sldj, reverse)
            x, sldj = self.norm(x, sldj, reverse)
        else:
            x, sldj = self.norm(x, sldj, reverse)
            x, sldj = self.conv(x, sldj, reverse)
            x, sldj = self.coup(x, sldj, reverse)

        return x, sldj
