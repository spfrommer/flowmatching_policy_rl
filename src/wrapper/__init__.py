from torch import Tensor

import torch
from flow_matching.utils import ModelWrapper
from torch.nn.modules import Module


class VanillaWrapper(ModelWrapper):
    def __init__(self, model: Module):
        super().__init__(model)

    def forward(self, x: Tensor, t: Tensor, extra: dict):
        assert t.shape == torch.Size([]) or t.shape == torch.Size([x.shape[0]])
        if t.shape == torch.Size([]):
            t = torch.zeros(x.shape[0], device=x.device) + t
        return self.model(x, t, conditioning=extra["conditioning"])
