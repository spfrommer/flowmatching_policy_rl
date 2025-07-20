# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Callable, Optional, Sequence, Tuple, Union

from einops import rearrange
import torch
from torch import Tensor
from torchdiffeq import odeint

from flow_matching.solver.solver import Solver
from flow_matching.utils import gradient, ModelWrapper


class SDESolver(Solver):
    def __init__(self, velocity_model: Union[ModelWrapper, Callable]):
        super().__init__()
        self.velocity_model = velocity_model

    def sample(
        self,
        x_init: Tensor,
        step_size: float,
        noise_level: float = 1.0,
        return_intermediates: bool = False,
        extra: dict = {},
    ) -> Union[Tensor, dict[str, Any]]:

        device = x_init.device
        
        steps = int(1.0 / step_size)
        time_grid = torch.linspace(0.0, 1.0 - step_size, steps, device=device)
        # time_grid = torch.linspace(1.0, step_size, steps, device=device)
        
        if return_intermediates:
            return_dict = {
                'intermediates': [x_init],
            }
        else:
            return_dict = {}
            
        x_t = x_init
        
        for i, t in enumerate(time_grid):
            posterior_mean, posterior_var = self.posterior_gaussian(
                x_t=x_t,
                t=t,
                step_size=step_size,
                noise_level=noise_level,
                extra=extra,
            )
            
            # if i == len(time_grid) - 1:
            #     x_t = posterior_mean
            # else:
            #     x_t = posterior_mean + torch.randn_like(x_t) * torch.sqrt(posterior_var)
            x_t = posterior_mean + torch.randn_like(x_t) * torch.sqrt(posterior_var) 

            if return_intermediates:
                return_dict['intermediates'].append(x_t.clone())

        return x_t, return_dict
        
    def posterior_gaussian(
        self,
        x_t: Tensor,
        t: float,
        step_size: float,
        noise_level: float,
        extra: dict = {},
    ) -> tuple[Tensor, Tensor]: # Return posterior mean and variance
        
        v_t = self.velocity_model(x=x_t, t=t, extra=extra)
        
        if isinstance(t, Tensor):
            assert t.shape == torch.Size([]) or t.shape == torch.Size([x_t.shape[0]])
            if t.shape == torch.Size([x_t.shape[0]]):
                t = rearrange(t, 'b -> b 1 1')

        sigma_t = noise_level * (1-t) ** 2
        drift_term = v_t + ((sigma_t ** 2) / (2 * (1 - t))) * (x_t + (t) * v_t)
        
        return x_t + drift_term * step_size, sigma_t ** 2 * step_size
