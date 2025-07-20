from einops import pack, rearrange, repeat, reduce
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float
import numpy as np

from core.specs import Observation, ObservationProcessed, ObservationProcessedAsVector, TimedActionTrajectory, TimedActionTrajectoryNormalized
from envs.unicycle.specs import UnicycleObservation
from nn.resnet import ResNet1D
from nn.timesnet import TimesNet, TimesNetConfig
from nn.unet import UNetModel
from training.args import TrainArgs


class QFunction(nn.Module):
    def forward(
        self,
        observation: ObservationProcessed,
        action: TimedActionTrajectoryNormalized,
    ) -> Float[Tensor, '... 1']:
        pass
    
    
class UnicycleQFunctionTimesNet(QFunction):
    def __init__(self, args: TrainArgs):
        super().__init__()
        
        input_channels = 7 + 2 + 1
        
        self.net = TimesNet(
            in_channels=input_channels,
            config=TimesNetConfig(
                seq_len=64,
                pred_len=0,
                e_layers=args.grpo_q_function_layers,
                d_model=args.grpo_q_function_dimension,
                d_ff=args.grpo_q_function_dimension,
            )
        )

    def forward(
        self,
        observation: ObservationProcessedAsVector,
        action: TimedActionTrajectoryNormalized,
    ) -> Float[Tensor, '... 1']:

        length = action.actions.shape[-1]

        x = torch.cat([
            action.unify(),
            repeat(observation.observations, 'b c -> b c l', l=length),
        ], dim=1)
        
        out = self.net(rearrange(x, 'b c l -> b l c'))
        return out