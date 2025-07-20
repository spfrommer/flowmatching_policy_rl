import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

from core.specs import Observation, ObservationProcessed, ObservationProcessedAsVector
from envs.unicycle.specs import UnicycleObservation


class ValueFunction(nn.Module):
    def forward(self, observation: ObservationProcessed) -> Float[Tensor, '... 1']:
        pass
    

class UnicycleValueFunction(ValueFunction):
    def __init__(self):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(
        self,
        observation: ObservationProcessedAsVector,
    ) -> Float[Tensor, '... 1']:

        return self.mlp(observation.observations)