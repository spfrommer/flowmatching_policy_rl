from einops import pack, rearrange
import torch
from torch import Tensor
from jaxtyping import Float
from torch import nn
from envs.unicycle.env import UnicycleEnvParams, UnicycleObservation
from core.specs import Action, TimedActionTrajectory, Observation, ObservationProcessed, ObservationProcessedAsActivations, ObservationProcessedAsVector, TimedActionTrajectoryNormalized


############################### OBSERVATION PROCESSORS #################################


class ObservationProcessor(nn.Module):
    def process(self, observation: Observation) -> ObservationProcessed:
        pass


class UnicycleObservationProcessor(ObservationProcessor):
    def __init__(self, env_params: UnicycleEnvParams):
        super().__init__()
        self.env_params = env_params
        
        self.normalizer = Normalizer(
            min_val=torch.tensor([
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                0.0,
                -1.0,
                -1.0,
            ]),
            max_val=torch.tensor([
                1.0,
                1.0,
                1.0,
                1.0,
                env_params.max_velocity,
                1.0,
                1.0,
            ]),
        )
    
    def process(self, observation: UnicycleObservation) -> ObservationProcessedAsVector:
        concatenated = torch.cat([
            observation.position,
            observation.heading,
            observation.velocity,
            observation.instruction.target,
        ], dim=-1)
        normalized = self.normalizer.normalize(concatenated)
        return ObservationProcessedAsVector(
            observations=normalized,
            batch_size=observation.batch_size,
        )
    


############################### ACTION NORMALIZERS #####################################
    
class Normalizer(nn.Module):
    def __init__(
        self,
        min_val: Float[Tensor, 'channels'],
        max_val: Float[Tensor, 'channels'],
    ):
        super().__init__()
            
        self.channels = min_val.shape[0]
        self.register_buffer('min_val', min_val)
        self.register_buffer('max_val', max_val)

    def normalize(
        self, x: Float[Tensor, 'b c ...']
    ) -> Float[Tensor, 'b c ...']:
        
        assert x.shape[1] == self.channels

        shape = [1, self.channels] + [1] * (x.ndim - 2)
        min_val = self.min_val.view(shape)
        max_val = self.max_val.view(shape)
        
        # Normalize to [-1, 1] range using min and max
        normalized = (x - min_val) / (max_val - min_val)
        normalized = normalized * 2 - 1
        assert normalized.min() >= -1.0 and normalized.max() <= 1.0
            
        return normalized

    def unnormalize(
        self, x: Float[Tensor, 'b c ...']
    ) -> Float[Tensor, 'b c ...']:

        assert x.shape[1] == self.channels

        assert x.min() >= -1.0 and x.max() <= 1.0, f'{x.min()} {x.max()}'

        shape = [1, self.channels] + [1] * (x.ndim - 2)
        min_val = self.min_val.view(shape)
        max_val = self.max_val.view(shape)
        
        # Unnormalize from [-1, 1] range back to original range
        normalized = (x + 1) / 2
        return normalized * (max_val - min_val) + min_val

    def __repr__(self) -> str:
        return f"Normalizer(min_val={self.min_val.shape}, max_val={self.max_val.shape})"


class ActionTrajectoryNormalizer(nn.Module):
    def __init__(
        self,
        min_action: Float[Tensor, 'channels'],
        max_action: Float[Tensor, 'channels'],
        min_total_time_steps: Float[Tensor, '1'],
        max_total_time_steps: Float[Tensor, '1'],
    ):
        super().__init__()

        self.normalizer = Normalizer(min_action, max_action)
        self.time_normalizer = Normalizer(min_total_time_steps, max_total_time_steps)

    def normalize(
        self,
        actions: TimedActionTrajectory,
    ) -> TimedActionTrajectoryNormalized:

        return TimedActionTrajectoryNormalized(
            actions=self.normalizer.normalize(actions.actions),
            total_time_steps=self.time_normalizer.normalize(actions.total_time_steps),
            batch_size=actions.batch_size
        )
    
    def unnormalize(
        self,
        normalized: TimedActionTrajectoryNormalized,
    ) -> TimedActionTrajectory:

        return TimedActionTrajectory(
            actions=self.normalizer.unnormalize(normalized.actions),
            total_time_steps=self.time_normalizer.unnormalize(
                normalized.total_time_steps
            ),
            batch_size=normalized.batch_size
        )
    

class UnicycleActionTrajectoryNormalizer(ActionTrajectoryNormalizer):
    def __init__(
        self,
        env_params: UnicycleEnvParams,
        max_total_time_steps: int,
    ):
        super().__init__(
            min_action=torch.tensor([
                -env_params.max_angular_velocity,
                -env_params.max_acceleration,
            ]),
            max_action=torch.tensor([
                env_params.max_angular_velocity,
                env_params.max_acceleration,
            ]),
            min_total_time_steps=torch.tensor([0.0]),
            max_total_time_steps=torch.tensor([float(max_total_time_steps)]),
        )

