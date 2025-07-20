from einops import pack, repeat
from tensordict import TensorClass
from typing import NewType
from torch import Tensor
from jaxtyping import Float, Int
import torch
import torch.nn.functional as F


class State(TensorClass):
    """ Instantiated on a per-environment basis. """
    pass


class Instruction(TensorClass):
    """ Instantiated on a per-environment basis. """
    pass

class Observation(TensorClass):
    """ Instantiated on a per-environment basis. """
    
    instruction: Instruction | None
    
    def strip_for_saving(self) -> None:
        """ Strip the observation to the minimum required for saving to file. """
        pass


class ObservationProcessed(TensorClass):
    pass

class ObservationProcessedAsVector(TensorClass):
    observations: Float[Tensor, '... d']

class ObservationProcessedAsActivations(TensorClass):
    observations: Float[Tensor, '... channels sequence']



class Action(TensorClass):
    """ Instantiated on a per-environment basis. """
    
class ActionTrajectory(TensorClass):
    """ Instantiated on a per-environment basis.
    
    Can only be batched if all horizons are the same."""
    actions: Action  # horizon x batch (optional) x action
    
    def __post_init__(self):
        assert self.actions.batch_dims >= 1

    def as_trajectory_tensor(self) -> Float[Tensor, '... action horizon']:
        pass
    
    @staticmethod
    def from_trajectory_tensor(
        trajectory: Float[Tensor, '... action horizon'],
    ) -> 'ActionTrajectory':
        pass


class TimedActionTrajectory(TensorClass):
    """ Can be batched as varying horizons have been interpolated. """
    actions: Float[Tensor, '... dim horizon']
    total_time_steps: Float[Tensor, '... 1']
    
    def interpolate(self, desired_horizon: int) -> 'TimedActionTrajectory':
        return TimedActionTrajectory(
            actions=interpolate_actions(self.actions, desired_horizon),
            total_time_steps=self.total_time_steps,
            batch_size=self.batch_size,
        )
    
    def to_unit_timesteps_batched(self) -> tuple[
        'TimedActionTrajectory',
        Int[Tensor, 'batch']
    ]:
        truncate_times = torch.clamp(self.total_time_steps.squeeze(-1).int(), min=1)
        horizons = truncate_times
        max_horizon = horizons.max()
        batch_size = self.batch_size

        interpolated_actions = torch.zeros(
            *batch_size,
            self.actions.shape[-2],
            max_horizon.item(),
            device=self.device,
            dtype=self.actions.dtype,
        )
        
        for i in range(batch_size[0]):
            truncate_time = truncate_times[i].item()
            interpolated = interpolate_actions(
                self[i].actions, 
                desired_horizon=truncate_time
            )
            interpolated_actions[i, :, :truncate_time] = interpolated[:, :truncate_time]
        
        result = TimedActionTrajectory(
            actions=interpolated_actions,
            total_time_steps=truncate_times.unsqueeze(-1).float(),
            batch_size=batch_size,
            device=self.device,
        )
        
        return result, horizons.to(self.device)

    
    def to_unit_timesteps_not_batched(self) -> 'TimedActionTrajectory':
        assert self.batch_dims == 0
        
        truncate_time = max(int(self.total_time_steps.item()), 1)
        # TODO: more exact interpolation
        interpolated = self.interpolate(desired_horizon=truncate_time)
        return TimedActionTrajectory(
            actions=interpolated.actions[..., :truncate_time],
            total_time_steps=torch.tensor([truncate_time]).to(interpolated.device),
            batch_size=interpolated.batch_size,
        )

class TimedActionTrajectoryNormalized(TensorClass):
    actions: Float[Tensor, '... dim horizon']
    total_time_steps: Float[Tensor, '... 1']
    
    @staticmethod
    def from_unified(
        unified: Float[Tensor, '... dim_p_1 h'],
    ) -> 'TimedActionTrajectoryNormalized':

        return TimedActionTrajectoryNormalized(
            actions=unified[..., :-1, :],
            total_time_steps=unified[..., -1, :].mean(dim=-1).unsqueeze(-1),
            batch_size=unified.shape[:-2],
        )
    
    def unify(self) -> Float[Tensor, '... dim_p_1 horizon']:
        return torch.cat(
            [
                self.actions,
                repeat(self.total_time_steps, '... 1 -> ... 1 h', h=self.actions.shape[-1]),
            ],
            dim=-2
        )
    
    

class Reward(TensorClass):
    rewards: Float[Tensor, '... 1']
    


def interpolate_actions(
    actions: Float[Tensor, 'batch action horizon'] | Float[Tensor, 'action horizon'],
    desired_horizon: int,
) -> Float[Tensor, 'batch action new_horizon'] | Float[Tensor, 'action new_horizon']:

    original_ndim = actions.ndim
    
    if original_ndim == 2:
        actions = actions.unsqueeze(0)
    
    interpolated = F.interpolate(
        actions,
        size=desired_horizon,
        mode='linear',
        align_corners=True,
    )
    
    if original_ndim == 2:
        return interpolated.squeeze(0)
    
    return interpolated
