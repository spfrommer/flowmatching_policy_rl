from core.specs import Action, ActionTrajectory, Instruction, Observation, State

from torch import Tensor

import torch
from jaxtyping import Float


class UnicycleInstruction(Instruction):
    target: Float[Tensor, '... 2']

class UnicycleState(State):
    position: Float[Tensor, '... 2']
    heading: Float[Tensor, '... 2']
    velocity: Float[Tensor, '... 1']

class UnicycleObservation(Observation):
    position: Float[Tensor, '... 2']
    heading: Float[Tensor, '... 2']
    velocity: Float[Tensor, '... 1']

    instruction: UnicycleInstruction | None


class UnicycleAction(Action):
    angular_velocity: Float[Tensor, '... 1']
    acceleration: Float[Tensor, '... 1']
    
    
class UnicycleActionTrajectory(ActionTrajectory):
    actions: UnicycleAction # [... horizon action]

    def as_trajectory_tensor(self) -> Float[Tensor, '... action horizon']:
        return torch.cat([
            self.actions.angular_velocity,
            self.actions.acceleration,
        ], dim=-1).transpose(-2, -1)
    
    @staticmethod
    def from_trajectory_tensor(
        trajectory: Float[Tensor, '... action horizon'],
    ) -> 'UnicycleActionTrajectory':

        # Batch size is [..., horizon]
        batch_size = tuple(trajectory.shape[:-2]) + (trajectory.shape[-1],) 

        return UnicycleActionTrajectory(
            actions=UnicycleAction(
                angular_velocity=trajectory[..., :1, :].transpose(-2, -1),
                acceleration=trajectory[..., 1:2, :].transpose(-2, -1),
                batch_size=batch_size,
            ),
            batch_size=batch_size,
        )