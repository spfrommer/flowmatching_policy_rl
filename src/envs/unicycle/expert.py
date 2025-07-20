from dataclasses import dataclass
import torch
from torch import Tensor
from jaxtyping import Float
from torch.nn import functional as F
from envs.unicycle.env import UnicycleEnv, UnicycleState, UnicycleAction
from envs.unicycle.specs import UnicycleObservation


@dataclass
class UnicycleEnvExpertParams:
    """Parameters for the UnicycleEnvExpert."""
    heading_kp: float = 1.0
    velocity_kp: float = 2.0
    goal_velocity: float = 0.5


class UnicycleEnvExpert:
    def __init__(
        self, 
        env: UnicycleEnv,
        params: UnicycleEnvExpertParams = UnicycleEnvExpertParams(),
    ):
        """Initialize the expert controller."""
        self.env = env
        self.params = params
    
    def compute_heading_error(self, observation: UnicycleObservation) -> Float[Tensor, '']:
        """Compute the heading error angle."""
        heading = observation.heading
        target = observation.instruction.target
        target_dir = F.normalize(target - observation.position, dim=0)
        
        return torch.atan2(
            heading[0] * target_dir[1] - heading[1] * target_dir[0],
            heading[0] * target_dir[0] + heading[1] * target_dir[1],
        )
    
    def reset(self):
        pass
    
    def compute_action(
        self,
        observation: UnicycleObservation,
    ) -> UnicycleAction:
        
        assert observation.batch_size == ()

        angle_error = self.compute_heading_error(observation)
        
        angular_velocity = self.params.heading_kp * angle_error
        angular_velocity = torch.clamp(
            angular_velocity,
            -self.env.params.max_angular_velocity,
            self.env.params.max_angular_velocity,
        )
        
        velocity_error = self.params.goal_velocity - observation.velocity[0]
        acceleration = self.params.velocity_kp * velocity_error
        acceleration = torch.clamp(
            acceleration,
            -self.env.params.max_acceleration,
            self.env.params.max_acceleration,
        )
        
        return UnicycleAction(
            angular_velocity=angular_velocity.reshape(1),
            acceleration=acceleration.reshape(1),
            batch_size=(),
        )