from envs import Environment, Renderer, SimResult
from envs.unicycle.render import UnicycleRenderer
from core.specs import Action, ActionTrajectory, Observation, State

from dataclasses import dataclass
from einops import pack, rearrange
from torch import Tensor

import torch
from jaxtyping import Float
    
from envs.unicycle.specs import UnicycleAction, UnicycleActionTrajectory, UnicycleInstruction, UnicycleObservation, UnicycleState


@dataclass
class UnicycleEnvParams:
    max_velocity: float = 1.0
    max_angular_velocity: float = 3.0
    max_acceleration: float = 1.0
    dt: float = 0.1


class UnicycleSimResult(SimResult):
    states: UnicycleState
    observations: UnicycleObservation
    action_trajectories: UnicycleActionTrajectory
    

class UnicycleEnv(Environment):
    STATE_TYPE: type[State] = UnicycleState
    OBSERVATION_TYPE: type[Observation] = UnicycleObservation
    ACTION_TYPE: type[Action] = UnicycleAction
    ACTION_TRAJECTORY_TYPE: type[ActionTrajectory] = UnicycleActionTrajectory
    SIM_RESULT_TYPE: type[SimResult] = UnicycleSimResult
    RENDERER_TYPE: type[Renderer] = UnicycleRenderer
    

    def __init__(
        self, 
        params: UnicycleEnvParams = UnicycleEnvParams(),
    ):
        self.params = params
        self.state = None
        
    def random_initial_state(self) -> UnicycleState:
        heading = torch.rand(2)
        heading = heading / torch.norm(heading)
        
        return UnicycleState(
            position=(torch.rand(2) * 2) - 1,
            heading=heading,
            velocity=torch.rand(1) * self.params.max_velocity,
            batch_size=(),
        )
        
    def random_instruction(self) -> UnicycleInstruction:
        return UnicycleInstruction(
            target=(torch.rand(2) * 2) - 1,
            batch_size=(),
        )
    
    def reset(self, initial_state: UnicycleState) -> None:
        self.state = initial_state
        
    def step(self, action: UnicycleAction) -> None:
        assert action.batch_size == self.state.batch_size

        state = self.state
        
        ang_vel = torch.clamp(
            action.angular_velocity.squeeze(-1),
            -self.params.max_angular_velocity, 
            self.params.max_angular_velocity
        )
        
        accel = torch.clamp(
            action.acceleration.squeeze(-1),
            -self.params.max_acceleration, 
            self.params.max_acceleration
        )
        
        x, y = state.position.unbind(dim=-1)
        heading_x, heading_y = state.heading.unbind(dim=-1)
        vel = state.velocity.squeeze(-1)
            
        new_vel = vel + accel * self.params.dt
        new_vel = torch.clamp(new_vel, 0.0, self.params.max_velocity)
        
        
        theta = torch.atan2(heading_y, heading_x)
        theta += ang_vel * self.params.dt
        heading_x, heading_y = torch.cos(theta), torch.sin(theta)
        
        new_x = x + new_vel * heading_x * self.params.dt
        new_y = y + new_vel * heading_y * self.params.dt
        
        
        out_of_bounds = (new_x < -1.0) | (new_x > 1.0) | (new_y < -1.0) | (new_y > 1.0)
        new_x = torch.clamp(new_x, -1.0, 1.0)
        new_y = torch.clamp(new_y, -1.0, 1.0)
        
        new_vel = torch.where(out_of_bounds, torch.zeros_like(new_vel), new_vel)
        
        self.state = UnicycleState(
            position=torch.stack([new_x, new_y], dim=-1),
            heading=torch.stack([heading_x, heading_y], dim=-1),
            velocity=torch.stack([new_vel], dim=-1),
            batch_size=state.batch_size,
        )

    def get_state(self) -> UnicycleState:
        return self.state
    
    def get_observation(
        self, 
        state: UnicycleState | None = None,
    ) -> UnicycleObservation:
        if state is None:
            state = self.state
            
        return UnicycleObservation(
            position=state.position,
            heading=state.heading,
            velocity=state.velocity,
            batch_size=state.batch_size,
            instruction=None,
        )
    
    def open_loop_sim(
        self,
        initial_state: UnicycleState,
        initial_observation: UnicycleObservation,
        action_trajectory: UnicycleActionTrajectory,
    ) -> UnicycleSimResult:

        return super().open_loop_sim(
            initial_state=initial_state,
            initial_observation=initial_observation,
            action_trajectory=action_trajectory,
        )