from dataclasses import dataclass, field
import time
from einops import rearrange, reduce
import torch
from torch import Tensor
from jaxtyping import Float
import numpy as np

from envs import SimResult
from envs.unicycle.env import UnicycleActionTrajectory, UnicycleEnv, UnicycleObservation, UnicycleState
from reward import RewardModel
from core.specs import TimedActionTrajectory, Observation, Reward, State

import logging
logger = logging.getLogger(__name__)


@dataclass
class UnicycleRewardModelParams:
    final_position_weight: float = 1.0
    final_velocity_weight: float = 0.0
    total_time_weight: float = 0.0
    pointing_north_weight: float = 0.0
    wall_contact_weight: float = 0.0
    control_magnitude_weight: float = 0.0


class UnicycleRewardModel(RewardModel):
    components: list[str] = [
        'final_position', 'final_velocity', 'total_time',
        'pointing_north', 'wall_contact', 'control_magnitude'
    ]
    
    def __init__(
        self,
        env: UnicycleEnv,
        params: UnicycleRewardModelParams,
    ):
        self.env = env
        self.params = params

    def __call__(
        self,
        states: UnicycleState,
        observations: UnicycleObservation,
        actions: TimedActionTrajectory,
    ) -> Reward:

        device = states.device
        
        states = states.detach().cpu()
        observations = observations.detach().cpu()
        actions = actions.detach().cpu()
        
        assert len(observations.batch_size) == 1
        batch_size = observations.batch_size[0]
        
        unit_timestep_action_sequences, horizons = actions.to_unit_timesteps_batched()
        action_trajectories = UnicycleActionTrajectory.from_trajectory_tensor(
            unit_timestep_action_sequences.actions
        )
        sim_results = self.env.open_loop_sim(
            initial_state=states,
            initial_observation=observations,
            action_trajectory=action_trajectories,
        )

        last_obs: UnicycleObservation = sim_results.observations[
            torch.arange(batch_size), horizons
        ]
        instruction = last_obs.instruction

        final_position_dist = (last_obs.position - instruction.target).norm(dim=-1)
        reward_final_position = -(
            self.params.final_position_weight * final_position_dist
        )
        
        reward_final_velocity = -(
            self.params.final_velocity_weight * last_obs.velocity.norm(dim=-1)
        )
        
        reward_total_time = -(
            self.params.total_time_weight * horizons * self.env.params.dt
        )

        reward_pointing_north = self.params.pointing_north_weight * (
            last_obs.heading[:, 1] - 1
        )

        all_positions: Float[Tensor, 'b h 2'] = sim_results.states.position
        mask = (
            torch.arange(0, all_positions.shape[1]).repeat(all_positions.shape[0],1) <
            horizons.unsqueeze(1)
        ).float() 
        all_positions = all_positions * rearrange(mask, 'b h -> b h 1')
        max_pos_component = reduce(all_positions.abs(), 'b h p -> b', 'max')
        reward_wall_contact = -(
            self.params.wall_contact_weight * (max_pos_component > 0.999).float()
        )
        
        angular_velocities = action_trajectories.actions.angular_velocity
        accelerations = action_trajectories.actions.acceleration
        # We don't have to consider horizons because padding is zero
        angular_velocities_max = reduce(angular_velocities.abs(), 'b h 1 -> b', 'max')
        accelerations_max = reduce(accelerations.abs(), 'b h 1 -> b', 'max')
        control_magnitude = (
            angular_velocities_max / self.env.params.max_angular_velocity +
            accelerations_max / self.env.params.max_acceleration
        )
        reward_control_magnitude = -(
            self.params.control_magnitude_weight * control_magnitude
        )
        
        def make_rewards(data: torch.Tensor) -> Reward:
            if data.ndim == 1:
                data = data.unsqueeze(-1)
            return Reward(rewards=data, batch_size=batch_size)
        
        rewards_final_position = make_rewards(reward_final_position).to(device)
        rewards_final_velocity = make_rewards(reward_final_velocity).to(device)
        rewards_total_time = make_rewards(reward_total_time).to(device)
        rewards_pointing_north = make_rewards(reward_pointing_north).to(device)
        rewards_wall_contact = make_rewards(reward_wall_contact).to(device)
        rewards_control_magnitude = make_rewards(reward_control_magnitude).to(device)
        
        return (
            (
                rewards_final_position +
                rewards_final_velocity +
                rewards_total_time +
                rewards_pointing_north +
                rewards_wall_contact +
                rewards_control_magnitude
            ),
            {
                'final_position': rewards_final_position,
                'final_velocity': rewards_final_velocity,
                'total_time': rewards_total_time,
                'pointing_north': rewards_pointing_north,
                'wall_contact': rewards_wall_contact,
                'control_magnitude': rewards_control_magnitude,
            }
        )
    
    