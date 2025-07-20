import torch
import numpy as np
from envs import Renderer
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrow

from envs.unicycle.specs import UnicycleAction, UnicycleActionTrajectory, UnicycleObservation, UnicycleState

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from envs.unicycle.env import UnicycleEnvParams, UnicycleSimResult


class UnicycleRenderer(Renderer):
    def __init__(
        self,
        env_params: 'UnicycleEnvParams',
    ):
        self.env_params = env_params
        
    def _render_sim_to_figure(
        self,
        ax: Axes,
        result: 'UnicycleSimResult',
        color: str = 'blue'
    ) -> None:
        """Render the open loop simulation to an existing matplotlib figure."""

        observations = result.observations

        total_points = observations.batch_size[0]
        select_points = 30
        if total_points <= select_points:
            indices = range(total_points)
        else:
            # Always include first and last point
            indices = [0] + [
                int(i * (total_points - 2) / (select_points - 1)) + 1
                for i in range(select_points - 1)
            ] + [total_points - 1]
        
        observation: UnicycleObservation
        for i, observation in enumerate(observations):
            if i not in indices:
                continue
            
            velocity_fraction = observation.velocity / self.env_params.max_velocity
            arrow_length = 0.05 + 0.1 * velocity_fraction.item()
            
            arrow = FancyArrow(
                observation.position[0].item(),
                observation.position[1].item(), 
                observation.heading[0].item() * arrow_length,
                observation.heading[1].item() * arrow_length,
                width=0.02, 
                head_width=0.04, 
                head_length=0.04, 
                color=color
            )
            ax.add_patch(arrow)
        
        target_x = observations.instruction.target[-1, 0]
        target_y = observations.instruction.target[-1, 1]
        ax.plot(target_x, target_y, '*', color=color, markersize=10)

    def render_batch(
        self,
        results: list['UnicycleSimResult'],
        alternative_actions: list[UnicycleActionTrajectory] | None = None,
    ) -> Figure:
        
        if alternative_actions is not None:
            assert len(alternative_actions) == len(results)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
        
        colors = [
            'blue', 'red', 'green', 'gray', 'purple',
            'orange', 'brown', 'pink', 'cyan', 'yellow',
        ]
        
        for i in range(len(results)):
            result = results[i]
            color = colors[i]
            
            self._render_sim_to_figure(ax1, result, color)
            
            t = np.arange(result.action_trajectories.actions.batch_size[0])
            ax2.plot(t, result.action_trajectories.actions.angular_velocity, color=color)
            ax4.plot(t, result.action_trajectories.actions.acceleration, color=color)
            states_t = np.arange(result.states.velocity.shape[0])
            ax3.plot(states_t, result.states.velocity, color=color)
            
            if alternative_actions is not None:
                actions: UnicycleAction = alternative_actions[i].actions
                correct_t = np.arange(actions.batch_size[0])
                angular_vel = actions.angular_velocity
                accel = actions.acceleration
                ax2.plot(correct_t, angular_vel, color=color, linestyle=':')
                ax4.plot(correct_t, accel, color=color, linestyle=':')
        

        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_xlim(-1.1, 1.1)
        ax1.set_ylim(-1.1, 1.1)
        ax1.set_aspect('equal')
        
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Angular Velocity')
        ax2.set_ylim(
            -self.env_params.max_angular_velocity - 0.1,
            self.env_params.max_angular_velocity + 0.1
        )
        
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Velocity')
        ax3.set_ylim(-0.1, self.env_params.max_velocity + 0.1)
        
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Acceleration')
        ax4.set_ylim(
            -self.env_params.max_acceleration - 0.1,
            self.env_params.max_acceleration + 0.1
        )
        
        fig.tight_layout()
        return fig


    def render(
        self,
        result: 'UnicycleSimResult',
        alternative_action: UnicycleActionTrajectory | None = None,
    ) -> Figure:
        
        return self.render_batch([result], [alternative_action])
