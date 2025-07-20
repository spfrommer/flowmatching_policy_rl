import copy
from matplotlib.figure import Figure
import torch
from core.specs import Action, ActionTrajectory, Observation, State

from dataclasses import dataclass
from tensordict import TensorClass




class SimResult(TensorClass):
    states: State
    observations: Observation
    action_trajectories: ActionTrajectory
    
    def __post_init__(self):
        assert self.states.batch_size[-1] == self.observations.batch_size[-1]
        assert self.states.batch_size[-1] == self.action_trajectories.batch_size[-1] + 1
        
    def truncate_to_horizon(self, horizon: int) -> 'SimResult':
        """Truncates the actions to the horizon, states and observations will be +1."""
        return SimResult(
            states=self.states[..., :horizon + 1],
            observations=self.observations[..., :horizon + 1],
            action_trajectories=self.action_trajectories[..., :horizon],
            batch_size=self.batch_size,
        )


class Environment:
    STATE_TYPE: type[State]
    OBSERVATION_TYPE: type[Observation]
    ACTION_TYPE: type[Action]
    ACTION_TRAJECTORY_TYPE: type[ActionTrajectory]
    SIM_RESULT_TYPE: type[SimResult]
    
    def random_initial_state(self) -> State:
        pass
    
    def reset(self, initial_state: State) -> None:
        pass

    def step(self, action: Action) -> None:
        pass
    
    def get_state(self) -> State:
        pass
    
    def get_observation(self) -> Observation:
        pass


    def open_loop_sim(
        self,
        initial_state: State,
        initial_observation: Observation,
        action_trajectory: ActionTrajectory,
    ) -> SimResult:
        
        initial_state = initial_state.detach().cpu()
        initial_observation = initial_observation.detach().cpu()
        action_trajectory = action_trajectory.detach().cpu()

        assert action_trajectory.batch_size[:-1] == initial_state.batch_size
        assert action_trajectory.actions.batch_dims >= 1
        
        self.reset(initial_state)

        states, observations = [self.get_state()], [self.get_observation()]
        
        # Really want to do movedim(-1, 0) but can't do that on a TensorClass
        if action_trajectory.actions.batch_dims == 1:
            actions = action_trajectory.actions
        elif action_trajectory.actions.batch_dims == 2:
            actions = action_trajectory.actions.transpose(-2, -1)
        else:
            raise ValueError(f"Action trajectory batch dimensions must be 1 or 2")
        
        action: Action
        for action in actions:
            self.step(action)
            states.append(self.get_state())
            observations.append(self.get_observation())
        
        for obs in observations:
            obs.instruction = initial_observation.instruction

        result = SimResult(
            states=torch.stack(states, dim=-1),
            observations=torch.stack(observations, dim=-1),
            action_trajectories=action_trajectory,
            batch_size=action_trajectory.batch_size[:-1],
        )
            
        return result


class Renderer:
    def render(self, result: SimResult) -> Figure:
        pass

    def render_batch(self, results: list[SimResult]) -> Figure:
        pass
