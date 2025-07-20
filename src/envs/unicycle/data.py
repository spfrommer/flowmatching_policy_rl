#!/usr/bin/env python3

import copy
import os
from pathlib import Path

import json
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from tap import Tap
from tensordict import TensorDict

import tqdm
import wandb

from core.artifact import upload_directory_to_wandb
from envs import Environment
from envs.unicycle.env import (
    UnicycleEnv, 
    UnicycleEnvParams,
)
from envs.unicycle.specs import (
    UnicycleAction,
    UnicycleActionTrajectory,
    UnicycleInstruction,
    UnicycleObservation,
    UnicycleState,
)
from envs.unicycle.expert import UnicycleEnvExpert, UnicycleEnvExpertParams
from envs.unicycle.render import UnicycleRenderer
from core.specs import Observation, Reward, State, TimedActionTrajectory
from training.data import DemoDataset
from core.utils import seed_everything



class UnicycleGenerateDataArgs(Tap):
    upload: bool = True  # Whether to upload data to W&B
    
    collect: bool = True  # Whether to collect expert trajectories
    trajectories: int = 30000  # Number of trajectories to collect
    
    data_dir: str | None = None  # Directory to save data
    
    expert_randomize: bool = True  # Whether to randomize expert parameters
    horizon_randomize: bool = True  # Whether to randomize horizon
    
    train_ratio: float = 0.95  # Proportion of data to use for training
    val_ratio: float = 0.01  # Proportion of data to use for validation
    
    max_horizon: int = 64
    
    def process_args(self) -> None:
        if self.data_dir is None:
            self.data_dir = f'data/unicycle'
        
        assert self.train_ratio + self.val_ratio <= 1.0


class UnicycleDemoDataset(DemoDataset):
    def __init__(self, chunks_path: Path):
        self.chunks_path = chunks_path
        self._load_data()
    
    def _load_data(self) -> None:
        data = torch.load(self.chunks_path, weights_only=False)
        
        if 'rewards' not in data:
            raise ValueError('Rewards not found in data')

        self.chunks_n = data['states'].shape[0]
        
        self.states: UnicycleState = data['states']
        self.observations: UnicycleObservation = data['observations']
        self.action_trajectories: TimedActionTrajectory = data['action_trajectories']
        self.rewards: Reward = data['rewards']
        self.reward_components: dict[str, Reward] = data['reward_components']
        
        assert self.rewards.rewards.ndim == 2
        assert self.rewards.rewards.shape[1] == 1

        d, h = self.action_trajectories[0].actions.shape
        # Add a dimension for the delta time 
        self.actions_shape = (d + 1, h)

        assert (
            self.states.batch_size
            == self.observations.batch_size
            == self.action_trajectories.batch_size
            == self.rewards.batch_size
        ), 'Batch sizes of all data components must match'

    def __len__(self) -> int:
        return self.states.batch_size[0]
    
    def __getitem__(self, idx: int) -> tuple[
        UnicycleState, UnicycleObservation, TimedActionTrajectory, Reward
    ]:
        return (
            self.states[idx],
            self.observations[idx],
            self.action_trajectories[idx],
            self.rewards[idx]
        )

    @staticmethod
    def collate_fn(
        batch: list[tuple[UnicycleState, UnicycleObservation, TimedActionTrajectory, Reward]],
    ) -> tuple[UnicycleState, UnicycleObservation, TimedActionTrajectory, Reward]:

        states = torch.stack([item[0] for item in batch])
        observations = torch.stack([item[1] for item in batch])
        actions = torch.stack([item[2] for item in batch])
        rewards = torch.stack([item[3] for item in batch])
        return states, observations, actions, rewards


def generate_expert_trajectories(args: UnicycleGenerateDataArgs) -> None:
    data_dir = Path(args.data_dir)
    expert_traj_dir = data_dir / 'expert_trajectories'
    expert_traj_dir.mkdir(parents=True, exist_ok=True)
    
    env = UnicycleEnv(params=UnicycleEnvParams()) 
    
    all_states: list[UnicycleState] = []
    all_observations: list[UnicycleObservation] = []
    all_actions: list[TimedActionTrajectory] = []
    
    for traj_idx in tqdm.tqdm(range(args.trajectories)):
        seed_everything(traj_idx)
        expert_params = UnicycleEnvExpertParams()
        if args.expert_randomize:
            expert_params.heading_kp = np.random.uniform(0.3, 3.0)
            expert_params.velocity_kp = np.random.uniform(0.3, 3.0)
            expert_params.goal_velocity = np.random.uniform(0.2, 1.0)
        expert = UnicycleEnvExpert(env, expert_params)
        
        if args.horizon_randomize:
            horizon = np.random.randint(1, args.max_horizon)
        else:
            horizon = args.max_horizon

        seed_everything(traj_idx)
        env.reset(env.random_initial_state())

        initial_state = env.get_state()
        instruction = env.random_instruction()
        initial_observation = env.get_observation()
        initial_observation.instruction = instruction
        
        observation = initial_observation
        actions = []
        
        for _ in range(horizon):
            action = expert.compute_action(observation)
            actions.append(action)
            env.step(action)
            observation = env.get_observation()
            observation.instruction = instruction

        total_time = torch.tensor([len(actions)]).float()
        action_trajectory = UnicycleActionTrajectory(actions=torch.stack(actions))
        timed_action_trajectory = TimedActionTrajectory(
            actions=action_trajectory.as_trajectory_tensor(),
            total_time_steps=total_time,
        )
        timed_action_trajectory = timed_action_trajectory.interpolate(args.max_horizon)

        all_states.append(initial_state)
        all_observations.append(initial_observation)
        all_actions.append(timed_action_trajectory)
        
    generated = TensorDict(
        **{
            'states': torch.stack(all_states),
            'observations': torch.stack(all_observations),
            'action_trajectories': torch.stack(all_actions),
        },
        batch_size=(len(all_states),),
    ).to('cpu')
    
    data_n = len(generated)
    train_idx = int(data_n * args.train_ratio)
    val_idx = train_idx + int(data_n * args.val_ratio)
    
    torch.save(generated[:train_idx], expert_traj_dir / 'trajectories_train.pt')
    torch.save(generated[train_idx:val_idx], expert_traj_dir / 'trajectories_val.pt')
    torch.save(generated[val_idx:], expert_traj_dir / 'trajectories_test.pt')
    
    with open(Path(args.data_dir) / 'env_params.json', 'w') as f:
        json.dump(env.params.__dict__, f)
        
    args.save(Path(args.data_dir) / 'generate_data_args.json')

    (Path(args.data_dir) / 'reward_model_params.json').unlink(missing_ok=True)


def load_env(args: UnicycleGenerateDataArgs) -> UnicycleEnv:
    with open(Path(args.data_dir) / 'env_params.json', 'r') as f:
        env_params = UnicycleEnvParams(**json.load(f))
    return UnicycleEnv(params=env_params)


def render_expert_trajectories(args: UnicycleGenerateDataArgs) -> None:
    expert_traj_dir = Path(args.data_dir) / 'expert_trajectories'
    data = torch.load(expert_traj_dir / 'trajectories_train.pt', weights_only=False)

    env = load_env(args)
    renderer = UnicycleRenderer(env.params)

    results = []
    
    for idx in np.random.choice(len(data['states']), size=5, replace=False):
        initial_state: UnicycleState = data['states'][idx]
        initial_observation: UnicycleObservation = data['observations'][idx]
        action_traj: TimedActionTrajectory = data['action_trajectories'][idx]
        
        unit_timestep_action_sequence = action_traj.to_unit_timesteps_not_batched()
        action_trajectory = UnicycleActionTrajectory.from_trajectory_tensor(
            unit_timestep_action_sequence.actions
        )
        
        results.append(env.open_loop_sim(
            initial_state=initial_state,
            initial_observation=initial_observation,
            action_trajectory=action_trajectory,
        ))
    
    fig = renderer.render_batch(results)
    
    plot_path = expert_traj_dir / 'plot.png'
    fig.savefig(plot_path)
    plt.close(fig)
    
    print(f'Saved trajectory plot to {plot_path}')


def main() -> None:
    args = UnicycleGenerateDataArgs(explicit_bool=True).parse_args()
    
    data_dir_root = Path(args.data_dir).parent
    dataset_name = Path(args.data_dir).name
    
    if args.collect:
        generate_expert_trajectories(args)

    render_expert_trajectories(args)
    
    if args.upload:
        print(f'Uploading {dataset_name} to W&B')
        
        if os.path.exists('/tmp/wandb'):
            shutil.rmtree('/tmp/wandb')

        run = wandb.init(
            project='flowmatchingrl',
            job_type='dataset-upload',
            dir='/tmp/wandb',
        )

        upload_directory_to_wandb(
            data_dir_root / dataset_name,
            artifact_kwargs={
                'name': dataset_name,
                'type': 'dataset',
                'description': 'Unicycle expert trajectory dataset',
                'metadata': args.as_dict(),
            },
            run=run,
        )
        print('Done!')
        run.finish()


if __name__ == '__main__':
    main() 