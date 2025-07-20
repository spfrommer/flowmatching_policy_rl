import pickle
import gc
import logging
from pathlib import Path
from typing import Any, Iterable

import torch
from torchmetrics import MeanMetric
import wandb
from core.specs import TimedActionTrajectory, Observation, Reward, State
from inference.policy import Policy
from envs import Environment
import matplotlib.pyplot as plt
from reward import RewardModel
from training.args import TrainArgs


logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 50

def eval_model(
    policy: Policy,
    reward_model: RewardModel,

    data_loader: Iterable[tuple[State, Observation, TimedActionTrajectory, Reward]],
    
    env: Environment,

    device: torch.device,
    epoch: int,

    log_subdir: str,
    args: TrainArgs,
) -> dict[str, float]:

    gc.collect()
    
    epoch_rewards = MeanMetric().to(device, non_blocking=True)
    epoch_reward_components = {
        component: MeanMetric().to(device, non_blocking=True)
        for component in reward_model.components
    }

    for data_step, (states, observations, actions, rewards) in enumerate(data_loader):
        observations = observations.to(device, non_blocking=True)
        
        synthetic_actions, _ = policy.infer(observations=observations, temperature=0.0)
        
        synthetic_rewards, reward_components = reward_model(
            states=states,
            observations=observations,
            actions=synthetic_actions,
        )
        
        epoch_rewards.update(synthetic_rewards.rewards)
        
        for component, reward in reward_components.items():
            epoch_reward_components[component].update(reward.rewards)

        if data_step == 0:
            log_media = log_snapshots(
                epoch=epoch,
                states=states,
                observations=observations.to(states.device),
                original_actions=actions,
                synthetic_actions=synthetic_actions,
                env=env,
                log_subdir=log_subdir,
                args=args,
            )

        if args.test_run:
            break
            
    mean_reward = epoch_rewards.compute().detach().cpu().item()
    
    log_data = {
        'reward': mean_reward,
        **{
            f'{component}_reward': metric.compute().detach().cpu().item()
            for component, metric in epoch_reward_components.items()
        },
        **log_media,
    }
        
    return log_data


def log_snapshots(
    epoch: int, 
    states: State,
    observations: Observation,
    original_actions: TimedActionTrajectory,
    synthetic_actions: TimedActionTrajectory,
    env: Environment,
    log_subdir: str,
    args: TrainArgs,
) -> dict[str, Any]:
    
    output_dir = Path(args.output_dir)
    (output_dir / 'snapshots' / log_subdir).mkdir(parents=True, exist_ok=True)

    results, alternative_actions = [], []
    for i in range(10):
        initial_state: State = states[i]
        initial_observation: Observation = observations[i]
        synthetic_action_traj = env.ACTION_TRAJECTORY_TYPE.from_trajectory_tensor(
            synthetic_actions[i].to_unit_timesteps_not_batched().actions
        )
        
        sim_result = env.open_loop_sim(
            initial_state=initial_state,
            initial_observation=initial_observation,
            action_trajectory=synthetic_action_traj,
        )
        results.append(sim_result)
        
        original_action_traj = env.ACTION_TRAJECTORY_TYPE.from_trajectory_tensor(
            original_actions[i].to_unit_timesteps_not_batched().actions
        )
        
        alternative_actions.append(original_action_traj.cpu())
        
    renderer = env.RENDERER_TYPE(env.params)
    fig = renderer.render_batch(results[:5], alternative_actions[:5])
    fig.savefig(output_dir / 'snapshots' / log_subdir / f'{epoch}.png')

    # For some reason this causes dataloaders to crash with multiple workers!
    # pickle_path = output_dir / 'snapshots' / log_subdir / f'{epoch}_results.pkl'
    # with open(pickle_path, 'wb') as f:
    #     pickle.dump({
    #         'results': results,
    #         'original_actions': alternative_actions,
    #     }, f)
    
    log_media = {
        'snapshot': wandb.Image(fig),
    }
    plt.close(fig)
    
    return log_media