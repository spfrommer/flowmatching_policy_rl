import gc
import logging
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor
from jaxtyping import Float
import wandb
from core.specs import TimedActionTrajectory, Observation, Reward, State
from inference.policy import Policy
from training.args import TrainArgs
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 50


def collect_samples(
    data_loader: Iterable[tuple[State, Observation, TimedActionTrajectory, Reward]],
    policy: Policy,
    max_samples: int,

    device: torch.device,
    epoch: int,

    args: TrainArgs,
) -> tuple[str, dict[str, Any]]:

    gc.collect()
    
    all_states, all_observations, all_synthetic_actions = [], [], []
    
    log_media = {}

    for data_step, (states, observations, _, _) in enumerate(data_loader):
        observations = observations.to(device, non_blocking=True)
            
        actions, info = policy.infer(
            observations=observations,
            temperature=args.collect_explore_amplitude,
        )
                
        
        if data_step == 0:
            log_media = plot_policy_info(info)
            
        observations.strip_for_saving()
        
        all_states.append(states.cpu())
        all_observations.append(observations.cpu())
        all_synthetic_actions.append(actions.cpu())
        
        if args.test_run:
            break
        
    assert sum(len(s) for s in all_states) >= max_samples

    samples_path = save_trajectory_samples(
        epoch=epoch,
        states=torch.cat(all_states, dim=0)[:max_samples],
        observations=torch.cat(all_observations, dim=0)[:max_samples],
        synthetic_actions=torch.cat(all_synthetic_actions, dim=0)[:max_samples],
        args=args,
    )

    logger.info(f"Epoch {epoch} COLLECT: computed {len(all_states)} batches")
            
    return samples_path, log_media 


def save_trajectory_samples(
    epoch: int, 
    states: State,
    observations: Observation,
    synthetic_actions: TimedActionTrajectory,
    args: TrainArgs,
) -> Path:
    
    online_chunk_dir = Path(args.data_dir) / 'online_trajectories_with_reward'
    online_chunk_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = online_chunk_dir / f'{epoch}.pt'
    save_data = {
        'states': states,
        'observations': observations,
        'action_trajectories': synthetic_actions,
    }
    torch.save(save_data, save_path)
    
    return save_path


def plot_policy_info(info: dict[str, Any]) -> dict[str, Any]:
    if 'synthetic_samples' not in info or 'synthetic_samples_similar' not in info:
        return {}
    
    samples: Float[Tensor, 'b c h'] = info['synthetic_samples']
    samples_similar: Float[Tensor, 'b s c h'] = info['synthetic_samples_similar']
    
    channel_n = samples.shape[1]
    samples_n = min(samples.shape[0], 4)
    
    fig, axes = plt.subplots(
        samples_n, channel_n, figsize=(5 * channel_n, 5 * samples_n)
    )
    
    for sample_idx in range(samples_n):
        for channel_idx in range(channel_n):
            ax = axes[sample_idx, channel_idx]
            
            ax.plot(
                samples[sample_idx, channel_idx].numpy(), 
                linestyle='--', 
                linewidth=2, 
            )
            
            similar_samples = samples_similar[sample_idx, :, channel_idx, :]
            for i in range(similar_samples.shape[0]):
                ax.plot(
                    similar_samples[i].numpy(),
                    linestyle='-',
                    linewidth=1,
                    alpha=0.3,
                    color=ax.lines[-1].get_color(),
                )
                
            ax.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    
    log_media = {'samples_with_variants': wandb.Image(fig)}
    plt.close(fig)
    
    return log_media
