import gc
import logging
from pathlib import Path
from typing import Any, Iterable
import numpy as np
from envs.unicycle.specs import UnicycleObservation

import torch
from torchmetrics import MeanMetric
import wandb
from core.specs import TimedActionTrajectory, Observation, Reward, State
from envs.unicycle.specs import UnicycleObservation
import matplotlib.pyplot as plt
from training.args import TrainArgs
from training.processor import ActionTrajectoryNormalizer, ObservationProcessor
from training.q_func import QFunction
from training.value_func import ValueFunction


logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 50


@torch.no_grad()
def qv_eval(
    v_function: ValueFunction | None,
    q_function: QFunction | None,
    
    observation_processor: ObservationProcessor,
    action_normalizer: ActionTrajectoryNormalizer,

    data_loader: Iterable[tuple[State, Observation, TimedActionTrajectory, Reward]],

    device: torch.device,
    epoch: int,

    args: TrainArgs,
) -> dict[str, float]:

    gc.collect()

    epoch_v_function_loss = MeanMetric().to(device, non_blocking=True)
    epoch_q_function_loss = MeanMetric().to(device, non_blocking=True)
    
    actual_rewards = []

    if v_function is not None:
        v_function.eval()
        v_function_predictions = []
        
    if q_function is not None:
        q_function.eval()
        q_function_predictions = []

    for data_step, (states, observations, actions, rewards) in enumerate(data_loader):
        observations = observations.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        rewards = rewards.to(device, non_blocking=True)

        observations_processed = observation_processor.process(observations)
        actions_normalized = action_normalizer.normalize(actions)
        
        actual_rewards.append(rewards.rewards.squeeze(-1).detach().cpu())

        if v_function is not None:
            v_function_preds = v_function(observations_processed)
            assert v_function_preds.shape == rewards.rewards.shape
            v_error = rewards.rewards - v_function_preds
            epoch_v_function_loss.update(v_error.pow(2).mean())

            v_function_predictions.append(v_function_preds.squeeze(-1).detach().cpu())

        if q_function is not None:
            q_function_preds = q_function(
                observations_processed,
                actions_normalized,
            )
            assert q_function_preds.shape == rewards.rewards.shape
            q_error = q_function_preds - rewards.rewards
            epoch_q_function_loss.update(q_error.pow(2).mean())

            q_function_predictions.append(q_function_preds.squeeze(-1).detach().cpu())

        if args.test_run:
            break


    log_data = {}

    all_actual_rewards = torch.cat(actual_rewards)

    if v_function is not None:
        log_data['v_function_loss'] = epoch_v_function_loss.compute()

        all_v_function_predictions = torch.cat(v_function_predictions)
        
        log_data.update(log_rewards_comparison(
            epoch=epoch,
            expected_rewards=all_v_function_predictions,
            actual_rewards=all_actual_rewards,
            prefix='v_function',
            args=args,
        ))
        
        log_data.update(log_value_function_sweep(
            epoch=epoch,
            v_function=v_function,
            observation_processor=observation_processor,
            args=args,
        ))
        
    if q_function is not None:
        log_data['q_function_loss'] = epoch_q_function_loss.compute()

        all_q_function_predictions = torch.cat(q_function_predictions)
        
        log_data.update(log_rewards_comparison(
            epoch=epoch,
            expected_rewards=all_q_function_predictions,
            actual_rewards=all_actual_rewards,
            prefix='q_function',
            args=args,
        ))
        
        
    return log_data


def log_rewards_comparison(
    epoch: int,
    expected_rewards: torch.Tensor,
    actual_rewards: torch.Tensor,
    prefix: str,
    args: TrainArgs,
) -> dict[str, Any]:

    output_dir = Path(args.output_dir)
    (output_dir / 'rewards').mkdir(parents=True, exist_ok=True)
    
    expected_rewards = expected_rewards.numpy()
    actual_rewards = actual_rewards.numpy()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    sample_n = min(1000, len(expected_rewards))
    indices = np.random.choice(len(expected_rewards), sample_n, replace=False)
    expected_rewards_sample = expected_rewards[indices]
    actual_rewards_sample = actual_rewards[indices]
    
    ax.scatter(
        actual_rewards_sample,
        expected_rewards_sample,
        color='blue',
        alpha=0.6
    )
    
    corr = np.corrcoef(actual_rewards_sample, expected_rewards_sample)[0, 1]
    
    min_val = min(expected_rewards_sample.min(), actual_rewards_sample.min())
    max_val = max(expected_rewards_sample.max(), actual_rewards_sample.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    
    ax.set_xlabel('Actual Reward')
    ax.set_ylabel('Expected Reward')
    ax.set_title(f'Expected vs Actual Rewards (Correlation: {corr:.4f})')
    ax.grid(True, alpha=0.3)
    
    ax.set_aspect('equal')
    
    fig_path = output_dir / 'rewards' / f'{prefix}_rewards_comparison_{epoch}.png'
    fig.savefig(fig_path)
    
    log_media = { f'{prefix}_rewards_comparison': wandb.Image(fig) }
    plt.close(fig)
    return log_media


def log_value_function_sweep(
    epoch: int,
    v_function: ValueFunction,
    observation_processor: ObservationProcessor,
    args: TrainArgs,
) -> dict[str, Any]:
    
    if args.env != 'unicycle':
        return {}
    
    output_dir = Path(args.output_dir)
    (output_dir / 'value_function').mkdir(parents=True, exist_ok=True)
    
    # Parameters for the sweep
    grid_size = 30
    x_range = np.linspace(-1.0, 1.0, grid_size)
    y_range = np.linspace(-1.0, 1.0, grid_size)
    
    # Create a meshgrid
    X, Y = np.meshgrid(x_range, y_range)
    positions = np.stack([X.flatten(), Y.flatten()], axis=1)
    
    # Fixed parameters
    target = torch.tensor([0.0, 0.0])  # Target at origin
    heading = torch.tensor([0.0, 1.0])  # Heading straight up
    velocity = torch.tensor([1.0])      # Velocity of 1.0
    
    # Create batch of observations
    batch_size = positions.shape[0]
    
    # Convert numpy positions to torch tensors
    positions_tensor = torch.tensor(positions, dtype=torch.float32)
    
    # Create batched parameters
    targets_batched = target.unsqueeze(0).expand(batch_size, -1)
    headings_batched = heading.unsqueeze(0).expand(batch_size, -1)
    velocities_batched = velocity.unsqueeze(0).expand(batch_size, -1)
    
    # Create UnicycleObservation
    observations = UnicycleObservation(
        position=positions_tensor,
        heading=headings_batched,
        velocity=velocities_batched,
        target=targets_batched,
        batch_size=(batch_size,),
    )
    
    device = v_function.mlp[0].weight.device
    observations = observations.to(device)
    
    # Process observations
    with torch.no_grad():
        observations_processed = observation_processor.process(observations)
        # Get value function predictions
        value_predictions = v_function(observations_processed).squeeze(-1)
    
    # Reshape predictions to grid
    value_grid = value_predictions.reshape(grid_size, grid_size).cpu().numpy()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    cmap = plt.cm.viridis
    im = ax.imshow(
        value_grid, 
        extent=[-1.0, 1.0, -1.0, 1.0],
        origin='lower',
        cmap=cmap,
        interpolation='bilinear',
        vmin=-1.0,
        vmax=0.0
    )
    
    # Mark the target position
    ax.plot(0, 0, 'ro', markersize=10)
    
    # Add colorbar
    fig.colorbar(im, ax=ax, label='Expected Reward')
    
    # Set title and labels
    ax.set_title(f'Value Function Prediction (Epoch {epoch})')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    
    # Set axis limits
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save figure
    fig_path = output_dir / 'value_function' / f'value_sweep_{epoch}.png'
    fig.savefig(fig_path)
    
    log_media = {
        'value_function_sweep': wandb.Image(fig)
    }
    
    plt.close(fig)
    
    return log_media
            
    