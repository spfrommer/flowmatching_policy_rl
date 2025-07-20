import gc
import logging
import math
from typing import Iterable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from core.specs import TimedActionTrajectory, Observation, Reward, State
from torchmetrics.aggregation import MeanMetric

from training.args import TrainArgs
from training.processor import ActionTrajectoryNormalizer, ObservationProcessor
from training.value_func import ValueFunction
from training.q_func import QFunction

logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 100


def qv_train_one_epoch(
    v_function: ValueFunction | None,
    q_function: QFunction | None,

    observation_processor: ObservationProcessor,
    action_normalizer: ActionTrajectoryNormalizer,

    data_loader: Iterable[tuple[State, Observation, TimedActionTrajectory, Reward]],

    optimizer: Optimizer,
    lr_schedule: LRScheduler,

    device: torch.device,
    epoch: int,

    args: TrainArgs,
):
    gc.collect()

    epoch_v_function_loss = MeanMetric().to(device, non_blocking=True)
    epoch_q_function_loss = MeanMetric().to(device, non_blocking=True)
    
    if v_function is not None:
        v_function.train(True)
        
    if q_function is not None:
        q_function.train(True)

    for data_step, (_, observations, actions, rewards) in enumerate(data_loader):
        optimizer.zero_grad()
        if data_step > 0 and args.test_run:
            break

        observations = observations.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        rewards = rewards.to(device, non_blocking=True)

        observations_processed = observation_processor.process(observations)
        actions_normalized = action_normalizer.normalize(actions)
        
        noised_observations_processed = (
            observations_processed +
            args.observation_noise * torch.randn_like(observations_processed)
        )


        if q_function is not None:
            q_function_preds = q_function(
                noised_observations_processed,
                actions_normalized,
            )
            assert q_function_preds.shape == rewards.rewards.shape
            q_error = q_function_preds - rewards.rewards
            q_function_loss = q_error.pow(2).mean()
            q_function_loss.backward()
            
            if not math.isfinite(q_function_loss.item()):
                raise ValueError(f"Q loss is {q_function_loss.item()}")

            epoch_q_function_loss.update(q_function_loss.item())

        
        if v_function is not None:
            v_function_preds = v_function(noised_observations_processed)
            assert v_function_preds.shape == rewards.rewards.shape
            v_error = rewards.rewards - v_function_preds
            v_function_loss = v_error.pow(2).mean()
            v_function_loss.backward()

            if not math.isfinite(v_function_loss.item()):
                raise ValueError(f"V loss is {v_function_loss.item()}")

            epoch_v_function_loss.update(v_function_loss.item())
        

        optimizer.step()

        # lr = optimizer.param_groups[0]["lr"]
        if data_step % PRINT_FREQUENCY == 0:
            logger.info(
                f"Epoch {epoch} QV TRAIN [{data_step}/{len(data_loader)}]"
            )
        
    optimizer.zero_grad()
    lr_schedule.step()

    log_data = {}

    if v_function is not None:
        log_data['value_function_loss'] = epoch_v_function_loss.compute().detach().cpu().item()
    if q_function is not None:
        log_data['q_function_loss'] = epoch_q_function_loss.compute().detach().cpu().item()

    return log_data
