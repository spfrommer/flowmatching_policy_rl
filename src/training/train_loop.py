import gc
import logging
import math
from typing import Iterable

from einops import rearrange, reduce, repeat
import torch
from torch import Tensor
from jaxtyping import Float
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from core.specs import TimedActionTrajectory, Observation, Reward, State, TimedActionTrajectoryNormalized
from flow_matching.path import CondOTProbPath
from flow_matching.solver.sde_solver import SDESolver
from flow_matching.utils.model_wrapper import ModelWrapper
from inference.policy import Policy
from nn.ema import EMA
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric

from nn.unet import UNetModel
from reward import RewardModel
from training.args import TrainArgs
from training.processor import ActionTrajectoryNormalizer, ObservationProcessor
from training.value_func import ValueFunction
from training.q_func import QFunction

logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 100


def train_one_epoch(
    model: UNetModel | EMA,
    v_function: ValueFunction | None,
    q_function: QFunction | None,
    policy: Policy,
    reward_model: RewardModel,

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
    epoch_model_loss = MeanMetric().to(device, non_blocking=True)
    
    model.train(True)
    if v_function is not None:
        v_function.train(False)
    if q_function is not None:
        q_function.train(False)

    accum_iter = args.accum_iter

    for data_step, (states, observations, actions, rewards) in enumerate(data_loader):
        if data_step % accum_iter == 0:
            optimizer.zero_grad()
            if data_step > 0 and args.test_run:
                break

        states = states.to(device, non_blocking=True)
        observations = observations.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        rewards = rewards.to(device, non_blocking=True)
        
        if epoch < args.grpo_start_epoch:
            model_loss = regular_update(
                model=model,
                value_function=v_function,

                observations=observations,
                actions=actions,
                rewards=rewards,

                observation_processor=observation_processor,
                action_normalizer=action_normalizer,

                args=args,
            )
        else:
            model_loss = grpo_update(
                model=model,
                q_function=q_function,
                policy=policy,
                reward_model=reward_model,

                states=states,
                observations=observations,

                observation_processor=observation_processor,
                action_normalizer=action_normalizer,

                args=args,
            )

        epoch_model_loss.update(model_loss.item())

        if not math.isfinite(model_loss.item()):
            raise ValueError(f"Loss is {model_loss.item()}, stopping training")

        model_loss /= accum_iter

        apply_update = (data_step + 1) % accum_iter == 0
        model_loss.backward()
        if apply_update:
            optimizer.step()

        if apply_update and isinstance(model, EMA):
            model.update_ema()
        elif (
            apply_update
            and isinstance(model, DistributedDataParallel)
            and isinstance(model.module, EMA)
        ):
            model.module.update_ema()

        # lr = optimizer.param_groups[0]["lr"]
        if data_step % PRINT_FREQUENCY == 0:
            logger.info(
                f"Epoch {epoch} TRAIN [{data_step}/{len(data_loader)}]"
            )
        
    optimizer.zero_grad()
    lr_schedule.step()

    return {
        "model_loss": epoch_model_loss.compute().detach().cpu().item(),
    }


def regular_update(
    model: UNetModel | EMA,
    value_function: ValueFunction | None,

    observations: Observation,
    actions: TimedActionTrajectory,
    rewards: Reward,

    observation_processor: ObservationProcessor,
    action_normalizer: ActionTrajectoryNormalizer,
    
    args: TrainArgs,
) -> None:
    
    device = observations.device
    path = CondOTProbPath()
    
    observations_processed = observation_processor.process(observations)
    actions_normalized = action_normalizer.normalize(actions)

    batch_n = observations_processed.shape[0]
    conditioning = observations_processed.observations
    samples = actions_normalized.unify()

    noise = torch.randn_like(samples).to(device)
    t = torch.rand(batch_n).to(device)

    path_sample = path.sample(t=t, x_0=noise, x_1=samples)
    x_t = path_sample.x_t
    u_t = path_sample.dx_t
    
    noised_conditioning = (
        conditioning + args.observation_noise * torch.randn_like(conditioning)
    )
    pred_u_t = model(x_t, t, noised_conditioning)
    
    if value_function is not None:
        v_function_preds = value_function(observations_processed)
        assert rewards.rewards.shape == v_function_preds.shape
        delta_reward = rewards.rewards - v_function_preds
        weight = torch.exp(args.rwfm_alpha * delta_reward.detach())
    else:
        weight = torch.exp(args.rwfm_alpha * rewards.rewards)
        
    weight = weight.clamp(max=20.0)
    error = reduce(torch.pow(pred_u_t - u_t, 2), 'b c t -> b 1', 'mean')
    assert error.shape == weight.shape
    return (error * weight).mean()


def grpo_update(
    model: UNetModel | EMA,
    q_function: QFunction | None,
    policy: Policy,
    reward_model: RewardModel,

    states: State,
    observations: Observation,

    observation_processor: ObservationProcessor,
    action_normalizer: ActionTrajectoryNormalizer,

    args: TrainArgs,
) -> None:
    
    batch_n = observations.shape[0]
    device = observations.device
    path = CondOTProbPath()
    
    observations_processed = observation_processor.process(observations)
    
    conditioning = observations_processed.observations
    
    samples_per_group = args.grpo_samples_per_group

    states_repeated = states.repeat_interleave(samples_per_group, dim=0)
    observations_repeated = observations.repeat_interleave(samples_per_group, dim=0)
    conditioning_repeated = conditioning.repeat_interleave(samples_per_group, dim=0)
    
    with torch.no_grad():
        if args.grpo_use_ddpo:
            generated_actions_repeated, policy_extra = policy.stochastic_infer(
                observations=observations_repeated,
            )
        else:
            generated_actions_repeated, _ = policy.infer(
                observations=observations_repeated,
                temperature=args.grpo_explore_amplitude,
            )
            
    
    if q_function is None:
        generated_rewards, _ = reward_model(
            states_repeated,
            observations_repeated,
            generated_actions_repeated,
        )
        generated_rewards = generated_rewards.rewards
    else:
        with torch.no_grad():
            generated_rewards = q_function(
                observation_processor.process(observations_repeated),
                action_normalizer.normalize(generated_actions_repeated),
            )

    generated_rewards = generated_rewards.view(batch_n, samples_per_group)
    
    mean_reward_per_group = reduce(generated_rewards, 'b g -> b 1', 'mean')
    std_reward_per_group = generated_rewards.std(dim=-1, keepdim=True) + 1e-5
    advantages = (generated_rewards - mean_reward_per_group) / std_reward_per_group
    advantages = rearrange(advantages, 'b g -> (b g) 1')

    if args.grpo_use_ddpo:
        # States and actions are the same thing
        state_trajectories: Float[Tensor, 'b t c h'] = torch.stack(
            policy_extra['intermediates'], dim=1
        )
        
        diffusion_steps = state_trajectories.shape[1]
        state_previous = rearrange(state_trajectories[:, :-1], 'b t ... -> (b t) ...')
        state_next = rearrange(state_trajectories[:, 1:], 'b t ... -> (b t) ...')
        advantages = repeat(advantages, 'b 1 -> (b t)', t=diffusion_steps-1)
        conditioning = repeat(conditioning_repeated, 'b d -> (b t) d', t=diffusion_steps-1)
        
        diffusion_t = repeat(
            torch.linspace(0, 1, diffusion_steps)[:-1].to(device),
            't -> (b g t)',
            b=batch_n,
            g=samples_per_group
        )
        
        posterior_mean, posterior_var = policy.stochastic_posterior_gaussian(
            x_t_unified=state_previous,
            t=diffusion_t,
            observations=observations_repeated.repeat_interleave(diffusion_steps-1, dim=0),
        )

        log_prob = -0.5 * (
            torch.log(2 * math.pi * posterior_var) + 
            torch.pow(state_next - posterior_mean, 2) / posterior_var
        )
        log_prob = reduce(log_prob, 'b c h -> b', 'sum')

        assert log_prob.shape == advantages.shape
        policy_loss = -(log_prob * advantages).mean()
        
        return policy_loss
    else:
        samples_repeated = action_normalizer.normalize(generated_actions_repeated).unify()

        noise = torch.randn_like(samples_repeated).to(device)
        t = torch.rand(batch_n * samples_per_group).to(device)

        path_sample = path.sample(t=t, x_0=noise, x_1=samples_repeated)
        x_t = path_sample.x_t
        u_t = path_sample.dx_t
        
        noised_conditioning = (
            conditioning_repeated +
            args.observation_noise * torch.randn_like(conditioning_repeated)
        )
        pred_u_t = model(x_t, t, noised_conditioning)
        
        weight = torch.exp(args.grpo_alpha * advantages).clamp(max=20.0)
        error = reduce(torch.pow(pred_u_t - u_t, 2), 'b c t -> b 1', 'mean')
        assert error.shape == weight.shape
        return (error * weight).mean()