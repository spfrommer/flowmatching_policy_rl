# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.

import gc
import itertools
import json
import logging
import math
import os
import re
import sys
import shutil
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import wandb
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, DistributedSampler, Dataset, ConcatDataset
from core.artifact import download_directory_from_wandb, upload_directory_to_wandb
from core.utils import DatasetTruncator
from envs.unicycle.data import UnicycleDemoDataset
from core.specs import TimedActionTrajectory, Observation, State
from envs import Environment
from envs.unicycle.env import UnicycleEnv, UnicycleEnvParams
from inference.policy import Policy
from reward import RewardModel
from reward.unicycle import UnicycleRewardModel, UnicycleRewardModelParams
import training.configs as training_configs
from training.processor import ActionTrajectoryNormalizer, ObservationProcessor, UnicycleActionTrajectoryNormalizer, UnicycleObservationProcessor

from training import distributed_mode
from training.data import DemoDataset
from training.args import TrainArgs
from training.eval_loop import eval_model
import training.load_and_save as load_and_save
from training.q_func import QFunction, UnicycleQFunctionTimesNet
from training.qv_eval_loop import qv_eval
from training.qv_train_loop import qv_train_one_epoch
from training.train_loop import train_one_epoch
from training.collect_loop import collect_samples
from training.value_func import UnicycleValueFunction, ValueFunction
from wrapper import VanillaWrapper

logger = logging.getLogger(__name__)



def main(args: TrainArgs) -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    distributed_mode.init_distributed_mode(args)


    logger.info(f'job dir: {os.path.dirname(os.path.realpath(__file__))}')
    logger.info(f'{args}'.replace(', ', ',\n'))
    
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)

    if distributed_mode.is_main_process():
        args_filepath = output_dir / 'args.json'
        logger.info(f'Saving args to {args_filepath}')
        with open(args_filepath, 'w') as f:
            json.dump(args.as_dict(), f, default=lambda o: '<not serializable>')
        
        run = wandb.init(
            project=args.wandb_project_name,
            name=args.wandb_run_name,
            config=args.as_dict(),
            dir='/tmp/wandb',
        )


    logger.info('Downloading Dataset Artifact')
    if not args.skip_data_download:
        download_directory_from_wandb(
            directory=data_dir,
            artifact_name=f'{data_dir.name}:latest',
            run=run,
        )

    logger.info('Intializing Device and Seed')
    device = torch.device(args.device)
    seed = args.seed + distributed_mode.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True


    logger.info('Intializing Environment')
    env = create_env(args)

    logger.info('Intializing Reward Model')
    reward_model = create_reward_model(env, args)
    reward_params_path = data_dir / 'reward_model_params.json'
    if not reward_params_path.exists():
        compute_rewards = True
    else:
        with open(reward_params_path, 'r') as f:
            compute_rewards = json.load(f) != args.reward_model_params
    
    expert_traj_with_reward_dir = data_dir / 'expert_trajectories_with_reward'
    if compute_rewards or args.reward_recompute:
        if expert_traj_with_reward_dir.exists():
            shutil.rmtree(expert_traj_with_reward_dir)
        
        shutil.copytree(
            data_dir / 'expert_trajectories',
            expert_traj_with_reward_dir,
        )

    expert_traj_train_path = expert_traj_with_reward_dir / 'trajectories_train.pt'
    expert_traj_val_path = expert_traj_with_reward_dir / 'trajectories_val.pt'
    expert_traj_test_path = expert_traj_with_reward_dir / 'trajectories_test.pt'
    if compute_rewards or args.reward_recompute:
        logger.info(f'Computing rewards...')
        add_rewards(reward_model, expert_traj_train_path)
        add_rewards(reward_model, expert_traj_val_path)
        add_rewards(reward_model, expert_traj_test_path)

    with open(reward_params_path, 'w') as f:
        json.dump(args.reward_model_params, f)
        

    logger.info('Intializing Value Function')
    if args.rwfm_v_function or args.test_run:
        v_function = create_v_function(args).to(device)
    else:
        v_function = None
        

    logger.info('Intializing Q Function')
    if args.grpo_q_function or args.test_run:
        q_function = create_q_function(args).to(device)
    else:
        q_function = None
    
    qv_enabled = args.rwfm_v_function or args.grpo_q_function


    logger.info('Intializing DataLoaders and Normalizers')
    online_traj_dir = data_dir / 'online_trajectories_with_reward'
    if online_traj_dir.exists():
        shutil.rmtree(online_traj_dir)
    online_traj_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_train = create_dataset(expert_traj_train_path, args, split='train')
    dataset_val = create_dataset(expert_traj_val_path, args, split='val')
    dataset_test = create_dataset(expert_traj_test_path, args, split='test')
    collate_fn = dataset_train.collate_fn
    observation_processor, action_normalizer = create_processors(args, env.params)

    data_loader_train, data_loader_val, data_loader_test = create_dataloaders(
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        dataset_test=dataset_test,
        collate_fn=collate_fn,
        args=args,
    )
    
    data_loader_train_batch_size_adjust = data_loader_train

    expert_log = {}
    for split, dataset in zip(
        ['train', 'val', 'test'], [dataset_train, dataset_val, dataset_test]
    ):
        for component_name, component_reward in dataset.reward_components.items():
            expert_log[f'expert/{split}_{component_name}_reward'] = (
                component_reward.rewards.mean().item()
            )
        expert_log[f'expert/{split}_reward'] = dataset.rewards.rewards.mean().item()

    logger.info(f'Expert statistics: {expert_log}')
    log_stats(expert_log)


    logger.info('Initializing Model and Optimizer')
    model = training_configs.instantiate_model(
        architecture=args.model_architecture,
        use_ema=args.use_ema,
    ).to(device)

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    optimizer = create_optimizer(
        model=model_without_ddp,
        processor=observation_processor,
        v_function=v_function,
        q_function=q_function,
        args=args,
    )
    lr_schedule = create_lr_scheduler(optimizer, args)
    

    logger.info('Initializing Policy')
    policy = Policy(
        ode_method=args.ode_method,
        ode_opts=args.ode_options,
        actions_shape=dataset_train.actions_shape,
        model_wrapper=VanillaWrapper(model=model),
        observation_processor=observation_processor,
        action_normalizer=action_normalizer,
    )

    # In case of EMA, model is a wrapper around the model
    logger.info(str(model.model if hasattr(model, 'model') else model))
    logger.info(f'Optimizer: {optimizer}')
    logger.info(f'Learning-Rate Schedule: {lr_schedule}')


    policy_reward_model_args = { 'policy': policy, 'reward_model': reward_model }
    qv_args = { 'v_function': v_function, 'q_function': q_function }
    processor_args = {
        'observation_processor': observation_processor,
        'action_normalizer': action_normalizer,
    }
    optimizer_args = { 'optimizer': optimizer, 'lr_schedule': lr_schedule }
    

    eval_freq = args.eval_freq
    batch_size = args.batch_size
    collect_stagnate_epochs = args.collect_stagnate_epochs
    collect_iter = 0

    for epoch in itertools.count() if (args.epochs == math.inf) else range(args.epochs):
        data_loader_train.sampler.set_epoch(epoch)

        if epoch == args.grpo_start_epoch:
            eval_freq = args.grpo_eval_freq
            batch_size = args.grpo_batch_size
            collect_stagnate_epochs = args.grpo_collect_stagnate_epochs

            data_loader_train_batch_size_adjust = recreate_train_dataloader(
                dataset=dataset_train,
                collate_fn=collate_fn,
                args=args,
                batch_size_override=batch_size,
            )
            
        log_data = {
            'misc/epoch': epoch,
            'misc/train_samples': len(dataset_train),
            'misc/collect_iter': collect_iter,
        }

        common_args = { 'device': device, 'epoch': epoch, 'args': args }
        

        # TRAINING

        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader_train_batch_size_adjust,
            **qv_args,
            **policy_reward_model_args,
            **processor_args,
            **optimizer_args,
            **common_args,
        )
        log_data.update({f'model_train/{k}': v for k, v in train_stats.items()})
        
        if (args.qv_train_loops > 0 and qv_enabled) or args.test_run:
            qv_train_loops = args.qv_train_loops if not args.test_run else 1
            for _ in range(qv_train_loops):
                qv_train_stats = qv_train_one_epoch(
                    data_loader=data_loader_train,
                    **qv_args,
                    **processor_args,
                    **optimizer_args,
                    **common_args,
                )

            log_data.update({f'qv_train/{k}': v for k, v in qv_train_stats.items()})
            

        # EVALUATION
        
        collect_epoch = False
        eval_epoch = eval_freq > 0 and (epoch) % eval_freq == 0
        if eval_epoch or args.test_run:
            data_loader_val.sampler.set_epoch(0)
            
            eval_val_stats = eval_model(
                data_loader=data_loader_val,
                env=env,
                **policy_reward_model_args,
                **common_args,
                log_subdir='val',
            )
            log_data.update({f'model_eval_val/{k}': v for k, v in eval_val_stats.items()})
            logger.info(f'Epoch {epoch} EVAL: reward val = {eval_val_stats["reward"]:.4f}')
            
            eval_train_samples = int(len(data_loader_val) * data_loader_val.batch_size)
            eval_train_stats = eval_model(
                data_loader=recreate_train_dataloader(
                    dataset=DatasetTruncator(dataset_train, eval_train_samples),
                    collate_fn=collate_fn,
                    args=args,
                ),
                env=env,
                **policy_reward_model_args,
                **common_args,
                log_subdir='train',
            )
            log_data.update({f'model_eval_train/{k}': v for k, v in eval_train_stats.items()})
            logger.info(f'Epoch {epoch} EVAL: reward train = {eval_train_stats["reward"]:.4f}')
            
            epochs_since_best = load_and_save.save_model(
                val_performance=eval_val_stats['reward'],
                args=args,
                epoch=epoch,
                collect_iter=collect_iter,
                model_without_dpp=model_without_ddp,
                **qv_args,
                **optimizer_args,
            )
            
            log_data['misc/epochs_since_best'] = epochs_since_best
            
            if epochs_since_best >= collect_stagnate_epochs:
                do_grpo = (
                    args.grpo_start_epoch is not None and 
                    math.isfinite(args.grpo_start_epoch)
                )
                if (
                    (not do_grpo) or
                    (do_grpo and epoch >= args.grpo_start_epoch + args.grpo_collect_stagnate_epochs)
                ):
                    collect_iter += 1
                    collect_epoch = True
                    
                    if collect_iter > args.collect_iters:
                        break
        
        qv_eval_epoch = args.qv_eval_freq > 0 and (epoch) % args.qv_eval_freq == 0
        if (qv_eval_epoch and qv_enabled) or args.test_run:
            qv_eval_test_stats = qv_eval(
                data_loader=data_loader_val,
                **qv_args,
                **processor_args,
                **common_args,
            )
            log_data.update({f'qv_eval_val/{k}': v for k, v in qv_eval_test_stats.items()})
            logger.info(f'Epoch {epoch} QV EVAL')

            
        # COLLECTION

        if args.test_run or collect_epoch:
            collect_n = int(len(dataset_train) * args.collect_dataset_fraction)
            collect_batches = math.ceil(collect_n / args.batch_size)
            if args.collect_condition_generate:
                data_loader = generate_collect_conditions(env, collect_batches)
            else:
                collect_samples = int(len(dataset_train) * args.collect_dataset_fraction)
                data_loader = recreate_train_dataloader(
                    dataset=DatasetTruncator(dataset_train, collect_samples),
                    collate_fn=collate_fn,
                    args=args,
                )

                
            samples_path, collect_log_data = collect_samples(
                data_loader=data_loader,
                policy=policy,
                max_samples=collect_n,
                **common_args,
            )
            log_data.update({f'collect/{k}': v for k, v in collect_log_data.items()})
            add_rewards(reward_model, samples_path)

            new_dataset = create_dataset(samples_path, args, split='train')
            dataset_train = ConcatDataset([dataset_train, new_dataset])

            del data_loader_train
            gc.collect()

            data_loader_train = recreate_train_dataloader(
                dataset=dataset_train,
                collate_fn=collate_fn,
                args=args,
            )
            
            data_loader_train_batch_size_adjust = recreate_train_dataloader(
                dataset=dataset_train,
                collate_fn=collate_fn,
                args=args,
                batch_size_override=batch_size,
            )

        log_stats(log_data)

        if args.test_run:
            break

    # TESTING
    
    logger.info('Testing')
    
    for collect_iter in range(args.collect_iters + 1):
        log_data = {}
        
        load_and_save.load_model(
            args=args,
            collect_iter=collect_iter,
            model_without_ddp=model_without_ddp,
            **qv_args,
        )
        
        qv_eval_test_stats = qv_eval(
            data_loader=data_loader_test,
            **qv_args,
            **processor_args,
            **common_args,
        )
        log_data.update({
            f'qv_eval_test/collect_iter_{collect_iter}/{k}': v
            for k, v in qv_eval_test_stats.items()
        })


        if args.distributed:
            data_loader_test.sampler.set_epoch(0)

        eval_test_stats = eval_model(
            data_loader=data_loader_test,
            env=env,
            **policy_reward_model_args,
            **common_args,
            log_subdir='test',
        )
        log_data.update({
            f'model_eval_test/collect_iter_{collect_iter}/{k}': v
            for k, v in eval_test_stats.items()
        })

        logger.info(f'TEST: Reward test = {eval_test_stats["reward"]:.4f}')

        log_stats(log_data)


    if distributed_mode.is_main_process():
        upload_directory_to_wandb(
            directory=Path(args.output_dir),
            artifact_kwargs={
                # Needed because there's a char max of 128 for artifact names
                'name': _abbreviate_run_name_dicts(Path(args.output_dir).name),
                'type': 'output',
                'description': 'Output directory',
                'metadata': args.as_dict(),
            },
            run=run,
        )

        wandb.finish()
        

def log_stats(log_data: dict):
    if distributed_mode.is_main_process():
        with open(Path(args.output_dir) / 'log.txt', mode='a') as f:
            log_stats_writeable = {
                k: v for k, v in log_data.items() if isinstance(v, (int, float))
            }
            f.write(json.dumps(log_stats_writeable) + '\n')
        
        wandb.log(log_data)


def create_processors(
    args: TrainArgs,
    env_params: UnicycleEnvParams,
) -> tuple[ObservationProcessor, ActionTrajectoryNormalizer]:

    device = torch.device(args.device)

    generate_data_args_path = Path(args.data_dir) / 'generate_data_args.json'
    with open(generate_data_args_path, 'r') as f:
        generate_data_args = json.load(f)

    if args.env == 'unicycle':
        observation_processor = UnicycleObservationProcessor(env_params=env_params)
        action_normalizer = UnicycleActionTrajectoryNormalizer(
            env_params=env_params,
            max_total_time_steps=generate_data_args['max_horizon'],
        )
    else:
        raise ValueError(f'Unknown environment: {args.env}')
    
    return observation_processor.to(device), action_normalizer.to(device)


def create_dataset(
    expert_traj_path: Path,
    args: TrainArgs,
    split: Literal['train', 'val', 'test'],
) -> DemoDataset:

    if args.env == 'unicycle':
        return UnicycleDemoDataset(expert_traj_path)
    else:
        raise ValueError(f'Unknown environment: {args.env}')


def create_dataloaders(
    dataset_train: Dataset,
    dataset_val: Dataset,
    dataset_test: Dataset,
    collate_fn: Callable,
    args: TrainArgs,
) -> tuple[DataLoader, DataLoader, DataLoader]:

    sampler_params, dataloader_params = _sampler_dataloader_params(
        collate_fn=collate_fn,
        args=args,
    )
    
    data_loader_train = DataLoader(
        dataset_train,
        sampler=DistributedSampler(dataset_train, shuffle=True, **sampler_params),
        drop_last=True,
        **dataloader_params,
    )
    
    data_loader_val = DataLoader(
        dataset_val,
        sampler=DistributedSampler(dataset_val, shuffle=False, **sampler_params),
        drop_last=False,
        **dataloader_params,
    )
    
    data_loader_test = DataLoader(
        dataset_test,
        sampler=DistributedSampler(dataset_test, shuffle=False, **sampler_params),
        drop_last=False,
        **dataloader_params,
    )
    
    return data_loader_train, data_loader_val, data_loader_test

def recreate_train_dataloader(
    dataset: Dataset,
    collate_fn: Callable,
    args: TrainArgs,
    batch_size_override: int | None = None,
) -> DataLoader:

    sampler_params, dataloader_params = _sampler_dataloader_params(
        collate_fn=collate_fn,
        args=args,
    )
    
    if batch_size_override is not None:
        dataloader_params['batch_size'] = batch_size_override
    
    return DataLoader(
        dataset,
        sampler=DistributedSampler(dataset, shuffle=True, **sampler_params),
        drop_last=True,
        **dataloader_params,
    )

def _sampler_dataloader_params(
    collate_fn: Callable,
    args: TrainArgs,
) -> tuple[dict, dict]:

    sampler_params = {
        'num_replicas': distributed_mode.get_world_size(),
        'rank': distributed_mode.get_rank(),
    }
    
    dataloader_params = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': args.pin_mem,
        'collate_fn': collate_fn,
        'prefetch_factor': args.prefetch_factor if args.num_workers > 0 else None,
    }

    return sampler_params, dataloader_params

def create_optimizer(
    model: nn.Module,
    processor: ObservationProcessor,
    v_function: nn.Module | None,
    q_function: nn.Module | None,
    args: TrainArgs,
) -> Optimizer:
    
    parameter_groups = [
        {'params': model.parameters(), 'lr': args.lr},
        {'params': processor.parameters(), 'lr': args.lr},
    ]
    if v_function is not None:
        parameter_groups.append({
            'params': v_function.parameters(),
            'lr': args.rwfm_v_function_lr,
        })
    if q_function is not None:
        parameter_groups.append({
            'params': q_function.parameters(),
            'lr': args.grpo_q_function_lr,
        })

    optimizer = torch.optim.AdamW(
        parameter_groups,
        betas=tuple(args.optimizer_betas),
    )
    return optimizer


def create_lr_scheduler(optimizer: Optimizer, args: TrainArgs) -> LRScheduler:
    if args.decay_lr:
        lr_schedule = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=args.epochs,
            start_factor=1.0,
            end_factor=1e-8 / args.lr,
        )
    else:
        lr_schedule = torch.optim.lr_scheduler.ConstantLR(
            optimizer, total_iters=args.epochs, factor=1.0
        )
    return lr_schedule


def create_env(args: TrainArgs) -> Environment:
    env_params_path = Path(args.data_dir) / 'env_params.json'
    with open(env_params_path, 'r') as f:
        env_params = json.load(f)

    if args.env == 'unicycle':
        return UnicycleEnv(params=UnicycleEnvParams(**env_params))
    else:
        raise ValueError(f'Unknown environment: {args.env}')


def create_reward_model(env: Environment, args: TrainArgs) -> RewardModel:
    if args.env == 'unicycle':
        return UnicycleRewardModel(
            env=env,
            params=UnicycleRewardModelParams(**args.reward_model_params),
        )
    else:
        raise ValueError(f'Unknown environment: {args.env}')
    

def create_v_function(args: TrainArgs) -> ValueFunction:
    if args.env == 'unicycle':
        return UnicycleValueFunction()
    else:
        raise ValueError(f'Unknown environment: {args.env}')


def create_q_function(args: TrainArgs) -> QFunction:
    if args.env == 'unicycle':
        # return UnicycleQFunctionUNet(args)
        return UnicycleQFunctionTimesNet(args)
        # return UnicycleQFunctionLSTM(args)
    else:
        raise ValueError(f'Unknown environment: {args.env}')


def add_rewards(reward_model: RewardModel, file_path: Path):
    data = torch.load(file_path, weights_only=False)
    
    states: State = data['states']
    observations: Observation = data['observations']
    action_trajectories: TimedActionTrajectory = data['action_trajectories']

    rewards, reward_components = reward_model(states, observations, action_trajectories)

    data['rewards'] = rewards
    data['reward_components'] = reward_components

    torch.save(data, file_path)
    

def generate_collect_conditions(
    env: Environment,
    collect_batches: int,
) -> Iterable[tuple[State, Observation, None, None]]:
    
    assert isinstance(env, UnicycleEnv)

    batch_size = args.batch_size
    states_list, observations_list = [], []
    
    for _ in range(collect_batches):
        states = torch.stack([env.random_initial_state() for _ in range(batch_size)])
        observations = []
        for state in states:
            observation = env.get_observation(state)
            observation.instruction = env.random_instruction()
            observations.append(observation)
        observations = torch.stack(observations)
        
        states_list.append(states)
        observations_list.append(observations)
    
    data_loader = (
        (states, observations, None, None) 
        for states, observations in zip(states_list, observations_list)
    )
    
    return data_loader

def _abbreviate_run_name_dicts(run_name: str) -> str:
    match = re.search(r'---(.*?)---', run_name)
    if not match:
        return run_name
    reward_section = match.group(1)
    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', reward_section)
    reward_values = '-'.join(numbers)
    new_str = re.sub(r'---.*?---', f'{reward_values}', run_name)
    return new_str


if __name__ == '__main__':
    args = TrainArgs(description='Training configuration', explicit_bool=True)
    args.parse_args()
    
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main(args)
