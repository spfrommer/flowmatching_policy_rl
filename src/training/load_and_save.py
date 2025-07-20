# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import json
from pathlib import Path

import torch
from .distributed_mode import is_main_process
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch import nn

# Import TrainArgs type for type annotations only
from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from training.args import TrainArgs

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def save_model(
    val_performance: float,
    args: 'TrainArgs',
    epoch: int,
    collect_iter: int,
    model_without_dpp: nn.Module,
    v_function: nn.Module | None,
    q_function: nn.Module | None,
    optimizer: Optimizer,
    lr_schedule: LRScheduler,
) -> int:
    """ Return epochs since best model for this collect iteration. """

    checkpoint_dir = Path(args.output_dir) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = checkpoint_dir / f'best_metadata_collect_iter_{collect_iter}.json'
    model_path = checkpoint_dir / f'best_model_collect_iter_{collect_iter}.pth'

    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            best_metadata = json.load(f)
        
        delta_performance = val_performance - best_metadata['val_performance']
        if delta_performance > args.collect_stagnate_threshold:
            best_metadata['val_performance'] = val_performance
            best_metadata['epoch'] = epoch
        else:
            return epoch - best_metadata['epoch']
    else:
        best_metadata = { 'val_performance': val_performance, 'epoch': epoch }

    with open(metadata_path, 'w') as f:
        json.dump(best_metadata, f)

    to_save = {
        'model': model_without_dpp.state_dict(),
        'v_function': v_function.state_dict() if v_function else None,
        'q_function': q_function.state_dict() if q_function else None,
        'optimizer': optimizer.state_dict(),
        'lr_schedule': lr_schedule.state_dict(),
        'epoch': epoch,
        'args': args.as_dict(),
    }

    save_on_master(to_save, model_path)

    return 0


def load_model(
    args: 'TrainArgs',
    collect_iter: int,
    model_without_ddp: nn.Module,
    v_function: nn.Module | None,
    q_function: nn.Module | None,
):
    checkpoint_dir = Path(args.output_dir) / 'checkpoints'
    file_name = f'best_model_collect_iter_{collect_iter}.pth'
    checkpoint_path = checkpoint_dir / file_name

    checkpoint = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=False
    )
    model_without_ddp.load_state_dict(checkpoint['model'])

    if v_function is not None:
        v_function.load_state_dict(checkpoint['v_function'])
    else:
        assert v_function is None

    if q_function is not None:
        q_function.load_state_dict(checkpoint['q_function'])
    else:
        assert q_function is None