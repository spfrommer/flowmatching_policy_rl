import os
from pathlib import Path
import shutil

from torch.utils.data import Dataset
import wandb

from core.specs import Observation, TimedActionTrajectory, Reward, State


class DemoDataset(Dataset):
    def __init__(self, chunks_path: Path):
        pass
    
    def __getitem__(self, idx: int) -> tuple[
        State, Observation, TimedActionTrajectory, Reward
    ]:
        pass
    
    def mean_reward(self) -> float:
        pass

    @staticmethod
    def collate_fn(
        batch: list[tuple[State, Observation, TimedActionTrajectory, Reward]],
    ) -> tuple[State, Observation, TimedActionTrajectory, Reward]:
        pass