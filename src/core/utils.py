import random
import numpy as np
import torch
from torch.utils.data import Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


class DatasetTruncator(Dataset):
    def __init__(self, dataset: Dataset, max_length: int):
        self.dataset = dataset
        self.max_length = max_length

    def __len__(self):
        return self.max_length
    
    def __getitem__(self, index):
        return self.dataset[index]