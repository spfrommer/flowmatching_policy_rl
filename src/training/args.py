import json
import math
import re
from pathlib import Path
from tap import Tap
from typing import Any
from torchdiffeq._impl.odeint import SOLVERS


class TrainArgs(Tap):
    """Training configuration arguments."""
    
    # Training parameters
    batch_size: int = 512  # Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)
    accum_iter: int = 1  # Accumulate gradient iterations (for increasing the effective batch size under memory constraints)
    epochs: int | None = None
    
    # Optimizer parameters
    lr: float = 0.0005  # Learning rate (absolute lr)
    optimizer_betas: list[float] = [0.9, 0.95]
    decay_lr: bool = False  # Adds a linear decay to the lr during training
    use_ema: bool = True  # When evaluating, use the model Exponential Moving Average weights
    
    # Env and directory parameters
    env: str = 'unicycle' # Environment to use
    data_dir: str | None = None  # Directory to load data
    output_dir: str | None = None  # Path where to save results
    
    # ODE and misc parameters
    ode_method: str = 'euler'  # ODE solver used to generate samples
    ode_options: dict[str, Any]  # ODE solver options
    device: str = 'cuda'  # Device to use for training / testing
    seed: int = 0
    skip_data_download: bool = False # Whether to skip downloading data
    
    # Model parameters
    model_architecture: str = 'unicycle'

    # Reward parameters
    reward_recompute: bool = True # Whether to force a recomputation of rewards
    reward_model_params: dict[str, Any] # Parameters for the reward model
    
    # Evaluation parameters
    eval_freq: int = 5  # Frequency (in number of epochs) for running evaluation. -1 to never run evaluation
    
    # Data parameters
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_mem: bool = False  # Pin CPU memory in DataLoader for more efficient transfer to GPU
    
    # Collecting parameters
    collect_condition_generate: bool = True # Whether to generate new states / observations when collecting
    collect_stagnate_epochs: int = 500 # Number of epochs with no improvement to wait before collecting new data
    collect_stagnate_threshold: float = 0.01 # Threshold for considering the performance to be stagnating
    collect_iters: int = 10 # Number of collect iterations to run (runs every time val performance stagnates)
    collect_dataset_fraction: float = 0.2 # How many new samples to collect each time (as a fraction of the dataset)
    collect_explore_amplitude: float = 0.0 # Temperature for the perturbed distribution used when collecting new samples
    
    # Value & q function parameters
    qv_train_loops: int = 2 # Number of QV training loops per epoch
    qv_eval_freq: int = 5  # Frequency (in number of epochs) for running QV evaluation. -1 to never run evaluation
    
    # RWFM parameters
    rwfm_alpha: float = 0.0 
    rwfm_v_function: bool = False # Value function baseline for regular training
    rwfm_v_function_lr: float = 0.005 # Learning rate for the value function
    
    # GRPO parameters
    grpo_alpha: float = 1.0
    grpo_q_function: bool = False # Whether to use a reward surrogate for GRPO training (Q function)
    grpo_q_function_lr: float = 0.001 # Learning rate for the Q function
    grpo_q_function_layers: int = 2 # Number of layers in the TimesNet Q function
    grpo_q_function_dimension: int = 64 # Intermediate hidden dimension of the TimesNet Q function

    grpo_samples_per_group: int = 10
    grpo_explore_amplitude: float = 0.0

    grpo_start_epoch: int | None = None # Epoch to start GRPO training. If None, GRPO training will start after the value function baseline is trained.
    grpo_eval_freq: int = 1  # Frequency (in number of epochs) for running evaluation after starting GRPO training
    grpo_collect_stagnate_epochs: int = 25 # Number of epochs to wait before collecting new data for GRPO training
    grpo_batch_size: int = 128  # Batch size for GRPO training
    grpo_use_ddpo: bool = False # Whether to use DDPO

    # Conditioning parameters
    observation_noise: float = 0.0
    
    # Debug parameters
    test_run: bool = False  # Only run one batch of training and evaluation
    
    # Weights & Biases parameters
    wandb_version: str = '1'  # Version of the wandb run
    wandb_project_name: str = 'flowmatchingrl'  # Project name for wandb
    wandb_run_name: str | None = None  # Run name for wandb, if None a unique name will be generated
    
    # Distributed training parameters
    world_size: int = 1  # Number of distributed processes
    local_rank: int = -1
    dist_on_itp: bool = False
    dist_url: str = 'env://'  # URL used to set up distributed training
    
    # Derived attributes (not command-line arguments)
    distributed: bool = False
    gpu: int | None = None
    

    def configure(self) -> None:
        """Set the model-specific configurations."""
        # Configure the ODE method choices
        self.add_argument(
            "--epochs",
            default=math.inf,
            type=lambda x: math.inf if x.lower() == 'none' else int(x),
            help="Number of epochs to train for.",
        )
        self.add_argument(
            '--ode_method', 
            choices=list(SOLVERS.keys())
        )
        self.add_argument(
            "--ode_options",
            default='{"step_size": 0.25}',
            type=json_loads,
            help="ODE solver options. Eg. the midpoint solver requires step-size, dopri5 has no options to set.",
        )
        self.add_argument(
            "--reward_model_params",
            default='{}',
            type=json_loads,
            help="Reward model parameters.",
        )
        self.add_argument(
            "--grpo_start_epoch",
            default=math.inf,
            type=lambda x: math.inf if x.lower() == 'none' else int(x),
            help="Epoch to start GRPO training.",
        )
        
    def process_args(self) -> None:
        if self.grpo_start_epoch is None:
            self.grpo_start_epoch = math.inf
        
        if self.data_dir is None:
            self.data_dir = f'data/{self.env}'

        if self.output_dir is None:
            self.output_dir = f'output/{self.env}'
        elif '${' in self.output_dir:
            # Process any template variables in the output_dir
            self.output_dir = self._process_template_string(self.output_dir)
            
        if self.wandb_run_name is None:
            self.wandb_run_name = Path(self.output_dir).name
        
    def _process_template_string(self, template_str: str) -> str:
        def replace_var(match):
            var_name = match.group(1)
            if hasattr(self, var_name):
                value = str(getattr(self, var_name))
                if value.startswith('{') and value.endswith('}'):
                    # Handle case where it's a dictionary
                    value = '---' + value[1:-1] + '---'
                    value = value.replace("'", '')
                    value = value.replace(':', '')
                    value = value.replace(' ', '')
                    value = value.replace(',', '-')
                return value
            return match.group(0)  # Keep original if variable not found
        
        # Find and replace all ${VAR} patterns
        pattern = r'\${(\w+)}'
        return re.sub(pattern, replace_var, template_str)

        

def json_loads(json_string: str) -> Any:
    if json_string.startswith("'") and json_string.endswith("'"):
        json_string = json_string[1:-1]
    return json.loads(json_string)
        