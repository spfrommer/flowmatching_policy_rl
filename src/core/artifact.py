import math
import os
from pathlib import Path
import shutil
from typing import Any
import wandb
from wandb.sdk.wandb_run import Run


def _parse_dict_for_wandb(d: dict[str, Any]) -> dict[str, Any]:
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = _parse_dict_for_wandb(value)
        elif value is None:
            d[key] = 'None'
        elif value == math.inf:
            d[key] = 'inf'
    return d

def upload_directory_to_wandb(
    directory: Path,
    artifact_kwargs: dict[str, Any],
    run: Run
) -> None:

    artifact = wandb.Artifact(**_parse_dict_for_wandb(artifact_kwargs))
    artifact.add_dir(str(directory))
    run.log_artifact(artifact)

def download_directory_from_wandb(
    directory: Path,
    artifact_name: str,
    run: Run
) -> None:

    artifact = run.use_artifact(artifact_name)
    artifact.download(root=str(directory))



def upload_directory_to_wandb_compressed(
    directory: Path,
    artifact_kwargs: dict[str, Any],
    run: Run
) -> None:

    parent_dir = directory.parent
    directory_name = directory.name
    
    current_dir = os.getcwd()
    os.chdir(parent_dir)
    os.system(f'tar -cf {directory_name}.tar {directory_name}')
    os.system(f'pv {directory_name}.tar | pbzip2 >> {directory_name}.tar.bz2')

    artifact = wandb.Artifact(**_parse_dict_for_wandb(artifact_kwargs))
    artifact.add_file(f'{directory_name}.tar.bz2')
    print(f'Uploading {artifact.name} to W&B')
    run.log_artifact(artifact)
    
    os.remove(f'{directory_name}.tar')
    os.remove(f'{directory_name}.tar.bz2')
    os.chdir(current_dir)


def download_directory_from_wandb_compressed(
    root_dir: Path,
    artifact_name: str,
    run: Run
) -> None:
    current_dir = os.getcwd()
    root_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(root_dir)
    
    if os.path.exists('/tmp/wandb'):
        shutil.rmtree('/tmp/wandb')
        
    artifact = run.use_artifact(f'flowmatchingrl/{artifact_name}:latest')
    artifact.download('./artifact')
    
    compressed_files = list(Path('./artifact').glob('*.tar.bz2'))
    assert len(compressed_files) == 1, 'Expected 1 .tar.bz2 file in the artifact dir'
    compressed_name = compressed_files[0].name
    directory_name = compressed_name.replace('.tar.bz2', '')
    
    shutil.copy(f'./artifact/{compressed_name}', f'./{compressed_name}')
    shutil.rmtree('./artifact')

    if os.path.exists(f'./{directory_name}'):
        shutil.rmtree(f'./{directory_name}')
    
    print('Extracting tar file...')
    os.system(f'tar xf {directory_name}.tar.bz2')

    os.remove(f'{directory_name}.tar.bz2')
    os.chdir(current_dir)
