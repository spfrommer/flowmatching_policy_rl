# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Optional, Union

import torch

from nn.ema import EMA
from nn.unet import UNetModel

MODEL_CONFIGS = {
    "unicycle": {
        "in_channels": 2 + 1, # 2 for ang vel and accel, 1 for total time
        "model_channels": 64,
        "out_channels": 2 + 1,
        "num_res_blocks": 1,
        "attention_resolutions": [2],
        "dropout": 0.0,
        "channel_mult": [2, 2],
        "dims": 1,
        "conditioning_dim": 7,
        "conditioning_on_sequence": False,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
}


def instantiate_model(
    architecture: str, use_ema: bool
) -> Union[UNetModel, EMA]:
    assert (
        architecture in MODEL_CONFIGS
    ), f"Model architecture {architecture} is missing its config."

    model = UNetModel(**MODEL_CONFIGS[architecture])

    return EMA(model=model) if use_ema else model