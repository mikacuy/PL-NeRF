from __future__ import annotations

from dataclasses import dataclass
from typing import Type

import torch
from nerfstudio.engine.optimizers import OptimizerConfig

@dataclass
class AdamWOptimizerConfig(OptimizerConfig):
    """Basic optimizer config with Adam"""

    _target: Type = torch.optim.AdamW
    weight_decay: float = 0
    """The weight decay to use."""
