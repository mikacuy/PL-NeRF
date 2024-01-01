from dataclasses import dataclass, field
from typing import Type

from torch.optim import Optimizer

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    # Backwards compatibility for PyTorch 1.x
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from nerfstudio.engine.schedulers import SchedulerConfig, Scheduler
import numpy as np

@dataclass
class LogDecaySchedulerConfig(SchedulerConfig):
    """Config for cosine decay schedule"""

    _target: Type = field(default_factory=lambda: LogDecayScheduler)
    """target class to instantiate"""
    lr_final: float = 1e-6
    """Final lr"""
    lr_delay_steps: int = 1
    """lr_delay_steps"""
    lr_delay_mult: int = 1
    """lr_delay_mult"""
    max_steps: int = 300000
    """The maximum number of steps."""


class LogDecayScheduler(Scheduler):
    config: LogDecaySchedulerConfig
    
    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        return LogDecayLR(optimizer, 
                          lr_init, 
                          self.config.lr_final, 
                          self.config.max_steps, 
                          self.config.lr_delay_steps, 
                          self.config.lr_delay_mult
                          )
        
class LogDecayLR(LRScheduler):
    def __init__(self, optimizer, lr_init, lr_final, max_steps, lr_delay_steps=0, lr_delay_mult=1):
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.max_steps = max_steps
        self.lr_delay_steps = lr_delay_steps
        self.lr_delay_mult = lr_delay_mult
        super(LogDecayLR, self).__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if self.lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(self.last_epoch / self.lr_delay_steps, 0, 1))
        else:
            delay_rate = 1.
        t = np.clip(step / self.max_steps, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        return [delay_rate * log_lerp]