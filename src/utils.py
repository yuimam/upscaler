from contextlib import contextmanager, nullcontext
from functools import lru_cache

import torch

from models.upscaler import NoiseLevelAndTextConditionedUpscaler


@lru_cache()
def get_available_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return torch.device(device)


@contextmanager
def amp_autocast(use_autocast, device):
    device_type = device.type
    if use_autocast and device_type in ('cuda',):
        precision_scope = torch.autocast
    else:
        precision_scope = nullcontext
    with precision_scope(device_type):
        yield
