import os
from contextlib import contextmanager, nullcontext
from functools import lru_cache

import click
import k_diffusion as K
import numpy as np
import requests
import torch
from tqdm import tqdm

from models.upscaler import NoiseLevelAndTextConditionedUpscaler


def download_model(dir, filename):
    URL = 'https://models.rivershavewings.workers.dev/laion_text_cond_latent_upscaler_2_1_00470000_slim.pth'

    filepath = os.path.join(dir, filename)
    if os.path.exists(filepath):
        click.echo('Model already downloaded')
        return

    with requests.get(URL, stream=True) as r:
        click.echo('Model download has started')
        total_size = int(r.headers.get('content-length', 0))
        with open(filepath, 'wb') as f:
            tqdm_params = {
                'desc': URL,
                'total': total_size,
                'miniters': 1,
                'unit': 'B',
                'unit_scale': True,
                'unit_divisor': 1024,
            }
            with tqdm(**tqdm_params) as progress:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    progress.update(len(chunk))
        click.echo('Model download was successful')


@lru_cache()
def make_upscaler_model(
    config_path, model_path, device, pooler_dim=768, train=False
):
    config = K.config.load_config(open(config_path))
    model = K.config.make_model(config)
    model = NoiseLevelAndTextConditionedUpscaler(
        model,
        sigma_data=config['model']['sigma_data'],
        embed_dim=config['model']['mapping_cond_dim'] - pooler_dim,
    )
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt['model_ema'])
    model = K.config.make_denoiser_wrapper(config)(model)
    if not train:
        model = model.eval().requires_grad_(False)
    return model.to(device)


def do_sample(
    model, noise, sampler, steps, eta, tol_scale, extra_args, device
):
    SIGMA_MIN, SIGMA_MAX = 0.029167532920837402, 14.614642143249512
    sigmas = (
        torch.linspace(np.log(SIGMA_MAX), np.log(SIGMA_MIN), steps + 1)
        .exp()
        .to(device)
    )
    if sampler == 'k_euler':
        return K.sampling.sample_euler(
            model, noise * SIGMA_MAX, sigmas, extra_args=extra_args
        )
    elif sampler == 'k_euler_ancestral':
        return K.sampling.sample_euler_ancestral(
            model, noise * SIGMA_MAX, sigmas, extra_args=extra_args, eta=eta
        )
    elif sampler == 'k_dpm_2_ancestral':
        return K.sampling.sample_dpm_2_ancestral(
            model, noise * SIGMA_MAX, sigmas, extra_args=extra_args, eta=eta
        )
    elif sampler == 'k_dpm_fast':
        return K.sampling.sample_dpm_fast(
            model,
            noise * SIGMA_MAX,
            SIGMA_MIN,
            SIGMA_MAX,
            steps,
            extra_args=extra_args,
            eta=eta,
        )
    else:
        sampler_opts = dict(
            s_noise=1.0,
            rtol=tol_scale * 0.05,
            atol=tol_scale / 127.5,
            pcoeff=0.2,
            icoeff=0.4,
            dcoeff=0,
        )
        return K.sampling.sample_dpm_adaptive(
            model,
            noise * SIGMA_MAX,
            SIGMA_MIN,
            SIGMA_MAX,
            extra_args=extra_args,
            eta=eta,
            **sampler_opts,
        )


@lru_cache()
def get_available_device():
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


@contextmanager
def amp_autocast(use_autocast, device):
    device_type = device.type
    if use_autocast and device_type in ('cuda',):
        precision_scope = torch.autocast
    else:
        precision_scope = nullcontext
    with precision_scope(device_type):
        yield
