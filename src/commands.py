import os
import time

import click
import torch
from diffusers import AutoencoderKL
from PIL import Image
from pytorch_lightning import seed_everything
from torchvision.transforms import functional as TF

from models.transformers import CLIPEmbedder, CLIPTokenizerTransform
from models.upscalers import CFGUpscaler
from utils import do_sample, download_model, make_upscaler_model


@click.group()
def _cli():
    pass


def get_cli():
    _cli.add_command(upscale)
    return _cli


@click.command()
@click.option(
    '--input-path',
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    '--output-path',
    required=True,
    type=click.Path(exists=False),
)
@click.option(
    '--batch-size',
    default=1,
    type=click.IntRange(1),
)
@click.option(
    '--guidance-scale',
    default=1,
    type=click.IntRange(0, 10),
)
@click.option(
    '--noise-aug-level',
    default=0.0,
    type=click.FloatRange(0.0, 0.6),
)
@click.option(
    '--noise-aug-type',
    default='gaussian',
    type=click.Choice(['fake', 'gaussian']),
)
@click.option(
    '--sampler',
    default='k_dpm_adaptive',
    type=click.Choice(
        [
            'k_euler',
            'k_euler_ancestral',
            'k_dpm_2_ancestral',
            'k_dpm_fast',
            'k_dpm_adaptive',
        ]
    ),
)
@click.option(
    '--sampler-steps',
    default=50,
    type=click.IntRange(1),
)
@click.option(
    '--sampler-tol-scale',
    default=0.25,
    type=click.FloatRange(0.0),
)
@click.option(
    '--sampler-eta',
    default=1.0,
    type=click.FloatRange(0.0),
)
def upscale(
    input_path,
    output_path,
    batch_size,
    guidance_scale,
    noise_aug_level,
    noise_aug_type,
    sampler,
    sampler_steps,
    sampler_tol_scale,
    sampler_eta,
):
    SD_Q = 0.18215
    PROMPT = (
        'the temple of fire by Ross Tran and Gerardo Dottori, oil on canvas'
    )
    UPSCALER_DIR = os.path.join(os.path.dirname(__file__), 'latent_upscaler')
    UPSCALER_FILENAME = 'latent_upscaler.pth'
    DEVICE = torch.device('cuda')

    download_model(UPSCALER_DIR, UPSCALER_FILENAME)
    seed_everything(int(time.time()))

    tokenizer, text_encoder = CLIPTokenizerTransform(), CLIPEmbedder()
    upscaler = CFGUpscaler(
        make_upscaler_model(
            os.path.join(UPSCALER_DIR, 'latent_upscaler.json'),
            os.path.join(UPSCALER_DIR, UPSCALER_FILENAME),
            DEVICE,
        ),
        text_encoder(tokenizer(batch_size * [''])),
        cond_scale=guidance_scale,
    )
    vae = (
        AutoencoderKL.from_pretrained(
            'runwayml/stable-diffusion-v1-5', subfolder='vae'
        )
        .eval()
        .requires_grad_(False)
        .to(DEVICE)
    )

    # Encode an image
    image = Image.open(input_path).convert('RGB')
    image = TF.to_tensor(image).to(DEVICE) * 2 - 1
    low_res_latent = vae.encode(image.unsqueeze(0)).latent_dist.sample() * SD_Q

    # Generate an initial noise
    [_, C, H, W] = low_res_latent.shape
    x_shape = [batch_size, C, 2 * H, 2 * W]
    noise = torch.randn(x_shape, device=DEVICE)

    # Sampling
    if noise_aug_type == 'fake':
        latent_noised = low_res_latent * (noise_aug_level**2 + 1) ** 0.5
    else:
        latent_noised = low_res_latent + noise_aug_level * torch.randn_like(
            low_res_latent
        )
    extra_args = {
        'low_res': latent_noised,
        'low_res_sigma': torch.full(
            [batch_size], noise_aug_level, device=DEVICE
        ),
        'c': text_encoder(tokenizer(batch_size * [PROMPT])),
    }
    up_latents = do_sample(
        upscaler,
        noise,
        sampler,
        sampler_steps,
        sampler_eta,
        sampler_tol_scale,
        extra_args,
        DEVICE,
    )

    # Save an upscaled image
    pixels = vae.decode(up_latents / SD_Q).sample
    pixels = pixels.add(1).div(2).clamp(0, 1)
    for i in range(pixels.shape[0]):
        img = TF.to_pil_image(pixels[i])
        img.save(output_path)
