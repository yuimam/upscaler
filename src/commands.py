import os
import time
import uuid

import click
import torch
from diffusers import StableDiffusionPipeline
from pytorch_lightning import seed_everything
from torchvision.transforms import functional as TF

from models.clip import CLIPEmbedder, CLIPTokenizerTransform
from models.upscaler import CFGUpscaler
from utils import (amp_autocast, do_sample, download_model,
                   get_available_device, make_upscaler_model)


@click.group()
def _cli():
    pass


def get_cli():
    _cli.add_command(upscale)
    return _cli


@click.command()
@click.option(
    '--prompt',
    required=True,
)
@click.option(
    '--negative-prompt',
)
@click.option(
    '--output-dir',
    required=True,
    type=click.Path(exists=False),
)
@click.option(
    '--num-images',
    default=1,
    type=click.IntRange(1, 10),
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
@click.option(
    '--autocast/--disable-autocast',
    default=True,
    type=bool,
)
@torch.no_grad()
def upscale(
    prompt,
    negative_prompt,
    output_dir,
    num_images,
    batch_size,
    guidance_scale,
    noise_aug_level,
    noise_aug_type,
    sampler,
    sampler_steps,
    sampler_tol_scale,
    sampler_eta,
    autocast,
):
    SCALE_FACTOR = 0.18215
    UPSCALER_DIR = os.path.join(os.path.dirname(__file__), 'latent_upscaler')
    UPSCALER_FILENAME = 'latent_upscaler.pth'
    UPSCALER_CONFIG = 'latent_upscaler.json'
    DEVICE = torch.device(get_available_device())
    DEFAULT_NEGATIVE_PROMPT = (
        'ugly,duplication,duplicates,mutilation,deformed,mutilated,mutation,'
        'twisted body,disfigured,bad anatomy,out of frame,extra fingers,'
        'mutated hands,poorly drawn hands,extra limbs,malformed limbs,missing arms,'
        'extra arms,missing legs,extra legs,extra hands,fused fingers,missing fingers,'
        'long neck,small head,closed eyes,rolling eyes,weird eyes,smudged face,'
        'blurred face,poorly drawn face,cloned face,strange mouth,grainy,blurred,blurry,'
        'writing,calligraphy,signature,text,watermark,bad art,nsfw,ugly,bad face,'
        'flat color,flat shading,retro style,poor quality,bad fingers,low res,cropped,'
        'username,artist name,low quality'
    )
    negative_prompt = negative_prompt or DEFAULT_NEGATIVE_PROMPT

    seed_everything(int(time.time()))
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5'
    ).to(DEVICE)

    # Generate images in a latent space
    with amp_autocast(autocast, DEVICE):
        images = sd_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=512,
            width=512,
            num_images_per_prompt=num_images,
            output_type='latent',
        ).images.to(DEVICE)

    # Tokenize the prompts
    tokenizer, text_encoder = CLIPTokenizerTransform(
        tokenizer=sd_pipe.tokenizer
    ), CLIPEmbedder(encoder=sd_pipe.text_encoder, device=DEVICE)
    with amp_autocast(autocast, DEVICE):
        encoded_c = text_encoder(tokenizer(batch_size * [prompt]))
        encoded_uc = text_encoder(tokenizer(batch_size * [negative_prompt]))

    # Create upscaler
    download_model(UPSCALER_DIR, UPSCALER_FILENAME)
    upscaler = CFGUpscaler(
        make_upscaler_model(
            os.path.join(UPSCALER_DIR, UPSCALER_CONFIG),
            os.path.join(UPSCALER_DIR, UPSCALER_FILENAME),
            DEVICE,
        ),
        encoded_uc,
        cond_scale=guidance_scale,
    )

    for image in images:
        # Sampling process
        low_res_latent = image.unsqueeze(0)
        if noise_aug_type == 'fake':
            latent_noised = low_res_latent * (noise_aug_level**2 + 1) ** 0.5
        else:
            latent_noised = (
                low_res_latent
                + noise_aug_level * torch.randn_like(low_res_latent)
            )
        extra_args = {
            'low_res': latent_noised,
            'low_res_sigma': torch.full(
                [batch_size], noise_aug_level, device=DEVICE
            ),
            'c': encoded_c,
        }
        with amp_autocast(autocast, DEVICE):
            [_, C, H, W] = low_res_latent.shape
            x_shape = [batch_size, C, 2 * H, 2 * W]
            noise = torch.randn(x_shape, device=DEVICE)
            up_latent = do_sample(
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
        vae = sd_pipe.vae
        pixels = vae.decode(up_latent / SCALE_FACTOR).sample
        pixels = pixels.add(1).div(2).clamp(0, 1)
        for i in range(pixels.shape[0]):
            img = TF.to_pil_image(pixels[i])
            file_path = os.path.join(output_dir, str(uuid.uuid4()) + '.jpeg')
            img.save(file_path)
