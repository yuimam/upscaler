import os
import time
import uuid
from functools import lru_cache

import click
import k_diffusion as K
import numpy as np
import requests
import torch
from diffusers import StableDiffusionPipeline
from pytorch_lightning import seed_everything
from torchvision.transforms import functional as TF
from tqdm import tqdm

from models.clip import CLIPEmbedder, CLIPTokenizerTransform
from models.upscaler import CFGUpscaler, NoiseLevelAndTextConditionedUpscaler
from src import ROOT_DIR


class ImageController:
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
    UPSCALER_DIR = os.path.join(ROOT_DIR, 'latent_upscaler')
    UPSCALER_FILENAME = 'latent_upscaler.pth'
    UPSCALER_CONFIG = 'latent_upscaler.json'
    UPSCALER_URL = 'https://models.rivershavewings.workers.dev/laion_text_cond_latent_upscaler_2_1_00470000_slim.pth'
    SCALE_FACTOR = 0.18215
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')

    def __init__(self, device):
        self.device = device
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(
            'runwayml/stable-diffusion-v1-5'
        ).to(self.device)
        seed_everything(int(time.time()))

    def _download_upscaler_model(self, dir, filename):
        filepath = os.path.join(dir, filename)
        if os.path.exists(filepath):
            click.echo('Upscaler model already downloaded')
            return

        RETRY_LIMIT = 3
        RETRY_DELAY = 5
        attempts = 0
        click.echo('Upscaler model download has started')
        while attempts < RETRY_LIMIT:
            try:
                with requests.get(self.UPSCALER_URL, stream=True) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get('content-length', 0))
                    with open(filepath, 'wb') as f:
                        tqdm_params = {
                            'desc': self.UPSCALER_URL,
                            'total': total_size,
                            'miniters': 1,
                            'unit': 'B',
                            'unit_scale': True,
                            'unit_divisor': 1024,
                        }
                        with tqdm(**tqdm_params) as progress:
                            for chunk in r.iter_content(
                                chunk_size=1024 * 1024
                            ):
                                f.write(chunk)
                                progress.update(len(chunk))
                    click.echo('Upscaler model download was successful')
                    return

            except Exception as e:
                click.echo(f'Download failed: {e}')
                attempts += 1
                if attempts < RETRY_LIMIT:
                    click.echo(f'Retrying in {RETRY_DELAY} seconds...')
                    time.sleep(RETRY_DELAY)
                if os.path.exists(filepath):
                    os.remove(filepath)

        click.echo(
            'Download failed after multiple attempts. Please check your connection and try again later.'
        )

    @lru_cache()
    def _make_upscaler_model(
        self, config_path, model_path, device, pooler_dim=768, train=False
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

    def _do_sample(
        self, model, noise, sampler, steps, eta, tol_scale, extra_args, device
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
                model,
                noise * SIGMA_MAX,
                sigmas,
                extra_args=extra_args,
                eta=eta,
            )
        elif sampler == 'k_dpm_2_ancestral':
            return K.sampling.sample_dpm_2_ancestral(
                model,
                noise * SIGMA_MAX,
                sigmas,
                extra_args=extra_args,
                eta=eta,
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

    def generate_latents(
        self,
        prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        num_images=1,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
    ):
        latents = self.sd_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=512,
            width=512,
            num_images_per_prompt=num_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            eta=eta,
            output_type='latent',
        ).images.to(self.device)
        return latents

    def upscale_latents(
        self,
        latents,
        prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        guidance_scale=1,
        noise_aug_level=0.0,
        noise_aug_type='gaussian',
        sampler='k_dpm_adaptive',
        steps=50,
        tol_scale=0.25,
        eta=1.0,
    ):
        tokenizer, text_encoder = CLIPTokenizerTransform(
            tokenizer=self.sd_pipe.tokenizer
        ), CLIPEmbedder(encoder=self.sd_pipe.text_encoder, device=self.device)
        encoded_c = text_encoder(tokenizer([prompt]))
        encoded_uc = text_encoder(tokenizer([negative_prompt]))

        self._download_upscaler_model(
            self.UPSCALER_DIR, self.UPSCALER_FILENAME
        )
        upscaler = CFGUpscaler(
            self._make_upscaler_model(
                os.path.join(self.UPSCALER_DIR, self.UPSCALER_CONFIG),
                os.path.join(self.UPSCALER_DIR, self.UPSCALER_FILENAME),
                self.device,
            ),
            encoded_uc,
            cond_scale=guidance_scale,
        )

        up_latent_list = []
        for latent in latents:
            low_res_latent = latent.unsqueeze(0)
            if noise_aug_type == 'fake':
                latent_noised = (
                    low_res_latent * (noise_aug_level**2 + 1) ** 0.5
                )
            else:
                latent_noised = (
                    low_res_latent
                    + noise_aug_level * torch.randn_like(low_res_latent)
                )
            extra_args = {
                'low_res': latent_noised,
                'low_res_sigma': torch.tensor(
                    [noise_aug_level], device=self.device
                ),
                'c': encoded_c,
            }
            [_, C, H, W] = low_res_latent.shape
            x_shape = [1, C, 2 * H, 2 * W]
            noise = torch.randn(x_shape, device=self.device)
            up_latent = self._do_sample(
                upscaler,
                noise,
                sampler,
                steps,
                eta,
                tol_scale,
                extra_args,
                self.device,
            )
            up_latent_list.append(up_latent)

        return up_latent_list

    def save_image_from_latent(self, latent, output_dir=OUTPUT_DIR):
        pixels = self.sd_pipe.vae.decode(latent / self.SCALE_FACTOR).sample
        pixels = pixels.add(1).div(2).clamp(0, 1)
        file_path = os.path.abspath(
            os.path.join(output_dir, str(uuid.uuid4()) + '.png')
        )
        for i in range(pixels.shape[0]):
            image = TF.to_pil_image(pixels[i])
            image.save(file_path)
        click.echo(f'Saved: {file_path}')
