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
    SCALE_FACTOR = 0.18215
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')

    def __init__(self, use_autocast=True):
        self.device = torch.device(get_available_device())
        self.use_autocast = use_autocast
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(
            'runwayml/stable-diffusion-v1-5'
        ).to(self.device)
        seed_everything(int(time.time()))

    def generate_latents(
        self,
        prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        num_images=1,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
    ):
        with amp_autocast(self.use_autocast, self.device):
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
        with amp_autocast(self.use_autocast, self.device):
            encoded_c = text_encoder(tokenizer([prompt]))
            encoded_uc = text_encoder(tokenizer([negative_prompt]))

        download_model(self.UPSCALER_DIR, self.UPSCALER_FILENAME)
        upscaler = CFGUpscaler(
            make_upscaler_model(
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
            with amp_autocast(self.use_autocast, self.device):
                [_, C, H, W] = low_res_latent.shape
                x_shape = [1, C, 2 * H, 2 * W]
                noise = torch.randn(x_shape, device=self.device)
                up_latent = do_sample(
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

            pixels = self.sd_pipe.vae.decode(up_latent / self.SCALE_FACTOR).sample
            pixels = pixels.add(1).div(2).clamp(0, 1)
            for i in range(pixels.shape[0]):
                image = TF.to_pil_image(pixels[i])
                file_path = os.path.join(self.OUTPUT_DIR, str(uuid.uuid4()) + '.png')
                image.save(file_path)

        return up_latent_list

    def save_image_from_latent(self, latent, output_dir=OUTPUT_DIR):
        pixels = self.sd_pipe.vae.decode(latent / self.SCALE_FACTOR).sample
        pixels = pixels.add(1).div(2).clamp(0, 1)
        for i in range(pixels.shape[0]):
            image = TF.to_pil_image(pixels[i])
            file_path = os.path.join(output_dir, str(uuid.uuid4()) + '.png')
            image.save(file_path)
