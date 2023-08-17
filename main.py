import os
import sys
import time

ROOT_DIR = os.path.dirname(__file__)
sys.path.extend([os.path.join(ROOT_DIR, 'models/taming-transformers')])
sys.path.extend([os.path.join(ROOT_DIR, 'models/stable-diffusion')])

import huggingface_hub
import k_diffusion as K
import numpy as np
import torch
import torch.nn.functional as F
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import nn
from torchvision.transforms import functional as TF
from transformers import CLIPTextModel, CLIPTokenizer, logging


class NoiseLevelAndTextConditionedUpscaler(nn.Module):
    def __init__(self, inner_model, sigma_data=1.0, embed_dim=256):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = sigma_data
        self.low_res_noise_embed = K.layers.FourierFeatures(
            1, embed_dim, std=2
        )

    def forward(self, input, sigma, low_res, low_res_sigma, c, **kwargs):
        cross_cond, cross_cond_padding, pooler = c
        c_in = 1 / (low_res_sigma**2 + self.sigma_data**2) ** 0.5
        c_noise = low_res_sigma.log1p()[:, None]
        c_in = K.utils.append_dims(c_in, low_res.ndim)
        low_res_noise_embed = self.low_res_noise_embed(c_noise)
        low_res_in = (
            F.interpolate(low_res, scale_factor=2, mode='nearest') * c_in
        )
        mapping_cond = torch.cat([low_res_noise_embed, pooler], dim=1)
        return self.inner_model(
            input,
            sigma,
            unet_cond=low_res_in,
            mapping_cond=mapping_cond,
            cross_cond=cross_cond,
            cross_cond_padding=cross_cond_padding,
            **kwargs,
        )


class CFGUpscaler(nn.Module):
    def __init__(self, model, uc, cond_scale):
        super().__init__()
        self.inner_model = model
        self.uc = uc
        self.cond_scale = cond_scale

    def forward(self, x, sigma, low_res, low_res_sigma, c):
        if self.cond_scale in (0.0, 1.0):
            # Shortcut for when we don't need to run both.
            if self.cond_scale == 0.0:
                c_in = self.uc
            else:
                c_in = c
            return self.inner_model(
                x, sigma, low_res=low_res, low_res_sigma=low_res_sigma, c=c_in
            )

        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        low_res_in = torch.cat([low_res] * 2)
        low_res_sigma_in = torch.cat([low_res_sigma] * 2)
        c_in = [
            torch.cat([uc_item, c_item]) for uc_item, c_item in zip(self.uc, c)
        ]
        uncond, cond = self.inner_model(
            x_in,
            sigma_in,
            low_res=low_res_in,
            low_res_sigma=low_res_sigma_in,
            c=c_in,
        ).chunk(2)
        return uncond + (cond - uncond) * self.cond_scale


class CLIPTokenizerTransform:
    def __init__(self, version='openai/clip-vit-large-patch14', max_length=77):

        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.max_length = max_length

    def __call__(self, text):
        indexer = 0 if isinstance(text, str) else ...
        tok_out = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding='max_length',
            return_tensors='pt',
        )
        input_ids = tok_out['input_ids'][indexer]
        attention_mask = 1 - tok_out['attention_mask'][indexer]
        return input_ids, attention_mask


class CLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(self, version='openai/clip-vit-large-patch14', device='cuda'):
        super().__init__()

        logging.set_verbosity_error()
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.transformer = (
            self.transformer.eval().requires_grad_(False).to(device)
        )

    @property
    def device(self):
        return self.transformer.device

    def forward(self, tok_out):
        input_ids, cross_cond_padding = tok_out
        clip_out = self.transformer(
            input_ids=input_ids.to(self.device), output_hidden_states=True
        )
        return (
            clip_out.hidden_states[-1],
            cross_cond_padding.to(self.device),
            clip_out.pooler_output,
        )


def make_upscaler_model(
    config_path, model_path, pooler_dim=768, train=False, device='cpu'
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


def do_sample(model, noise, extra_args):
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


# 固定値
SD_C = 4   # Latent dimension
SD_F = 8   # Latent patch size (pixels per latent)
SD_Q = 0.18215   # sd_model.scale_factor; scaling for latents in first stage models
SIGMA_MIN, SIGMA_MAX = 0.029167532920837402, 14.614642143249512
cpu = torch.device('cpu')
device = torch.device('cuda')

# パラメータ
input_image_path = os.path.join(ROOT_DIR, 'input.jpeg')
output_image_path = os.path.join(ROOT_DIR, 'output.jpeg')
batch_size = 1
seed = int(time.time())
prompt = 'the temple of fire by Ross Tran and Gerardo Dottori, oil on canvas'
sampler = 'k_dpm_adaptive'
steps = 50
eta = 1.0
tol_scale = 0.25
num_samples = 1
noise_aug_type = 'gaussian'
guidance_scale = 1
noise_aug_level = 0

# Tokenizer 作成
tokenizer = CLIPTokenizerTransform()

# Text Encoder 作成
text_encoder = CLIPEmbedder(device=device)

# UpScaler Model 作成
model = make_upscaler_model(
    os.path.join(ROOT_DIR, 'configs/latent_upscaler.json'),
    os.path.join(ROOT_DIR, 'models/latent_upscaler.pth'),
).to(device)
model = CFGUpscaler(
    model,
    text_encoder(tokenizer(batch_size * [''])),
    cond_scale=guidance_scale,
)

# VAE 作成
vae_path = huggingface_hub.hf_hub_download(
    'stabilityai/sd-vae-ft-mse-original', 'vae-ft-mse-840000-ema-pruned.ckpt'
)
pl_sd = torch.load(vae_path, map_location='cpu')
sd = pl_sd['state_dict']
config = OmegaConf.load(
    os.path.join(ROOT_DIR, 'configs/kl_f8.yaml')
)
vae = instantiate_from_config(config.model)
m, u = vae.load_state_dict(sd, strict=False)
vae = vae.to(cpu).eval().requires_grad_(False).to(device)


###
# 実行
###
seed_everything(seed)

# 画像を開きエンコード
image = Image.open(input_image_path).convert('RGB')
image = TF.to_tensor(image).to(device) * 2 - 1
low_res_latent = vae.encode(image.unsqueeze(0)).sample() * SD_Q

# ノイズ生成
[_, C, H, W] = low_res_latent.shape
x_shape = [batch_size, C, 2 * H, 2 * W]
noise = torch.randn(x_shape, device=device)

# サンプラーにわたす extra args の作成
low_res_sigma = torch.full([batch_size], noise_aug_level, device=device)
if noise_aug_type == 'fake':
    latent_noised = low_res_latent * (noise_aug_level**2 + 1) ** 0.5
else:
    latent_noised = low_res_latent + noise_aug_level * torch.randn_like(
        low_res_latent
    )
extra_args = {
    'low_res': latent_noised,
    'low_res_sigma': low_res_sigma,
    'c': text_encoder(tokenizer(batch_size * [prompt])),
}

# サンプリング
up_latents = do_sample(model, noise, extra_args)

# デコード
pixels = vae.decode(up_latents / SD_Q)
pixels = pixels.add(1).div(2).clamp(0, 1)

# 画像の保存
for j in range(pixels.shape[0]):
    img = TF.to_pil_image(pixels[j])
    img.save(output_image_path)
