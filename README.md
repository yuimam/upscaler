# Upscaler

## System Requirements

- Python: >= 3.10. This script has been tested with Python 3.10.6.
- [Poetry](https://github.com/python-poetry/poetry)
- CUDA: >= 12.0. This script has been tested with CUDA 12.0.

## Usage

### 1. Install Dependencies

First, install the necessary dependencies using `poetry` from the root directory of this application.

```sh
poetry install
```

### 2. Generate 1024 x 1024 images

Execute the following command from the root directory:

```sh
poetry run python src/main.py generate --prompt "Marilyn Monroe" 
```

The initial execution might take several minutes because the script needs to load various libraries and download models.

The progress will be displayed in the stdout.
Images will be saved in the `output` directory by default.

```sh
--- CLI started ---
--- Launch Stable Diffusion ---
Loading pipeline components...:  14%|████████████▎                                                                         | 1/7 [00:00<00:02,  2.85it/s]
`text_config_dict` is provided which will be used to initialize `CLIPTextConfig`. The value `text_config["id2label"]` will be overriden.
`text_config_dict` is provided which will be used to initialize `CLIPTextConfig`. The value `text_config["bos_token_id"]` will be overriden.
`text_config_dict` is provided which will be used to initialize `CLIPTextConfig`. The value `text_config["eos_token_id"]` will be overriden.
Loading pipeline components...: 100%|██████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:01<00:00,  6.32it/s]
Global seed set to 1693096428
--- Generate images in a latent space ---
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:14<00:00,  3.47it/s]
--- Sample latents ---
Upscaler model already downloaded
45it [00:03, 14.49it/s]
45it [00:03, 14.89it/s]
51it [00:03, 14.90it/s]
51it [00:03, 14.90it/s]
45it [00:03, 14.90it/s]
--- Save images ---
Saved: /home/ec2-user/upscaler/output/5b3f7a2b-bac7-421c-9a0a-64656224aeda.png
Saved: /home/ec2-user/upscaler/output/0c2a728c-5bc4-4388-a68b-bea46d9f4c1a.png
Saved: /home/ec2-user/upscaler/output/f006466f-1734-4728-9a15-4ee0384ee9a7.png
Saved: /home/ec2-user/upscaler/output/5d83612d-cd9c-474c-9dfe-694ed72cb319.png
Saved: /home/ec2-user/upscaler/output/dfa5b510-d38c-4252-9a50-78aa4fa4864d.png
```

### 3. Execute with options

You can also specify other options such as:

- `--num-images`: to determine the number of images to generate
- `--output-dir`: to specify a directory where the images will be saved.

```sh
poetry run python src/main.py generate --prompt "Marilyn Monroe" --num-images 10 --output-dir ../output
```

To view all available options, append the --help flag:

```sh
poetry run python src/main.py generate --help
```
