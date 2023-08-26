import click

from image_controller import ImageController


@click.group()
def _cli():
    pass


def get_cli():
    _cli.add_command(generate)
    return _cli


def _build_kwargs(required, optional):
    kwargs = required
    for key, value in optional.items():
        if value:
            kwargs[key] = value
    return kwargs


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
    type=click.Path(exists=True),
)
@click.option(
    '--num-images',
    default=5,
    type=click.IntRange(1, 15),
)
@click.option(
    '--num-inference-steps',
    default=50,
    type=click.IntRange(1, 100),
)
@click.option(
    '--guidance-scale',
    default=7.5,
    type=click.FloatRange(0.0, 20.0),
)
@click.option(
    '--eta',
    default=0.0,
    type=click.FloatRange(0.0),
)
@click.option(
    '--autocast/--disable-autocast',
    default=True,
    type=bool,
)
def generate(
    prompt,
    negative_prompt,
    output_dir,
    num_images,
    num_inference_steps,
    guidance_scale,
    eta,
    autocast,
):
    click.echo('--- Launch Stable Diffusion ---')
    image_controller = ImageController(use_autocast=autocast)

    click.echo('--- Generate images in a latent space ---')
    kwargs = _build_kwargs(
        {
            'prompt': prompt,
            'num_images': num_images,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'eta': eta,
        },
        {
            'negative_prompt': negative_prompt,
        },
    )
    latent_list = image_controller.generate_latents(**kwargs)

    click.echo('--- Sample latents ---')
    kwargs = _build_kwargs(
        {
            'latents': latent_list,
            'prompt': prompt,
        },
        {
            'negative_prompt': negative_prompt,
        },
    )
    up_latent_list = image_controller.upscale_latents(**kwargs)

    # click.echo('--- Save Images ---')
    # for up_latent in up_latent_list:
    #     kwargs = _build_kwargs(
    #         {
    #             'latent': up_latent,
    #         },
    #         {
    #             'output_dir': output_dir,
    #         },
    #     )
    #     image_controller.save_image_from_latent(**kwargs)

    click.echo('--- Finished ---')
