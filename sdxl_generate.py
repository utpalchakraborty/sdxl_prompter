import math
import os

import PIL
import torch
from PIL import Image, ImageEnhance
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from dotenv import dotenv_values
from loguru import logger

from gradio_prompter.private_logger import log
from gradio_prompter.upscaler.esrgan_model import UpscalerESRGAN

config = dotenv_values(".env")
sdxl_model_path = config["SDXL_MODEL_PATH"]
upscaler_model_path = config["UPSCALER_MODEL_PATH"]
refiner_model_path = config["REFINER_MODEL_PATH"]
lora_path = config["LORA_PATH"]

logger.info(f"SDXL model path: {sdxl_model_path}")
logger.info(f"Upscaler model path: {upscaler_model_path}")
logger.info(f"Lora path: {lora_path}")
logger.info(f"Refiner model path: {refiner_model_path}")

base_model_name = os.path.basename(sdxl_model_path)
refiner_model_name = os.path.basename(refiner_model_path)
base_lora_name = os.path.basename(lora_path)
refiner_switch = 0.75
refiner_cfg = 3.5

cache_dir = None
force_download = False
resume_download = "resume_download"
proxies = "proxies"
token = "token"
local_files_only = False
revision = None


load_config_kwargs = {
    "cache_dir": cache_dir,
    "force_download": force_download,
    "resume_download": resume_download,
    "proxies": proxies,
    "token": token,
    "local_files_only": local_files_only,
    "revision": revision,
}


def load_pipeline() -> StableDiffusionXLPipeline:
    logger.info("Loading pipeline...")
    sdxl_pipe = StableDiffusionXLPipeline.from_single_file(
        sdxl_model_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    logger.info("Loading Lora weights...")
    sdxl_pipe.load_lora_weights(lora_path)
    return sdxl_pipe


def load_refiner() -> StableDiffusionXLImg2ImgPipeline:
    logger.info("Loading refiner...")
    refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
        refiner_model_path,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        load_safety_checker=False,
    ).to("cuda")

    return refiner_pipe


pipeline = None
upscaler = None
refiner_pipeline = None


def enhance(img: PIL.Image) -> PIL.Image:
    logger.info("Sharpening image...")
    img = ImageEnhance.Sharpness(img).enhance(1.25)
    logger.info("Increasing contrast...")
    img = ImageEnhance.Contrast(img).enhance(1.25)
    return img


def create_image_data(
    prompt, negative_prompt, guidance_scale, num_inference_steps, seed, use_refiner
) -> list[tuple[str, str]]:
    # d = [
    #     ("Prompt", task["log_positive_prompt"]),
    #     ("Negative Prompt", task["log_negative_prompt"]),
    #     ("Fooocus V2 Expansion", task["expansion"]),
    #     ("Styles", str(raw_style_selections)),
    #     ("Performance", performance_selection),
    #     ("Resolution", str((width, height))),
    #     ("Sharpness", sharpness),
    #     ("Guidance Scale", guidance_scale),
    #     (
    #         "ADM Guidance",
    #         str(
    #             (
    #                 modules.patch.positive_adm_scale,
    #                 modules.patch.negative_adm_scale,
    #                 modules.patch.adm_scaler_end,
    #             )
    #         ),
    #     ),
    #     ("Base Model", base_model_name),
    #     ("Refiner Model", refiner_model_name),
    #     ("Refiner Switch", refiner_switch),
    #     ("Sampler", sampler_name),
    #     ("Scheduler", scheduler_name),
    #     ("Seed", task["task_seed"]),
    # ]
    data = [
        ("Prompt", prompt),
        ("Negative Prompt", negative_prompt),
        ("Guidance Scale", guidance_scale),
        ("Inference Steps", num_inference_steps),
        ("Seed", seed),
        ("Base Model", base_model_name),
        ("lora ", base_lora_name),
    ]
    if use_refiner:
        data.extend(
            [
                ("Refiner Model", refiner_model_name),
                ("Refiner Switch", refiner_switch),
            ]
        )
    return data


@torch.no_grad()
@torch.inference_mode()
def generate_image(
    prompt: str,
    negative_prompt: str = None,
    guidance_scale: float = 7.0,
    num_inference_steps: int = 50,
    seed: int = -1,
    use_refiner: bool = False,
):
    if seed == -1:
        seed = torch.Generator(device="cuda").seed()

    generator = torch.Generator(device="cuda").manual_seed(seed)
    logger.info(
        f"Generating image with prompt: {prompt}, -ve prompt: {negative_prompt}, guidance scale: {guidance_scale}, "
        f"seed: {seed}"
    )
    global pipeline, upscaler, refiner_pipeline, refiner_switch
    if pipeline is None:
        pipeline = load_pipeline()
    if upscaler is None:
        logger.info("Loading upscaler...")
        upscaler = UpscalerESRGAN()
    if use_refiner and refiner_pipeline is None:
        refiner_pipeline = load_refiner()

    logger.info("Generating image...")
    sdxl_output = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        output_type="latent" if use_refiner else "pil",
        denoising_end=refiner_switch if use_refiner else 1.0,
    )
    sdxl_output_img = sdxl_output.images[0]

    if use_refiner:
        logger.info("Refining image...")
        global refiner_cfg
        refiner_steps = math.ceil(num_inference_steps * (1 - refiner_switch))
        sdxl_output_img = refiner_pipeline(
            prompt=prompt,
            image=sdxl_output_img,
            negative_prompt=negative_prompt,
            strength=0.3,
            guidance_scale=refiner_cfg,
            latents=sdxl_output_img,
            generator=generator,
            num_inference_steps=refiner_steps,
            output_type="pil",
            vae=True,
        ).images[0]

    logger.info("Upscaling image...")

    img = enhance(
        upscaler.upscale(
            sdxl_output_img,
            1.5,
            upscaler_model_path,
        )
    )
    logger.info("Saving image...")
    log(
        img,
        create_image_data(
            prompt,
            negative_prompt,
            guidance_scale,
            num_inference_steps,
            seed,
            use_refiner,
        ),
    )
    # return list for gallery

    return [img]
