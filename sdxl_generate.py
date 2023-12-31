import math
import os

import PIL
import torch
from PIL import Image, ImageEnhance
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from dotenv import dotenv_values
from loguru import logger

from gradio_prompter.generation_data import GenerationData
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


def enhance(img: PIL.Image, sharpness: float, contrast: float) -> PIL.Image:
    if sharpness > 0:
        logger.info("Sharpening image...")
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
    if contrast > 0:
        logger.info("Increasing contrast...")
        img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def create_image_data(
    generation_data: GenerationData,
) -> list[tuple[str, str]]:
    # d = [

    #     ("Fooocus V2 Expansion", task["expansion"]),
    #     ("Styles", str(raw_style_selections)),
    #     ("Resolution", str((width, height))),
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
    #     ("Sampler", sampler_name),
    #     ("Scheduler", scheduler_name),
    # ]
    data = [
        ("Prompt", generation_data.prompt),
        ("Negative Prompt", generation_data.negative_prompt),
        ("Guidance Scale", generation_data.guidance_scale),
        ("Inference Steps", generation_data.num_inference_steps),
        ("Seed", generation_data.seed),
        ("Base Model", generation_data.base_model),
        ("lora ", generation_data.lora),
        ("Sharpness", generation_data.sharpness),
        ("Contrast", generation_data.contrast),
        ("Upscaled", generation_data.upscale_by),
    ]
    if generation_data.use_refiner:
        data.extend(
            [
                ("Refiner Model", generation_data.refiner_model),
                ("Refiner Switch", generation_data.refiner_switch),
            ]
        )
    data.append(("json", generation_data.model_dump_json()))
    return data


@torch.no_grad()
@torch.inference_mode()
def run_sdxl_pipelines(generation_data: GenerationData) -> list[PIL.Image]:
    global pipeline, upscaler, refiner_pipeline
    generator = torch.Generator(device="cuda").manual_seed(generation_data.seed)
    if pipeline is None:
        pipeline = load_pipeline()
    if upscaler is None and generation_data.upscale_by > 1:
        logger.info("Loading upscaler...")
        upscaler = UpscalerESRGAN()
    if generation_data.use_refiner and refiner_pipeline is None:
        refiner_pipeline = load_refiner()

    logger.info("Generating image...")
    sdxl_output = pipeline(
        prompt=generation_data.prompt,
        negative_prompt=generation_data.negative_prompt,
        guidance_scale=generation_data.guidance_scale,
        num_inference_steps=generation_data.num_inference_steps,
        generator=generator,
        output_type="latent" if generation_data.use_refiner else "pil",
        denoising_end=generation_data.refiner_switch
        if generation_data.use_refiner
        else 1.0,
    )
    sdxl_output_img = sdxl_output.images[0]

    if generation_data.use_refiner:
        logger.info("Refining image...")
        global refiner_cfg
        refiner_steps = math.ceil(
            generation_data.num_inference_steps * (1 - generation_data.refiner_switch)
        )
        sdxl_output_img = refiner_pipeline(
            prompt=generation_data.prompt,
            image=sdxl_output_img,
            negative_prompt=generation_data.negative_prompt,
            strength=0.3,
            guidance_scale=refiner_cfg,
            latents=sdxl_output_img,
            generator=generator,
            num_inference_steps=refiner_steps,
            output_type="pil",
            vae=True,
        ).images[0]

    if generation_data.upscale_by > 1:
        logger.info("Upscaling image...")
        sdxl_output_img = upscaler.upscale(
            sdxl_output_img,
            generation_data.upscale_by,
            upscaler_model_path,
        )

    sdxl_output_img = enhance(
        sdxl_output_img,
        generation_data.sharpness,
        generation_data.contrast,
    )
    logger.info("Saving image...")
    log(
        sdxl_output_img,
        create_image_data(generation_data),
    )
    # return list for gallery

    return [sdxl_output_img]


def generate_image(
    prompt: str,
    negative_prompt: str = None,
    guidance_scale: float = 7.0,
    num_inference_steps: int = 50,
    seed: int = -1,
    use_refiner: bool = False,
    sharpness: float = 0,
    contrast: float = 0,
    upscale_by: float = 1.5,
):
    if seed == -1:
        seed = torch.Generator(device="cuda").seed()

    generation_data = GenerationData(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        use_refiner=use_refiner,
        sharpness=sharpness,
        contrast=contrast,
        upscale_by=upscale_by,
        base_model=base_model_name,
        lora=base_lora_name,
    )
    if use_refiner:
        generation_data.refiner_model = refiner_model_name
        generation_data.refiner_switch = refiner_cfg
    else:
        generation_data.refiner_model = None
        generation_data.refiner_switch = None

    logger.info(
        f"Generating image with generation data: {generation_data.model_dump_json()} "
    )
    return run_sdxl_pipelines(generation_data), generation_data.model_dump_json(
        indent=2
    )
