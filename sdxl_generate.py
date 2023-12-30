import torch
from PIL import ImageEnhance
from diffusers import StableDiffusionXLPipeline
from dotenv import dotenv_values
from loguru import logger

from gradio_prompter.upscaler.esrgan_model import UpscalerESRGAN

config = dotenv_values(".env")
sdxl_model_path = config["SDXL_MODEL_PATH"]
upscaler_model_path = config["UPSCALER_MODEL_PATH"]
lora_path = config["LORA_PATH"]
logger.info(f"SDXL model path: {sdxl_model_path}")
logger.info(f"Upscaler model path: {upscaler_model_path}")
logger.info(f"Lora path: {lora_path}")

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


def load_pipeline():
    logger.info("Loading pipeline...")
    sdxl_pipe = StableDiffusionXLPipeline.from_single_file(
        sdxl_model_path,
        torch_dtype=torch.float16,
    ).to("cuda")
    logger.info("Loading Lora weights...")
    sdxl_pipe.load_lora_weights(lora_path)
    return sdxl_pipe


pipeline = None
upscaler = None


def enhance(img):
    logger.info("Sharpening image...")
    img = ImageEnhance.Sharpness(img).enhance(1.25)
    logger.info("Increasing contrast...")
    img = ImageEnhance.Contrast(img).enhance(1.25)
    return img


@torch.no_grad()
@torch.inference_mode()
def generate_image(
    prompt: str, negative_prompt: str = None, guidance_scale: float = 7.0
):
    logger.info(
        f"Generating image with prompt: {prompt} and -ve prompt: {negative_prompt} and guidance scale: {guidance_scale}"
    )
    global pipeline, upscaler
    if pipeline is None:
        pipeline = load_pipeline()
    if upscaler is None:
        logger.info("Loading upscaler...")
        upscaler = UpscalerESRGAN()

    logger.info("Generating image...")
    original = pipeline(
        prompt=prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale
    ).images[0]

    logger.info("Upscaling image...")
    # return list for gallery
    return [
        enhance(
            upscaler.upscale(
                original,
                1.5,
                upscaler_model_path,
            )
        )
    ]
