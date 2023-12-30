import torch
from PIL import ImageEnhance
from diffusers import StableDiffusionXLPipeline
from dotenv import dotenv_values
from loguru import logger

from gradio_prompter.private_logger import log
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


def create_image_data(prompt, negative_prompt, guidance_scale, num_inference_steps):
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
    return [
        ("Prompt", prompt),
        ("Negative Prompt", negative_prompt),
        ("Guidance Scale", guidance_scale),
        ("Inference Steps", num_inference_steps),
    ]


@torch.no_grad()
@torch.inference_mode()
def generate_image(
    prompt: str,
    negative_prompt: str = None,
    guidance_scale: float = 7.0,
    num_inference_steps: int = 50,
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
    sdxl_output = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )
    logger.info(sdxl_output)
    sdxl_output_img = sdxl_output.images[0]

    logger.info("Upscaling image...")

    img = enhance(
        upscaler.upscale(
            sdxl_output_img,
            1.5,
            upscaler_model_path,
        )
    )
    logger.info("Saving image...")
    log(img, create_image_data(prompt, negative_prompt, guidance_scale))
    # return list for gallery

    return [img]
