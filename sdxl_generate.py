import torch
from diffusers import StableDiffusionXLPipeline

from gradio_prompter.upscaler.esrgan_model import UpscalerESRGAN

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
    return StableDiffusionXLPipeline.from_single_file(
        "H:\\ai\\stable-diffusion-webui\\models\\Stable-diffusion\\SDXL\\juggernautXL_v7Rundiffusion.safetensors",
        torch_dtype=torch.float16,
    ).to("cuda")


pipeline = None
upscaler = None


@torch.no_grad()
@torch.inference_mode()
def generate_image(
    prompt: str, negative_prompt: str = None, guidance_scale: float = 7.0
):
    global pipeline, upscaler
    if pipeline is None:
        pipeline = load_pipeline()
    if upscaler is None:
        upscaler = UpscalerESRGAN()

    original = pipeline(
        prompt=prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale
    ).images[0]

    return upscaler.upscale(
        original,
        1.5,
        "H:\\ai\\stable-diffusion-webui\\models\\ESRGAN\\4x-UltraSharp.pth",
    )
