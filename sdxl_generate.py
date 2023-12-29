import torch
from diffusers import StableDiffusionXLPipeline

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
    return StableDiffusionXLPipeline.from_pretrained(
        "dataautogpt3/OpenDalleV1.1", torch_dtype=torch.float16
    ).to("cuda")


pipeline = None


@torch.no_grad()
@torch.inference_mode()
def generate_image(prompt: str):
    global pipeline
    if pipeline is None:
        pipeline = load_pipeline()
    return pipeline(prompt).images[0]
