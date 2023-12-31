from typing import Optional

from pydantic import BaseModel


class GenerationData(BaseModel):
    prompt: str
    negative_prompt: str = None
    guidance_scale: float
    num_inference_steps: int
    seed: int
    use_refiner: bool
    sharpness: float
    contrast: float
    upscale_by: float
    base_model: str
    lora: str
    refiner_model: Optional[str] = None
    refiner_switch: Optional[float] = None
