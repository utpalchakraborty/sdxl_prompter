from threading import Thread

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    pipeline,
    Pipeline,
    TextIteratorStreamer,
)

logger.info(f"cuda is available: {torch.cuda.is_available()}")

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

llm: Pipeline = None
streamer: TextIteratorStreamer = None
default_top_p = 0.9
default_top_k = 0
default_temperature = 0.0001


def init_llm():
    logger.info(f"Initializing LLM{MODEL_NAME}")
    global llm
    global streamer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )

    generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
    generation_config.max_new_tokens = 1024
    generation_config.temperature = default_temperature
    generation_config.do_sample = True
    generation_config.top_k = default_top_k
    generation_config.top_p = default_top_p

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    logger.info("Creating LLM pipeline")
    llm = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        generation_config=generation_config,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )


def format_prompt(prompt, system_prompt=""):
    if system_prompt.strip():
        return f"[INST] {system_prompt} {prompt} [/INST]"
    return f"[INST] {prompt} [/INST]"


SYSTEM_PROMPT = """
You're an awesome stable diffusion prompt generator. Given any prompt you can analyze it and generate a better version 
of it by modifying it in subtle ways. Given the following stable diffusion prompt, enhance it in three different ways. 
Put each new prompt on a new line adding an extra line in between them. Do not add line numberings to the prompt.
""".strip()


def extract_post_instruction(text):
    """
    Extracts and returns the part of the text following the [/INST] tag.
    """
    # Splitting the text at the [/INST] tag
    parts = text.split("[/INST]")
    # Returning the part after the [/INST] tag, if it exists
    return parts[1].strip() if len(parts) > 1 else ""


def generate_from_llm(
    prompt: str, system_prompt: str, top_p: float, top_k: int, temperature: float
) -> str:
    if not llm:
        init_llm()
    if not system_prompt:
        system_prompt = SYSTEM_PROMPT

    llm.model.generation_config.top_p = top_p
    llm.model.generation_config.top_k = top_k
    llm.model.generation_config.temperature = temperature

    thread = Thread(
        target=lambda: llm(format_prompt(prompt, system_prompt), return_full_text=True)
    )
    thread.start()
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield generated_text
