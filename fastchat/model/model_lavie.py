import gc
from threading import Thread
import torch
from diffusers import DDIMScheduler

from fastchat.utils import build_logger

logger = build_logger("diffusion_infer", 'diffusion_infer.log')

@torch.inference_mode()
def generate_stream_lavie(
    model,
    tokenizer,
    params,
    device,
):
    prompt = params["prompt"]
    encoding = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = encoding.input_ids
    input_echo_len = len(input_ids)

    logger.info(f"prompt: {prompt}")
    logger.info(f"model.scheduler: {model.pipe.scheduler}")
    logger.info(f"model.type: {type(model)}")
    # logger.info(f"prompt: {prompt}")
    output = model(prompt=prompt,
                   video_length=16,
                   height=360,
                   width=512,
                   num_inference_steps=50,
                   guidance_scale=7.5).video

    yield {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": 0,
            "total_tokens": input_echo_len,
        },
        "finish_reason": "stop",
    }
    # thread.join()

    # clean
    gc.collect()
    torch.cuda.empty_cache()
    if device == "xpu":
        torch.xpu.empty_cache()
    if device == "npu":
        torch.npu.empty_cache()
