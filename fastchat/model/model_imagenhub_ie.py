import gc
import numpy as np
from threading import Thread
import torch
from diffusers import DDIMScheduler
import transformers
from transformers import (
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)
from fastchat.utils import build_logger
import PIL

logger = build_logger("diffusion_infer", 'diffusion_infer.log')

def generate_stream_imagen_ie(
    model,
    tokenizer,
    params,
    device,
    context_len=256,
    stream_interval=2,
):
    prompt_source = params["prompt_source"]
    prompt_target = params["prompt_target"]
    prompt_instruct = params["prompt_instruct"]
    grey = np.array(params["image_source"])
    image_source = PIL.Image.fromarray(grey.astype(np.uint8)).resize((256, 256))
    # image_source = PIL.Image.fromarray(np.array(params["image_source"]))
    encoding = tokenizer(prompt_source, return_tensors="pt").to(device)
    input_ids = encoding.input_ids
    # encoding["decoder_input_ids"] = encoding["input_ids"].clone()
    input_echo_len = len(input_ids)

    logger.info(f"prompt source: {prompt_source}")
    logger.info(f"prompt target: {prompt_target}")
    logger.info(f"image source shape: {image_source.size}")
    logger.info(f"model.scheduler: {model.pipe.scheduler}")
    logger.info(f"model.type: {type(model)}")
    # logger.info(f"prompt: {prompt}")
    # logger.info(f"prompt: {prompt}")
    output = model.infer_one_image(src_image=image_source, src_prompt=prompt_source, target_prompt=prompt_target,
                                   instruct_prompt=prompt_instruct, seed=42)

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



