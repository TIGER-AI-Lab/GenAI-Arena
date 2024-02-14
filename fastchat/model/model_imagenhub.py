import gc
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

logger = build_logger("diffusion_infer", 'diffusion_infer.log')

@torch.inference_mode()
def generate_stream_imagen(
    model,
    tokenizer,
    params,
    device,
    context_len=256,
    stream_interval=2,
):
    prompt = params["prompt"]
    encoding = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = encoding.input_ids
    # encoding["decoder_input_ids"] = encoding["input_ids"].clone()
    input_echo_len = len(input_ids)
    #
    # generation_config = GenerationConfig(
    #     max_new_tokens=max_new_tokens,
    #     do_sample=temperature >= 1e-5,
    #     temperature=temperature,
    #     repetition_penalty=repetition_penalty,
    #     no_repeat_ngram_size=10,
    #     top_p=top_p,
    #     top_k=top_k,
    #     eos_token_id=stop_token_ids,
    # )
    #
    # class CodeBlockStopper(StoppingCriteria):
    #     def __call__(
    #         self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    #     ) -> bool:
    #         # Code-completion is open-end generation.
    #         # We check \n\n to stop at end of a code block.
    #         if list(input_ids[0][-2:]) == [628, 198]:
    #             return True
    #         return False

    # gen_kwargs = dict(
    #     **encoding,
    #     streamer=streamer,
    #     generation_config=generation_config,
    #     stopping_criteria=StoppingCriteriaList([CodeBlockStopper()]),
    # )
    # generation_kwargs = {"prompt": prompt}
    #
    # model.pipe.scheduler = DDIMScheduler.from_config(model.pipe.scheduler.config)
    # thread = Thread(target=model.infer_one_image, kwargs=generation_kwargs)
    # thread.start()
    # i = 0
    # output = ""
    # for new_text in streamer:
    #     i += 1
    #     output += new_text
    #     if i % stream_interval == 0 or i == max_new_tokens - 1:
    #         yield {
    #             "text": output,
    #             "usage": {
    #                 "prompt_tokens": input_echo_len,
    #                 "completion_tokens": i,
    #                 "total_tokens": input_echo_len + i,
    #             },
    #             "finish_reason": None,
    #         }
    #     if i >= max_new_tokens:
    #         break
    #
    # if i >= max_new_tokens:
    #     finish_reason = "length"
    # else:
    #     finish_reason = "stop"
    logger.info(f"prompt: {prompt}")
    logger.info(f"model.scheduler: {model.pipe.scheduler}")
    logger.info(f"model.type: {type(model)}")
    # logger.info(f"prompt: {prompt}")
    output = model.infer_one_image(prompt=prompt, seed=42)

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



