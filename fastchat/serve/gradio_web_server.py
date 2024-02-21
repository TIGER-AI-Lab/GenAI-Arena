"""
The gradio demo server for chatting with a single model.
"""

import argparse
from collections import defaultdict
import datetime
import json
import os
import random
import time
from PIL import Image
import numpy as np

import uuid

import gradio as gr
import requests

from fastchat.conversation import SeparatorStyle
from fastchat.constants import (
    LOGDIR,
    WORKER_API_TIMEOUT,
    ErrorCode,
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    SERVER_ERROR_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
    SESSION_EXPIRATION_TIME,
)
from fastchat.model.model_adapter import (
    get_conversation_template,
    ANTHROPIC_MODEL_LIST,
)
from fastchat.model.model_registry import get_model_info, model_info
from fastchat.serve.api_provider import (
    anthropic_api_stream_iter,
    openai_api_stream_iter,
    palm_api_stream_iter,
    init_palm_chat,
)
from fastchat.utils import (
    build_logger,
    moderation_filter,
    get_window_url_params_js,
    get_window_url_params_with_tos_js,
    parse_gradio_auth_creds,
)
from fastchat.serve.model_worker import ModelWorker
from fastchat.serve.base_model_worker import app
import uvicorn


logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "FastChat Client"}

# no_change_btn = gr.Button.update()
# enable_btn = gr.Button.update(interactive=True, visible=True)
# disable_btn = gr.Button.update(interactive=False)
# invisible_btn = gr.Button.update(interactive=False, visible=False)
enable_btn = gr.Button(interactive=True, visible=True)
disable_btn = gr.Button(interactive=False)
invisible_btn = gr.Button(interactive=False, visible=False)
no_change_btn = gr.Button(value="No Change", interactive=True, visible=True)
# enable_btn = gr.Button(value="Enable", interactive=True, visible=True)
# disable_btn = gr.Button(value="Disable", interactive=False)
# invisible_btn = gr.Button(value="Invisible", interactive=False, visible=False)

controller_url = None
enable_moderation = False

acknowledgment_md = """
### Acknowledgment
<div class="image-container">
    <p> Our code base is built upon <a href="https://github.com/lm-sys/FastChat" target="_blank">FastChat</a> and <a href="https://github.com/TIGER-AI-Lab/ImagenHub" target="_blank">ImagenHub</a></p>.
</div>
"""

ip_expiration_dict = defaultdict(lambda: 0)

# Information about custom OpenAI compatible API models.
# JSON file format:
# {
#     "vicuna-7b": {
#         "model_name": "vicuna-7b-v1.5",
#         "api_base": "http://8.8.8.55:5555/v1",
#         "api_key": "password"
#     },
# }
openai_compatible_models_info = {}


class State:
    def __init__(self, model_name):
        self.conv = get_conversation_template(model_name)
        self.conv_id = uuid.uuid4().hex
        self.skip_next = False
        self.model_name = model_name

        if model_name == "palm-2":
            # According to release note, "chat-bison@001" is PaLM 2 for chat.
            # https://cloud.google.com/vertex-ai/docs/release-notes#May_10_2023
            self.palm_chat = init_palm_chat("chat-bison@001")

    def to_gradio_chatbot(self):
        return self.conv.to_gradio_chatbot()

    def dict(self):
        base = self.conv.dict()
        base.update(
            {
                "conv_id": self.conv_id,
                "model_name": self.model_name,
            }
        )
        return base

class ImageState:
    def __init__(self, model_name):
        self.conv = get_conversation_template(model_name)
        self.conv_id = uuid.uuid4().hex
        self.skip_next = False
        self.model_name = model_name
        self.online_load = False
        self.prompt = None
        # self.conv = prompt
        self.output = None

        # if model_name == "palm-2":
        #     # According to release note, "chat-bison@001" is PaLM 2 for chat.
        #     # https://cloud.google.com/vertex-ai/docs/release-notes#May_10_2023
        #     self.palm_chat = init_palm_chat("chat-bison@001")

    # def to_gradio_chatbot(self):
    #     return self.conv.to_gradio_chatbot()

    def dict(self):
        base = {
                "conv_id": self.conv_id,
                "model_name": self.model_name,
                "online_load": self.online_load
                }
        return base


def set_global_vars(controller_url_, enable_moderation_):
    global controller_url, enable_moderation
    controller_url = controller_url_
    enable_moderation = enable_moderation_


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list(
    controller_url, register_openai_compatible_models, add_chatgpt, add_claude, add_palm, post_load=False
):
    if not post_load:
        if controller_url:
            ret = requests.post(controller_url + "/refresh_all_workers")
            assert ret.status_code == 200
            ret = requests.post(controller_url + "/list_models")
            models = ret.json()["models"]
        else:
            models = []

        # Add API providers
        if register_openai_compatible_models:
            global openai_compatible_models_info
            openai_compatible_models_info = json.load(
                open(register_openai_compatible_models)
            )
            models += list(openai_compatible_models_info.keys())

        if add_chatgpt:
            models += ["gpt-3.5-turbo", "gpt-3.5-turbo-1106"]
        if add_claude:
            models += ["claude-2.0", "claude-2.1", "claude-instant-1"]
        if add_palm:
            models += ["palm-2"]
        models = list(set(models))

        if "deluxe-chat-v1" in models:
            del models[models.index("deluxe-chat-v1")]
        if "deluxe-chat-v1.1" in models:
            del models[models.index("deluxe-chat-v1.1")]

        priority = {k: f"___{i:02d}" for i, k in enumerate(model_info)}
        models.sort(key=lambda x: priority.get(x, x))
    else:
        models = ['imagenhub_SD', 'imagenhub_SDXL', 'imagenhub_SSD']
    logger.info(f"Models: {models}")
    return models


def load_demo_single(models, url_params):
    logger.info("load_demo_single")
    selected_model = models[0] if len(models) > 0 else ""
    logger.info(f"url_params: {url_params}")
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            selected_model = model
    logger.info(f"selected_model: {selected_model}")

    dropdown_update = gr.Dropdown.update(
        choices=models, value=selected_model, visible=True
    )

    state = None
    return state, dropdown_update


def load_demo(url_params, request: gr.Request):
    global models

    ip = get_ip(request)
    logger.info(f"load_demo. ip: {ip}. params: {url_params}")
    ip_expiration_dict[ip] = time.time() + SESSION_EXPIRATION_TIME

    if args.model_list_mode == "reload":
        models = get_model_list(
            controller_url,
            args.register_openai_compatible_models,
            args.add_chatgpt,
            args.add_claude,
            args.add_palm,
        )

    return load_demo_single(models, url_params)


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, model_selector, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"upvote. ip: {ip}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"downvote. ip: {ip}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"flag. ip: {ip}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


# def regenerate(state, request: gr.Request):
#     ip = get_ip(request)
#     logger.info(f"regenerate. ip: {ip}")
#     state.conv.update_last_message(None)
#     return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5

def regenerate(state, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"regenerate. ip: {ip}")
    # state.conv.update_last_message(None)
    return (state, gr.Image(height=512, width=512), "") + (disable_btn,) * 5


# def clear_history(request: gr.Request):
#     ip = get_ip(request)
#     logger.info(f"clear_history. ip: {ip}")
#     state = None
#     return (state, gr.Image(height=512, width=512), "") + (disable_btn,) * 5

def clear_history(request: gr.Request):
    ip = get_ip(request)
    logger.info(f"clear_history. ip: {ip}")
    state = None
    return (state, None, "") + (disable_btn,) * 5


def get_ip(request: gr.Request):
    if "cf-connecting-ip" in request.headers:
        ip = request.headers["cf-connecting-ip"]
    else:
        ip = request.client.host
    return ip


def add_text(state, model_selector, text, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"add_text. ip: {ip}. len: {len(text)}")

    if state is None:
        state = ImageState(model_selector)

    # add for online
    state.online_load = False

    if len(text) <= 0:
        state.skip_next = True
        return (state, "", "") + (no_change_btn,) * 5

    flagged = moderation_filter(text, [state.model_name])
    if flagged:
        logger.info(f"violate moderation. ip: {ip}. text: {text}")
        # overwrite the original text
        text = MODERATION_MSG

    conv = state.conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {ip}. text: {text}")
        state.skip_next = True
        return (state, "", CONVERSATION_LIMIT_MSG) + (
            no_change_btn,
        ) * 5

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    logger.info("=====text====")
    logger.info(text)
    conv.append_message(conv.roles[0], text)
    # state.conv = conv
    # conv.append_message(conv.roles[1], None)
    return (state, gr.Image(height=512, width=512), text) + (disable_btn,) * 5

# def add_text(state, model_selector, text, request: gr.Request):
#     ip = get_ip(request)
#     logger.info(f"add_text. ip: {ip}. len: {len(text)}")
#
#     if state is None:
#         state = ImageState(model_selector)
#
#     if len(text) <= 0:
#         state.skip_next = True
#         return (state, "", "") + (no_change_btn,) * 5
#
#     flagged = moderation_filter(text, [state.model_name])
#     if flagged:
#         logger.info(f"violate moderation. ip: {ip}. text: {text}")
#         # overwrite the original text
#         text = MODERATION_MSG
#
#     # if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
#     #     logger.info(f"conversation turn limit. ip: {ip}. text: {text}")
#     #     state.skip_next = True
#     #     return (state, state.to_gradio_chatbot(), CONVERSATION_LIMIT_MSG) + (
#     #         no_change_btn,
#     #     ) * 5
#
#     text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
#     conv.append_message(conv.roles[0], text)
#     return (state, "", "") + (disable_btn,) * 5


def post_process_code(code):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)
    return code


def model_diffusion_worker_stream_iter(
    model_name,
    worker_addr,
    prompt,
):
    # Make requests
    gen_params = {
        "model": model_name,
        "prompt": prompt
    }
    logger.info(f"==== request ====\n{gen_params}")

    # Stream output
    response = requests.post(
        worker_addr + "/worker_generate_stream",
        headers=headers,
        json=gen_params,
        stream=True,
        timeout=WORKER_API_TIMEOUT,
    )

    logger.info(f"==== 2request2 ====\n{gen_params}")

    for chunk in response.iter_lines(decode_unicode=False, chunk_size=32768, delimiter=b"\0"):
        if chunk:
            logger.info(f"before")
            data = json.loads(chunk.decode())
            logger.info(f"after")
            yield data



def model_worker_stream_iter(
    conv,
    model_name,
    worker_addr,
    prompt,
    temperature,
    repetition_penalty,
    top_p,
    max_new_tokens,
):
    # Make requests
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "stop": conv.stop_str,
        "stop_token_ids": conv.stop_token_ids,
        "echo": False,
    }
    logger.info(f"==== request ====\n{gen_params}")

    # Stream output
    response = requests.post(
        worker_addr + "/worker_generate_stream",
        headers=headers,
        json=gen_params,
        stream=True,
        timeout=WORKER_API_TIMEOUT,
    )
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode())
            yield data


def load_model_worker(model_path):
    worker_id = str(uuid.uuid4())[:8]
    controller_addr = "http://localhost:21001"
    worker_addr = "http://localhost:21002"
    model_names = ''
    max_gpu_memory = ''
    host = "localhost"
    port = 21002
    limit_worker_concurrency = 10
    worker = ModelWorker(
        controller_addr,
        worker_addr,
        worker_id,
        model_path,
        model_names,
        limit_worker_concurrency,
        no_register=False,
        device='cuda',
        num_gpus=1,
        max_gpu_memory=max_gpu_memory
    )
    logger.info("before")
    # uvicorn.run(app, host=host, port=port, log_level="info")
    logger.info("after")
    return

def diffusion_response(state, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"diffusion_response. ip: {ip}")
    start_tstamp = time.time()
    conv, model_name = state.conv, state.model_name
    prompt = conv.messages[-1][1]
    logger.info(f'prompt message: {prompt}')
    if state.online_load:
        load_model_worker(model_name)

    logger.info(f'controller_url: {controller_url}')

    ret = requests.post(
        controller_url + "/get_worker_address", json={"model": model_name}
    )
    logger.info(f"ret: {ret}")

    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")


    # Construct prompt.
    # We need to call it here, so it will not be affected by "▌".
    # prompt = conv.get_prompt()

    # Set repetition_penalty
    # if "t5" in model_name:
    #     repetition_penalty = 1.2
    # else:
    #     repetition_penalty = 1.0

    stream_iter = model_diffusion_worker_stream_iter(
        model_name,
        worker_addr,
        prompt
    )

    # try:
    for i, data in enumerate(stream_iter):
        error_code = data["error_code"]
        logger.info(f"error_code: {error_code}")
        if error_code == 0:
            logger.info(f"yes")
            # if data:
            grey = np.array(data['text'])
            logger.info(f"grey.shape: {grey.shape}")
            output = Image.fromarray(grey.astype(np.uint8))
            output = output.resize((512, 512))
            logger.info(f"output.size {output.size}")
            # yield (state, output) + (enable_btn,) * 5
        else:
            output = data + f"\n\n(error_code: {data['error_code']})"
            state.output = output
            yield (state, output) + (
                disable_btn,
                disable_btn,
                disable_btn,
                enable_btn,
                enable_btn,
            )
            return
    # output = data
    # if "vicuna" in model_name:
    #     output = post_process_code(output)
    state.output = output
    # yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5
    yield (state, output) + (enable_btn,) * 5
    # except requests.exceptions.RequestException as e:
    #     # conv.update_last_message(
    #     #     f"{SERVER_ERROR_MSG}\n\n"
    #     #     f"(error_code: {ErrorCode.GRADIO_REQUEST_ERROR}, {e})"
    #     # )
    #     logger.error(f"{SERVER_ERROR_MSG}\n\n"
    #                  f"(error_code: {ErrorCode.GRADIO_REQUEST_ERROR}, {e})")
    #     yield (state, "") + (
    #         disable_btn,
    #         disable_btn,
    #         disable_btn,
    #         enable_btn,
    #         enable_btn,
    #     )
    #     return
    # except Exception as e:
    #     logger.error(
    #         f"{SERVER_ERROR_MSG}\n\n"
    #         f"(error_code: {ErrorCode.GRADIO_STREAM_UNKNOWN_ERROR}, {e})"
    #     )
    #     yield (state, "") + (
    #         disable_btn,
    #         disable_btn,
    #         disable_btn,
    #         enable_btn,
    #         enable_btn,
    #     )
    #     return

    finish_tstamp = time.time()
    # logger.info(f"===output===: {output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "gen_params": {},
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")


def bot_response(state, temperature, top_p, max_new_tokens, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"bot_response. ip: {ip}")
    start_tstamp = time.time()
    temperature = float(temperature)
    top_p = float(top_p)
    max_new_tokens = int(max_new_tokens)

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        state.skip_next = False
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    conv, model_name = state.conv, state.model_name
    if model_name in openai_compatible_models_info:
        model_info = openai_compatible_models_info[model_name]
        prompt = conv.to_openai_api_messages()
        stream_iter = openai_api_stream_iter(
            model_info["model_name"],
            prompt,
            temperature,
            top_p,
            max_new_tokens,
            api_base=model_info["api_base"],
            api_key=model_info["api_key"],
        )
    elif model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo-1106"]:
        # avoid conflict with Azure OpenAI
        assert model_name not in openai_compatible_models_info
        prompt = conv.to_openai_api_messages()
        stream_iter = openai_api_stream_iter(
            model_name, prompt, temperature, top_p, max_new_tokens
        )
    elif model_name in ANTHROPIC_MODEL_LIST:
        prompt = conv.get_prompt()
        stream_iter = anthropic_api_stream_iter(
            model_name, prompt, temperature, top_p, max_new_tokens
        )
    elif model_name == "palm-2":
        stream_iter = palm_api_stream_iter(
            state.palm_chat, conv.messages[-2][1], temperature, top_p, max_new_tokens
        )
    else:
        # Query worker address
        ret = requests.post(
            controller_url + "/get_worker_address", json={"model": model_name}
        )
        worker_addr = ret.json()["address"]
        logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

        # No available worker
        if worker_addr == "":
            conv.update_last_message(SERVER_ERROR_MSG)
            yield (
                state,
                state.to_gradio_chatbot(),
                disable_btn,
                disable_btn,
                disable_btn,
                enable_btn,
                enable_btn,
            )
            return

        # Construct prompt.
        # We need to call it here, so it will not be affected by "▌".
        prompt = conv.get_prompt()

        # Set repetition_penalty
        if "t5" in model_name:
            repetition_penalty = 1.2
        else:
            repetition_penalty = 1.0

        stream_iter = model_worker_stream_iter(
            conv,
            model_name,
            worker_addr,
            prompt,
            temperature,
            repetition_penalty,
            top_p,
            max_new_tokens,
        )

    conv.update_last_message("▌")
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        for i, data in enumerate(stream_iter):
            if data["error_code"] == 0:
                output = data["text"].strip()
                conv.update_last_message(output + "▌")
                yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
            else:
                output = data["text"] + f"\n\n(error_code: {data['error_code']})"
                conv.update_last_message(output)
                yield (state, state.to_gradio_chatbot()) + (
                    disable_btn,
                    disable_btn,
                    disable_btn,
                    enable_btn,
                    enable_btn,
                )
                return
        output = data["text"].strip()
        if "vicuna" in model_name:
            output = post_process_code(output)
        conv.update_last_message(output)
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5
    except requests.exceptions.RequestException as e:
        conv.update_last_message(
            f"{SERVER_ERROR_MSG}\n\n"
            f"(error_code: {ErrorCode.GRADIO_REQUEST_ERROR}, {e})"
        )
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return
    except Exception as e:
        conv.update_last_message(
            f"{SERVER_ERROR_MSG}\n\n"
            f"(error_code: {ErrorCode.GRADIO_STREAM_UNKNOWN_ERROR}, {e})"
        )
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "gen_params": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
            },
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")


block_css = """
#notice_markdown {
    font-size: 110%
}
#notice_markdown th {
    display: none;
}
#notice_markdown td {
    padding-top: 6px;
    padding-bottom: 6px;
}
#model_description_markdown {
    font-size: 110%
}
#leaderboard_markdown {
    font-size: 110%
}
#leaderboard_markdown td {
    padding-top: 6px;
    padding-bottom: 6px;
}
#leaderboard_dataframe td {
    line-height: 0.1em;
}
#about_markdown {
    font-size: 110%
}
#ack_markdown {
    font-size: 110%
}
#input_box textarea {
}
footer {
    display:none !important
}
.image-container {
    display: flex;
    align-items: center;
    padding: 1px;
}
.image-container img {
    margin: 0 30px;
    height: 30px;
    max-height: 100%;
    width: auto;
    max-width: 30%;
}
.image-about img {
    margin: 0 30px;
    margin-top:  30px;
    height: 60px;
    max-height: 100%;
    width: auto;
    float: left;
}
.input-image, .image-preview {
    margin: 0 30px;
    height: 30px;
    max-height: 100%;
    width: auto;
    max-width: 30%;}
"""


def get_model_description_md(models):
    model_description_md = """
| | | |
| ---- | ---- | ---- |
"""
    ct = 0
    visited = set()
    for i, name in enumerate(models):
        minfo = get_model_info(name)
        if minfo.simple_name in visited:
            continue
        visited.add(minfo.simple_name)
        one_model_md = f"[{minfo.simple_name}]({minfo.link}): {minfo.description}"

        if ct % 3 == 0:
            model_description_md += "|"
        model_description_md += f" {one_model_md} |"
        if ct % 3 == 2:
            model_description_md += "\n"
        ct += 1
    return model_description_md


def build_about():
    about_markdown = f"""
# About Us
This is a project from TIGER Lab at University of Waterloo. 

## Contributors:
[Tianle Li](https://scholar.google.com/citations?user=g213g7YAAAAJ&hl=en), [Dongfu Jiang](https://jdf-prog.github.io/), Yuansheng Ni.

## Contact:
Email: t29li@uwaterloo.ca (Tianle Li)

## Advisors
[Wenhu Chen](https://wenhuchen.github.io/)

## Sponsorship
We are keep looking for sponsorship to support the arena project for the long term. Please contact us if you are interested in supporting this project.
"""

    # state = gr.State()
    gr.Markdown(about_markdown, elem_id="about_markdown")

    # return [state]



def build_single_model_ui(models, add_promotion_links=False, image_editing_task=False):
    promotion = (
        """
- | [GitHub](https://github.com/TIGER-AI-Lab/ImagenHub) | [Paper](https://arxiv.org/abs/2310.01596) | [Dataset](https://huggingface.co/ImagenHub) |
"""
        if add_promotion_links
        else ""
    )

    notice_markdown = f"""
# 🏔️ Play with Image Generation Models
{promotion}

## 🤖 Choose any model to generate
"""
    state = gr.State()
    gr.Markdown(notice_markdown, elem_id="notice_markdown")
    # with gr.Group(elem_id="share-region-named"):
    with gr.Box(elem_id="share-region-named"):
        with gr.Row(elem_id="model_selector_row"):
            model_selector = gr.Dropdown(
                choices=models,
                value=models[0] if len(models) > 0 else "",
                interactive=True,
                show_label=False,
                container=False,
            )
            logger.info(f"model_selector: {model_selector}")
        with gr.Row():
            with gr.Accordion(
                "🔍 Expand to see all model descriptions",
                open=False,
                elem_id="model_description_accordion",
            ):
                model_description_md = get_model_description_md(models)
                gr.Markdown(model_description_md, elem_id="model_description_markdown")

    with gr.Row():

        if not image_editing_task:
            textbox = gr.Textbox(
                show_label=False,
                placeholder="👉 Enter your prompt and press ENTER",
                elem_id="input_box",
            )

        send_btn = gr.Button(value="Send", variant="primary", scale=0)

    # if image_editing_task:
    #     source_image = gr.Image(height=512, width=512, type="pil")

    with gr.Row():
        chatbot = gr.Image(height=512, width=512, type="pil")

    with gr.Row() as button_row:
        upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
        downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
        flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    # with gr.Accordion("Parameters", open=False) as parameter_row:
    #     temperature = gr.Slider(
    #         minimum=0.0,
    #         maximum=1.0,
    #         value=0.7,
    #         step=0.1,
    #         interactive=True,
    #         label="Temperature",
    #     )
    #     top_p = gr.Slider(
    #         minimum=0.0,
    #         maximum=1.0,
    #         value=1.0,
    #         step=0.1,
    #         interactive=True,
    #         label="Top P",
    #     )
    #     max_output_tokens = gr.Slider(
    #         minimum=16,
    #         maximum=1024,
    #         value=512,
    #         step=64,
    #         interactive=True,
    #         label="Max output tokens",
    #     )

    if add_promotion_links:
        gr.Markdown(acknowledgment_md, elem_id="ack_markdown")

    gr.Examples(
        examples=[
            ["a cute dog is playing a ball", os.path.join("./examples", "dog.jpg")],
            ["Buildings on fire, old film still",
             os.path.join("./examples", "fire.jpg")],
            ["Lonely evil bananas on a table, hard light chiaroscuro, realistic",
             os.path.join("./examples", "banana.jpg")],
            ["A futuristic hopeful busy city, purple and green color scheme",
             os.path.join("./examples", "city.jpg")]],
        inputs=[textbox, chatbot])

    # Register listeners
    btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
    upvote_btn.click(
        upvote_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    downvote_btn.click(
        downvote_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    flag_btn.click(
        flag_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    # regenerate_btn.click(regenerate, state, [state, chatbot, textbox] + btn_list).then(
    #     bot_response,
    #     [state, temperature, top_p, max_output_tokens],
    #     [state, chatbot] + btn_list,
    # )

    regenerate_btn.click(regenerate, state, [state, chatbot, textbox] + btn_list).then(
        diffusion_response,
        [state],
        [state, chatbot] + btn_list,
    )
    clear_btn.click(clear_history, None, [state, chatbot, textbox] + btn_list)

    model_selector.change(clear_history, None, [state, chatbot, textbox] + btn_list)

    # textbox.submit(
    #     add_text, [state, model_selector, textbox], [state, chatbot, textbox] + btn_list
    # ).then(
    #     bot_response,
    #     [state, temperature, top_p, max_output_tokens],
    #     [state, chatbot] + btn_list,
    # )
    textbox.submit(
        add_text, [state, model_selector, textbox], [state, chatbot, textbox] + btn_list
    ).then(
        diffusion_response,
        [state],
        [state, chatbot] + btn_list,
    )
    # send_btn.click(
    #     add_text,
    #     [state, model_selector, textbox],
    #     [state, chatbot, textbox] + btn_list,
    # ).then(
    #     bot_response,
    #     [state, temperature, top_p, max_output_tokens],
    #     [state, chatbot] + btn_list,
    # )
    send_btn.click(
        add_text,
        [state, model_selector, textbox],
        [state, chatbot, textbox] + btn_list,
    ).then(
        diffusion_response,
        [state],
        [state, chatbot] + btn_list,
    )

    return [state, model_selector]


def build_demo(models):
    with gr.Blocks(
        title="Chat with Open Large Language Models",
        theme=gr.themes.Default(),
        css=block_css,
    ) as demo:
        logger.info("begin")
        logger.info(f"======models=====: {models}")

        state, model_selector = build_single_model_ui(models, add_promotion_links=True)

        if args.model_list_mode not in ["once", "reload"]:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

        if args.show_terms_of_use:
            load_js = get_window_url_params_with_tos_js
        else:
            load_js = get_window_url_params_js

        url_params = gr.JSON(visible=True)

        logger.info(f"======url_params=====: {url_params}")

        demo.load(
            load_demo,
            [url_params],
            [
                state,
                model_selector,
            ],
            _js=load_js,
        )
        logger.info(f"======url_params=====: {url_params}")
        logger.info("end")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Whether to generate a public, shareable link",
    )
    parser.add_argument(
        "--controller-url",
        type=str,
        default="http://localhost:21001",
        help="The address of the controller",
    )
    parser.add_argument(
        "--concurrency-count",
        type=int,
        default=10,
        help="The concurrency count of the gradio queue",
    )
    parser.add_argument(
        "--model-list-mode",
        type=str,
        default="once",
        choices=["once", "reload"],
        help="Whether to load the model list once or reload the model list every time",
    )
    parser.add_argument(
        "--moderate",
        action="store_true",
        help="Enable content moderation to block unsafe inputs",
    )
    parser.add_argument(
        "--show-terms-of-use",
        action="store_true",
        help="Shows term of use before loading the demo",
    )
    parser.add_argument(
        "--add-chatgpt",
        action="store_true",
        help="Add OpenAI's ChatGPT models (gpt-3.5-turbo, gpt-4)",
    )
    parser.add_argument(
        "--add-claude",
        action="store_true",
        help="Add Anthropic's Claude models (claude-2, claude-instant-1)",
    )
    parser.add_argument(
        "--add-palm",
        action="store_true",
        help="Add Google's PaLM model (PaLM 2 for Chat: chat-bison@001)",
    )
    parser.add_argument(
        "--register-openai-compatible-models",
        type=str,
        help="Register custom OpenAI API compatible models by loading them from a JSON file",
    )
    parser.add_argument(
        "--gradio-auth-path",
        type=str,
        help='Set the gradio authentication file path. The file should contain one or more user:password pairs in this format: "u1:p1,u2:p2,u3:p3"',
    )
    parser.add_argument(
        "--online-load",
        action="store_true",
        help="Whether to load the model online",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # Set global variables
    set_global_vars(args.controller_url, args.moderate)
    models = get_model_list(
        args.controller_url,
        args.register_openai_compatible_models,
        args.add_chatgpt,
        args.add_claude,
        args.add_palm,
        post_load=args.online_load
    )

    # Set authorization credentials
    auth = None
    if args.gradio_auth_path is not None:
        auth = parse_gradio_auth_creds(args.gradio_auth_path)

    # Launch the demo
    demo = build_demo(models)
    # concurrency_count=args.concurrency_count,
    demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=200,
        auth=auth,
    )
