"""
Chatbot Arena (side-by-side) tab.
Users chat with two chosen models.
"""

import json
import time
import os

import gradio as gr
import numpy as np

from fastchat.constants import (
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
)
from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.gradio_web_image_editing_server import (
    ImageState,
    diffusion_response,
    bot_response,
    get_conv_log_filename,
    no_change_btn,
    enable_btn,
    disable_btn,
    invisible_btn,
    acknowledgment_md,
    get_model_description_md,
    ip_expiration_dict,
    get_ip,
)
from fastchat.utils import (
    build_logger,
    moderation_filter,
)


logger = build_logger("gradio_web_server_multi_ie", "gradio_web_server_multi_ie.log")

num_sides = 2
enable_moderation = False


def set_global_vars_named_ie(enable_moderation_):
    global enable_moderation
    global enable_moderation
    enable_moderation = enable_moderation_


def load_demo_side_by_side_named_ie(models, url_params):
    logger.info("load_demo_side_by_side_named_ie")
    states = (None,) * num_sides

    model_left = models[0] if len(models) > 0 else ""
    if len(models) > 1:
        weights = ([8] * 4 + [4] * 8 + [1] * 32)[: len(models) - 1]
        weights = weights / np.sum(weights)
        model_right = np.random.choice(models[1:], p=weights)
    else:
        model_right = model_left

    selector_updates = (
        gr.Dropdown.update(choices=models, value=model_left, visible=True),
        gr.Dropdown.update(choices=models, value=model_right, visible=True),
    )

    return states + selector_updates


def vote_last_response(states, vote_type, model_selectors, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "models": [x for x in model_selectors],
            "states": [x.dict() for x in states],
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
    for state in states:
        output_file = f'/ML-A100/team/mm/zhangge/FastChat/image_results/edition/{state.conv_id}_{state.model_name}.jpg'
        source_file = f'/ML-A100/team/mm/zhangge/FastChat/image_results/edition/{state.conv_id}_{state.model_name}_source.jpg'
        with open(output_file, 'w') as f:
            state.output.save(f, 'JPEG')
        with open(source_file, 'w') as sf:
            state.conv[2].save(sf, 'JPEG')


def leftvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"leftvote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "leftvote", [model_selector0, model_selector1], request
    )
    return ("", "", gr.Image(height=512, width=512, type="pil"), "") + (disable_btn,) * 4


def rightvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"rightvote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "rightvote", [model_selector0, model_selector1], request
    )
    return ("", "", gr.Image(height=512, width=512, type="pil"), "") + (disable_btn,) * 4


def tievote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"tievote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "tievote", [model_selector0, model_selector1], request
    )
    return ("", "", gr.Image(height=512, width=512, type="pil"), "") + (disable_btn,) * 4


def bothbad_vote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"bothbad_vote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "bothbad_vote", [model_selector0, model_selector1], request
    )
    return ("", "", gr.Image(height=512, width=512, type="pil"), "") + (disable_btn,) * 4


# def regenerate(state0, state1, request: gr.Request):
#     logger.info(f"regenerate (named). ip: {get_ip(request)}")
#     states = [state0, state1]
#     for i in range(num_sides):
#         states[i].conv.update_last_message(None)
#     return states + [x.to_gradio_chatbot() for x in states] + [""] + [disable_btn] * 6

def regenerate(state0, state1, request: gr.Request):
    logger.info(f"regenerate (anony). ip: {get_ip(request)}")
    states = [state0, state1]
    # for i in range(num_sides):
    #     states[i].conv.update_last_message(None)
    return states + [gr.Image(height=512, width=512) for x in states] + ["", "", gr.Image(height=512, width=512, type="pil"), ""] + [disable_btn] * 6


def clear_history(request: gr.Request):
    logger.info(f"clear_history (named). ip: {get_ip(request)}")
    return (
        [None] * num_sides
        + [None] * num_sides
        + ["", "", None, ""]
        + [invisible_btn] * 4
        + [disable_btn] * 2
    )


def share_click(state0, state1, model_selector0, model_selector1, request: gr.Request):
    logger.info(f"share (named). ip: {get_ip(request)}")
    if state0 is not None and state1 is not None:
        vote_last_response(
            [state0, state1], "share", [model_selector0, model_selector1], request
        )


def add_text(
    state0, state1, model_selector0, model_selector1, text_source, text_target, image_source, text_instruct, request: gr.Request
):
    ip = get_ip(request)
    logger.info(f"add_text (anony). ip: {ip}. len: {len(text_source)}")
    states = [state0, state1]
    model_selectors = [model_selector0, model_selector1]

    # Init states if necessary
    if states[0] is None:
        assert states[1] is None

        for i in range(num_sides):
            if states[i] is None:
                states[i] = ImageState(model_selectors[i])

    if len(text_source) <= 0:
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [gr.Image(height=512, width=512) for x in states]
            + ["", "", gr.Image(height=512, width=512, type="pil"), ""]
            + [
                no_change_btn,
            ]
            * 6
        )


    conv = states[0].conv

    text_source = text_source[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    text_target = text_target[:INPUT_CHAR_LEN_LIMIT]
    text_instruct = text_instruct[:INPUT_CHAR_LEN_LIMIT]
    for i in range(num_sides):
        states[i].conv = []
        states[i].conv.append(text_source)
        states[i].conv.append(text_target)
        states[i].conv.append(image_source)
        states[i].conv.append(text_instruct)
        # states[i].conv.append_message(states[i].conv.roles[1], None)
        states[i].skip_next = False

    return (
        states
        + [gr.Image(height=512, width=512) for x in states]
        + [text_source, text_target, image_source, text_instruct]
        + [
            disable_btn,
        ]
        * 6
    )

# def add_text(
#     state0, state1, model_selector0, model_selector1, text, request: gr.Request
# ):
#     ip = get_ip(request)
#     logger.info(f"add_text (named). ip: {ip}. len: {len(text)}")
#     states = [state0, state1]
#     model_selectors = [model_selector0, model_selector1]
#
#     # Init states if necessary
#     for i in range(num_sides):
#         if states[i] is None:
#             states[i] = State(model_selectors[i])
#
#     if len(text) <= 0:
#         for i in range(num_sides):
#             states[i].skip_next = True
#         return (
#             states
#             + [x.to_gradio_chatbot() for x in states]
#             + [""]
#             + [
#                 no_change_btn,
#             ]
#             * 6
#         )
#
#     model_list = [states[i].model_name for i in range(num_sides)]
#     flagged = moderation_filter(text, model_list)
#     if flagged:
#         logger.info(f"violate moderation (named). ip: {ip}. text: {text}")
#         # overwrite the original text
#         text = MODERATION_MSG
#
#     conv = states[0].conv
#     if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
#         logger.info(f"conversation turn limit. ip: {ip}. text: {text}")
#         for i in range(num_sides):
#             states[i].skip_next = True
#         return (
#             states
#             + [x.to_gradio_chatbot() for x in states]
#             + [CONVERSATION_LIMIT_MSG]
#             + [
#                 no_change_btn,
#             ]
#             * 6
#         )
#
#     text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
#     for i in range(num_sides):
#         states[i].conv.append_message(states[i].conv.roles[0], text)
#         states[i].conv.append_message(states[i].conv.roles[1], None)
#         states[i].skip_next = False
#
#     return (
#         states
#         + [x.to_gradio_chatbot() for x in states]
#         + [""]
#         + [
#             disable_btn,
#         ]
#         * 6
#     )

def diffusion_response_multi(
    state0,
    state1,
    request: gr.Request,
):
    logger.info(f"bot_response_multi (anony). ip: {get_ip(request)}")

    if state0 is None or state0.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (
            state0,
            state1,
            gr.Image(height=512, width=512),
            gr.Image(height=512, width=512),
        ) + (no_change_btn,) * 6
        return

    states = [state0, state1]
    gen = []
    for i in range(num_sides):
        gen.append(
            diffusion_response(
                states[i],
                request,
            )
        )

    chatbots = [None] * num_sides
    while True:
        stop = True
        for i in range(num_sides):
            try:
                ret = next(gen[i])
                states[i], chatbots[i] = ret[0], ret[1]
                stop = False
            except StopIteration:
                pass
        yield states + chatbots + [disable_btn] * 6
        if stop:
            break


# def bot_response_multi(
#     state0,
#     state1,
#     temperature,
#     top_p,
#     max_new_tokens,
#     request: gr.Request,
# ):
#     logger.info(f"bot_response_multi (named). ip: {get_ip(request)}")
#
#     if state0.skip_next:
#         # This generate call is skipped due to invalid inputs
#         yield (
#             state0,
#             state1,
#             state0.to_gradio_chatbot(),
#             state1.to_gradio_chatbot(),
#         ) + (no_change_btn,) * 6
#         return
#
#     states = [state0, state1]
#     gen = []
#     for i in range(num_sides):
#         gen.append(
#             bot_response(
#                 states[i],
#                 temperature,
#                 top_p,
#                 max_new_tokens,
#                 request,
#             )
#         )
#
#     chatbots = [None] * num_sides
#     while True:
#         stop = True
#         for i in range(num_sides):
#             try:
#                 ret = next(gen[i])
#                 states[i], chatbots[i] = ret[0], ret[1]
#                 stop = False
#             except StopIteration:
#                 pass
#         yield states + chatbots + [disable_btn] * 6
#         if stop:
#             break


def flash_buttons():
    btn_updates = [
        [disable_btn] * 4 + [enable_btn] * 2,
        [enable_btn] * 6,
    ]
    for i in range(4):
        yield btn_updates[i % 2]
        time.sleep(0.5)


def build_side_by_side_ui_named_ie(models):
    logger.info("build_side_by_side_ui_named")
    notice_markdown = """
# ⚔️ GenAI-Arena ⚔️ : Benchmarking Visual Generative Models in the Wild
| [GitHub](https://github.com/TIGER-AI-Lab/ImagenHub) | [Paper](https://arxiv.org/abs/2310.01596) | [Dataset](https://huggingface.co/ImagenHub) | 

## 📜 Rules
- Edit with two selected models side-by-side and vote!
- Upload a source image that you want to edit.
- Input source prompt, target prompt and instruct prompt.
- Wait to see the results after edition.
- Click "Clear history" to start a new round.
- The model could output a totally black image or noise. It's not a bug but the failure case of the model.

## 🤖 Choose two models to compare
"""

    states = [gr.State() for _ in range(num_sides)]
    model_selectors = [None] * num_sides
    chatbots = [None] * num_sides

    notice = gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Group(elem_id="share-region-named"):
        with gr.Row():
            for i in range(num_sides):
                with gr.Column():
                    model_selectors[i] = gr.Dropdown(
                        choices=models,
                        value=models[i] if len(models) > i else "",
                        interactive=True,
                        show_label=False,
                        container=False,
                    )
        with gr.Row():
            with gr.Accordion("🔍 Expand to see all model descriptions", open=False):
                model_description_md = get_model_description_md(models)
                gr.Markdown(model_description_md, elem_id="model_description_markdown")

        with gr.Row():
            for i in range(num_sides):
                label = "Model A" if i == 0 else "Model B"
                with gr.Column():
                    chatbots[i] = gr.Image(height=512, width=512)

        with gr.Row():
            leftvote_btn = gr.Button(
                value="👈  A is better", visible=False, interactive=False
            )
            rightvote_btn = gr.Button(
                value="👉  B is better", visible=False, interactive=False
            )
            tie_btn = gr.Button(value="🤝  Tie", visible=False, interactive=False)
            bothbad_btn = gr.Button(
                value="👎  Both are bad", visible=False, interactive=False
            )

    # with gr.Row():
    #     textbox = gr.Textbox(
    #         show_label=False,
    #         placeholder="👉 Enter your prompt and press ENTER",
    #         elem_id="input_box",
    #     )
    #     send_btn = gr.Button(value="Send", variant="primary", scale=0)

    with gr.Row():

        textbox_source = gr.Textbox(
            show_label=False,
            placeholder="👉 Enter your source prompt",
            elem_id="input_box_s",
        )
        textbox_target = gr.Textbox(
            show_label=False,
            placeholder="👉 Enter your target prompt",
            elem_id="input_box_t",
        )
        textbox_instruct = gr.Textbox(
            show_label=False,
            placeholder="👉 Enter your instruct prompt",
            elem_id="input_box_t",
        )

    # if image_editing_task:
    with gr.Row():
        source_image = gr.Image(type="pil")
        send_btn = gr.Button(value="Send", variant="primary", scale=0)

    with gr.Row() as button_row:
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        share_btn = gr.Button(value="📷  Share")

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

    gr.Markdown(acknowledgment_md, elem_id="ack_markdown")
    gr.Examples(
        examples=[
            ["a bowl of strawberries", "a bowl of oranges", "change strawberries to oranges",
             os.path.join("./examples", "strawberries.jpg")],
            ["a pig is eating an ice cream", "a rabbit is eating an ice cream", "change pig to rabbit",
             os.path.join("./examples", "pig.jpg")],
            ["a rubber duck in a swimming pool", "a rubber duck with a hat in a swimming pool", "add a hat to the duck",
             os.path.join("./examples", "duck.jpg")],
            ["a photo of a cat", "a photo of a mouse", "change cat to mouse",
             os.path.join("./examples", "cat.jpeg")]],
        inputs=[textbox_source, textbox_target, textbox_instruct, source_image])
    # Register listeners
    btn_list = [
        leftvote_btn,
        rightvote_btn,
        tie_btn,
        bothbad_btn,
        regenerate_btn,
        clear_btn,
    ]
    leftvote_btn.click(
        leftvote_last_response,
        states + model_selectors,
        [textbox_source, textbox_target, source_image, textbox_instruct, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    rightvote_btn.click(
        rightvote_last_response,
        states + model_selectors,
        [textbox_source, textbox_target, source_image, textbox_instruct, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    tie_btn.click(
        tievote_last_response,
        states + model_selectors,
        [textbox_source, textbox_target, source_image, textbox_instruct, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    bothbad_btn.click(
        bothbad_vote_last_response,
        states + model_selectors,
        [textbox_source, textbox_target, source_image, textbox_instruct, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    regenerate_btn.click(
        regenerate, states, states + chatbots + [textbox_source, textbox_target, source_image, textbox_instruct] + btn_list
    ).then(
        diffusion_response_multi,
        states,
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )
    clear_btn.click(clear_history, None, states + chatbots + [textbox_source, textbox_target, source_image, textbox_instruct] + btn_list)

    share_js = """
function (a, b, c, d) {
    const captureElement = document.querySelector('#share-region-named');
    html2canvas(captureElement)
        .then(canvas => {
            canvas.style.display = 'none'
            document.body.appendChild(canvas)
            return canvas
        })
        .then(canvas => {
            const image = canvas.toDataURL('image/png')
            const a = document.createElement('a')
            a.setAttribute('download', 'chatbot-arena.png')
            a.setAttribute('href', image)
            a.click()
            canvas.remove()
        });
    return [a, b, c, d];
}
"""
    share_btn.click(share_click, states + model_selectors, [], _js=share_js)

    for i in range(num_sides):
        model_selectors[i].change(
            clear_history, None, states + chatbots + [textbox_source, textbox_target, source_image, textbox_instruct] + btn_list
        )

    # textbox.submit(
    #     add_text,
    #     states + model_selectors + [textbox],
    #     states + chatbots + [textbox] + btn_list,
    # ).then(
    #     diffusion_response_multi,
    #     states,
    #     states + chatbots + btn_list,
    # ).then(
    #     flash_buttons, [], btn_list
    # )

    send_btn.click(
        add_text,
        states + model_selectors + [textbox_source, textbox_target, source_image, textbox_instruct],
        states + chatbots + [textbox_source, textbox_target, source_image, textbox_instruct] + btn_list,
    ).then(
        diffusion_response_multi,
        states,
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )

    return states + model_selectors
