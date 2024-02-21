"""
The gradio demo server with multiple tabs.
It supports chatting with a single model or chatting with two models side-by-side.
"""

import argparse
import pickle
import time
from pathlib import Path
import gradio as gr

from fastchat.constants import (
    SESSION_EXPIRATION_TIME,
)
from fastchat.serve.gradio_block_arena_anony_ie import (
    build_side_by_side_ui_anony_ie,
    load_demo_side_by_side_anony_ie,
    set_global_vars_anony_ie,
)
from fastchat.serve.gradio_block_arena_named_ie import (
    build_side_by_side_ui_named_ie,
    load_demo_side_by_side_named_ie,
    set_global_vars_named_ie,
)

from fastchat.serve.gradio_block_arena_anony import (
    build_side_by_side_ui_anony,
    load_demo_side_by_side_anony,
    set_global_vars_anony,
)
from fastchat.serve.gradio_block_arena_named import (
    build_side_by_side_ui_named,
    load_demo_side_by_side_named,
    set_global_vars_named,
)
from fastchat.serve.gradio_web_image_editing_server import (
    set_global_vars_ie,
    block_css,
    build_single_model_ui_ie,
    build_about,
    get_model_list,
    load_demo_single_ie,
    ip_expiration_dict,
    get_ip,
)
from fastchat.serve.gradio_web_server import load_demo_single, build_single_model_ui, set_global_vars
from fastchat.serve.monitor.monitor import build_leaderboard_tab
from fastchat.utils import (
    build_logger,
    get_window_url_params_js,
    get_window_url_params_with_tos_js,
    parse_gradio_auth_creds,
)
import pdb

logger = build_logger("gradio_web_server_multi_ie", "gradio_web_server_multi_ie.log")


def load_demo(url_params, request: gr.Request):
    logger.info("load_demo_multi")
    global models

    ip = get_ip(request)
    logger.info(f"load_demo. ip: {ip}. params: {url_params}")
    ip_expiration_dict[ip] = time.time() + SESSION_EXPIRATION_TIME

    selected = 0
    if "arena" in url_params:
        selected = 0
    elif "compare" in url_params:
        selected = 1
    elif "single" in url_params:
        selected = 2
    elif "leaderboard" in url_params:
        selected = 3

    if args.model_list_mode == "reload":
        if args.anony_only_for_proprietary_model:
            models = get_model_list(
                args.controller_url,
                args.register_openai_compatible_models,
                False,
                False,
                False,
            )
        else:
            models = get_model_list(
                args.controller_url,
                args.register_openai_compatible_models,
                args.add_chatgpt,
                args.add_claude,
                args.add_palm,
            )

    single_updates = load_demo_single(models, url_params)

    models_anony = list(models)
    if args.anony_only_for_proprietary_model:
        # Only enable these models in anony battles.
        if args.add_chatgpt:
            models_anony += [
                "gpt-4",
                "gpt-3.5-turbo",
                "gpt-4-turbo",
                "gpt-3.5-turbo-1106",
            ]
        if args.add_claude:
            models_anony += ["claude-2.1", "claude-2.0", "claude-1", "claude-instant-1"]
        if args.add_palm:
            models_anony += ["palm-2"]
    models_anony = list(set(models_anony))

    side_by_side_anony_updates = load_demo_side_by_side_anony(models_anony, url_params)
    side_by_side_named_updates = load_demo_side_by_side_named(models, url_params)
    return (
        (gr.Tabs.update(selected=selected),)
        + single_updates
        + side_by_side_anony_updates
        + side_by_side_named_updates
    )


def load_combine_demo(url_params, request: gr.Request):
    logger.info("load_demo_multi")
    global models

    ip = get_ip(request)
    logger.info(f"load_demo. ip: {ip}. params: {url_params}")
    ip_expiration_dict[ip] = time.time() + SESSION_EXPIRATION_TIME

    selected_combine = 0
    if "generation" in url_params:
        selected_combine = 0
    elif "edition" in url_params:
        selected_combine = 1


    selected_edition = 0
    # if "arena" in url_params and "edition" in url_params:
    #     selected_edition = 0
    if "arena" in url_params and "edition" in url_params:
        selected_edition = 0
    elif "compare" in url_params and "edition" in url_params:
        selected_edition = 1
    elif "single" in url_params and "edition" in url_params:
        selected_edition = 2
    elif "leaderboard" in url_params and "edition" in url_params:
        selected_edition = 3

    selected_generation = 0
    if "arena" in url_params and "generation" in url_params:
        selected_generation = 0
    elif "compare" in url_params and "generation" in url_params:
        selected_generation = 1
    elif "single" in url_params and "generation" in url_params:
        selected_generation = 2
    elif "leaderboard" in url_params and "generation" in url_params:
        selected_generation = 3

    logger.info(f"selected_combine {selected_combine}")
    logger.info(f"selected_generation {selected_generation}")
    logger.info(f"selected_edition {selected_edition}")

    if args.model_list_mode == "reload":
        if args.anony_only_for_proprietary_model:
            models = get_model_list(
                args.controller_url,
                args.register_openai_compatible_models,
                False,
                False,
                False,
            )
        else:
            models = get_model_list(
                args.controller_url,
                args.register_openai_compatible_models,
                args.add_chatgpt,
                args.add_claude,
                args.add_palm,
            )

    models_anony = list(models)

    models_anony_ie = [x for x in models_anony if "edition" in x]
    models_anony_ig = [x for x in models_anony if "generation" in x]

    models_anony_ig = list(set(models_anony_ig))
    models_anony_ie = list(set(models_anony_ie))

    models_ig = list(set(models_anony_ig))
    models_ie = list(set(models_anony_ie))

    single_updates_ig = load_demo_single(models_ig, url_params)
    single_updates_ie = load_demo_single_ie(models_ie, url_params)

    side_by_side_anony_updates_ie = load_demo_side_by_side_anony_ie(models_anony_ie, url_params)
    side_by_side_named_updates_ie = load_demo_side_by_side_named_ie(models_ie, url_params)
    side_by_side_anony_updates_ig = load_demo_side_by_side_anony(models_anony_ig, url_params)
    side_by_side_named_updates_ig = load_demo_side_by_side_named(models_ig, url_params)
    return (
        (gr.Tabs.update(selected=selected_combine), gr.Tabs.update(selected=selected_generation),
         gr.Tabs.update(selected=selected_edition))
        + single_updates_ig
        + side_by_side_anony_updates_ig
        + side_by_side_named_updates_ig
        + single_updates_ie
        + side_by_side_anony_updates_ie
        + side_by_side_named_updates_ie
    )


def build_demo(models, elo_results_file, leaderboard_table_file):
    # text_size = gr.themes.sizes.text_md
    with gr.Blocks(
        title="Play with Open Vision Models",
        theme=gr.themes.Default(),
        css=block_css,
    ) as demo:
        logger.info("build demo")
        url_params = gr.JSON(visible=False)
        with gr.Tabs() as tabs:
            with gr.Tab("Arena (battle)", id=0):
                side_by_side_anony_list = build_side_by_side_ui_anony(models)

            with gr.Tab("Arena (side-by-side)", id=1):
                side_by_side_named_list = build_side_by_side_ui_named(models)

            with gr.Tab("Direct Chat", id=2):
                single_model_list = build_single_model_ui(
                    models, add_promotion_links=True
                )
            if elo_results_file:
                with gr.Tab("Leaderboard", id=3):
                    build_leaderboard_tab(elo_results_file, leaderboard_table_file)
            with gr.Tab("About Us", id=4):
                about = build_about()


        logger.info(f"url_param: {url_params}")

        if args.model_list_mode not in ["once", "reload"]:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

        if args.show_terms_of_use:
            load_js = get_window_url_params_with_tos_js
        else:
            load_js = get_window_url_params_js

        logger.info(f"url_param: {url_params}")

        demo.load(
            load_demo,
            [url_params],
            [tabs]
            + single_model_list
            + side_by_side_anony_list
            + side_by_side_named_list,
            _js=load_js,
        )
        # pdb.set_trace()
        logger.info("build demo end")

    return demo


def build_combine_demo(models, elo_results_file, leaderboard_table_file):
    with gr.Blocks(
        title="Play with Open Vision Models",
        theme=gr.themes.Default(),
        css=block_css,
    ) as demo:
        logger.info("build demo")
        url_params = gr.JSON(visible=False)
        models_ie = [x for x in models if "edition" in x]
        models_ig = [x for x in models if "generation" in x]
        with gr.Tabs() as tabs_combine:
            with gr.Tab("Image Generation", id=0):
                with gr.Tabs() as tabs_ig:
                    with gr.Tab("Generation Arena (battle)", id=0):
                        side_by_side_anony_list_ig = build_side_by_side_ui_anony(models_ig)

                    with gr.Tab("Generation Arena (side-by-side)", id=1):
                        side_by_side_named_list_ig = build_side_by_side_ui_named(models_ig)

                    with gr.Tab("Generation Direct Chat", id=2):
                        single_model_list_ig = build_single_model_ui(
                            models_ig, add_promotion_links=True
                        )
                    if elo_results_file:
                        with gr.Tab("Generation Leaderboard", id=3):
                            build_leaderboard_tab(elo_results_file['t2i_generation'], leaderboard_table_file['t2i_generation'])
                    with gr.Tab("About Us", id=4):
                        about = build_about()
            with gr.Tab("Image Edition", id=5):
                with gr.Tabs() as tabs_ie:
                    with gr.Tab("Edition Arena (battle)", id=5):
                        side_by_side_anony_list_ie = build_side_by_side_ui_anony_ie(models_ie)

                    with gr.Tab("Edition Arena (side-by-side)", id=6):
                        side_by_side_named_list_ie = build_side_by_side_ui_named_ie(models_ie)

                    with gr.Tab("Edition Direct Chat", id=7):
                        single_model_list_ie = build_single_model_ui_ie(
                            models_ie, add_promotion_links=True
                        )
                    if elo_results_file:
                        with gr.Tab("Edition Leaderboard", id=8):
                            build_leaderboard_tab(elo_results_file['image_editing'], leaderboard_table_file['image_editing'])
                    with gr.Tab("About Us", id=9):
                        about = build_about()


        logger.info(f"url_param: {url_params}")

        if args.model_list_mode not in ["once", "reload"]:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

        if args.show_terms_of_use:
            load_js = get_window_url_params_with_tos_js
        else:
            load_js = get_window_url_params_js

        logger.info(f"url_param: {url_params}")

        demo.load(
            load_combine_demo,
            [url_params],
            [tabs_combine, tabs_ig, tabs_ie]
            + single_model_list_ig
            + side_by_side_anony_list_ig
            + side_by_side_named_list_ig
            + single_model_list_ie
            + side_by_side_anony_list_ie
            + side_by_side_named_list_ie,
            _js=load_js,
        )
        # pdb.set_trace()
        logger.info("build demo end")

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
        help="Whether to load the model list once or reload the model list every time.",
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
        "--anony-only-for-proprietary-model",
        action="store_true",
        help="Only add ChatGPT, Claude, Bard under anony battle tab",
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
        default=None,
    )
    parser.add_argument(
        "--elo_results_dir",
        type=str,
        help="Load leaderboard results and plots"
    )
    # parser.add_argument(
    #     "--elo-results-file", type=str, help="Load leaderboard results and plots"
    # )
    # parser.add_argument(
    #     "--leaderboard-table-file", type=str, help="Load leaderboard results and plots"
    # )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # Set global variables
    set_global_vars(args.controller_url, args.moderate)
    set_global_vars_named(args.moderate)
    set_global_vars_anony(args.moderate)

    set_global_vars_ie(args.controller_url, args.moderate)
    set_global_vars_named_ie(args.moderate)
    set_global_vars_anony_ie(args.moderate)
    if args.anony_only_for_proprietary_model:
        models = get_model_list(
            args.controller_url,
            args.register_openai_compatible_models,
            False,
            False,
            False,
        )
    else:
        models = get_model_list(
            args.controller_url,
            args.register_openai_compatible_models,
            args.add_chatgpt,
            args.add_claude,
            args.add_palm,
        )

    logger.info(f'models: {models}')

    # Set authorization credentials
    auth = None
    if args.gradio_auth_path is not None:
        auth = parse_gradio_auth_creds(args.gradio_auth_path)

    from collections import defaultdict
    args.elo_results_file = defaultdict(lambda: None)
    args.leaderboard_table_file = defaultdict(lambda: None)
    if args.elo_results_dir is not None:
        args.elo_results_dir = Path(args.elo_results_dir)
        args.elo_results_file = {}
        args.leaderboard_table_file = {}
        for file in args.elo_results_dir.glob('elo_results_*.pkl'):
            if 't2i_generation' in file.name:
                args.elo_results_file['t2i_generation'] = file
            elif 'image_editing' in file.name:
                args.elo_results_file['image_editing'] = file
            else:
                raise ValueError(f"Unknown file name: {file.name}")
        for file in args.elo_results_dir.glob('*_leaderboard.csv'):
            if 't2i_generation' in file.name:
                args.leaderboard_table_file['t2i_generation'] = file
            elif 'image_editing' in file.name:
                args.leaderboard_table_file['image_editing'] = file
            else:
                raise ValueError(f"Unknown file name: {file.name}")
        
    # Launch the demo
    # demo = build_demo(models, args.elo_results_file, args.leaderboard_table_file)
    demo = build_combine_demo(models, args.elo_results_file, args.leaderboard_table_file)
    # demo.launch(
    #     server_name=args.host,
    #     server_port=args.port,
    #     share=args.share,
    #     max_threads=400,
    #     auth=auth,
    # )
    demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False,
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=800,
        root_path="/AIGenArena",
        auth=auth,
    )

    # demo.queue(
    #     status_update_rate=10, api_open=False
    # ).launch(
    #     server_name=args.host,
    #     server_port=args.port,
    #     root_path="/chatbot",
    #     share=args.share,
    #     max_threads=400,
    #     auth=auth,
    #     # share_server_address="visual-arena.com:7000",
    #     show_error=True
    # )

