import fire
import time
import json
from collections import defaultdict
from .basic_stats import get_log_files, NUM_SERVERS, LOG_ROOT_DIR
from .utils import detect_language, get_time_stamp_from_date, get_input_image_path, load_image_from_path
from tqdm import tqdm
VOTES = ["tievote", "leftvote", "rightvote", "bothbad_vote", "chat"]


def remove_html(raw):
    if raw.startswith("<h3>"):
        return raw[raw.find(": ") + 2 : -len("</h3>\n")]
    if raw.startswith("### Model A: ") or raw.startswith("### Model B: "):
        return raw[13:]
    return raw


def read_file(filename):
    data = []
    for retry in range(5):
        try:
            # lines = open(filename).readlines()
            for l in open(filename):
                row = json.loads(l)
                if row["type"] in VOTES:
                    data.append(row)
            break
        except FileNotFoundError:
            time.sleep(2)
    return data


def read_file_parallel(log_files, num_threads=16):
    data_all = []
    from multiprocessing import Pool

    with Pool(num_threads) as p:
        ret_all = list(tqdm(p.imap(read_file, log_files), total=len(log_files)))
        for ret in ret_all:
            data_all.extend(ret)
    return data_all

def num_tokens(s:str):
    if s is None:
        return 0
    return len(s) / 4

def main(
):
    log_files = get_log_files()
    data = read_file_parallel(log_files)
    
    all_model_counts = defaultdict(int)
    all_model_input_tokens_counts = defaultdict(list)
    all_model_output_tokens_counts = defaultdict(list)
    all_model_image_sizes = defaultdict(list)
    chat_battle_counts = defaultdict(int)
    for row in tqdm(data, desc="counting"):
        if row['type'] == "chat":
            chat_battle_counts["chat"] += 1
            all_model_counts[row['model']] += 1
            tstamp = row["tstamp"]
            conv_id = row["state"]["conv_id"]
            
            image = load_image_from_path(get_input_image_path(tstamp, conv_id))
            if image is None:
                image_size = None
            else:
                image_size = load_image_from_path(get_input_image_path(tstamp, conv_id)).size
            all_model_image_sizes[row['model']].append(image_size)
            try:
                for message in row["state"]["messages"][row["state"]["offset"] :: 2]:
                    all_model_input_tokens_counts[row['model']].append(num_tokens(message[1]))
                for message in row["state"]["messages"][row["state"]["offset"] + 1 :: 2]:
                    all_model_output_tokens_counts[row['model']].append(num_tokens(message[1]))
            except Exception as e:
                print(row)
                raise e
            
        else:
            chat_battle_counts[row['type']] += 1
            if row["models"][0] is None or row["models"][1] is None:
                continue

            # Resolve model names
            models_public = [remove_html(row["models"][0]), remove_html(row["models"][1])]
            if "model_name" in row["states"][0]:
                models_hidden = [
                    row["states"][0]["model_name"],
                    row["states"][1]["model_name"],
                ]
                if models_hidden[0] is None:
                    models_hidden = models_public
            else:
                models_hidden = models_public

            if (models_public[0] == "" and models_public[1] != "") or (
                models_public[1] == "" and models_public[0] != ""
            ):
                continue

            if models_public[0] == "" or models_public[0] == "Model A":
                anony = True
                models = models_hidden
            else:
                anony = False
                models = models_public
                if not models_public == models_hidden:
                    continue
            
            all_model_counts[models[0]] += 1
            all_model_counts[models[1]] += 1
            tstamp = row["tstamp"]
            conv_id1 = row["states"][0]["conv_id"]
            conv_id2 = row["states"][1]["conv_id"]
            
            image1 = load_image_from_path(get_input_image_path(tstamp, conv_id1))
            image2 = load_image_from_path(get_input_image_path(tstamp, conv_id2))
            all_model_image_sizes[models[0]].append(None if image1 is None else image1.size)
            all_model_image_sizes[models[1]].append(None if image2 is None else image2.size)
            
            for message in row["states"][0]["messages"][row["states"][0]["offset"] :: 2]:
                all_model_input_tokens_counts[models[0]].append(num_tokens(message[1]))
            for message in row["states"][0]["messages"][row["states"][0]["offset"] + 1 :: 2]:
                all_model_output_tokens_counts[models[0]].append(num_tokens(message[1]))
            for message in row["states"][1]["messages"][row["states"][1]["offset"] :: 2]:
                all_model_input_tokens_counts[models[1]].append(num_tokens(message[1]))
            for message in row["states"][1]["messages"][row["states"][1]["offset"] + 1 :: 2]:
                all_model_output_tokens_counts[models[1]].append(num_tokens(message[1]))

    print("### Chat battle counts (requests)")
    print(json.dumps(chat_battle_counts, indent=4))
    
    print("### Model counts (requests)")
    print(json.dumps(all_model_counts, indent=4))
    
    print("### Model Avg input tokens counts (tokens)")
    average_input_tokens_counts = {}
    for model, counts in all_model_input_tokens_counts.items():
        average_input_tokens_counts[model] = sum(counts) / len(counts)
    print(json.dumps(average_input_tokens_counts, indent=4))
    
    print("### Model AVg output tokens counts (tokens)")
    average_output_tokens_counts = {}
    for model, counts in all_model_output_tokens_counts.items():
        average_output_tokens_counts[model] = sum(counts) / len(counts)
    print(json.dumps(average_output_tokens_counts, indent=4))
    
    print("### Model Avg image sizes (height, width)")
    average_image_sizes = {}
    for model, sizes in all_model_image_sizes.items():
        avg_height = sum([size[0] for size in sizes if size is not None]) / len(sizes)
        avg_width = sum([size[1] for size in sizes if size is not None]) / len(sizes)
        average_image_sizes[model] = (avg_height, avg_width)
    print(json.dumps(average_image_sizes, indent=4))
    
    print("### GPT-4V estimated cost (USD)")
    gpt_4v_name = "gpt-4-vision-preview"
    gpt_4v_cost = {}
    gpt_4v_cost['input'] = sum(all_model_input_tokens_counts[gpt_4v_name]) / 1000 * 0.01
    gpt_4v_cost['output'] = sum(all_model_output_tokens_counts[gpt_4v_name]) / 1000 * 0.03
    
    all_image_cost = 0
    for size in all_model_image_sizes[gpt_4v_name]:
        if size is None:
            continue
        all_image_tokens = (size[0] // 512 + 1) * (size[1] // 512 + 1) * 170 + 85
        all_image_cost += all_image_tokens / 1000 * 0.01
    gpt_4v_cost['image'] = all_image_cost
    print(json.dumps(gpt_4v_cost, indent=4))
        
    
    
    
if __name__ == "__main__":
    fire.Fire(main)