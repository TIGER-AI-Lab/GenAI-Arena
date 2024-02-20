import argparse
import code
import datetime
import json
import os
from pytz import timezone
import time

import pandas as pd
from tqdm import tqdm
import csv

import base64
from icecream import ic
from openai import OpenAI

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_log_files(max_num_files=None):
    dates = []
    for month in [2, 3]:
        for day in range(1, 32):
            dates.append(f"2024-{month:02d}-{day:02d}")

    num_servers = 1
    filenames = []
    for d in dates:
        for i in range(num_servers):
            # name = os.path.expanduser(f"~/fastchat_logs/server{i}/{d}-conv.json")
            name = os.path.expanduser(f"vision-arena-logs/{d}-conv.json")
            if os.path.exists(name):
                filenames.append(name)
    max_num_files = max_num_files or len(filenames)
    filenames = filenames[-max_num_files:]
    return filenames


def pretty_print_conversation(messages):
    for role, msg in messages:
        print(f"[[{role}]]: {msg}")


def get_gpt4v_response(client, img_bs64=None, text_prompt="", use_vision=False):
    if use_vision:
        response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_bs64}"
                        }
                    },
                ],
            }
        ],
        max_tokens=100,
        )
    else:
        response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                ],
            }
        ],
        max_tokens=100,
        )
    return response.choices[0].message.content

task_template_map = {
    "image_caption": "Give me the semantic alignment score between the given image and the given caption: \"{generated_sentence}\" on a scale of 0-100. Only reply the score value.",
    "vqa": "Rate the answer correctness regarding the question within the context of the given image on a scale of 0-100. Only reply the score value.",
    "pair_rate_old": "[Instruction]\n\"{instruction}\"\n\n\"{generated_sentence}\"\n\n[System]\nGiven the instruction and the image, please compare the correctness of responses A and B. Reply with \"leftvote\" if you find A better, \"rightvote\" if B is better, \"bothbad_vote\" if both responses are wrong, and \"tievote\" if both responses are equally satisfactory. If you are unable to make a decision, please reply with \"NA\".",
    "pair_rate_wexplanation": "[Instruction]\n\"{instruction}\"\n\n\"{generated_sentence}\"[System]\nPlease act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any positional biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.",
    "pair_rate": "[Instruction]\n\"{instruction}\"\n\n\"{generated_sentence}\"\n\n[System]\nPlease act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any positional biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Reply with \"leftvote\" if you find assistant A better, \"rightvote\" if assistant B is better, \"bothbad_vote\" if both responses are wrong, and \"tievote\" if both assistants provide equally satisfactory answers. If you are unable to make a decision, please reply with \"NA\"."
}

def inspect_convs(log_files):
    ic(log_files)
    data = []
    total_vote = 0
    correct_vote = 0
        
    client = OpenAI()
    with open('all_pairvote_log_wgpt_prtchatbot.csv', 'w', newline='') as csvfile:
        # fieldnames = ['tstamp', 'type', 'model_1', 'model_2', 'template_name_1', 'template_name_2', 'system_message_1', 'system_message_2', 'role_1', 'role_2', 'instruction_1', 'instruction_2', 'message_1', 'message_2', 'offset_1', 'offset_2', 'conv_id_1', 'conv_id_2', 'model_name_1', 'model_name_2', 'ip']
        fieldnames = ['tstamp', 'type', 'models', 'states', 'ip', 'gpt_vote']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header
        writer.writeheader()

        for filename in tqdm(log_files, desc="read files"):
            for retry in range(5):
                try:
                    lines = open(filename).readlines()
                    break
                except FileNotFoundError:
                    time.sleep(2)
            
            for l in lines:
                row = json.loads(l)
                
                if "states" not in row:
                    continue
                if row["type"] not in ["leftvote", "rightvote", "bothbad_vote", "tievote"]:
                    continue
                
                model_names = row["states"][0]["model_name"], row["states"][1]["model_name"]
                

                # Iterate through each state and write the relevant information
                if not len(row["states"][0]['messages']): continue
                # ic(row["states"][0]['messages'][1][1])

                if row["states"][0]['messages'][1][1] is None or row["states"][1]['messages'][1][1] is None or "NETWORK ERROR" in row["states"][0]['messages'][1][1] or "NETWORK ERROR" in row["states"][1]['messages'][1][1]: continue
                total_vote += 1
                # row = {
                #     'tstamp': row['tstamp'],
                #     'type': row['type'],
                #     'model_1': row['models'][0],
                #     'model_2': row['models'][1],
                #     'template_name_1': row["states"][0]['template_name'],
                #     'system_message_1': row["states"][0]['system_message'],
                #     'template_name_2': row["states"][1]['template_name'],
                #     'system_message_2': row["states"][1]['system_message'],
                #     'role_1': row["states"][0]['roles'],
                #     'role_2': row["states"][1]['roles'],
                #     'instruction_1': row["states"][0]['messages'][0][1],
                #     'instruction_2': row["states"][1]['messages'][0][1],
                #     'message_1': row["states"][0]['messages'][1][1],
                #     'message_2': row["states"][1]['messages'][1][1],
                #     'offset_1': row["states"][0]['offset'],
                #     'offset_2': row["states"][1]['offset'],
                #     'conv_id_1': row["states"][0]['conv_id'],
                #     'conv_id_2': row["states"][1]['conv_id'],
                #     'model_name_1': row["states"][0]['model_name'],
                #     'model_name_2': row["states"][1]['model_name'],
                #     'ip': row['ip']
                # }
                # writer.writerow(row)
                # Convert complex objects to JSON strings
                # TODO: check two image are the same
                conv_id = row["states"][0]['conv_id']
                image_path = os.path.join("/local/home/yujielu/project/Arena-Elo/vision-arena-logs", os.path.basename(filename)[:-5]+"input_images", f"input_image_{conv_id}.png")
                if not os.path.exists(image_path):
                    response = "NA"
                    ic(image_path)
                else:
                    base64_image = encode_image(image_path)
                    left_response = row["states"][0]['messages'][1][1]
                    right_response = row["states"][1]['messages'][1][1]
                    sep = "-" * 20
                    instruction = row["states"][0]['messages'][0][1]
                    generated_sentence = f"[The Start of Assistant A’s Answer]\n{left_response}\n[The End of Assistant A’s Answer]\n\n[The Start of Assistant B’s Answer]\n{right_response}\n[The End of Assistant B’s Answer]"
                    text_prompt = task_template_map["pair_rate"].format(instruction=instruction, generated_sentence=generated_sentence)
                    # ic(text_prompt)
                    try:
                        response = get_gpt4v_response(client, img_bs64=base64_image, text_prompt=text_prompt, use_vision=True)
                    except:
                        ic(">>> skip")
                        response = "NA"
                    
                    # response = get_gpt4v_response(client, img_bs64=base64_image, text_prompt=text_prompt, use_vision=True)
                    ic(row['type'], response)
                    if response.strip() not in ["leftvote", "rightvote", "bothbad_vote", "tievote"]:
                        response = "NA"
                    # ic(generated_sentence)
                    
                    # if row['type'] == "leftvote":
                    #     row['type'] = "A"
                    # elif row['type'] == "rightvote":
                    #     row['type'] = "B"
                    # elif row['type'] in ["bothbad_vote", "tievote"]:
                    #     row['type'] = "C"
                    if row['type'] == response.strip():
                        correct_vote += 1
                row['models'] = json.dumps(row['models'])
                row['states'] = json.dumps(row['states'], ensure_ascii=False)
                row['gpt_vote'] = response
                
                # Write the modified row to the CSV file
                writer.writerow(row)
                # if row["type"] == "leftvote":
                #     winner, loser = model_names[0], model_names[1]
                #     winner_conv, loser_conv = row["states"][0], row["states"][1]
                # elif row["type"] == "rightvote":
                #     loser, winner = model_names[0], model_names[1]
                #     loser_conv, winner_conv = row["states"][0], row["states"][1]

                # if loser == "llava-v1.5-13b" and winner == "llava-v1.5-13b":
                #     print("=" * 20)
                #     print(f"Winner: {winner}")
                #     pretty_print_conversation(winner_conv["messages"])
                #     print(f"Loser: {loser}")
                #     pretty_print_conversation(loser_conv["messages"])
                #     print("=" * 20)
                #     input()
                # if row['type'] == 'bothbad_vote':
                #     from icecream import ic
                #     ic(model_names)
                # if row["type"] == "bothbad_vote" and "gpt-4-vision-preview" in model_names:
                #    print("=" * 20)
                #    print(f"Model A: {model_names[0]}")
                #    pretty_print_conversation(row["states"][0]["messages"])
                #    print(f"Model B: {model_names[1]}")
                #    pretty_print_conversation(row["states"][1]["messages"])
                #    print("=" * 20)
                #    input()
                # if correct_vote >= 300: break
    ic(total_vote, correct_vote)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-num-files", type=int)
    args = parser.parse_args()

    log_files = get_log_files(args.max_num_files)
    

            
    inspect_convs(log_files)
