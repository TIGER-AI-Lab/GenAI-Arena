import argparse
import json
import os
import time
from pytz import timezone
from tqdm import tqdm
import base64
from icecream import ic
from PIL import Image


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

task_template_map = {
    "image_caption": "Give me the semantic alignment score between the given image and the given caption: \"{generated_sentence}\" on a scale of 0-100. Only reply the score value.",
    "vqa": "Rate the answer correctness regarding the question within the context of the given image on a scale of 0-100. Only reply the score value.",
    "pair_rate_old": "[Instruction]\n\"{instruction}\"\n\n\"{generated_sentence}\"\n\n[System]\nGiven the instruction and the image, please compare the correctness of responses A and B. Reply with \"leftvote\" if you find A better, \"rightvote\" if B is better, \"bothbad_vote\" if both responses are wrong, and \"tievote\" if both responses are equally satisfactory. If you are unable to make a decision, please reply with \"NA\".",
    "pair_rate_wexplanation": "<image>[Instruction]\n\"{instruction}\"\n\n\"{generated_sentence}\"[System]\nPlease act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any positional biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.",
    "pair_rate": "<image>[Instruction]\n\"{instruction}\"\n\n\"{generated_sentence}\"\n\n[System]\nPlease act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any positional biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Reply with \"leftvote\" if you find assistant A better, \"rightvote\" if assistant B is better, \"bothbad_vote\" if both responses are wrong, and \"tievote\" if both assistants provide equally satisfactory answers. If you are unable to make a decision, please reply with \"NA\"."
}

def inspect_convs(log_files):
    json_data = []

    ic(log_files)
    total_vote = 0

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

            conv_id = row["states"][0]['conv_id']
            image_path = os.path.join("/local/home/yujielu/project/Arena-Elo/vision-arena-logs", os.path.basename(filename)[:-5]+"input_images", f"input_image_{conv_id}.png")
            if not os.path.exists(image_path) :
                continue
            try:
                image = Image.open(image_path).convert("RGB")
            except:
                continue

            left_response = row["states"][0]['messages'][1][1]
            right_response = row["states"][1]['messages'][1][1]
            instruction = row["states"][0]['messages'][0][1]
            generated_sentence = f"[The Start of Assistant A’s Answer]\n{left_response}\n[The End of Assistant A’s Answer]\n\n[The Start of Assistant B’s Answer]\n{right_response}\n[The End of Assistant B’s Answer]"
            text_prompt = task_template_map["pair_rate"].format(instruction=instruction, generated_sentence=generated_sentence)
            
            user_input = text_prompt
            # Create the conversation structure
            conversation = [
                {
                    "from": "human",
                    "value": user_input
                },
                {
                    "from": "gpt",
                    "value": row["type"]
                }
            ]
            
            # Create the JSON object for each row
            json_obj = {
                "id": conv_id,
                "image": image_path,
                "conversations": conversation
            }
            
            # Append the JSON object to the list
            json_data.append(json_obj)

        # Write the JSON data to a file
        with open('output_evaluator_data.json', 'w') as json_file:
            json.dump(json_data, json_file, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-num-files", type=int)
    args = parser.parse_args()

    log_files = get_log_files(args.max_num_files)
    

            
    inspect_convs(log_files)



