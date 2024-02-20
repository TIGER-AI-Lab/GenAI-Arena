import fire
import json
import os
import datasets
import datetime
from pathlib import Path
from datetime import datetime
from PIL import Image

datasets.config.DEFAULT_MAX_BATCH_SIZE = 500
def create_hf_dataset(data_file: str, split="test"):
    hf_dataset = datasets.Dataset.from_list(
        data_file,
        features=datasets.Features(
            {
                "question_id": datasets.Value("string"),
                "model": datasets.Value("string"),
                "conversation": [
                    {
                        "role": datasets.Value("string"),
                        "content": datasets.Value("string"),
                    }
                ],
                "language": datasets.Value("string"),
                "image": datasets.Image(),
                "turn": datasets.Value("int32"),
            }
        ),
        split=split,
    )
    return hf_dataset

def create_hf_battle_dataset(data_file: str, split="test"):
    hf_dataset = datasets.Dataset.from_list(
        data_file,
        features=datasets.Features(
            {
                "question_id": datasets.Value("string"),
                "model_a": datasets.Value("string"),
                "model_b": datasets.Value("string"),
                "conversation_a": [
                    {
                        "role": datasets.Value("string"),
                        "content": datasets.Value("string"),
                    }
                ],
                "conversation_b": [
                    {
                        "role": datasets.Value("string"),
                        "content": datasets.Value("string"),
                    }
                ],
                "language": datasets.Value("string"),
                "image": datasets.Image(),
                "turn": datasets.Value("int32"),
                "anony": datasets.Value("bool"),
            }
        ),
        split=split,
    )
    return hf_dataset
                           
                                            


def load_image(path:str):
    try:
        return Image.open(path)
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

def get_date_from_time_stamp(unix_timestamp: int):
    # Create a datetime object from the Unix timestamp
    dt = datetime.fromtimestamp(unix_timestamp)

    # Convert the datetime object to a string with the desired format
    date_str = dt.strftime("%Y-%m-%d")
    return date_str

def load_battle_image(battle, log_dir):
    image_path = Path(log_dir) / f"{get_date_from_time_stamp(battle['tstamp'])}-convinput_images" / f"input_image_{battle['question_id']}.png"
    return load_image(image_path)
    

def main(
    data_file: str = "./results/latest/clean_battle_conv.json",
    repo_id: str = "DongfuTingle/wildvision-bench",
    log_dir: str = os.getenv("LOGDIR", "./vision-arena-logs/"),
    mode="battle",
    token = os.environ.get("HUGGINGFACE_TOKEN", None)
):
    with open(data_file, "r") as f:
        data = json.load(f)
    
    
    
    has_image_stats = {
        "has_image": 0,
        "no_image": 0,
    }
    if mode == "keep_bad_only":
        # anony only
        data = [d for d in data if d["anony"]]
        
        new_data = []
        for battle in data:
            image = load_battle_image(battle, log_dir)
            if image is None:
                has_image_stats["no_image"] += 1
                # we don't keep the data without image
                continue
            has_image_stats["has_image"] += 1
            
            if battle["winner"] in ["model_a", "model_b"]:
                if battle["winner"] == "model_a":
                    worse_model = "model_b"
                    worse_conv = "conversation_b"
                if battle["winner"] == "model_b":
                    worse_model = "model_a"
                    worse_conv = "conversation_a"
                    
                new_data.append({
                    "question_id": battle["question_id"],
                    "model": battle[worse_model],
                    "conversation": battle[worse_conv],
                    "language": battle["language"],
                    "image": image,
                    "turn": battle["turn"],
                })
            elif battle["winner"] == "tie (bothbad)":
                
                new_data.append({
                    "question_id": battle["question_id"],
                    "model": battle["model_a"],
                    "conversation": battle["conversation_a"],
                    "language": battle["language"],
                    "image": image,
                    "turn": battle["turn"],
                })

                new_data.append({
                    "question_id": battle["question_id"],
                    "model": battle["model_b"],
                    "conversation": battle["conversation_b"],
                    "language": battle["language"],
                    "image": image,
                    "turn": battle["turn"],
                })
                
        split = "test"
        hf_dataset = create_hf_dataset(new_data, "test")
    
    elif mode == "battle":
        new_data = []
        for battle in data:
            image = load_battle_image(battle, log_dir)
            if image is None:
                has_image_stats["no_image"] += 1
                continue
            has_image_stats["has_image"] += 1
            new_data.append({
                "question_id": battle["question_id"],
                "model_a": battle["model_a"],
                "model_b": battle["model_b"],
                "conversation_a": battle["conversation_a"],
                "conversation_b": battle["conversation_b"],
                "language": battle["language"],
                "image": image,
                "turn": battle["turn"],
                "anony": battle["anony"],
            })
        split = "test"
        hf_dataset = create_hf_battle_dataset(new_data, "test")
    else:
        raise ValueError(f"Invalid mode: {mode}")

    print(f"Stats: {has_image_stats}")
    print(hf_dataset)
    print(f"Uploading to part {repo_id}:{split}...")
    hf_dataset.push_to_hub(
        repo_id=repo_id,
        config_name=mode,
        split=split,
        token=token,
        commit_message=f"Add vision-arena {split} dataset",
    )
    
    print("Done!")
    
    
if __name__ == "__main__":
    fire.Fire(main)