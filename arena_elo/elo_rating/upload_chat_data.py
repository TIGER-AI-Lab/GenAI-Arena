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

def load_chat_image(chat, log_dir):
    image_path = Path(log_dir) / f"{get_date_from_time_stamp(chat['tstamp'])}-convinput_images" / f"input_image_{chat['conversation_id']}.png"
    return load_image(image_path)
    

def main(
    data_file: str = "./results/latest/clean_chat_conv.json",
    repo_id: str = "DongfuTingle/wildvision-bench",
    log_dir: str = os.getenv("LOGDIR", "./vision-arena-logs/"),
    token = os.environ.get("HUGGINGFACE_TOKEN", None)
):
    with open(data_file, "r") as f:
        data = json.load(f)
    
    
    new_data = []
    for chat in data:
        image = load_chat_image(chat, log_dir)
        if image is None:
            # we don't keep the data without image
            continue
        new_data.append({
            "question_id": chat["conversation_id"],
            "model": chat['model'],
            "conversation": chat['conversation'],
            "language": chat["language"],
            "image": image,
            "turn": chat["turn"],
        })
    split = "test"
    hf_dataset = create_hf_dataset(new_data, "test")
    
    print(hf_dataset)
    print(f"Uploading to part {repo_id}:{split}...")
    hf_dataset.push_to_hub(
        repo_id=repo_id,
        config_name="chat",
        split=split,
        token=token,
        commit_message=f"Add vision-arena {split} dataset",
    )
    
    print("Done!")
    
    
if __name__ == "__main__":
    fire.Fire(main)