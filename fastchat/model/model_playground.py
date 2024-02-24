import requests
import io, base64, json
from PIL import Image
def generate_stream_playground(
    model,
    tokenizer,
    params,
    device,
    context_len=256,
    stream_interval=2,
):
    prompt = params["prompt"]
    if model == "Playground v2":
        model_name = "Playground_v2"
    elif model == "Playground v2.5":
        model_name = "Playground_v2.5"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer pg_0061b0a63475918714c4be28ec9a4a861a5012b57b12b77adaee97677cb35a87',
    }

    data = json.dumps({"prompt": prompt, "filter_model": model_name, "scheduler": "DPMPP_2M_K", "guidance_scale": 3})

    response = requests.post('https://playground.com/api/models/external/v1', headers=headers, data=data)
    json_obj = response.json()
    image = json_obj['images'][0]
    img = Image.open(io.BytesIO(base64.decodebytes(bytes(image, "utf-8"))))
    yield {
        "text": img,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "finish_reason": "stop",
    }