### Installation

```
pip install -e .
pip install -r imagenhub_requirements.txt
cd ImageHub && pip install -e .
```

### run


```
# run controller
python3 -m fastchat.serve.controller > controller.log &

# run worker
CUDA_VISIBLE_DEVICES=0 nohup python3 -m fastchat.serve.model_worker --model-path imagenhub_LCM_generation --controller http://localhost:21001 --port 31005 --worker http://localhost:31005 >  model_log/lcm.log &

CUDA_VISIBLE_DEVICES=0 nohup python3 -m fastchat.serve.model_worker --model-path imagenhub_SDXLTurbo_generation --controller http://localhost:21001 --port 31010 --worker http://localhost:31010 > model_log/SDXLTurbo.log &

CUDA_VISIBLE_DEVICES=2 nohup python3 -m fastchat.serve.model_worker --model-path imagenhub_SDXL_generation --controller http://localhost:21001 --port 31017 --worker http://localhost:31017 > model_log/SDXL.log &

CUDA_VISIBLE_DEVICES=1 nohup python3 -m fastchat.serve.model_worker --model-path imagenhub_OpenJourney_generation --controller http://localhost:21001 --port 31011 --worker http://localhost:31011 > model_log/openjourney.log &

CUDA_VISIBLE_DEVICES=1 nohup python3 -m fastchat.serve.model_worker --model-path imagenhub_PixArtAlpha_generation --controller http://localhost:21001 --port 31012 --worker http://localhost:31012 --limit-worker-concurrency 1 > model_log/PixArtAlpha.log &

CUDA_VISIBLE_DEVICES=6 nohup python3 -m fastchat.serve.model_worker --model-path imagenhub_SDXLLightning_generation --controller http://localhost:21001 --port 31022 --worker http://localhost:31022 --limit-worker-concurrency 1 > model_log/SDXLLightning.log &

CUDA_VISIBLE_DEVICES=5 nohup python3 -m fastchat.serve.model_worker --model-path imagenhub_StableCascade_generation --controller http://localhost:21001 --port 31023 --worker http://localhost:31023 --limit-worker-concurrency 1 > model_log/StableCascade.log &

nohup python3 -m fastchat.serve.model_worker --model-path “Playground v2.5” --controller http://localhost:21001 --port 31024 --worker http://localhost:31024 --limit-worker-concurrency 1 > model_log/PlayGroundV2.5.log &

nohup python3 -m fastchat.serve.model_worker --model-path “Playground v2” --controller http://localhost:21001 --port 31021 --worker http://localhost:31021 --limit-worker-concurrency 1 > model_log/PlayGroundV2.log &



CUDA_VISIBLE_DEVICES=2 nohup python3 -m fastchat.serve.model_worker --model-path imagenhub_CycleDiffusion_edition --controller http://localhost:21001 --port 31013 --worker http://localhost:31013 > model_log/CycleDiffusion.log &

CUDA_VISIBLE_DEVICES=4 nohup python3 -m fastchat.serve.model_worker --model-path imagenhub_Pix2PixZero_edition --controller http://localhost:21001 --port 31014 --worker http://localhost:31014 --limit-worker-concurrency 1 > model_log/Pix2PixZero.log &

CUDA_VISIBLE_DEVICES=7 nohup python3 -m fastchat.serve.model_worker --model-path imagenhub_Prompt2prompt_edition --controller http://localhost:21001 --port 31015 --worker http://localhost:31015 --limit-worker-concurrency 1 > model_log/Prompt2prompt.log &

CUDA_VISIBLE_DEVICES=5 nohup python3 -m fastchat.serve.model_worker --model-path imagenhub_SDEdit_edition --controller http://localhost:21001 --port 31016 --worker http://localhost:31016 > model_log/SDEdit.log &

CUDA_VISIBLE_DEVICES=5 nohup python3 -m fastchat.serve.model_worker --model-path imagenhub_InstructPix2Pix_edition --controller http://localhost:21001 --port 31018 --worker http://localhost:31018 > model_log/InstructPix2Pix.log &

CUDA_VISIBLE_DEVICES=6 nohup python3 -m fastchat.serve.model_worker --model-path imagenhub_MagicBrush_edition --controller http://localhost:21001 --port 31019 --worker http://localhost:31019 > model_log/MagicBrush.log &

CUDA_VISIBLE_DEVICES=6 nohup python3 -m fastchat.serve.model_worker --model-path imagenhub_PNP_edition --controller http://localhost:21001 --port 31020 --worker http://localhost:31020 --limit-worker-concurrency 1 > model_log/PNP.log &

# run web server UI (without leaderboard)
python3 -m fastchat.serve.gradio_web_server_image_editing_multi --share --controller-url http://localhost:21001

# run web server UI (with leaderboard)
python3 -m fastchat.serve.gradio_web_server_image_editing_multi --share --controller-url http://localhost:21001 --elo_results_dir ./arena_elo/results/latest/
```

### update leaderboard data

```
cd arena_elo 
export LOGDIR="/home/tianle/arena_vote"
bash update_elo.sh
```
then results are updated in `arena_elo/results/latest/`

### WishList

1. LEDITS: https://huggingface.co/spaces/editing-images/ledits
2. InfEdit: https://huggingface.co/spaces/sled-umich/InfEdit
3. MGIE: https://huggingface.co/spaces/tsujuifu/ml-mgie
4. OpenDalle: https://huggingface.co/dataautogpt3/OpenDalleV1.1
