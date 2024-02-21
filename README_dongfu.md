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