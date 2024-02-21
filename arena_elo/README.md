## Computing the Elo Ratings


```bash
apt-get -y install pkg-config
pip install -r requirements.txt
```


### to update the leaderboard

```bash
export LOGDIR="/path/to/your/logdir"
bash update_elo_rating.sh
```

### to inspect the leaderboard status
```bash
python -m elo_rating.inspect_elo_rating_pkl
```

### to inspect the collected data status and cost
```bash
export LOGDIR="/path/to/your/logdir"
python -m elo_rating.inspect_cost
```

### to upload the battle data to hugging face🤗 
```bash
export HUGGINGFACE_TOKEN="your_huggingface_token"
bash get_latest_data.sh
python -m elo_rating.upload_battle_data --repo_id "WildVision/wildvision-bench" --log_dir "./vision-arena-logs/"
```

### to upload the chat data to hugging face🤗 
```bash
export HUGGINGFACE_TOKEN="your_huggingface_token"
bash get_latest_data.sh
python -m elo_rating.upload_chat_data --repo_id "WildVision/wildvision-bench" --log_dir "./vision-arena-logs/"
```


### to get the collected data
```bash
python -m 

