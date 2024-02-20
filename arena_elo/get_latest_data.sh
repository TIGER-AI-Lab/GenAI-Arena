
# set LOGDIR to default if not set before
if [ -z "$LOGDIR" ]; then
    export LOGDIR="./vision-arena-logs"
fi
mkdir -p results

# for the direct chat data
python -m elo_rating.clean_chat_data
chat_cutoff_date=`cat cut_off_date.txt` && rm cut_off_date.txt && echo "Chat date last updated on $chat_cutoff_date"

# # for battle data
python -m elo_rating.clean_battle_data --model_infos_file "./model_infos.json" --mode conv_release
battle_cutoff_date=`cat cut_off_date.txt` && rm cut_off_date.txt && echo "Battle data last updated on $battle_cutoff_date"


mkdir -p ./results/latest
mkdir -p ./results/$chat_cutoff_date && mv ./clean_chat_conv_$chat_cutoff_date.json ./results/$chat_cutoff_date/clean_chat_conv.json
mkdir -p ./results/$battle_cutoff_date && mv ./clean_battle_conv_$battle_cutoff_date.json ./results/$battle_cutoff_date/clean_battle_conv.json
cp ./results/$chat_cutoff_date/clean_chat_conv.json ./results/latest/clean_chat_conv.json
cp ./results/$battle_cutoff_date/clean_battle_conv.json ./results/latest/clean_battle_conv.json

echo "Chat data last updated on $chat_cutoff_date" > ./results/latest/latest_updated_date.txt
echo "Battle data last updated on $battle_cutoff_date" >> ./results/latest/latest_updated_date.txt
