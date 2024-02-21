
# set LOGDIR to default if not set before
if [ -z "$LOGDIR" ]; then
    export LOGDIR="./vision-arena-logs"
fi
mkdir -p results


# # for battle data
python -m elo_rating.clean_battle_data --model_infos_file "./model_infos.json" --mode conv_release
battle_cutoff_date=`cat cut_off_date.txt` && rm cut_off_date.txt && echo "Battle data last updated on $battle_cutoff_date"

mkdir -p ./results/latest
mkdir -p ./results/$battle_cutoff_date && mv ./clean_battle_conv_$battle_cutoff_date.json ./results/$battle_cutoff_date/clean_battle_conv.json
cp ./results/$battle_cutoff_date/clean_battle_conv.json ./results/latest/clean_battle_conv.json

echo "Battle data last updated on $battle_cutoff_date" >> ./results/latest/latest_updated_date.txt
