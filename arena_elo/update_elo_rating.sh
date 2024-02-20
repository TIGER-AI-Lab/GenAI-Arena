export LOGDIR="/home/tianle/arena_vote" # for local
# set LOGDIR to default if not set before
if [ -z "$LOGDIR" ]; then
    export LOGDIR="./vision-arena-logs"
fi

mkdir -p results

# # for battle data
python -m elo_rating.clean_battle_data --model_infos_file "./model_infos.json"
battle_cutoff_date=`cat cut_off_date.txt` && rm cut_off_date.txt && echo "Battle data last updated on $battle_cutoff_date"

python3 -m elo_rating.elo_analysis --clean-battle-file clean_battle_$battle_cutoff_date.json

# mv them to results folder
mkdir -p ./results/$battle_cutoff_date
mv ./elo_results_$battle_cutoff_date.pkl ./results/$battle_cutoff_date/elo_results.pkl

python -m elo_rating.generate_leaderboard \
    --model_info_file "./model_infos.json" \
    --elo_rating_pkl "./results/$battle_cutoff_date/elo_results.pkl" \
    --output_csv "./results/$battle_cutoff_date/leaderboard.csv"

mkdir -p ./results/latest

cp ./results/$battle_cutoff_date/elo_results.pkl ./results/latest/elo_results.pkl
cp ./results/$battle_cutoff_date/leaderboard.csv ./results/latest/leaderboard.csv


