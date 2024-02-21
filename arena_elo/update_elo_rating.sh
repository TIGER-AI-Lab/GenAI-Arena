export LOGDIR="/home/tianle/arena_vote" # for local
# set LOGDIR to default if not set before
if [ -z "$LOGDIR" ]; then
    export LOGDIR="./vision-arena-logs"
fi

mkdir -p results

# # for battle data
python -m elo_rating.clean_battle_data --task_name "image_editing"
edition_battle_cutoff_date=`cat cut_off_date.txt` && rm cut_off_date.txt && echo "Image editing battle data last updated on $battle_cutoff_date"

python -m elo_rating.clean_battle_data --task_name "t2i_generation"
generation_battle_cutoff_date=`cat cut_off_date.txt` && rm cut_off_date.txt && echo "T2I image generation battle data last updated on $battle_cutoff_date"

mkdir -p ./results/$battle_cutoff_date

python3 -m elo_rating.elo_analysis --clean-battle-file clean_battle_image_editing_$edition_battle_cutoff_date.json
mv ./elo_results_$edition_battle_cutoff_date.pkl ./results/$edition_battle_cutoff_date/elo_results_image_editing.pkl

python3 -m elo_rating.elo_analysis --clean-battle-file clean_battle_t2i_generation_$generation_battle_cutoff_date.json
mv ./elo_results_$generation_battle_cutoff_date.pkl ./results/$generation_battle_cutoff_date/elo_results_t2i_generation.pkl

# generat the leaderboard

python -m elo_rating.generate_leaderboard \
    --model_info_file "./edition_model_info.json" \
    --elo_rating_pkl "./results/$edition_battle_cutoff_date/elo_results_image_editing.pkl" \
    --output_csv "./results/$edition_battle_cutoff_date/image_editing_leaderboard.csv"

python -m elo_rating.generate_leaderboard \
    --model_info_file "./generation_model_info.json" \
    --elo_rating_pkl "./results/$generation_battle_cutoff_date/elo_results_t2i_generation.pkl" \
    --output_csv "./results/$generation_battle_cutoff_date/t2i_generation_leaderboard.csv"

mkdir -p ./results/latest
cp ./results/$edition_battle_cutoff_date/image_editing_leaderboard.csv ./results/latest/image_editing_leaderboard.csv
cp ./results/$generation_battle_cutoff_date/t2i_generation_leaderboard.csv ./results/latest/t2i_generation_leaderboard.csv
cp ./results/$edition_battle_cutoff_date/elo_results_image_editing.pkl ./results/latest/elo_results_image_editing.pkl
cp ./results/$generation_battle_cutoff_date/elo_results_t2i_generation.pkl ./results/latest/elo_results_t2i_generation.pkl


