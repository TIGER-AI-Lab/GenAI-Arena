import fire
import json
import pandas as pd
import pickle

def main(
    model_info_file: str,
    elo_rating_pkl: str,
    output_csv: str
):
    model_info = json.load(open(model_info_file))
    
    
    with open(elo_rating_pkl, "rb") as fin:
        elo_rating_results = pickle.load(fin)
    
    anony_elo_rating_results = elo_rating_results["anony"]
    full_elo_rating_results = elo_rating_results["full"]
    anony_leaderboard_data = anony_elo_rating_results["leaderboard_table_df"]
    full_leaderboard_data = full_elo_rating_results["leaderboard_table_df"]
    
    # Model,MT-bench (score),Arena Elo rating,MMLU,License,Link
    fields = ["key", "Model", "Arena Elo rating (anony)", "Arena Elo rating (full)", "MMMU", "License", "Organization", "Link"]
    # set Organization and license to empty for now
    all_models = anony_leaderboard_data.index.tolist()
    
    for model in all_models:
        if not model in model_info:
            model_info[model] = {}
            model_info[model]["MMMU"] = "-"
            model_info[model]["License"] = "N/A"
            model_info[model]["Organization"] = "N/A"
            model_info[model]["Link"] = "N/A"
        model_info[model]["Model"] = model
        model_info[model]["key"] = model
        
        if model in anony_leaderboard_data.index:
            model_info[model]["Arena Elo rating (anony)"] = anony_leaderboard_data.loc[model, "rating"]
        else:
            model_info[model]["Arena Elo rating (anony)"] = 0
        
        if model in full_elo_rating_results["leaderboard_table_df"].index:
            model_info[model]["Arena Elo rating (full)"] = full_leaderboard_data.loc[model, "rating"]
        else:
            model_info[model]["Arena Elo rating (full)"] = 0
    
    final_model_info = {}
    for model in model_info:
        if "Model" in model_info[model]:
            final_model_info[model] = model_info[model]
    model_info = final_model_info
    
    exclude_keys = ['starting_from']
    for key in exclude_keys:
        for model in model_info:
            if key in model_info[model]:
                del model_info[model][key]
    df = pd.DataFrame(model_info).T
    df = df[fields]
    # sort by anony rating
    df = df.sort_values(by=["Arena Elo rating (anony)"], ascending=False)
    df.to_csv(output_csv, index=False)
    print("Leaderboard data saved to", output_csv)
    print(df)
    

if __name__ == "__main__":
    fire.Fire(main)