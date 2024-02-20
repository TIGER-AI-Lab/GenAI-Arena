import pickle
with open("./results/latest/elo_results.pkl",'rb') as f:
    data = pickle.load(f)
    print()
    df = data["anony"]["leaderboard_table_df"]
    # sort by rating 
    df = df.sort_values(by=["rating"], ascending=False)
    print(df)

    print()

    df = data["full"]["leaderboard_table_df"]
    # sort by rating 
    df = df.sort_values(by=["rating"], ascending=False)
    print(df)
    print('done') 