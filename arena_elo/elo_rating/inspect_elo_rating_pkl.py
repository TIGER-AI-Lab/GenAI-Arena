import pickle
import plotly.graph_objects as go

def output_figure(data, figure_name="battle_count_heatmap", label="annoy"):
    fig = data[label][figure_name]
    fig.update_layout(
        height=700,
        width=700,
        title={'text': f'{figure_name}', 'x': 0.5, 'y': 0.07},
        xaxis_title="Model B",
        yaxis_title="Model A",
        # coloraxis_colorscale=[[0.0, '#0d0887'], [1.0, '#f0f921']],
        margin={'t': 60}
    )
    fig.write_image(f"{figure_name}.png")

with open("./results/latest/elo_results.pkl",'rb') as f:
    data = pickle.load(f)
    print()
    df = data["anony"]["leaderboard_table_df"]
    # sort by rating 
    print(data["anony"].keys())
    
    for figure_name in [ 'win_fraction_heatmap', 'battle_count_heatmap',]:
        output_figure(data, figure_name, "anony")
        
    df = df.sort_values(by=["rating"], ascending=False)
    print(df) 
    df = data["full"]["leaderboard_table_df"]
    # sort by rating 
    df = df.sort_values(by=["rating"], ascending=False)
    print(df)
    print('done') 
