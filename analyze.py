import wandb
import json

api = wandb.Api()
run = api.run("ysl2683-seoul-national-university-ofscience-and-technology/square_residual_rl/zfh1cpzv")

keys = [
    "eval/success_rate", 
    "train/critic_qt", 
    "train/critic_loss",
    "lane/PBRS_dense_avg", 
    "lane/Phi_next_avg", 
    "lane/S_main_next_avg", 
    "lane/S_wrist_next_avg",
    "lane/action_l2_penalty",
    "data/batch_R",
    "data/batch_terminal_R"
]

history = run.history(keys=keys, samples=500)
stats = {}

for k in keys:
    if k in history.columns:
        valid_vals = history[k].dropna()
        if len(valid_vals) > 0:
            stats[k] = {
                "min": valid_vals.min(),
                "max": valid_vals.max(),
                "mean": valid_vals.mean(),
                "start": valid_vals.iloc[0],
                "end": valid_vals.iloc[-1]
            }

print(json.dumps(stats, indent=2))
