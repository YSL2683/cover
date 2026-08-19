import wandb
import pandas as pd

api = wandb.Api()
runs = api.runs("square_residual_rl")

# Get the most recent run (no_step_penalty) and the original scale100 run
recent_runs = []
for run in runs:
    if "SquareID" in run.name:
        recent_runs.append(run)
        
print("Found runs:")
for r in recent_runs[:5]:
    print(f"- {r.name} (ID: {r.id}, State: {r.state})")

# Let's extract metrics for the current run and the old scale100 run
# The current run is likely "SquareID_reward_pbrs_no_step_penalty_beta1.0_scale100.0"
# The old run is likely "SquareID_reward_pbrs_beta1.0_scale100.0"

target_runs = {
    "NEW (No Step Penalty, Scale 100)": api.run("square_residual_rl/1oo5s05u")
}


for label, run in target_runs.items():
    if run is None:
        continue
    print(f"\n=======================================================")
    print(f"Metrics for {label}: {run.name}")
    print(f"=======================================================")
    
    # Get summary metrics
    summary = run.summary
    keys_to_check = [
        "eval/success_rate", 
        "data/batch_terminal_R", 
        "data/batch_R",
        "train/critic_qt", 
        "lane/PBRS_dense_avg",
        "lane/action_l2_penalty",
        "train/actor_loss_base",
        "train/critic_loss"
    ]
    
    print(f"Current Step: {summary.get('_step', 'N/A')}")
    for k in keys_to_check:
        val = summary.get(k, "N/A")
        if isinstance(val, float):
            print(f"  {k}: {val:.4f}")
        else:
            print(f"  {k}: {val}")
    
    # Let's get history for the last 50 steps to see the trend
    history = run.history(keys=["eval/success_rate", "train/critic_qt", "lane/PBRS_dense_avg", "train/actor_loss_base", "data/batch_terminal_R"], samples=30)
    print("\nRecent History (Last 30 samples):")
    print(history.to_string(index=False))
