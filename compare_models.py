import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend to avoid blocking
import matplotlib.pyplot as plt
import os

def get_rl_params(file_path):
    params = {}
    if not os.path.exists(file_path):
        return {"alpha": "N/A", "gamma": "N/A", "epsilon": "N/A", "min_epsilon": "N/A", "decay_rate": "N/A"}
    with open(file_path, "r") as f:
        for line in f:
            line_clean = line.strip()
            # Stop parsing when reaching training phase section to avoid overwriting variables in loops
            if line_clean.startswith("# 1. TRAINING") or line_clean.startswith("# 2. TRAINING"):
                break
            if "#" in line_clean:
                line_clean = line_clean.split("#")[0].strip()
            if "=" in line_clean:
                parts = line_clean.split("=")
                key = parts[0].strip()
                val = parts[1].strip()
                if key in ["alpha", "gamma", "epsilon", "min_epsilon", "decay_rate"]:
                    try:
                        params[key] = float(val)
                    except ValueError:
                        params[key] = val
    return params

def analyze_and_compare():
    rl_path = "rl_greenhouse_training_logs.csv"
    dqn_path = "dqn_greenhouse_training_logs.csv"
    rl_val_path = "rl_greenhouse_validation_logs.csv"
    dqn_val_path = "dqn_greenhouse_validation_logs.csv"
    
    # Handle potentially capitalized filenames on Windows case-sensitivity checks
    if not os.path.exists(dqn_path) and os.path.exists("DQN_greenhouse_training_logs.csv"):
        dqn_path = "DQN_greenhouse_training_logs.csv"
    if not os.path.exists(dqn_val_path) and os.path.exists("DQN_greenhouse_validation_logs.csv"):
        dqn_val_path = "DQN_greenhouse_validation_logs.csv"
        
    if not all(os.path.exists(p) for p in [rl_path, dqn_path, rl_val_path, dqn_val_path]):
        print("Error: Make sure all training and validation CSV log files exist:")
        print(f" - {rl_path}: {os.path.exists(rl_path)}")
        print(f" - {dqn_path}: {os.path.exists(dqn_path)}")
        print(f" - {rl_val_path}: {os.path.exists(rl_val_path)}")
        print(f" - {dqn_val_path}: {os.path.exists(dqn_val_path)}")
        return

    # Load data
    df_rl = pd.read_csv(rl_path)
    df_dqn = pd.read_csv(dqn_path)
    df_rl_val = pd.read_csv(rl_val_path)
    df_dqn_val = pd.read_csv(dqn_val_path)
    
    # Calculate average reward per episode
    rl_episodes = df_rl.groupby("Episode")["Reward"].mean().reset_index()
    dqn_episodes = df_dqn.groupby("Episode")["Reward"].mean().reset_index()
    
    # Calculate training statistics
    rl_overall_avg = df_rl["Reward"].mean()
    dqn_overall_avg = df_dqn["Reward"].mean()
    rl_late_avg = df_rl[df_rl["Episode"] > 80]["Reward"].mean()
    dqn_late_avg = df_dqn[df_dqn["Episode"] > 80]["Reward"].mean()
    
    # Count penalties (rewards < 0)
    rl_penalties_early = df_rl[df_rl["Episode"] <= 20]["Reward"].apply(lambda r: 1 if r < 0 else 0).sum()
    dqn_penalties_early = df_dqn[df_dqn["Episode"] <= 20]["Reward"].apply(lambda r: 1 if r < 0 else 0).sum()
    rl_penalties_late = df_rl[df_rl["Episode"] > 80]["Reward"].apply(lambda r: 1 if r < 0 else 0).sum()
    dqn_penalties_late = df_dqn[df_dqn["Episode"] > 80]["Reward"].apply(lambda r: 1 if r < 0 else 0).sum()
    
    rl_reduction = ((rl_penalties_early - rl_penalties_late) / rl_penalties_early * 100) if rl_penalties_early > 0 else 0
    dqn_reduction = ((dqn_penalties_early - dqn_penalties_late) / dqn_penalties_early * 100) if dqn_penalties_early > 0 else 0

    # Calculate validation statistics
    rl_val_overall_avg = df_rl_val["Val_Reward"].mean()
    dqn_val_overall_avg = df_dqn_val["Val_Reward"].mean()
    rl_val_late_avg = df_rl_val[df_rl_val["Episode"] > 80]["Val_Reward"].mean()
    dqn_val_late_avg = df_dqn_val[df_dqn_val["Episode"] > 80]["Val_Reward"].mean()
    
    rl_val_penalties_early = df_rl_val[df_rl_val["Episode"] <= 20]["Val_Penalties"].sum()
    dqn_val_penalties_early = df_dqn_val[df_dqn_val["Episode"] <= 20]["Val_Penalties"].sum()
    rl_val_penalties_late = df_rl_val[df_rl_val["Episode"] > 80]["Val_Penalties"].sum()
    dqn_val_penalties_late = df_dqn_val[df_dqn_val["Episode"] > 80]["Val_Penalties"].sum()
    
    rl_val_reduction = ((rl_val_penalties_early - rl_val_penalties_late) / rl_val_penalties_early * 100) if rl_val_penalties_early > 0 else 0
    dqn_val_reduction = ((dqn_val_penalties_early - dqn_val_penalties_late) / dqn_val_penalties_early * 100) if dqn_val_penalties_early > 0 else 0

    # Extract RL parameters
    rl_params = get_rl_params("GreenhouseAI.py")
    dqn_params = get_rl_params("GreenhouseAIDQN.py")
    
    def format_val(p_dict, key):
        val = p_dict.get(key, 'N/A')
        return f"{val:.4f}" if isinstance(val, float) else str(val)
        
    rl_alpha = format_val(rl_params, "alpha")
    dqn_alpha = format_val(dqn_params, "alpha")
    rl_gamma = format_val(rl_params, "gamma")
    dqn_gamma = format_val(dqn_params, "gamma")
    rl_epsilon = format_val(rl_params, "epsilon")
    dqn_epsilon = format_val(dqn_params, "epsilon")
    rl_min_eps = format_val(rl_params, "min_epsilon")
    dqn_min_eps = format_val(dqn_params, "min_epsilon")
    rl_decay = format_val(rl_params, "decay_rate")
    dqn_decay = format_val(dqn_params, "decay_rate")

    # Write report to text file
    report_content = f"""====================================================================================
                              MODEL COMPARISON METRICS SUMMARY           
====================================================================================
Metric                              | Tabular Q-Learning     | Deep Q-Network (DQN)
------------------------------------------------------------------------------------
RL PARAMETERS (CONFIGURED):
  Learning Rate (alpha)             | {rl_alpha:22} | {dqn_alpha:22}
  Discount Factor (gamma)           | {rl_gamma:22} | {dqn_gamma:22}
  Initial Exploration (epsilon)     | {rl_epsilon:22} | {dqn_epsilon:22}
  Minimum Epsilon (min_epsilon)     | {rl_min_eps:22} | {dqn_min_eps:22}
  Epsilon Decay Rate (decay_rate)   | {rl_decay:22} | {dqn_decay:22}
------------------------------------------------------------------------------------
TRAINING PHASE:
  Overall Avg Reward/Step           | {rl_overall_avg:22.4f} | {dqn_overall_avg:22.4f}
  Final Avg Reward/Step (Ep 81-100) | {rl_late_avg:22.4f} | {dqn_late_avg:22.4f}
  Total Penalties (Ep 1-20)         | {rl_penalties_early:22d} | {dqn_penalties_early:22d}
  Total Penalties (Ep 81-100)       | {rl_penalties_late:22d} | {dqn_penalties_late:22d}
  Penalty Reduction Rate (%)        | {rl_reduction:21.2f}% | {dqn_reduction:21.2f}%
------------------------------------------------------------------------------------
VALIDATION PHASE (Periodic exploitation evaluation):
  Overall Avg Reward/Step           | {rl_val_overall_avg:22.4f} | {dqn_val_overall_avg:22.4f}
  Final Avg Reward/Step (Ep 81-100) | {rl_val_late_avg:22.4f} | {dqn_val_late_avg:22.4f}
  Total Avg Penalties (Ep 1-20)     | {rl_val_penalties_early:22.2f} | {dqn_val_penalties_early:22.2f}
  Total Avg Penalties (Ep 81-100)   | {rl_val_penalties_late:22.2f} | {dqn_val_penalties_late:22.2f}
  Penalty Reduction Rate (%)        | {rl_val_reduction:21.2f}% | {dqn_val_reduction:21.2f}%
====================================================================================
"""
    print(report_content)
    
    with open("comparison_report.txt", "w") as f:
        f.write(report_content)

    # 1. Training/Testing Comparison Chart (comparison_testing.png)
    fig1, axes1 = plt.subplots(2, 1, figsize=(12, 10))
    
    # Training Learning Curves
    axes1[0].plot(rl_episodes["Episode"], rl_episodes["Reward"], color='#1abc9c', alpha=0.3, label='Tabular RL (Raw)')
    axes1[0].plot(rl_episodes["Episode"], rl_episodes["Reward"].rolling(5, min_periods=1).mean(), color='#16a085', linewidth=2.5, label='Tabular RL (5-Ep SMA)')
    axes1[0].plot(dqn_episodes["Episode"], dqn_episodes["Reward"], color='#3498db', alpha=0.3, label='DQN (Raw)')
    axes1[0].plot(dqn_episodes["Episode"], dqn_episodes["Reward"].rolling(5, min_periods=1).mean(), color='#2980b9', linewidth=2.5, label='DQN (5-Ep SMA)')
    axes1[0].set_title('Training Phase: Learning Curves (Tabular Q-Learning vs. DQN)', fontsize=14, fontweight='bold', pad=15)
    axes1[0].set_xlabel('Episodes', fontsize=12)
    axes1[0].set_ylabel('Average Reward per Step', fontsize=12)
    axes1[0].grid(True, linestyle='--', alpha=0.5)
    axes1[0].legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')
    
    # Training Safety Penalties
    rl_penalties_per_ep = df_rl.groupby("Episode")["Reward"].apply(lambda x: (x < 0).sum()).reset_index()
    dqn_penalties_per_ep = df_dqn.groupby("Episode")["Reward"].apply(lambda x: (x < 0).sum()).reset_index()
    
    axes1[1].plot(rl_penalties_per_ep["Episode"], rl_penalties_per_ep["Reward"].rolling(5, min_periods=1).mean(), color='#e67e22', linewidth=2.5, label='Tabular RL Penalties')
    axes1[1].plot(dqn_penalties_per_ep["Episode"], dqn_penalties_per_ep["Reward"].rolling(5, min_periods=1).mean(), color='#e74c3c', linewidth=2.5, label='DQN Penalties')
    axes1[1].set_title('Training Phase: Safety Penalty Count per Episode', fontsize=14, fontweight='bold', pad=15)
    axes1[1].set_xlabel('Episodes', fontsize=12)
    axes1[1].set_ylabel('Penalties per Episode (Max 50 Steps)', fontsize=12)
    axes1[1].grid(True, linestyle='--', alpha=0.5)
    axes1[1].legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    
    fig1.tight_layout()
    fig1.savefig('comparison_testing.png', dpi=300)
    print("Training comparison visualization saved successfully as 'comparison_testing.png'!")

    # 2. Validation Comparison Chart (comparison_validation.png)
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 10))
    
    # Validation Learning Curves
    axes2[0].plot(df_rl_val["Episode"], df_rl_val["Val_Reward"], color='#2ecc71', alpha=0.3, label='Tabular RL Val (Raw)')
    axes2[0].plot(df_rl_val["Episode"], df_rl_val["Val_Reward"].rolling(5, min_periods=1).mean(), color='#27ae60', linewidth=2.5, label='Tabular RL Val (5-Ep SMA)')
    axes2[0].plot(df_dqn_val["Episode"], df_dqn_val["Val_Reward"], color='#9b59b6', alpha=0.3, label='DQN Val (Raw)')
    axes2[0].plot(df_dqn_val["Episode"], df_dqn_val["Val_Reward"].rolling(5, min_periods=1).mean(), color='#8e44ad', linewidth=2.5, label='DQN Val (5-Ep SMA)')
    axes2[0].set_title('Validation Phase: Learning Curves (Tabular Q-Learning vs. DQN)', fontsize=14, fontweight='bold', pad=15)
    axes2[0].set_xlabel('Episodes', fontsize=12)
    axes2[0].set_ylabel('Average Reward per Step', fontsize=12)
    axes2[0].grid(True, linestyle='--', alpha=0.5)
    axes2[0].legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')
    
    # Validation Safety Penalties
    axes2[1].plot(df_rl_val["Episode"], df_rl_val["Val_Penalties"].rolling(5, min_periods=1).mean(), color='#f1c40f', linewidth=2.5, label='Tabular RL Val Penalties')
    axes2[1].plot(df_dqn_val["Episode"], df_dqn_val["Val_Penalties"].rolling(5, min_periods=1).mean(), color='#d35400', linewidth=2.5, label='DQN Val Penalties')
    axes2[1].set_title('Validation Phase: Safety Penalty Count per Episode', fontsize=14, fontweight='bold', pad=15)
    axes2[1].set_xlabel('Episodes', fontsize=12)
    axes2[1].set_ylabel('Penalties per Episode (Max 50 Steps)', fontsize=12)
    axes2[1].grid(True, linestyle='--', alpha=0.5)
    axes2[1].legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    
    fig2.tight_layout()
    fig2.savefig('comparison_validation.png', dpi=300)
    print("Validation comparison visualization saved successfully as 'comparison_validation.png'!")

if __name__ == "__main__":
    analyze_and_compare()
