import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd  # Added for CSV generation

# ==========================================
# 1. ENVIRONMENT & REWARD SYSTEM SELECTION
# ==========================================
moisture_states = ["Dry", "Optimal", "Wet"] 
temp_states = ["Low", "Optimal", "High"]
sun_states = ["Low", "Medium", "High"]

actions = [
    "Do nothing", "Water Pump ON", "Water Pump OFF", 
    "Heater ON", "Heater OFF", "Open shade curtains", "Close shade curtains"
]

state_space = [(m, t, s) for m in moisture_states for t in temp_states for s in sun_states]

# DQN Parameters
alpha = 0.01          # Learning rate for the neural network optimizer
gamma = 0.9           # Discount factor for future rewards
epsilon = 0.9         # Exploration rate
min_epsilon = 0.05    # Minimum exploration floor
decay_rate = 0.02     # Epsilon decay per episode

num_episodes = 100
steps_per_episode = 50

def get_reward(state, action_idx):
    m, t, s = state
    action = actions[action_idx]
    reward = 0
    
    # Positive Reinforcements
    if m == "Optimal": reward += 3
    if t == "Optimal": reward += 3
    if s == "Medium":  reward += 4
    
    # Penalty Safeguards
    if m == "Wet" and action == "Water Pump ON": reward -= 5
    if m == "Dry" and action == "Water Pump OFF": reward -= 2
    if t == "High" and action == "Heater ON": reward -= 5
    return reward

def simulate_environment(state, action_idx):
    m, t, s = state
    action = actions[action_idx]
    if random.random() < 0.9: 
        if action == "Water Pump ON": m = "Optimal" if m == "Dry" else "Wet"
        if action == "Water Pump OFF": m = "Dry" if m == "Optimal" else "Optimal"
        if action == "Heater ON": t = "Optimal" if t == "Low" else "High"
        if action == "Heater OFF": t = "Low" if t == "Optimal" else "Optimal"
        if action == "Open shade curtains": s = "Medium" if s == "Low" else "High"
        if action == "Close shade curtains": s = "Medium" if s == "High" else "Low"
    
    if random.random() < 0.05: 
        s = random.choice(sun_states)
    return (m, t, s)

# Helper: Convert categorical state tuple into a One-Hot encoded tensor for the Neural Network
def state_to_tensor(state):
    state_idx = state_space.index(state)
    one_hot = np.zeros(len(state_space))
    one_hot[state_idx] = 1.0
    return torch.FloatTensor(one_hot)

# ==========================================
# REPLACEMENT: DEEP Q-NETWORK ARCHITECTURE
# ==========================================
class GreenhouseDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GreenhouseDQN, self).__init__()
        # Simple feedforward network: 27 Inputs -> 64 Hidden Nodes -> 7 Output Action Values
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

# Instantiate the Network Brain and its Optimizer
dqn_brain = GreenhouseDQN(input_dim=len(state_space), output_dim=len(actions))
optimizer = optim.Adam(dqn_brain.parameters(), lr=alpha)
loss_fn = nn.MSELoss()

# ==========================================
# 2. TRAINING PHASE (100 Episodes)
# ==========================================
print("Running Deep Q-Network Training Phase (100 Episodes)...")
episode_avg_rewards = []
episode_penalties = []
episode_val_rewards = []
episode_val_penalties = []

# Container to hold step-by-step logs for CSV export
training_logs = []

for episode in range(num_episodes):
    current_state = random.choice(state_space)
    total_episode_reward = 0
    training_penalties_count = 0
    
    for step in range(steps_per_episode):
        state_tensor = state_to_tensor(current_state)
        
        # Predict Q-values from Neural Network
        q_values = dqn_brain(state_tensor)
        
        # Epsilon-Greedy Exploration Action Choice
        if random.random() < epsilon:
            action_idx = random.randint(0, len(actions)-1)
        else:
            action_idx = torch.argmax(q_values).item()
            
        action = actions[action_idx]
        reward = get_reward(current_state, action_idx)
        total_episode_reward += reward
        if reward < 0:
            training_penalties_count += 1
            
        next_state = simulate_environment(current_state, action_idx)
        next_tensor = state_to_tensor(next_state)
        
        # Calculate Target Q-value using Bellman targets
        with torch.no_grad():
            next_q_values = dqn_brain(next_tensor)
            max_next_q = torch.max(next_q_values)
            target_q = reward + gamma * max_next_q
            
        # Current Prediction vector copy
        current_q_target = q_values.clone().detach()
        current_q_target[action_idx] = target_q
        
        # Log data before running backpropagation & transitioning state variables
        training_logs.append({
            "Episode": episode + 1,
            "Step": step + 1,
            "State_Moisture": current_state[0],
            "State_Temperature": current_state[1],
            "State_Sunlight": current_state[2],
            "Action": action,
            "Reward": reward,
            "Next_Moisture": next_state[0],
            "Next_Temperature": next_state[1],
            "Next_Sunlight": next_state[2],
            "Epsilon": round(epsilon, 4)
        })
        
        # Compute Loss & Update Neural Network Weights via Backpropagation
        optimizer.zero_grad()
        loss = loss_fn(q_values, current_q_target)
        loss.backward()
        optimizer.step()
        
        current_state = next_state
        
    epsilon = max(min_epsilon, epsilon - decay_rate)
    episode_avg_rewards.append(total_episode_reward / steps_per_episode)
    episode_penalties.append(training_penalties_count)

    # Periodic Validation Check (exploitation only)
    dqn_brain.eval()
    val_rewards_this_ep = []
    val_penalties_this_ep = []
    with torch.no_grad():
        for _ in range(5):
            val_state = random.choice(state_space)
            val_ep_reward = 0
            val_ep_penalties = 0
            for _ in range(steps_per_episode):
                val_tensor = state_to_tensor(val_state)
                val_q_values = dqn_brain(val_tensor)
                val_action_idx = torch.argmax(val_q_values).item()
                val_reward = get_reward(val_state, val_action_idx)
                val_ep_reward += val_reward
                if val_reward < 0:
                    val_ep_penalties += 1
                val_state = simulate_environment(val_state, val_action_idx)
            val_rewards_this_ep.append(val_ep_reward / steps_per_episode)
            val_penalties_this_ep.append(val_ep_penalties)
    dqn_brain.train()
    episode_val_rewards.append(np.mean(val_rewards_this_ep))
    episode_val_penalties.append(np.mean(val_penalties_this_ep))

# Export collected DQN logs to a CSV File
df_logs = pd.DataFrame(training_logs)
df_logs.to_csv("dqn_greenhouse_training_logs.csv", index=False)
print("DQN Training data successfully saved to 'dqn_greenhouse_training_logs.csv'!")

# Export validation logs to a CSV File
df_val_logs = pd.DataFrame({
    "Episode": range(1, num_episodes + 1),
    "Val_Reward": episode_val_rewards,
    "Val_Penalties": episode_val_penalties
})
df_val_logs.to_csv("dqn_greenhouse_validation_logs.csv", index=False)
print("DQN Validation data successfully saved to 'dqn_greenhouse_validation_logs.csv'!")

# ==========================================
# 3. VALIDATION & TESTING PHASE
# ==========================================
print("Running Live Testing Phase with Frozen Parameters...")
test_steps = 100
test_state = random.choice(state_space)

moisture_history, temp_history, sun_history = [], [], []
state_map = {"Dry": 0, "Low": 0, "Optimal": 1, "Medium": 1, "Wet": 2, "High": 2}

# Turn off gradient calculation for pure exploitation evaluation
dqn_brain.eval()
with torch.no_grad():
    for _ in range(test_steps):
        state_tensor = state_to_tensor(test_state)
        q_values = dqn_brain(state_tensor)
        action_idx = torch.argmax(q_values).item() # Purely choose highest network prediction
        
        moisture_history.append(state_map[test_state[0]])
        temp_history.append(state_map[test_state[1]])
        sun_history.append(state_map[test_state[2]])
        
        test_state = simulate_environment(test_state, action_idx)

print("Evaluation Complete. Rendering Visual Plots...")

# ==========================================
# PLOTTING DIAGRAMS
# ==========================================
# 1. Training vs Validation Reward & Penalty Curves
fig1, axes1 = plt.subplots(2, 1, figsize=(12, 9))

# Training vs Validation Reward Curves
axes1[0].plot(range(1, num_episodes + 1), episode_avg_rewards, color='teal', marker='o', markersize=3, linestyle='--', alpha=0.7, label='DQN Training Avg Reward')
axes1[0].plot(range(1, num_episodes + 1), episode_val_rewards, color='darkgreen', marker='s', markersize=3, linestyle='-', linewidth=2, label='DQN Validation Avg Reward')
axes1[0].set_title('DQN Performance: Training vs. Validation Avg Reward')
axes1[0].set_xlabel('Episodes')
axes1[0].set_ylabel('Average Reward (Per Step)')
axes1[0].grid(True, linestyle='--', alpha=0.5)
axes1[0].legend()

# Training vs Validation Penalty Counts
axes1[1].plot(range(1, num_episodes + 1), episode_penalties, color='orange', marker='o', markersize=3, linestyle='--', alpha=0.7, label='DQN Training Penalties')
axes1[1].plot(range(1, num_episodes + 1), episode_val_penalties, color='red', marker='s', markersize=3, linestyle='-', linewidth=2, label='DQN Validation Penalties')
axes1[1].set_title('DQN Safety: Training vs. Validation Penalty Count per Episode')
axes1[1].set_xlabel('Episodes')
axes1[1].set_ylabel('Number of Penalties (Steps with Reward < 0)')
axes1[1].grid(True, linestyle='--', alpha=0.5)
axes1[1].legend()

fig1.tight_layout()
fig1.savefig('dqn_val_results.png', dpi=300)
print("DQN training/validation visualization saved successfully as 'dqn_val_results.png'!")

plt.show()