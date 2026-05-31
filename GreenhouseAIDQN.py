import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

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
epsilon = 1.0         # Exploration rate
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

# Container to hold step-by-step logs for CSV export
training_logs = []

for episode in range(num_episodes):
    current_state = random.choice(state_space)
    total_episode_reward = 0
    
    for step in range(steps_per_episode):
        # Predict Q-values from Neural Network
        optimizer.zero_grad()
        state_tensor = state_to_tensor(current_state)
        q_values = dqn_brain(state_tensor)
        
        # Epsilon-Greedy Exploration Action Choice
        if random.random() < epsilon:
            action_idx = random.randint(0, len(actions)-1)
        else:
            action_idx = torch.argmax(q_values).item()
            
        reward = get_reward(current_state, action_idx)
        total_episode_reward += reward
        
        next_state = simulate_environment(current_state, action_idx)
        next_tensor = state_to_tensor(next_state)
        
        # Log data before transitioning state variables
        training_logs.append({
            "Episode": episode + 1,
            "Step": step + 1,
            "State_Moisture": current_state[0],
            "State_Temperature": current_state[1],
            "State_Sunlight": current_state[2],
            "Action": actions[action_idx],
            "Reward": reward,
            "Next_Moisture": next_state[0],
            "Next_Temperature": next_state[1],
            "Next_Sunlight": next_state[2],
            "Epsilon": round(epsilon, 4)
        })
        
        # Calculate Target Q-value using Bellman targets
        with torch.no_grad():
            next_q_values = dqn_brain(next_tensor)
            max_next_q = torch.max(next_q_values)
            target_q = reward + gamma * max_next_q
            
        # Current Prediction vector copy
        current_q_target = q_values.clone().detach()
        current_q_target[action_idx] = target_q
        
        # Compute Loss & Update Neural Network Weights via Backpropagation
        loss = loss_fn(q_values, current_q_target)
        loss.backward()
        optimizer.step()
        
        current_state = next_state
        
    epsilon = max(min_epsilon, epsilon - decay_rate)
    episode_avg_rewards.append(total_episode_reward / steps_per_episode)

# Export collected logs to a CSV File
df_logs = pd.DataFrame(training_logs)
df_logs.to_csv("DQN_greenhouse_training_logs.csv", index=False)
print("Training data successfully saved to 'DQN_greenhouse_training_logs.csv'!")

# ==========================================
# 3. VALIDATION PHASE
# ==========================================
print("Running Validation Phase...")
validation_rewards = []
dqn_brain.eval()
with torch.no_grad():
    for _ in range(10):
        current_state = random.choice(state_space)
        total_val_reward = 0
        for _ in range(steps_per_episode):
            state_tensor = state_to_tensor(current_state)
            q_values = dqn_brain(state_tensor)
            action_idx = torch.argmax(q_values).item()
            total_val_reward += get_reward(current_state, action_idx)
            current_state = simulate_environment(current_state, action_idx)
        validation_rewards.append(total_val_reward / steps_per_episode)

# ==========================================
# 4. LIVE TESTING PHASE
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
plt.figure(figsize=(10, 4))
plt.plot(range(1, num_episodes + 1), episode_avg_rewards, color='teal', marker='o', markersize=4, label='DQN Avg Reward per Episode')
plt.title('Training Phase: DQN Performance Convergence (100 Episodes)')
plt.xlabel('Episodes')
plt.ylabel('Average Reward (Per Step)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(12, 5))
plt.step(range(test_steps), moisture_history, label='Moisture Level', color='blue', alpha=0.8, where='mid')
plt.step(range(test_steps), temp_history, label='Temperature Level', color='red', alpha=0.8, where='mid')
plt.step(range(test_steps), sun_history, label='Sunlight Level', color='gold', alpha=0.8, where='mid')
plt.axhline(y=1, color='green', linestyle=':', linewidth=2, label='Target Zone (Optimal/Medium)')
plt.yticks([0, 1, 2], ['Low / Dry', 'Optimal / Medium', 'High / Wet'])
plt.title('Validation Phase: DQN Controlled Environment Stability Tracker')
plt.xlabel('Test Step Duration')
plt.ylabel('Environmental Status')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')
plt.tight_layout()

plt.show()