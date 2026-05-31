import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd  # Added for CSV generation

# 1. Environment Setup
moisture_states = ["Dry", "Optimal", "Wet"] 
temp_states = ["Low", "Optimal", "High"]
sun_states = ["Low", "Medium", "High"]

actions = [
    "Do nothing", "Water Pump ON", "Water Pump OFF", 
    "Heater ON", "Heater OFF", "Open shade curtains", "Close shade curtains"
]

state_space = [(m, t, s) for m in moisture_states for t in temp_states for s in sun_states]
q_table = np.zeros((len(state_space), len(actions)))

# RL Parameters
alpha = 0.2
gamma = 0.9
epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.02 

num_episodes = 100
steps_per_episode = 50

def get_reward(state, action):
    m, t, s = state
    reward = 0
    if m == "Optimal": reward += 3
    if t == "Optimal": reward += 3
    if s == "Medium":  reward += 4
    
    if m == "Wet" and action == "Water Pump ON": reward -= 5
    if m == "Dry" and action == "Water Pump OFF": reward -= 2
    if t == "High" and action == "Heater ON": reward -= 5
    return reward

def simulate_environment(state, action):
    m, t, s = state
    if random.random() < 0.9: # 90% action success
        if action == "Water Pump ON": m = "Optimal" if m == "Dry" else "Wet"
        if action == "Water Pump OFF": m = "Dry" if m == "Optimal" else "Optimal"
        if action == "Heater ON": t = "Optimal" if t == "Low" else "High"
        if action == "Heater OFF": t = "Low" if t == "Optimal" else "Optimal"
        if action == "Open shade curtains": s = "Medium" if s == "Low" else "High"
        if action == "Close shade curtains": s = "Medium" if s == "High" else "Low"
    
    if random.random() < 0.05: # Weather drift
        s = random.choice(sun_states)
    return (m, t, s)

# 1. TRAINING PHASE (100 Episodes)

print("Running Training Phase (100 Episodes)...")
episode_avg_rewards = []

# Container to hold step-by-step logs for CSV export
training_logs = []

for episode in range(num_episodes):
    current_state = random.choice(state_space)
    total_episode_reward = 0
    
    for step in range(steps_per_episode):
        state_idx = state_space.index(current_state)
        
        if random.random() < epsilon:
            action_idx = random.randint(0, len(actions)-1)
        else:
            action_idx = np.argmax(q_table[state_idx])
            
        action = actions[action_idx]
        reward = get_reward(current_state, action)
        total_episode_reward += reward
        
        next_state = simulate_environment(current_state, action)
        next_idx = state_space.index(next_state)
        best_future_q = np.max(q_table[next_idx])
        
        # Log data before transitioning state variables
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
        
        q_table[state_idx, action_idx] += alpha * (reward + gamma * best_future_q - q_table[state_idx, action_idx])
        current_state = next_state
        
    epsilon = max(min_epsilon, epsilon - decay_rate)
    
    # Calculate the average reward for this specific episode
    avg_reward = total_episode_reward / steps_per_episode
    episode_avg_rewards.append(avg_reward)

# Export collected logs to a CSV File
df_logs = pd.DataFrame(training_logs)
df_logs.to_csv("rl_greenhouse_training_logs.csv", index=False)
print("Training data successfully saved to 'rl_greenhouse_training_logs.csv'!")

# 2. VALIDATION PHASE
print("Running Validation Phase...")
validation_rewards = []
for _ in range(10):
    current_state = random.choice(state_space)
    total_val_reward = 0
    for _ in range(steps_per_episode):
        state_idx = state_space.index(current_state)
        action_idx = np.argmax(q_table[state_idx])
        total_val_reward += get_reward(current_state, actions[action_idx])
        current_state = simulate_environment(current_state, actions[action_idx])
    validation_rewards.append(total_val_reward / steps_per_episode)

# 3. TESTING PHASE WITH ENVIRONMENT TRACKING
print("Running Live Testing Phase...")
test_steps = 100
test_state = random.choice(state_space)

moisture_history = []
temp_history = []
sun_history = []

# Map states to numbers for clear graphing: Low/Dry=0, Optimal/Medium=1, High/Wet=2
state_map = {"Dry": 0, "Low": 0, "Optimal": 1, "Medium": 1, "Wet": 2, "High": 2}

for _ in range(test_steps):
    state_idx = state_space.index(test_state)
    action_idx = np.argmax(q_table[state_idx]) 
    action = actions[action_idx]
    
    moisture_history.append(state_map[test_state[0]])
    temp_history.append(state_map[test_state[1]])
    sun_history.append(state_map[test_state[2]])
    
    test_state = simulate_environment(test_state, action)

print("Testing complete. Plotting results...")

# PLOTTING
# Training Evaluation (Average Reward per Episode)
plt.figure(figsize=(10, 4))
plt.plot(range(1, num_episodes + 1), episode_avg_rewards, color='teal', marker='o', markersize=4, linestyle='-', linewidth=1.5, label='Avg Reward per Episode')
plt.title('Training Phase: Performance Evaluation (100 Episodes)')
plt.xlabel('Episodes')
plt.ylabel('Average Reward (Per Step)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

# Testing Phase (Greenhouse Stability Tracker)
plt.figure(figsize=(12, 5))
plt.step(range(test_steps), moisture_history, label='Moisture Level', color='blue', alpha=0.8, where='mid')
plt.step(range(test_steps), temp_history, label='Temperature Level', color='red', alpha=0.8, where='mid')
plt.step(range(test_steps), sun_history, label='Sunlight Level', color='gold', alpha=0.8, where='mid')

plt.axhline(y=1, color='green', linestyle=':', linewidth=2, label='Target Zone (Optimal/Medium)')
plt.yticks([0, 1, 2], ['Low / Dry', 'Optimal / Medium', 'High / Wet'])
plt.title('Testing Phase: Automated Greenhouse Resource Metrics over 100 Steps')
plt.xlabel('Test Step Duration')
plt.ylabel('Environmental Status')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')
plt.tight_layout()

plt.show()