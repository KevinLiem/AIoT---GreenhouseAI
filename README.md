# AIoT---GreenhouseAI
Midterm for AIoT

========================================================================
           GREENHOUSE AUTOMATION VIA Q-LEARNING (RL)
========================================================================

1. OVERVIEW
-----------
This project implements a prototype Reinforcement Learning (RL) agent 
trained via Q-Learning to manage and optimize environmental conditions 
inside a smart greenhouse[cite: 1]. The agent observes three dynamic 
environmental variables—Moisture, Temperature, and Sunlight—and learns 
to take strategic actions to maximize plant comfort while minimizing 
hazardous operational mistakes[cite: 2].

The system has been updated to transition from an infinite live loop to a 
structured pipeline containing distinct Training, Validation, and Testing phases.

2. SYSTEM ARCHITECTURE
----------------------

### STATE SPACE (27 Total Combinations)
The environment tracks three features, each containing three discrete levels[cite: 3]:
* Moisture:    ["Dry", "Optimal", "Wet"] [cite: 3]
* Temperature: ["Low", "Optimal", "High"] [cite: 3]
* Sunlight:    ["Low", "Medium", "High"] [cite: 3]

### ACTION SPACE (7 Available Actions)
* "Do nothing" [cite: 3]
* "Water Pump ON"   / "Water Pump OFF" [cite: 3]
* "Heater ON"       / "Heater OFF" [cite: 3]
* "Open shade curtains" / "Close shade curtains" [cite: 3]

3. REWARD SYSTEM MECHANICS
--------------------------
The agent's objective is driven entirely by a feedback-driven reward system[cite: 3]:

* Positive Reinforcements (Per Step):
  + Optimal Moisture:    +3 [cite: 3]
  + Optimal Temperature: +3 [cite: 3]
  + Medium Sunlight:     +4 [cite: 4]

* Penalty Safeguards (Negative Rewards):
  - Activating "Water Pump ON" when state is already "Wet":  -5 [cite: 3]
  - Activating "Water Pump OFF" when state is already "Dry":  -2 [cite: 3]
  - Activating "Heater ON" when state is already "High":     -5 [cite: 3]

4. PIPELINE PHASES & ALGORITHM
------------------------------
The model uses the classic Bellman Optimality Equation to iteratively update 
its Q-table mapping[cite: 4]:

    Q(s, a) = Q(s, a) + alpha * [Reward + gamma * max(Q(s', a')) - Q(s, a)] [cite: 4]

* Learning Rate (alpha): 0.2    (Controls how fast new data replaces old info) [cite: 4, 5]
* Discount Factor (gamma): 0.9  (Balances immediate rewards vs. long-term gains) [cite: 5]
* Exploration Strategy: Epsilon-Greedy [cite: 5]
  - Starts at 1.0 (100% random exploration) [cite: 5]
  - Decays by 0.001 per episode down to a baseline floor of 0.05 [cite: 5]

### THE 3-STEP PIPELINE RUNTIME:
1. TRAINING PHASE: Runs for 2,000 discrete episodes (50 steps each). The agent 
   actively alternates between exploration and exploitation to fill the Q-table.
2. VALIDATION PHASE: Freezes learning parameters (Epsilon = 0). Evaluates the 
   agent across 20 diagnostic episodes to gather summary baseline statistics.
3. TESTING PHASE: Simulates a live 100-step test deployment tracking real-time 
   state adjustments against random environmental weather drifts.

5. REQUISITES & DEPENDENCIES
----------------------------
Make sure you have Python 3.x installed alongside the following libraries[cite: 5]:
* NumPy [cite: 5]
* Matplotlib (Required for evaluation plotting)

To install dependencies via pip:
$ pip install numpy matplotlib

6. HOW TO RUN THE SIMULATION & EVALUATION
-----------------------------------------
1. Open your terminal or command prompt[cite: 6].
2. Navigate to the directory containing the script[cite: 6].
3. Run the script using Python[cite: 6]:
   
   $ python GreenhouseAI.py

4. The pipeline will automatically cycle through Training, Validation logging, 
   and Testing tracking.
5. Upon completion, two interactive visualization windows will render:
   - Plot 1: The Training Learning Curve (Cumulative Rewards vs Episodes)
   - Plot 2: The Testing Environment Tracker (Moisture, Temp, and Sun metrics 
             mapped step-by-step to demonstrate target zone stability)
========================================================================
