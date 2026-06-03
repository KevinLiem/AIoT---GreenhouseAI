========================================================================
         GREENHOUSE AUTOMATION VIA Q-LEARNING (RL) AND DQN
========================================================================

1. OVERVIEW
-----------
This project implements a prototype Reinforcement Learning (RL) agent 
trained via Q-Learning to manage and optimize environmental conditions 
inside a smart greenhouse[cite: 1]. The agent observes three dynamic 
environmental variables—Moisture, Temperature, and Sunlight—and learns [cite: 2]
to take strategic actions to maximize plant comfort while minimizing [cite: 2]
hazardous operational mistakes[cite: 2].

The system architecture features a structured evaluation pipeline divided 
into three strict academic segments: Reward Function, Training Phase, 
and Validation.

2. SYSTEM ARCHITECTURE
----------------------

### STATE SPACE (27 Total Combinations)
The environment tracks three features, each containing three discrete levels: [cite: 3]
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
The agent's objective is driven entirely by a feedback-driven reward system: [cite: 3]

* Positive Reinforcements (Per Step):
  + Optimal Moisture:    +3 [cite: 3]
  + Optimal Temperature: +3 [cite: 3]
  + Medium Sunlight:     +4 [cite: 4]

* Penalty Safeguards (Negative Rewards):
  - Activating "Water Pump ON" when state is already "Wet":  -5 [cite: 4]
  - Activating "Water Pump OFF" when state is already "Dry":  -2 [cite: 4]
  - Activating "Heater ON" when state is already "High":     -5 [cite: 4]

4. EVALUATION PIPELINE
----------------------
The model uses the classic Bellman Optimality Equation to iteratively update [cite: 4]
its Q-table mapping matrix: [cite: 4]

    Q(s, a) = Q(s, a) + alpha * [Reward + gamma * max(Q(s', a')) - Q(s, a)] [cite: 4]

* Learning Rate (alpha): 0.2    (Controls how fast new data replaces old info) [cite: 4, 5]
* Discount Factor (gamma): 0.9  (Balances immediate rewards vs. long-term gains) [cite: 5]
* Exploration Strategy: Epsilon-Greedy (Starts at 1.0, decays by 0.02 down to 0.05 floor) [cite: 5]

### THE 3-STEP FLOW:
1. REWARD FUNCTION: Defines the mathematical objective boundaries (+3/+4 targets vs. -5 penalties). [cite: 3, 4]
2. TRAINING PHASE: Runs for exactly 100 discrete episodes (50 steps each). The agent 
   explores and exploits the state-action space to systematically build out the Q-table.
3. VALIDATION PHASE: Freezes learning parameters and locks exploration (Epsilon = 0). 
   Evaluates the fixed Q-table across 10 diagnostic episodes starting from randomized 
   initial states to calculate objective performance baselines.

5. REQUISITES & DEPENDENCIES
----------------------------
Make sure you have Python 3.x installed alongside the following libraries:
* NumPy
* Matplotlib (Required for evaluation plotting)

To install dependencies via pip:
$ pip install numpy matplotlib

6. HOW TO RUN THE SIMULATION & EVALUATION
-----------------------------------------
1. Open your terminal or command prompt.
2. Navigate to the directory containing the scripts.
3. Run either training script:
   
   $ python GreenhouseAI.py
   $ python GreenhouseAIDQN.py

4. The training scripts run training alongside periodic validation checks.
5. Upon completion, a visualization chart containing the training and validation learning curves (Avg Reward per Step and Safety Penalty Counts) is saved and displayed:
   - For Q-learning: saved as 'qlearning_val_results.png'
   - For DQN: saved as 'dqn_val_results.png'
   - Log files are also generated: '*_training_logs.csv' and '*_validation_logs.csv'

6. Compare the models by running the comparison script:

   $ python compare_models.py

   Upon completion, the script:
   - Generates 'comparison_report.txt' documenting the configured RL parameters (Learning Rate, Discount, Epsilon, etc.) and metrics side-by-side.
   - Saves 'comparison_testing.png' comparing learning curves & penalties in the training phase.
   - Saves 'comparison_validation.png' comparing learning curves & penalties in the validation phase.
========================================================================