import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import libsumo as traci
import pandas as pd

# --- Configuration ---
# Update these paths and IDs to match your specific SUMO network
SUMO_CMD = ["sumo-gui",
    "-c", "Traci.sumocfg",
    '--step-length', '0.05',
    '--delay', '100',
    '--lateral-resolution', '0.1'
    # "--no-step-log", "true",
    # "--waiting-time-memory", "1000"
]
TL_ID = "J7"  
INCOMING_LANES = ["E1_0", "-E2_0", "-E3_0", "-E4_0"] 
ACTION_PHASES = [0, 2]  # Green North-South, Green East-West (Example)

# Green Phase Indices (Check your net.xml). 
# Example: 0 is Green N-S, 2 is Green E-W. (1 and 3 are usually Yellow)
PHASE_CYCLE = [0, 2] 
NUM_PHASES = len(PHASE_CYCLE)

# Model Dimensions
# State = (Queue + Wait per lane) + One-Hot Phase ID
STATE_DIM = (len(INCOMING_LANES) * 2) + NUM_PHASES
ACTION_DIM = 2  # 0: Stay, 1: Switch

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.001
YELLOW_DURATION = 100  # Seconds
GREEN_DURATION = 300  # Seconds (Minimum time between decisions)

ALPHA_QUEUE = 1.0
BETA_WAIT = 0.5


SIM_TIME = 8100

data = []
def save_data():
    wait = 0
    queue_len = 0
    for v in traci.vehicle.getIDList():
        wait += traci.vehicle.getWaitingTime(v)
    for e in traci.edge.getIDList():
        queue_len += traci.edge.getLastStepHaltingNumber(e)
    data.append({
        'time': traci.simulation.getTime(),
        'queue_len': queue_len,
        'wait': wait
    })

def sim_step():
    save_data()
    traci.simulationStep()

# --- 3. Neural Network (Actor-Critic) ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Shared Layers
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # Actor Head (Policy: Probability of Stay vs Switch)
        self.actor = nn.Linear(64, action_dim)
        
        # Critic Head (Value: Estimate of state quality)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Actor outputs probabilities (Softmax)
        action_probs = F.softmax(self.actor(x), dim=-1)
        
        # Critic outputs scalar value
        state_value = self.critic(x)
        
        return action_probs, state_value

# --- 4. Environment Helper Functions ---
def get_state(current_phase_idx):
    """
    Constructs the state vector:
    1. Queue Length (Normalized)
    2. Waiting Time (Normalized)
    3. Current Phase (One-Hot Encoded)
    """
    state = []
    
    # A. Lane Statistics
    for lane in INCOMING_LANES:
        queue = traci.lane.getLastStepHaltingNumber(lane)
        wait_time = traci.lane.getWaitingTime(lane)
        
        # Normalize to keep neural network inputs roughly 0-1
        state.append(queue / 50.0)      # Assume max queue ~50
        state.append(wait_time / 1000.0) # Assume max wait ~1000s
        
    # B. Phase One-Hot Encoding
    phase_one_hot = [0.0] * NUM_PHASES
    phase_one_hot[current_phase_idx] = 1.0
    state.extend(phase_one_hot)
        
    return np.array(state, dtype=np.float32)

def get_reward():
    """
    Reward is negative sum of queue and waiting time.
    We want to maximize reward => minimize queue/wait.
    """
    total_queue = 0
    total_wait = 0
    for lane in INCOMING_LANES:
        total_queue += traci.lane.getLastStepHaltingNumber(lane)
        total_wait += traci.lane.getWaitingTime(lane)
        
    # Weights: Alpha=1.0 for Queue, Beta=0.5 for Wait
    reward = - (1.0 * total_queue + 0.5 * total_wait)
    return reward

# --- 5. Main Training Loop ---
def run_simulation():
    # Initialize Model & Optimizer
    model = ActorCritic(STATE_DIM, ACTION_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Start Simulation
    traci.start(SUMO_CMD)

    step = 0
    current_phase_idx = 0 
    
    # Set initial phase
    traci.trafficlight.setPhase(TL_ID, PHASE_CYCLE[current_phase_idx])
    
    # Run for 1 hour (3600 steps)
    while step < SIM_TIME:
        
        # --- A. Observe State ---
        state_np = get_state(current_phase_idx)
        state_tensor = torch.from_numpy(state_np).unsqueeze(0)
        
        # --- B. Agent Decision ---
        action_probs, value = model(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_val = action.item() # 0 = Stay, 1 = Switch
        
        reward_accumulated = 0
        
        # --- C. Execute Action ---
        if action_val == 1:
            # === SWITCH SEQUENCE ===
            # 1. Determine Yellow Phase (Assuming GreenID + 1 is Yellow)
            current_green_id = PHASE_CYCLE[current_phase_idx]
            yellow_phase_id = current_green_id + 1
            
            # 2. Set Yellow & Simulate
            traci.trafficlight.setPhase(TL_ID, yellow_phase_id)
            for _ in range(YELLOW_DURATION):
                sim_step()
                step += 1
                reward_accumulated += get_reward()
            
            # 3. Update Index to Next Green
            current_phase_idx = (current_phase_idx + 1) % NUM_PHASES
            next_green_id = PHASE_CYCLE[current_phase_idx]
            traci.trafficlight.setPhase(TL_ID, next_green_id)
            
        # If action_val == 0 (Stay), we do nothing here and just continue the current Green.

        # --- D. Green Duration (Minimum Hold Time) ---
        # We hold the (new or existing) Green light for 10 seconds
        for _ in range(GREEN_DURATION):
            if step >= SIM_TIME: break
            sim_step()
            step += 1
            reward_accumulated += get_reward()
            
        # --- E. Learn (Actor-Critic Update) ---
        # 1. Observe New State
        next_state_np = get_state(current_phase_idx)
        next_state_tensor = torch.from_numpy(next_state_np).unsqueeze(0)
        
        # 2. Calculate Target (TD Target)
        _, next_value = model(next_state_tensor)
        target_value = reward_accumulated + GAMMA * next_value.item()
        
        # 3. Calculate Advantage (TD Error)
        advantage = target_value - value.item()
        
        # 4. Compute Losses
        # Critic attempts to minimize prediction error
        critic_loss = F.mse_loss(value, torch.tensor([[target_value]]))
        
        # Actor attempts to maximize expected reward (minimize -log_prob * advantage)
        log_prob = dist.log_prob(action)
        actor_loss = -log_prob * advantage
        
        total_loss = actor_loss + critic_loss
        
        # 5. Backpropagate
        optimizer.zero_grad()
        total_loss.backward()
        # Clip gradients to prevent explosion during heavy traffic jams
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # --- F. Logging ---
        if step % 100 < GREEN_DURATION + YELLOW_DURATION: # Print roughly every 100s
            print(f"Time: {step}s | Phase: {current_phase_idx} | "
                  f"Action: {'Switch' if action_val==1 else 'Stay'} | "
                  f"Reward: {reward_accumulated:.2f} | Loss: {total_loss.item():.4f}")

    torch.save(model.state_dict(), "ac.pth")
    traci.close()
    print("Simulation finished.")

    df = pd.DataFrame(data)
    df.to_csv("ac.csv")

if __name__ == "__main__":
    run_simulation()
