import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import libsumo as traci

# --- Configuration ---
SUMO_CMD = ["sumo-gui", "-c", "sim.sumocfg", "--start", "--quit-on-end"]

# IDS
TL_ID = "GS_cluster_254515315_32892163_32892164_614336808_#4more"
INCOMING_LANES = [
    "686994635#0_0", "686994635#0_1", "686994635#0_2",
    "114183414#2_0", "114183414#2_1", "114183414#2_2",
    "912524824#0_0", "912524824#0_1",
    "1292667565#3_0", "1292667565#3_1", "1292667565#3_2"
]

# Phase Configuration
PHASE_CYCLE = [0, 2, 4] 
NUM_PHASES = len(PHASE_CYCLE)

# Hyperparameters
GAMMA = 0.95
LEARNING_RATE = 0.001 
ENTROPY_BETA = 0.01   
YELLOW_DURATION = 4

# --- AC MODIFICATION 1: Switching Penalty ---
# This discourages the agent from changing lights too frequently.
# It effectively tells the AI: "Changing the light costs energy, only do it if the queue is big."
SWITCH_PENALTY = 5.0 

# Constraints
# We can relax these slightly now that the AC Agent has a "Switching Cost"
MAIN_ROAD_PHASE_IDX = 0 
MAIN_ROAD_MIN_GREEN = 20 # Relaxed from 40, letting the Agent learn the rest
SIDE_ROAD_MIN_GREEN = 5  # Relaxed from 10

MAX_GREEN_DURATION = 500
DECISION_INTERVAL = 5   

SIM_TIME = 80000

# --- AC MODIFICATION 2: State Dimension ---
# We add +1 to the input size to include "Time since last switch"
STATE_DIM = (len(INCOMING_LANES) * 2) + NUM_PHASES + 1
ACTION_DIM = 2  

# --- Neural Network ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        
        # Actor
        self.actor = nn.Linear(64, action_dim)
        
        # Critic
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        
        return action_probs, state_value

# --- Helper Functions ---
def get_state(current_phase_idx, time_since_switch):
    state = []
    
    # 1. Lane Stats
    for lane in INCOMING_LANES:
        try:
            queue = traci.lane.getLastStepHaltingNumber(lane)
            wait_time = traci.lane.getWaitingTime(lane)
        except:
            queue = 0
            wait_time = 0
        
        state.append(queue / 20.0)      
        state.append(wait_time / 100.0) 
        
    # 2. Phase One-Hot
    phase_one_hot = [0.0] * NUM_PHASES
    phase_one_hot[current_phase_idx] = 1.0
    state.extend(phase_one_hot)

    # 3. Time Awareness (New Feature)
    # Normalize by Max Duration so it's between 0 and 1
    state.append(time_since_switch / MAX_GREEN_DURATION)
        
    return np.array(state, dtype=np.float32)

def get_reward():
    total_queue = 0
    total_wait = 0
    for lane in INCOMING_LANES:
        total_queue += traci.lane.getLastStepHaltingNumber(lane)
        total_wait += traci.lane.getWaitingTime(lane)
    
    reward = - (total_queue + (0.5 * total_wait)) / 100.0
    return reward

def apply_yellow_phase(current_green_idx):
    yellow_idx = PHASE_CYCLE[current_green_idx] + 1
    traci.trafficlight.setPhase(TL_ID, yellow_idx)
    traci.trafficlight.setPhaseDuration(TL_ID, YELLOW_DURATION)

def run_simulation():
    model = ActorCritic(STATE_DIM, ACTION_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    traci.start(SUMO_CMD)
    
    step = 0
    current_phase_idx = 0 
    
    traci.trafficlight.setPhase(TL_ID, PHASE_CYCLE[current_phase_idx])
    traci.trafficlight.setPhaseDuration(TL_ID, 1e9) 
    
    history = []
    action_last_switched = 0 

    while step < SIM_TIME:
        # 1. Get State (Now includes time_since_switch)
        time_since_switch = step - action_last_switched
        state_np = get_state(current_phase_idx, time_since_switch)
        state_tensor = torch.from_numpy(state_np).unsqueeze(0)
        
        # 2. Agent Decision
        action_probs, value = model(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_val = action.item() 
        
        # Hard Constraints (Safety Guards)
        required_min_green = MAIN_ROAD_MIN_GREEN if current_phase_idx == MAIN_ROAD_PHASE_IDX else SIDE_ROAD_MIN_GREEN

        if time_since_switch < required_min_green:
            action_val = 0 
            
        if time_since_switch > MAX_GREEN_DURATION:
            action_val = 1

        # 3. Execute Action logic
        reward_accumulated = 0
        
        if action_val == 1:
            # === SWITCH SEQUENCE ===
            
            # Apply AC Modification: Switching Penalty
            # This teaches the agent that switching is "expensive"
            reward_accumulated -= SWITCH_PENALTY

            apply_yellow_phase(current_phase_idx)
            
            for _ in range(YELLOW_DURATION):
                traci.simulationStep()
                step += 1
                reward_accumulated += get_reward()
                if step >= SIM_TIME: break
            
            current_phase_idx = (current_phase_idx + 1) % NUM_PHASES
            traci.trafficlight.setPhase(TL_ID, PHASE_CYCLE[current_phase_idx])
            traci.trafficlight.setPhaseDuration(TL_ID, 1e9)
            action_last_switched = step
            
            for _ in range(DECISION_INTERVAL):
                traci.simulationStep()
                step += 1
                reward_accumulated += get_reward()
                if step >= SIM_TIME: break

        else:
            # === STAY SEQUENCE ===
            for _ in range(DECISION_INTERVAL):
                traci.simulationStep()
                step += 1
                reward_accumulated += get_reward()
                if step >= SIM_TIME: break

        # 4. Learning Step
        if step < SIM_TIME:
            # Get next state (also includes time awareness)
            time_since_switch_next = step - action_last_switched
            next_state_np = get_state(current_phase_idx, time_since_switch_next)
            next_state_tensor = torch.from_numpy(next_state_np).unsqueeze(0)
            
            _, next_value = model(next_state_tensor)
            
            target_value = reward_accumulated + (GAMMA * next_value.item())
            advantage = target_value - value.item()
            
            critic_loss = F.mse_loss(value, torch.tensor([[target_value]]))
            
            log_prob = dist.log_prob(action)
            actor_loss = -log_prob * advantage
            
            entropy = dist.entropy().mean()
            
            total_loss = actor_loss + critic_loss - (ENTROPY_BETA * entropy)
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if step % 100 == 0:
                print(f"Time: {step} | Phase: {current_phase_idx} | "
                      f"Act: {action_val} | Reward: {reward_accumulated:.2f} | "
                      f"Loss: {total_loss.item():.4f}")

            history.append({'time': step, 'reward': reward_accumulated})

    traci.close()
    pd.DataFrame(history).to_csv("training_log.csv")
    torch.save(model.state_dict(), "model.pth")
    print("Training finished.")

if __name__ == "__main__":
    run_simulation()
