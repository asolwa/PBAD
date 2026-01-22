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
    "-c", "sim.sumocfg",
    # '--step-length', '0.05',
    # '--delay', '100',
    # '--lateral-resolution', '0.1'
    # "--no-step-log", "true",
    # "--waiting-time-memory", "1000"
]
TL_ID = "joinedS_3808244892_cluster_12590986567_254060091_3808244846_7788460158_#2more"  
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

data = []
def analysis():
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

def run_simulation():
    traci.start(SUMO_CMD)
    import pdb; pdb.set_trace()

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        analysis()

    df = pd.DataFrame(data)
    df.to_csv("baseline.csv")

if __name__ == "__main__":
    run_simulation()
