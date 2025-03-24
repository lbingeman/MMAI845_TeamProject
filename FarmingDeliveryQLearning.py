import numpy as np
import gym
from FarmingEnvironment import FarmEnv
import time
import pickle
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
alpha = 0.05
gamma = 0.9
EPISLON = 0.1
           
class QTable:
    def __init__(self, actions):
        self.actions = actions
        self.q_table = {}

    def get_q_value(self, state, action, action_mask):
        if state.__hash__() not in self.q_table:
            self._create_q_values(state, action_mask)
        return self.q_table[state.__hash__()][action]
    
    def _create_q_values(self, state, action_mask):
        self.q_table[state.__hash__()] = []
        for index in range(len(self.actions)):
            if action_mask[index] == 0.0:
                self.q_table[state.__hash__()].append(-1000000000)
            else:
                self.q_table[state.__hash__()].append(0.0)
    
    def get_q_values(self, state, action_mask):
        # Create q-table if it does not exist
        if state.__hash__() not in self.q_table:
            self._create_q_values(state, action_mask)
        return self.q_table[state.__hash__()]
            
    def set_q_value(self, state, action, value):
        self.q_table[state.__hash__()][action] = value

def q_learning(q_table, file_name, num_episodes):
    epsilon = EPISLON
    
    for episode in range(num_episodes):
        (state, info) = env.reset()
        state = state.copy()
        terminated = False
        rewards = 0.0
        step_count = 0
        while terminated == False:
            step_count += 1
            action_mask = info["action_mask"].copy()
            # Choose an action using epsilon-greedy
            q_values = q_table.get_q_values(state, action_mask)
            if np.random.rand() < epsilon:
                valid_actions = np.where(action_mask == 1)[0]
                action = np.random.choice(valid_actions)
            else:
                action = 0
                top_value = np.iinfo(np.int32).min
                for potential_action, value in enumerate(q_values):
                    if value >= top_value:
                        action = potential_action
                        top_value = value
            
            # Take the action and observe the next state and reward
            (next_state, reward, terminated, _, info) = env.step(action)
            
            action_mask = info["action_mask"].copy()
            # Update the Q-table
            next_state_q_value = q_table.get_q_values(next_state, action_mask)
            new_q_value = q_values[action] + alpha * (reward + gamma * np.max(next_state_q_value) - q_values[action])
            q_table.set_q_value(state, action,  new_q_value)

            # Update the state and rewards
            state = next_state.copy()
            rewards += reward
            
            # Check if the episode is done
            if terminated is True:
                break
    
        # Print the episode rewards
        print(f'Episode {episode+1}, Rewards: {rewards:.2f}')
    
    # Save the object to a file
    with open(file_name, "wb") as f:
        pickle.dump(q_table, f)

def q_table_execution(model_name, fix_location=True, fix_orders=True):
    # Load the object from the file
    with open(model_name, "rb") as f:
        q_table = pickle.load(f)
    env = FarmEnv(render_mode='human', fix_location=fix_location, fix_orders=fix_orders)
    (next_state, info) = env.reset()
    terminated = False
    count = 0
    while terminated is False and count < 200:
        count += 1
        action_mask = info["action_mask"]
        q_values = q_table.get_q_values(next_state, action_mask)
        action = 0
        top_value = np.iinfo(np.int32).min
        for potential_action, value in enumerate(q_values):
            if value >= top_value:
                action = potential_action
                top_value = value
        (next_state, reward, terminated, _, info) = env.step(action)
        if terminated is True:
            break

if __name__=='__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        q_table_execution("current_q_table_fixed_location.pkl", fix_location=True, fix_orders=False)
    elif len(sys.argv) > 1 and sys.argv[1] == "train":
        file_name = "current_q_table_1.pkl"
        env = FarmEnv(render_mode='not_human')
        
        if len(sys.argv) > 2 and sys.argv[2] == "continue":
            with open(file_name, "rb") as f:
                q_table = pickle.load(f)
            print("Loading previous table")
        else:
            q_table = QTable(actions=range(env.action_space.n))
        
        q_learning(q_table, file_name, 10000)
    elif len(sys.argv) > 1 and sys.argv[1] == "train_fix_location":
        file_name = "current_q_table_fixed_location.pkl"
        env = FarmEnv(render_mode='not_human', fix_location=True)
        
        if len(sys.argv) > 2 and sys.argv[2] == "continue":
            with open(file_name, "rb") as f:
                q_table = pickle.load(f)
            print("Loading previous table")
        else:
            q_table = QTable(actions=range(env.action_space.n))
        
        q_learning(q_table, file_name, 100000)
    elif len(sys.argv) > 1 and sys.argv[1] == "train_fix_location_fix_space":
        file_name = "current_q_table_fixed_location_fix_space.pkl"
        env = FarmEnv(render_mode='not_human', fix_location=True, fix_orders=True)
        
        if len(sys.argv) > 2 and sys.argv[2] == "continue":
            with open(file_name, "rb") as f:
                q_table = pickle.load(f)
            print("Loading previous table")
        else:
            q_table = QTable(actions=range(env.action_space.n))
        
        q_learning(q_table, file_name, 200)