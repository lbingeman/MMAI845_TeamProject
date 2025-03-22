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
gamma = 0.99
epsilon = 0.05
num_episodes = 100
num_steps = 1000
batch_size = 64

# Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(([state], [action], [reward], [next_state], [done]))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        self.output_dim = output_dim
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def get_action(self, state, epsilon, action_map):
        if random.random() < epsilon:
            return np.random.choice(range(self.output_dim))
        else:
            state = torch.tensor(state, dtype=torch.float32).to(device)
            q_values = self.forward(state).to(device)
            action = 0
            top_value = torch.finfo(torch.float32).min
            for potential_action, value in enumerate(q_values):
                if value >= top_value:
                    action = potential_action
                    top_value = value
            return action
    
# Training Process
class DQN:
    def __init__(self, input_dim, output_dim, capacity):
        self.q_network = QNetwork(input_dim, output_dim).to(device)
        self.target_network = QNetwork(input_dim, output_dim).to(device)
        self.replay_memory = ReplayMemory(capacity)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.update_target_network = 5

    def train(self, env, episodes):
        for episode in range(num_episodes):
            (state, info) = env.reset()
            terminated = False
            reward_count = 0.0
            step_count = 0
            
            while terminated == False and step_count < 200:
                step_count += 1
                action_map = info["action_mask"]
                action = self.q_network.get_action(state.vector_representation(), epsilon, action_map)
                (next_state, reward, terminated, _, info) = env.step(action)
                reward_count += reward
                self.replay_memory.push(state.vector_representation(), action, reward, next_state.vector_representation(), terminated)
                state = next_state

                if len(self.replay_memory.buffer) > self.batch_size:
                    states, actions, rewards, next_states, dones = self.replay_memory.sample(self.batch_size)
                    states = torch.tensor(states, dtype=torch.float32).to(device)
                    actions = torch.tensor(actions, dtype=torch.float32).to(device)
                    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                    dones = torch.tensor(dones, dtype=torch.float32).to(device)
                    q_values = self.q_network(states)
                    next_q_values = self.target_network(next_states)
                    q_value = q_values.gather(2, actions.long().unsqueeze(2)).squeeze(1)
                    next_q_value = next_q_values.max(2)[0]
                    expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)
                    loss = self.loss_fn(q_value, expected_q_value)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if episode % self.update_target_network == 0:
                        self.target_network.load_state_dict(self.q_network.state_dict())
                if terminated:
                    break
            print(f'Episode: {episode+1}, Rewards: {reward_count}')
        # Save the object to a file
        with open("current_q_table_nn.pkl", "wb") as f:
            pickle.dump(self, f)
            
class QTable:
    def __init__(self, actions):
        self.actions = actions
        self.q_table = {}

    def get_q_value(self, state, action):
        if state not in self.q_table:
            self._create_q_values(state)
        return self.q_table[state][action]
    
    def _create_q_values(self, state):
        self.q_table[state] = []
        for _ in self.actions:
            self.q_table[state].append(np.random.rand())
    
    def get_q_values(self, state):
        # Create q-table if it does not exist
        if state not in self.q_table:
            self._create_q_values(state)
        return self.q_table[state]
            
    def set_q_value(self, state, action, value):
        self.q_table[state][action] = value

def q_learning(q_table):
    for episode in range(10000):
        (state, info) = env.reset()
        terminated = False
        rewards = 0.0
        step_count = 0
        while terminated == False and step_count < 200:
            step_count += 1
            action_mask = info["action_mask"]
            # Choose an action using epsilon-greedy
            q_values = q_table.get_q_values(state)
            if np.random.rand() < epsilon:
                action = np.random.choice(range(len(action_mask)))
            else:
                action = 0
                top_value = np.iinfo(np.int32).min
                for potential_action, value in enumerate(q_values):
                    if value >= top_value:
                        action = potential_action
                        top_value = value

            # Take the action and observe the next state and reward
            (next_state, reward, terminated, _, info) = env.step(action)
            
            # Update the Q-table
            next_state_q_value = q_table.get_q_values(next_state)
            new_q_value = q_values[action] + alpha * (reward + gamma * np.max(next_state_q_value) - q_values[action])
            q_table.set_q_value(state, action,  new_q_value)
            
            # Update the state and rewards
            state = next_state
            rewards += reward
            
            # Check if the episode is done
            if terminated is True:
                break
    
        # Print the episode rewards
        print(f'Episode {episode+1}, Rewards: {rewards:.2f}')
    
    # Save the object to a file
    with open("current_q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)

def q_table_execution():
            # Load the object from the file
    with open("current_q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
    env = FarmEnv(render_mode='human')
    (next_state, info) = env.reset()
    terminated = False
    while terminated is False:
        action_mask = info["action_mask"]
        q_values = q_table.get_q_values(next_state)
        print(q_values)
        action = 0
        top_value = np.iinfo(np.int32).min
        valid_actions = np.nonzero(action_mask)[0]
        non_zero_rewards = np.nonzero(q_values)
        if len(non_zero_rewards) == 0:
            action = np.random.choice(valid_actions)
        else:
            for potential_action, value in enumerate(q_values):
                if value >= top_value and action_mask[potential_action] != 0:
                    action = potential_action
                    top_value = value
        (next_state, reward, terminated, _, info) = env.step(action)
        if terminated is True:
            break

def q_learning_nn():
    env = FarmEnv(render_mode='not_human')
    
    if len(sys.argv) > 2 and sys.argv[2] == "continue":
        print("loading neural network")
        with open("current_q_table_nn.pkl", "rb") as f:
            dqn = pickle.load(f)
    else:
        dqn = DQN(23, env.action_space.n, 100000)
    dqn.train(env, num_episodes)

if __name__=='__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "q_learning":
        env = FarmEnv(render_mode='not_human')
        
        if len(sys.argv) > 2 and sys.argv[2] == "continue":
            with open("current_q_table.pkl", "rb") as f:
                q_table = pickle.load(f)
            print("Loading previous table")
        else:
            q_table = QTable(actions=range(env.action_space.n))
        
        q_learning(q_table)
    elif len(sys.argv) > 1 and sys.argv[1] == "q_learning_nn":
        q_learning_nn()
    elif len(sys.argv) > 1 and sys.argv[1] == "prev_trained_table":
        q_table_execution()
    else:
        env = FarmEnv(render_mode='human')
        (next_state, info) = env.reset()
        
        total_reward = 0
        while True:
            action_mask = info["action_mask"]
            
            # choose an action from valid actions
            action = np.random.choice(np.nonzero(action_mask)[0])
            (next_state, reward, terminated, truncated, info) = env.step(action)
            print(next_state.__hash__())
            total_reward += reward
            if terminated:
                break
        env.close()