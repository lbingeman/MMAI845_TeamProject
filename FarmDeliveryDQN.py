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
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
alpha = 0.05
gamma = 0.99
num_episodes = 10
num_steps = 1000
batch_size = 32
file_name = 'current_q_table_nn.pkl'
replay_buffer_size = 10000
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.75
target_update_interval = 10

# Define the experience replay buffer
class ExperienceReplay:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
# Q-Network
class QNetwork:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

    def create_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        return model

    def predict(self, state):
        return self.model.predict(state, verbose=0)

    def target_predict(self, state):
        return self.target_model.predict(state, verbose=0)

    def fit(self, states, targets):
        return self.model.fit(states, targets, epochs=10, verbose=0)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
 
# Training Process
class DQN:
    def __init__(self, input_dim, output_dim, capacity):
        self.q_network = QNetwork(input_dim, output_dim)
        self.replay_memory = ExperienceReplay(capacity)
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.target_update_interval = target_update_interval
        self.epsilon = epsilon

    def train(self, env):
        for episode in range(num_episodes):
            (state, info) = env.reset()
            state = state.copy()
            action_mask = info["action_mask"].copy()
            
            terminated = False
            reward_count = 0.0
            step_count = 0.0
            
            while terminated == False and step_count < 1000:
                step_count += 1
                # Choose action
                state_vector = state.vector_representation()
                # Epsilon-greedy policy
                if np.random.rand() <= self.epsilon:
                    action = np.random.choice(self.output_dim, p=action_mask / np.sum(action_mask))
                else:
                    # get q values
                    q_values = self.q_network.predict(np.array([state_vector]))[0]
                    action = 0
                    max_value = -np.inf
                    for index, value in enumerate(q_values):
                        if action_mask[index] == 0:
                            continue
                        elif value > max_value:
                            max_value = value
                            action = index

                # Take action
                (next_state, reward, terminated, _, info) = env.step(action)
                next_state_vector = next_state.vector_representation()
                next_action_mask = info["action_mask"].copy()
                reward_count += reward
                self.replay_memory.add((state_vector, action, reward, next_state_vector, terminated, action_mask, next_action_mask))

                # Sample a batch from the replay buffer and update the DQN model
                if len(self.replay_memory.buffer) >= batch_size:
                    experiences = self.replay_memory.sample(batch_size)
                    states = np.array([e[0] for e in experiences])
                    actions = np.array([e[1] for e in experiences])
                    rewards = np.array([e[2] for e in experiences])
                    next_states = np.array([e[3] for e in experiences])
                    dones = np.array([e[4] for e in experiences])
                    next_action_masks = np.array([e[6] for e in experiences])
                    next_q_values = self.q_network.target_predict(next_states)
                    next_q_values = np.where(next_action_masks == 1, next_q_values, -1e9)
                    current_q_values = self.q_network.target_predict(states)
                    current_q_values[np.arange(batch_size), actions] = rewards + (1 - dones) * gamma * np.max(next_q_values, axis=1)
                    self.q_network.fit(states, current_q_values)

                # Update the target model
                if episode % self.target_update_interval == 0:
                    self.q_network.update_target_model()
                    
                state = next_state.copy()
                action_mask = next_action_mask
                
                self.epsilon *= self.epsilon
                self.epsilon = max(self.epsilon, epsilon_min)
                
                if terminated:
                    break
            print(f'Episode: {episode+1}, Rewards: {reward_count}')
        # Save the object to a file
        with open(file_name, "wb") as f:
            pickle.dump(self, f)
            

def q_table_execution():
            # Load the object from the file
    with open(file_name, "rb") as f:
        q_table = pickle.load(f)
    env = FarmEnv(render_mode='human')
    (next_state, info) = env.reset()
    terminated = False
    count = 0
    while terminated is False and count < 200:
        count += 1
        action_mask = info["action_mask"]
        q_values = q_table.get_q_values(next_state, action_mask)
        print(action_mask)
        print(q_values)
        print(next_state.vector_representation())
        action = 0
        top_value = np.iinfo(np.int32).min
        for potential_action, value in enumerate(q_values):
            if value >= top_value:
                action = potential_action
                top_value = value
        print(action)
        (next_state, reward, terminated, _, info) = env.step(action)
        if terminated is True:
            break

def q_learning_nn():
    env = FarmEnv(render_mode='not_human', fix_location=True, fix_orders=True)
    
    if len(sys.argv) > 2 and sys.argv[2] == "continue":
        print("loading neural network")
        with open(file_name, "rb") as f:
            dqn = pickle.load(f)
    else:
        dqn = DQN(23, env.action_space.n, 100000)
    dqn.train(env)


if __name__=='__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        q_table_execution()
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        q_learning_nn()