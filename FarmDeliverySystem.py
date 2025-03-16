import numpy as np
import gym
from FarmingEnvironment import FarmEnv
import time
import pickle
import sys

alpha = 0.1
gamma = 0.9
epsilon = 0.1
num_episodes = 10000000
num_steps = 1000

class QTable:
    def __init__(self, actions):
        self.actions = actions
        self.q_table = {}

    def get_q_value(self, state, action):
        if state not in self.q_table:
            self._create_q_values()
        return self.q_table[state][action]
    
    def _create_q_values(self):
        self.q_table[state] = []
        for action in self.actions:
            self.q_table[state].append(0.0)
    
    def get_q_values(self, state):
        # Create q-table if it does not exist
        if state not in self.q_table:
            self._create_q_values()
        return self.q_table[state]
            
    def set_q_value(self, state, action, value):
        self.q_table[state][action] = value

if __name__=='__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "q_learning":
        env = FarmEnv(render_mode='not_human')
        
        q_table = QTable(actions=range(env.action_space.n))
        
        for episode in range(num_episodes):
            (state, info) = env.reset()
            done = False
            rewards = 0.0
            
            for step in range(num_steps):
                action_mask = info["action_mask"]
                # Choose an action using epsilon-greedy
                q_values = q_table.get_q_values(state)
                valid_actions = np.nonzero(action_mask)[0]
                if np.random.rand() < epsilon:
                    action = np.random.choice(valid_actions)
                else:
                    # Remove illegal actions 
                    action = 0
                    top_value = np.iinfo(np.int32).min
                    for potential_action, value in enumerate(q_values):
                        if value >= top_value and action_mask[potential_action] != 0:
                            action = potential_action
                            top_value = value

                # Take the action and observe the next state and reward
                (next_state, reward, terminated, _, info) = env.step(action)
                
                # Update the Q-table
                next_state_q_value = q_table.get_q_values(state)
                new_q_value = q_values[action] + alpha * (reward + gamma * np.max(next_state_q_value) - q_values[action])
                q_table.set_q_value(state, action,  new_q_value)
                
                # Update the state and rewards
                state = next_state
                rewards += reward
                
                # Check if the episode is done
                if terminated:
                    break
        
            # Print the episode rewards
            print(f'Episode {episode+1}, Rewards: {rewards:.2f}')
        
        # Save the object to a file
        with open("current_q_table.pkl", "wb") as f:
            pickle.dump(q_table, f)
    else:
        env = FarmEnv(render_mode='human')
        (next_state, info) = env.reset()
        
        total_reward = 0
        while True:
            env.render()
            action_mask = info["action_mask"]
            
            # choose an action from valid actions
            action = np.random.choice(np.nonzero(action_mask)[0])
            (next_state, reward, terminated, truncated, info) = env.step(action)
            total_reward += reward
            if terminated:
                break
        env.close()