import numpy as np
import gym
from FarmingEnvironment import FarmEnv
import time

if __name__=='__main__':
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