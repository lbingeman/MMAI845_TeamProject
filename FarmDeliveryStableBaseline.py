from FarmingEnvironment import FarmEnv
import pandas as pd
from FarmDeliveryRLSystem import FarmDeliveryRLSystemBase, ModelConfig
from StateObject import StateObject
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from gymnasium import spaces

class EnvWrapper(gym.Env):
    def __init__(self, env: FarmEnv):
        """
        Initialize the environment wrapper for the gym environment.

        Args:
            env (FarmEnv): The environment to wrap.
        """
        self.env = env
        self.observation_space = spaces.Box(low=0.0, high=8.0, shape=(len(self.env.s.vector_representation()),), dtype=float)
        self.action_space = spaces.Discrete(self.env.action_space.n)
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        
        Args:
            seed (optional): Random seed for initialization.
            options (optional): Additional options for the reset.

        Returns:
            tuple: The initial processed state and additional info.
        """
        (state, info) = self.env.reset()
        return (self.process_state(state), info)
    
    def step(self, action):
        """
        Take a step in the environment using the given action.
        
        Args:
            action (int): The action to take.

        Returns:
            tuple: The processed new state, reward, done flag, truncated flag, and info.
        """
        (state, reward, done, truncated, info) = self.env.step(action)
        return (self.process_state(state), reward, done, truncated, info)
    
    def process_state(self, state_obj: StateObject):
        """
        Extracts the vector from the state object and converts it into a NumPy array.

        Args:
            state_obj (StateObject): The state object to process.

        Returns:
            np.array: The vector representation of the state as a NumPy array.
        """
        return np.array(state_obj.vector_representation(), dtype=np.float32)

class FarmStableBaselineDQNModel(FarmDeliveryRLSystemBase):
    def get_config(self) -> ModelConfig:
        """
        Retrieve the configuration for the DQN model.

        Returns:
            ModelConfig: The configuration settings for the model.
        """
        return ModelConfig(
            model_name='dqn_model_stable_baseline',
            max_steps_per_episode=1000,
            max_episodes_per_training=5000,
            min_epsilon=0.05,
            max_epsilon=1.0,
            epsilon_discount=0.0001,
            gamma=0.99,
            should_save_checkpoint=True,
            checkpoint_frequency=1000
        )
    
    def initialize_custom(self):
        """
        Initialize the custom environment wrapper and the DQN model.

        Returns:
            None
        """
        self.env.use_large_rewards = True
        self.wrapper_env = EnvWrapper(self.env)
        self.model = DQN("MlpPolicy", self.wrapper_env)
    
    def reload_model(self):
        """
        Reload a pre-trained DQN model.

        Returns:
            None
        """
        self.wrapper_env = EnvWrapper(self.env)
        self.model = DQN.load(self.filename, self.wrapper_env)
        
    def train(self):
        """
        Train the DQN model on the environment.

        Returns:
            None
        """
        self.model.learn(total_timesteps=(self.config.max_episodes_per_training * self.config.max_steps_per_episode), log_interval=4, progress_bar=True)
        self.model.save(self.filename)
        
    def evaluate(self, max_steps:int=100):
        """
        Evaluate the performance of the model after training.
        
        Args:
            max_steps (int): The maximum number of steps per episode during evaluation.

        Returns:
            None
        """
        self.wrapper_env.render_mode = "not_human"
        try:
            total_steps, total_rewards = 0, 0
            episodes = 100

            for _ in range(episodes):
                (state, _) = self.wrapper_env.reset()  # reset environment to a new, random state
                nb_steps, reward = 0, 0

                done = False

                while not done:
                    action, _ = self.model.predict(state, deterministic=True)
                    action = action.item()
                    _, reward, done, _, _ = self.wrapper_env.step(action)

                    total_rewards += reward

                    nb_steps += 1
                    if nb_steps == max_steps:
                        done = True
                        
                total_steps += nb_steps

            print(f"Results after {episodes} episodes:")
            print(f"Average timesteps per episode: {total_steps / episodes}")
            print(f"Average reward per episode: {total_rewards / episodes}")    
        except KeyboardInterrupt:
            pass 
        
    def run(self, max_steps:int=100):
        """
        Run the model for a specified number of steps and render the environment.
        
        Args:
            max_steps (int): The maximum number of steps to run the model.

        Returns:
            None
        """
        self.env.render_mode = "human"
        state, _ = self.wrapper_env.reset()
        step = 0
        while step < max_steps:
            step += 1
            action, _ = self.model.predict(state, deterministic=True)
            action = action.item()
            _, _, done, _, _ = self.wrapper_env.step(action)
            if done:
                break
