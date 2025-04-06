from FarmDeliveryModelMode import FarmDeliveryModelType, FarmDeliveryModelMode
import pickle
import pandas as pd
from StateObject import StateObject
from dataclasses import dataclass
import numpy as np

# Class that represents the hyperparameters of a model
@dataclass
class ModelConfig:
    """
    A dataclass that holds the hyperparameters for the RL model.
    """
    model_name: str
    max_steps_per_episode: int  # Max steps per each episode
    max_episodes_per_training: int  # Max number of episodes per training loop
    min_epsilon: float  # Minimum epsilon value for exploration-exploitation tradeoff
    max_epsilon: float  # Maximum epsilon value for exploration-exploitation tradeoff
    epsilon_discount: float  # Rate at which epsilon decays over time
    gamma: float  # Discount factor for future rewards
    should_save_checkpoint: bool  # Whether to save model checkpoints
    checkpoint_frequency: int  # Frequency of saving checkpoints


class FarmDeliveryRLSystemBase:
    """
    A base class for managing the farm delivery reinforcement learning system.
    This class provides methods for training, saving, and evaluating the model.
    """
    
    def __init__(self, environment_mode: str):
        """
        Initializes the RL system with a given environment mode.
        
        Args:
        - environment_mode: A string representing the mode of the environment 
                             (e.g., 'fixed_location', 'non_fixed_state', etc.)
        """
        self.config = self.get_config()  # Get model configuration
        delivery_mode = FarmDeliveryModelMode(model_type=FarmDeliveryModelType.from_string(environment_mode))
        self.file_prefix = self.config.model_name + "_" + delivery_mode.get_file_prefix()
        self.filename = self.file_prefix + ".pkl"
        self.training_log_file = self.file_prefix + ".csv"
        self.env = delivery_mode.get_environment()  # Get the environment based on mode
        
        # Get initial action space and state space dimensions
        self.action_space = self.env.action_space.n
        self.state_vector_space = self.env.state_vector_space
        
        self.current_training_episode = 0
        self.last_episode = 0
        self.historical_logs = []
        
        # Perform any custom initialization needed for the child class
        self.initialize_custom()
    
    ## These ensure that pickle saves correctly
    def __getstate__(self):
        """
        Ensures that all necessary attributes are included when pickling the object.
        
        Returns:
        - dict: The current state of the object for pickling.
        """
        return self.__dict__  # Ensure all attributes are included

    def __setstate__(self, state):
        """
        Updates the object's state when unpickling.
        
        Args:
        - state: The state dictionary to restore.
        """
        self.__dict__.update(state)
    
    ## Methods that child classes must implement
    
    def get_config(self) -> ModelConfig:
        """
        Returns the configuration for the RL system.
        This method should be implemented in a subclass.
        """
        pass
    
    def initialize_custom(self):
        """
        Any custom initialization specific to the child class.
        This method should be implemented in a subclass.
        """
        pass
    
    def _get_action_for_state(self, state: StateObject) -> int:
        """
        Returns the action based on the current policy for a given state.
        
        Args:
        - state: The current state.
        
        Returns:
        - int: The chosen action.
        """
        return 0
    
    def _choose_action(self, state, epsilon, info, prev_action) -> int:
        """
        Chooses an action based on the current policy and epsilon value.
        
        Args:
        - state: The current state.
        - epsilon: The epsilon value for exploration vs exploitation.
        - info: Additional information from the environment.
        - prev_action: The previous action taken (if applicable).
        
        Returns:
        - int: The chosen action.
        """
        return 0
    
    def _perform_training_step(self, current_state: StateObject, next_state: StateObject, action: int, reward: float, done: bool, next_info: dict, prev_info: dict) -> int:
        """
        Performs the training step based on the current state and the transition.
        
        Args:
        - current_state: The state before the action was taken.
        - next_state: The state after the action was taken.
        - action: The action that was taken.
        - reward: The reward received from taking the action.
        - done: A boolean indicating if the episode is done.
        - next_info: Additional information from the next state.
        - prev_info: Additional information from the previous state.
        
        Returns:
        - int: The next action to take.
        """
        pass
    
    def _perform_post_training_step(self, step_number):
        """
        Perform any actions needed at the end of a training step.
        
        Args:
        - step_number: The current training step number.
        """
        pass
    
    ## Parent Class methods
    
    def save_checkpoint(self):
        """
        Saves a checkpoint of the model at the current episode.
        The checkpoint is saved as a pickle file.
        """
        if not self.config.should_save_checkpoint:
            return
        checkpoint_name = "model_checkpoints/" + self.file_prefix + "_" + str(self.last_episode) + ".pkl"
        with open(checkpoint_name, "wb") as f:
            pickle.dump(self, f)
            
    def save_system(self):
        """
        Saves the current system (model and training logs) to disk.
        """
        # Let's clear the transition table before saving
        self.env.prepare_for_saving()
        
        with open(self.filename, "wb") as f:
            pickle.dump(self, f)
        
        # Save training records to a CSV file
        df = pd.DataFrame(self.historical_logs)
        df.to_csv(self.training_log_file, index=False)
    
    def log_training_instance(self, steps_in_episode, reward_in_episode):
        """
        Logs the performance of a training instance (episode).
        
        Args:
        - steps_in_episode: The number of steps in the current episode.
        - reward_in_episode: The total reward accumulated in the episode.
        """
        current_log = {"Episode": len(self.historical_logs), "Steps": steps_in_episode, "Reward": reward_in_episode}
        self.historical_logs.append(current_log)
        print(current_log)  # Print the log to the console
    
    def _get_epsilon(self, episode):
        """
        Computes the epsilon value for exploration based on the current episode number.
        
        Args:
        - episode: The current training episode number.
        
        Returns:
        - float: The computed epsilon value.
        """
        epsilon = self.config.min_epsilon + (self.config.max_epsilon - self.config.min_epsilon) * np.exp(-self.config.epsilon_discount * episode)
        return epsilon
    
    def train(self):
        """
        Trains the RL model by running episodes and updating the model based on experience.
        """
        self.env.render_mode = "not_human"  # Disable rendering
        self.config = self.get_config() # reset config
        try:
            # Initialize environment and training variables
            (state, info) = self.env.reset()
            done = False
            config = self.config
            epsilon = self._get_epsilon(self.last_episode)
            reward_in_episode = 0
            steps_in_episode = 0
            prev_action = None
            step = 0
            
            for episode in range(config.max_episodes_per_training):
                while not done:
                    step += 1
                    steps_in_episode += 1
                    
                    # Select action using epsilon-greedy policy
                    action = self._choose_action(state, epsilon, info, prev_action)
                    
                    # Take the action and observe the next state, reward, and termination
                    (next_state, reward, done, _, next_info) = self.env.step(action)
                    
                    # Perform a training step
                    next_action = self._perform_training_step(state, next_state, action, reward, done, next_info, info)
                    
                    reward_in_episode += reward
                    if steps_in_episode == config.max_steps_per_episode:
                        done = True

                    if done and step < 10:
                        for delivery in state.delivery_states:
                            print(delivery.has_fulfill_order())
                    
                    state = next_state
                    prev_action = next_action
                    info = next_info
                    
                    self._perform_post_training_step(step)
                    
                    if done:
                        self.log_training_instance(steps_in_episode=steps_in_episode, reward_in_episode=reward_in_episode)
                        steps_in_episode = 0
                        (state, _) = self.env.reset()
                        done = False
                        reward_in_episode = 0
                        
                        self.last_episode += 1
                        epsilon = self._get_epsilon(self.last_episode)
                        break
                if self.last_episode % self.config.checkpoint_frequency == 0:
                    self.save_checkpoint()
                    
        except KeyboardInterrupt:
            print("Training has been interrupted")
        
        self.save_system()
    
    def evaluate(self, max_steps: int = 100):
        """
        Evaluates the trained model by running it in the environment without training.
        
        Args:
        - max_steps: Maximum steps per episode during evaluation.
        """
        self.env.render_mode = "not_human"
        try:
            total_steps, total_rewards = 0, 0
            episodes = 1000

            for _ in range(episodes):
                (state, info) = self.env.reset()  # Reset the environment
                nb_steps, reward = 0, 0

                done = False

                while not done:
                    action = self._get_action_for_state(state, info)
                    (state, reward, done, _, _) = self.env.step(action)

                    total_rewards += reward
                    nb_steps += 1
                    if nb_steps == max_steps:
                        done = True
                        
                total_steps += nb_steps

            # Print evaluation results
            print(f"Results after {episodes} episodes:")
            print(f"Average timesteps per episode: {total_steps / episodes}")
            print(f"Average reward per episode: {total_rewards / episodes}")    
        except KeyboardInterrupt:
            pass
    
    def run(self, max_steps: int = 100):
        """
        Runs the RL model interactively with the environment.
        
        Args:
        - max_steps: Maximum steps per episode during execution.
        """
        self.env.render_mode = "human"
        (state, info) = self.env.reset()
        total_rewards = 0
        total_steps = 0

        done = False

        while not done:
            action = self._get_action_for_state(state, info)
            (state, reward, done, _, _) = self.env.step(action)

            total_rewards += reward
            total_steps += 1
            if total_steps == max_steps:
                done = True
