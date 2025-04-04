import numpy as np
from FarmingEnvironment import FarmEnv
from collections import deque
from keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape, Input
import tensorflow as tf
import pandas as pd
from FarmDeliveryRLSystem import FarmDeliveryRLSystemBase, ModelConfig
from StateObject import StateObject

# Define the experience replay buffer
class ExperienceReplay:
    """
    A class to store and sample experiences for training the agent.
    
    Attributes:
    - max_size: Maximum size of the replay buffer.
    - buffer: A deque to store experiences (state, action, reward, next_state, done).
    """
    def __init__(self, max_size):
        """
        Initializes the ExperienceReplay with a maximum buffer size.
        
        Args:
        - max_size: The maximum number of experiences to store in the buffer.
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        """
        Adds an experience to the replay buffer.
        
        Args:
        - experience: A tuple (state, action, reward, next_state, done).
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Samples a batch of experiences from the replay buffer.
        
        Args:
        - batch_size: The number of experiences to sample from the buffer.
        
        Returns:
        - A list of sampled experiences.
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

def create_initial_mode(input_length, action_length):
    """
    Creates a simple neural network model for the DQN agent.
    
    Args:
    - input_length: The size of the input (state vector length).
    - action_length: The number of possible actions the agent can take.
    
    Returns:
    - A compiled model with the specified input and output dimensions.
    """
    model = tf.keras.Sequential()
    model.add(Input(shape=(input_length,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(action_length, activation='linear'))
    return model

class FarmDQNModel(FarmDeliveryRLSystemBase):
    """
    A custom DQN model class for farming environment simulation.
    
    This class implements the methods necessary for deep Q-learning including:
    - Network architecture definition.
    - Training procedures.
    - Experience replay.
    """
    
    def get_config(self) -> ModelConfig:
        """
        Returns the model configuration parameters.
        
        Returns:
        - ModelConfig: Configuration settings for the model (such as epsilon, gamma, etc.).
        """
        return ModelConfig(
            model_name = 'dqn_model',
            max_steps_per_episode = 1000,
            max_episodes_per_training = 5000,
            min_epsilon = 0.05,
            max_epsilon = 1.0,
            epsilon_discount = 0.0001,
            gamma = 0.99,
            should_save_checkpoint = True,
            checkpoint_frequency = 1000
        )

    def initialize_custom(self):
        """
        Initializes the custom settings for the DQN model.
        
        This includes setting the learning rate, batch size, target model update frequency,
        and compiling the model. It also sets up the experience replay buffer.
        """
        # Initialization of important parameters
        self.replay_memory_capacity = 100000
        self.learning_rate = 0.001
        self.batch_size = 64
        self.target_model_update = 1000
        
        # Creating the model and target model
        self.model = create_initial_mode(self.state_vector_space, self.action_space)
        self.target_model = None
        self.replay_memory = ExperienceReplay(self.replay_memory_capacity)
        
        # Enabling large rewards in the environment
        self.env.use_large_rewards = True
        
        # Compile the model
        self.compile()

    def _get_action_for_state(self, state, _):
        """
        Predicts the best action for a given state using the current model.
        
        Args:
        - state: The current state object.
        
        Returns:
        - action: The action with the highest predicted Q-value.
        """
        predicted = self.model.predict_on_batch(np.array([state.vector_representation()]))
        action = np.argmax(predicted[0])  # Choose the action with the highest Q-value
        return action
    
    def _choose_action(self, state, epsilon, info, prev_action):
        """
        Chooses an action based on epsilon-greedy policy.
        
        Args:
        - state: The current state.
        - epsilon: The exploration factor (probability of choosing a random action).
        - prev_action: The previous action taken (not used here but might be useful in some cases).
        
        Returns:
        - action: The chosen action (either from exploration or exploitation).
        """
        if np.random.default_rng().uniform() < epsilon:
            # Explore: Random action
            action = self.env.action_space.sample()
        else:
            # Exploit: Choose the action predicted by the model
            action = self._get_action_for_state(state, info)
        return action
    
    def _perform_training_step(self, current_state: StateObject, next_state: StateObject, action: int, reward: float, done: bool, next_info: dict, prev_info: dict) -> int:
        """
        Perform a training step using the current experience.
        
        Args:
        - current_state: The current state object.
        - next_state: The next state after taking the action.
        - action: The action taken by the agent.
        - reward: The reward received for the action.
        - done: A flag indicating if the episode is over.
        
        Returns:
        - int: The next action to take (based on the updated model).
        """
        # Add the transition to the replay memory
        self._add_to_memory(current_state.vector_representation(), action, reward, next_state.vector_representation(), done)
        
        # Train the model with the current batch
        self._train_model()
    
    def _perform_post_training_step(self, step_number):
        """
        Perform any post-training steps, such as updating the target model.
        
        Args:
        - step_number: The current training step number.
        """
        if step_number % self.target_model_update == 0:
            print("Updating Target Weights")
            # Update the target model weights
            self.target_model.set_weights(self.model.get_weights())

    ## DQN Specific Helper Methods
    
    def compile(self):
        """
        Compiles the model and target model with the optimizer and loss function.
        """
        optimizer = Adam(learning_rate=self.learning_rate)
        
        # Initialize the target model and copy weights from the main model
        self.target_model = Sequential.from_config(self.model.get_config())
        self.target_model.set_weights(self.model.get_weights())
        self.target_model.compile(loss='mse', optimizer=optimizer)
        
        # Compile the main model
        self.model.compile(loss='mse', optimizer=optimizer)
    
    def _add_to_memory(self, state, action, reward, new_state, done):
        """
        Adds a new experience (transition) to the replay memory.
        
        Args:
        - state: The current state of the environment.
        - action: The action taken.
        - reward: The reward received for the action.
        - new_state: The next state after taking the action.
        - done: A flag indicating if the episode is over.
        """
        self.replay_memory.add([state, action, reward, new_state, done])
    
    def _train_model(self):
        """
        Trains the model using a batch of experiences sampled from the replay buffer.
        """
        # Ensure there are enough experiences in the buffer
        if len(self.replay_memory.buffer) < self.batch_size:
            return
        
        # Sample a batch of experiences
        current_batch = self.replay_memory.sample(self.batch_size)
        
        # Extract the states, actions, rewards, next states, and dones
        states = np.array([e[0] for e in current_batch])
        actions = np.array([e[1] for e in current_batch])
        rewards = np.array([e[2] for e in current_batch])
        next_states = np.array([e[3] for e in current_batch])
        dones = np.array([e[4] for e in current_batch])
        
        # Get Q-values for the current states
        q_values = self.model.predict_on_batch(states)
        
        # Get predicted Q-values for the next states from the target model
        target_q_values = self.target_model.predict_on_batch(next_states)
        
        # Get the maximum Q-value for the next state
        q_batch = np.max(target_q_values, axis=1).flatten()
        
        # Update the Q-values for the actions taken in the current batch
        indices = (np.arange(self.batch_size), actions)
        q_values[indices] = rewards + (1 - dones) * self.config.gamma * q_batch
        
        # Train the model on the updated Q-values
        self.model.train_on_batch(states.astype(np.float32), q_values.astype(np.float32))
