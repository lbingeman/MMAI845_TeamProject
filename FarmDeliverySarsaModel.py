import numpy as np
from FarmDeliveryRLSystem import FarmDeliveryRLSystemBase, ModelConfig
from StateObject import StateObject
from QTable import QTable

class FarmSarsaModel(FarmDeliveryRLSystemBase):
    def __init__(self, environment_mode, use_action_mask=False):
        """
        Initialize FarmQLearningModel instance.

        Parameters:
        environment_mode (str): Mode of the environment (training or testing).
        use_action_mask (bool): Whether to use an action mask to restrict valid actions.
        """
        super().__init__(environment_mode)
        self.use_action_mask = use_action_mask  # Whether to use action mask
    
    def get_config(self) -> ModelConfig:
        """
        Get the configuration for the Q-learning model.

        Returns:
        ModelConfig: Configuration object containing model parameters.
        """
        return ModelConfig(
            model_name='sarsa', 
            max_steps_per_episode=5000, 
            max_episodes_per_training=5000, 
            min_epsilon=0.1, 
            max_epsilon=0.1, 
            epsilon_discount=0.0001, 
            gamma=0.9, 
            should_save_checkpoint=True, 
            checkpoint_frequency=250
        )
    
    def initialize_custom(self):
        """
        Initialize custom settings for Q-learning model.
        
        This includes initializing the Q-table and setting the learning rate (alpha).
        """
        self.q_table = QTable(actions=range(self.env.action_space.n))  # Initialize Q-table with actions
        self.alpha = 0.05  # Learning rate for Q-value updates
        
        # Enabling large rewards in the environment
        self.env.use_large_rewards = True
    
    def _get_action_for_state(self, state, info):
        """
        Select the best action for a given state using Q-values.
        
        Parameters:
        state (StateObject): The current state.
        info (dict): Additional information, including the action mask.

        Returns:
        int: The action selected based on the Q-values.
        """
        # Apply action mask if specified
        if self.use_action_mask:
            action_mask = info["action_mask"]
        else:
            action_mask = np.ones(info["action_mask"].shape)  # All actions are valid by default
        
        q_values = self.q_table.get_q_values(state, action_mask)
        
        # Choose the action with the highest Q-value
        action = 0
        top_value = np.iinfo(np.int32).min
        for potential_action, value in enumerate(q_values):
            if value >= top_value:
                action = potential_action
                top_value = value
        
        return action

    def _choose_action(self, state, epsilon, info, previous_action):
        """
        Choose an action based on epsilon-greedy strategy.

        Parameters:
        state (StateObject): The current state.
        epsilon (float): The probability of choosing a random action.
        info (dict): Additional information, including the action mask.

        Returns:
        int: The action selected.
        """
        
        if previous_action is None:   
            if self.use_action_mask:
                action_mask = info["action_mask"]
            else:
                action_mask = np.ones(info["action_mask"].shape)  # All actions are valid by default
                
            q_values = self.q_table.get_q_values(state, action_mask)
            
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                valid_actions = np.where(action_mask == 1)[0]  # Find valid actions based on the mask
                action = np.random.choice(valid_actions)  # Select a random valid action
            else:
                # Choose the action with the highest Q-value
                action = 0
                top_value = np.iinfo(np.int32).min
                for potential_action, value in enumerate(q_values):
                    if value >= top_value:
                        action = potential_action
                        top_value = value
            
            return action
        
        return previous_action
    
    def _perform_training_step(self, current_state: StateObject, next_state: StateObject, action: int, reward: float, done: bool, next_info: dict, prev_info: dict) -> int:
        """
        Perform a training step by updating the Q-values using the Bellman equation.

        Parameters:
        current_state (StateObject): The current state before taking the action.
        next_state (StateObject): The next state after taking the action.
        action (int): The action taken in the current state.
        reward (float): The reward received after taking the action.
        done (bool): Whether the episode is finished.
        next_info (dict): Additional information from the next state.
        prev_info (dict): Additional information from the current state.

        Returns:
        int: The updated Q-value for the state-action pair.
        """
        q_values = self.q_table.get_q_values(current_state, prev_info["action_mask"])  # Get current Q-values
        next_state_q_value = self.q_table.get_q_values(next_state, next_info["action_mask"])  # Get Q-values for next state
        
        # Randomly choose the next action
        next_action = self._choose_action(next_state, self._get_epsilon(self.last_episode), next_info, None)

        # Update Q-value using the Bellman equation
        new_q_value = q_values[action] + self.alpha * (reward + self.config.gamma * next_state_q_value[next_action] - q_values[action])
        self.q_table.set_q_value(current_state, action, new_q_value)  # Set the updated Q-value
        return next_action
        
    def _perform_post_training_step(self, step_number):
        """
        Perform any necessary actions after each training step.

        This function can be used for additional processing after each training step if needed.
        
        Parameters:
        step_number (int): The current training step number.
        """
        pass