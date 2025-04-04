from StateObject import StateObject

class QTable:
    """
    A class to represent the Q-table for Q-learning.
    
    The Q-table stores Q-values for each state-action pair. It supports querying and updating Q-values.
    """
    def __init__(self, actions):
        """
        Initialize QTable instance.
        
        Parameters:
        actions (list): List of possible actions in the environment.
        """
        self.actions = actions  # List of possible actions
        self.q_table = {}  # Dictionary to store Q-values for state-action pairs

    def get_q_value(self, state: StateObject, action, action_mask):
        """
        Retrieve the Q-value for a specific state-action pair.
        
        Parameters:
        state (StateObject): The current state.
        action (int): The action taken.
        action_mask (np.array): A mask indicating which actions are valid.

        Returns:
        float: The Q-value for the given state-action pair.
        """
        if state.__hash__() not in self.q_table:
            self._create_q_values(state, action_mask)
        return self.q_table[state.__hash__()][action]
    
    def _create_q_values(self, state: StateObject, action_mask):
        """
        Initialize the Q-values for a new state.
        
        For each action, initialize the Q-value to a large negative value if it's invalid (based on action_mask),
        or zero otherwise.

        Parameters:
        state (StateObject): The current state.
        action_mask (np.array): A mask indicating which actions are valid.
        """
        self.q_table[state.__hash__()] = []
        for index in range(len(self.actions)):
            if action_mask[index] == 0.0:
                self.q_table[state.__hash__()].append(-1000000000)  # Assign large negative value for invalid actions
            else:
                self.q_table[state.__hash__()].append(0.0)  # Initialize valid actions with zero Q-value
    
    def get_q_values(self, state: StateObject, action_mask):
        """
        Retrieve the Q-values for all actions in the current state.

        Parameters:
        state (StateObject): The current state.
        action_mask (np.array): A mask indicating which actions are valid.

        Returns:
        list: A list of Q-values for each action in the current state.
        """
        # Create q-table for the state if it does not exist
        if state.__hash__() not in self.q_table:
            self._create_q_values(state, action_mask)
        return self.q_table[state.__hash__()]
            
    def set_q_value(self, state: StateObject, action, value):
        """
        Set a new Q-value for a specific state-action pair.

        Parameters:
        state (StateObject): The current state.
        action (int): The action taken.
        value (float): The new Q-value to assign.
        """
        self.q_table[state.__hash__()][action] = value