import copy

class StateObject:
    """
    A class to represent the state of the farm during a delivery task.
    
    This object holds the current inventory state, delivery states, and the farm's position
    on a grid (row and column).
    """
    
    def __init__(self, current_inventory_state, delivery_states, row, column):
        """
        Initialize the StateObject with the current inventory, delivery states, and position.

        Parameters:
        current_inventory_state (InventoryState): The state of the farm's inventory.
        delivery_states (list of DeliveryState): List of delivery states for the farm.
        row (int): The row position of the farm on the grid.
        column (int): The column position of the farm on the grid.
        """
        self.current_inventory_state = current_inventory_state  # Current inventory state
        self.delivery_states = delivery_states  # List of delivery states
        self.row = row  # Row position on the grid
        self.column = column  # Column position on the grid
    
    def copy(self):
        """
        Create a deep copy of the current StateObject.
        
        Returns:
        StateObject: A new StateObject that is a copy of the current one.
        """
        return copy.deepcopy(self)
    
    def __hash__(self):
        """
        Generate a unique hash for the state based on its attributes.
        
        The hash is generated using the row, column, current inventory state, and delivery states.
        
        Returns:
        int: The hash value for the state object.
        """
        state_list = [self.row, self.column, self.current_inventory_state]
        
        # Add delivery states to the list for hashing
        for state in self.delivery_states:
            state_list.append(state)
        
        # Return a hash of the tuple representation of the state
        return hash(tuple(state_list))
    
    def vector_representation(self):
        """
        Convert the state into a flat list of values (vector representation).
        
        This representation includes the current inventory state, the delivery states,
        and the farm's position on the grid (row and column).

        Returns:
        list: A list of values representing the state.
        """
        result = []
        
        # Add current inventory state vector representation
        result.extend(self.current_inventory_state.vector_representation())
        
        # Add each delivery state's vector representation
        for state in self.delivery_states:
            result.extend(state.vector_representation())
        
        # Add row and column as the farm's position on the grid
        result.append(self.row)
        result.append(self.column)
        
        return result

    def __eq__(self, other):
        """
        Check if two StateObject instances are equal based on their attributes.
        
        This comparison includes the current inventory state, delivery states, and position
        on the grid (row and column).

        Parameters:
        other (StateObject): Another instance of StateObject to compare with.

        Returns:
        bool: True if both states are equal, False otherwise.
        """
        # First, check if the delivery states are the same length
        if len(self.delivery_states) != len(other.delivery_states):
            return False
        
        # Check if all delivery states are equal
        for i in range(len(self.delivery_states)):
            if self.delivery_states[i] != other.delivery_states[i]:
                return False
        
        # Check if row, column, and current inventory state are equal
        return self.row == other.row and self.column == other.column and self.current_inventory_state == other.current_inventory_state
