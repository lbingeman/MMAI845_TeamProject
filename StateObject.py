import copy

class StateObject:
    def __init__(self, current_inventory_state, delivery_states, row, column):
        self.current_inventory_state = current_inventory_state
        self.delivery_states = delivery_states
        self.row = row
        self.column = column
    
    def copy(self):
        return copy.deepcopy(self)
    
    def __hash__(self):
        list = [self.row, self.column, self.current_inventory_state]
        for state in self.delivery_states:
            list.append(state)
        return hash(tuple(list))

    def __eq__(self, other):
        ## first check if the delivery states are the same
        if len(self.delivery_states) != len(other.delivery_states):
            return False
        
        for i in range(len(self.delivery_states)):
            if self.delivery_states[i] != other.delivery_states[i]:
                return False
        
        return self.row == other.row and self.column == other.column and self.current_inventory_state == other.current_inventory_state
    