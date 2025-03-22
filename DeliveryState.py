class DeliveryState:
    def __init__(self, union_farm=False, currie_farm=False, smith_farm=False):
        self.union_farm = union_farm
        self.currie_farm = currie_farm
        self.smith_farm = smith_farm
        
        self.order_fulfilled = False
    
    def __hash__(self):
        return hash((self.union_farm, self.currie_farm, self.smith_farm, self.order_fulfilled))

    def __eq__(self, other):
        return (self.union_farm, self.currie_farm, self.smith_farm, self.order_fulfilled) == (other.union_farm, other.currie_farm, other.smith_farm, other.order_fulfilled)

    def vector_representation(self):
        result = []
        result.append(int(self.union_farm))
        result.append(int(self.currie_farm))
        result.append(int(self.smith_farm))
        result.append(int(self.order_fulfilled))
        
        return result
    
    def encoding(self):
        bool_tuple = (self.union_farm, self.currie_farm, self.smith_farm)
        # Convert bool tuple to binary string
        binary_string = ''.join('1' if b else '0' for b in bool_tuple)

        # Convert binary string to decimal number
        return int(binary_string, 2)
    
    def mark_delivered(self):
        self.order_fulfilled = True

    def is_all_delivered(self):
        """
        Check if all farms have been delivered to.

        Returns:
            bool: True if all farms have been delivered to, False otherwise.
        """
        return self.order_fulfilled        

class InventoryState:
    def __init__(self, union_farm=False, currie_farm=False, smith_farm=False):
        self.union_farm = union_farm
        self.currie_farm = currie_farm
        self.smith_farm = smith_farm
    
    def __hash__(self):
        return hash((self.union_farm, self.currie_farm, self.smith_farm))

    def __eq__(self, other):
        return (self.union_farm, self.currie_farm, self.smith_farm) == (other.union_farm, other.currie_farm, other.smith_farm)
    
    def vector_representation(self):
        result = []
        result.append(int(self.union_farm))
        result.append(int(self.currie_farm))
        result.append(int(self.smith_farm))
        
        return result
    
    def pickup_inventory(self, farm):
        if farm == 'union_farm':
            self.union_farm = True
        elif farm == 'currie_farm':
            self.currie_farm = True
        elif farm == 'smith_farm':
            self.smith_farm = True
        else:
            raise ValueError("Invalid farm name")
    
    def has_inventory(self, farm):
        if farm == 'union_farm':
            return self.union_farm
        elif farm == 'currie_farm':
            return self.currie_farm
        elif farm == 'smith_farm':
            return self.smith_farm
        else:
            raise ValueError("Invalid farm name")
    
    def can_fulfill_order(self, order: DeliveryState):
        if (order.smith_farm and not self.smith_farm):
            return False
        if (order.currie_farm and not self.currie_farm):
            return False
        if (order.union_farm and not self.union_farm):
            return False
        return True

class Delivery:
    def __init__(self, deliveryState: DeliveryState, deliveryRow: int, deliveryColumn: int):
        self.delivery_state = deliveryState
        self.delivery_row = deliveryRow
        self.delivery_column = deliveryColumn
    
    def __hash__(self):
        return hash((self.delivery_state, self.delivery_row, self.delivery_column))

    def __eq__(self, other):
        return (self.delivery_state, self.delivery_row, self.delivery_column) == (other.delivery_state, other.delivery_row, other.delivery_column)
    
    def vector_representation(self):
        result = []
        result.extend(self.delivery_state.vector_representation())
        result.append(int(self.delivery_row))
        result.append(int(self.delivery_column))
        
        return result
    
    def is_at_location(self, location_row, location_column):
        return location_row == self.delivery_row and location_column == self.delivery_column
    
    def can_fulfill_order(self, inventory: InventoryState):
        return inventory.can_fulfill_order(self.delivery_state)
    
    def has_fulfill_order(self):
        return self.delivery_state.is_all_delivered()
    
    def fulfill_order(self, inventory: InventoryState):
        if self.can_fulfill_order(inventory):
            self.delivery_state.mark_delivered()
            return True 
        else:
            return False

class Farm:
    def __init__(self, farm_name: str, farm_row: int, farm_column: int):
        self.farm_name = farm_name
        self.farm_row = farm_row
        self.farm_column = farm_column
    
    def __hash__(self):
        return hash((self.farm_name, self.farm_row, self.farm_column))

    def __eq__(self, other):
        return (self.farm_name, self.farm_row, self.farm_column) == (other.farm_name, other.farm_row, other.farm_column)
    
    
    def is_at_location(self, row, column):
        return self.farm_row == row and self.farm_column == column
    
    def pickup_inventory(self, inventory: InventoryState):
        inventory.pickup_inventory(self.farm_name)
    
    def has_picked_up_inventory(self, inventory: InventoryState):
        return inventory.has_inventory(self.farm_name)