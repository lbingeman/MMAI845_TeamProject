from os import path
import pygame

# Object representing a given delivery order state. It contains the order contents and whether the order has been delivered
class DeliveryState:
    def __init__(self, union_farm=False, currie_farm=False, smith_farm=False):
        """
        Initialize the state of the order: which farms need to be delivered to.
        
        Args:
            union_farm (bool): Whether the delivery has an order from union farm 
            currie_farm (bool): Whether the delivery has an order from currie farm
            smith_farm (bool): Whether the delivery has an order from smith farm
        """
        self.union_farm = union_farm
        self.currie_farm = currie_farm
        self.smith_farm = smith_farm
        
        # Initially, the order is not fulfilled
        self.order_fulfilled = False
    
    def __hash__(self):
        """
        Generate a unique hash value for the delivery state based on farm delivery status and fulfillment.

        Returns:
            int: The hash value representing the delivery state.
        """
        return hash((self.union_farm, self.currie_farm, self.smith_farm, self.order_fulfilled))

    def __eq__(self, other):
        """
        Check if two DeliveryState objects are equal by comparing their attributes.
        
        Args:
            other (DeliveryState): The other object to compare with.

        Returns:
            bool: True if the attributes of both objects match, False otherwise.
        """
        return (self.union_farm, self.currie_farm, self.smith_farm, self.order_fulfilled) == (other.union_farm, other.currie_farm, other.smith_farm, other.order_fulfilled)

    def vector_representation(self):
        """
        Return a list of integers representing the farm delivery state and fulfillment status.
        
        Returns:
            list: A list of integers representing (has_order_for_union_farm, has_order_for_currie_farm, has_order_for_smith_farm, is_order_fulfilled)
        """
        result = []
        result.append(int(self.union_farm))
        result.append(int(self.currie_farm))
        result.append(int(self.smith_farm))
        result.append(int(self.order_fulfilled))
        
        return result
    
    def mark_delivered(self):
        """
        Mark the delivery as fulfilled.
        
        Returns:
            None
        """
        self.order_fulfilled = True

    def is_all_delivered(self):
        """
        Check if all farms have been delivered to.

        Returns:
            bool: True if all farms have been delivered to, False otherwise.
        """
        return self.order_fulfilled        

# Object representing the inventory of the delivery truck
class InventoryState:
    def __init__(self, union_farm=False, currie_farm=False, smith_farm=False):
        """
        Initialize the truck's inventory with the specified farm delivery status.
        
        Args:
            union_farm (bool): Whether the truck has inventory from the union farm.
            currie_farm (bool): Whether the truck has inventory from the currie farm.
            smith_farm (bool): Whether the truck has inventory from the smith farm.
        """
        self.union_farm = union_farm
        self.currie_farm = currie_farm
        self.smith_farm = smith_farm
    
    def __hash__(self):
        """
        Generate a unique hash value for the inventory state based on what inventory the truck has

        Returns:
            int: The hash value representing the inventory state.
        """
        return hash((self.union_farm, self.currie_farm, self.smith_farm))

    def __eq__(self, other):
        """
        Check if two InventoryState objects are equal by comparing their attributes.
        
        Args:
            other (InventoryState): The other object to compare with.

        Returns:
            bool: True if the attributes of both objects match, False otherwise.
        """
        return (self.union_farm, self.currie_farm, self.smith_farm) == (other.union_farm, other.currie_farm, other.smith_farm)
    
    def vector_representation(self):
        """
        Return a list of integers representing the truck's inventory state (True=1, False=0).
        
        Returns:
            list: A list of integers representing the inventory state.
        """
        result = []
        result.append(int(self.union_farm))
        result.append(int(self.currie_farm))
        result.append(int(self.smith_farm))
        
        return result
    
    def pickup_inventory(self, farm):
        """
        Add inventory for the specified farm to the truck's inventory.
        
        Args:
            farm (str): The farm name to add to the inventory.
        
        Raises:
            ValueError: If the farm name is invalid.
        """
        if farm == 'union_farm':
            self.union_farm = True
        elif farm == 'currie_farm':
            self.currie_farm = True
        elif farm == 'smith_farm':
            self.smith_farm = True
        else:
            raise ValueError("Invalid farm name")
    
    def has_inventory(self, farm):
        """
        Check if the truck has inventory from the specified farm.
        
        Args:
            farm (str): The farm name to check.

        Returns:
            bool: True if the truck has inventory for the farm, False otherwise.
        
        Raises:
            ValueError: If the farm name is invalid.
        """
        if farm == 'union_farm':
            return self.union_farm
        elif farm == 'currie_farm':
            return self.currie_farm
        elif farm == 'smith_farm':
            return self.smith_farm
        else:
            raise ValueError("Invalid farm name")
    
    def can_fulfill_order(self, order: DeliveryState):
        """
        Check if the truck's inventory can fulfill the given delivery order's requirements.
        
        Args:
            order (DeliveryState): The delivery order to check against.

        Returns:
            bool: True if the inventory can fulfill the order, False otherwise.
        """
        if (order.smith_farm and not self.smith_farm):
            return False
        if (order.currie_farm and not self.currie_farm):
            return False
        if (order.union_farm and not self.union_farm):
            return False
        return True

# Object representing a given delivery. It contains the delivery order state and the location of the drop off
class Delivery:
    def __init__(self, deliveryState: DeliveryState, deliveryRow: int, deliveryColumn: int):
        """
        Initialize the delivery with the given delivery state and its location (row, column).
        
        Args:
            deliveryState (DeliveryState): The state of the delivery order.
            deliveryRow (int): The row where the delivery needs to be made.
            deliveryColumn (int): The column where the delivery needs to be made.
        """
        self.delivery_state = deliveryState
        self.delivery_row = deliveryRow
        self.delivery_column = deliveryColumn
    
    def __hash__(self):
        """
        Generate a unique hash value for the delivery based on the delivery state and its location.

        Returns:
            int: The hash value representing the delivery.
        """
        return hash((self.delivery_state, self.delivery_row, self.delivery_column))

    def __eq__(self, other):
        """
        Check if two Delivery objects are equal by comparing their state and location.

        Args:
            other (Delivery): The other object to compare with.

        Returns:
            bool: True if the delivery state and location match, False otherwise.
        """
        return (self.delivery_state, self.delivery_row, self.delivery_column) == (other.delivery_state, other.delivery_row, other.delivery_column)
    
    def vector_representation(self):
        """
        Return a list of integers representing the delivery state and its location (row, column).
        
        Returns:
            list: A list of integers representing the delivery state and location.
        """
        result = []
        result.extend(self.delivery_state.vector_representation())
        result.append(int(self.delivery_row))
        result.append(int(self.delivery_column))
        
        return result
    
    def is_at_location(self, location_row, location_column):
        """
        Check if the delivery is at the specified location (row, column).
        
        Args:
            location_row (int): The row to check.
            location_column (int): The column to check.

        Returns:
            bool: True if the delivery is at the specified location, False otherwise.
        """
        return location_row == self.delivery_row and location_column == self.delivery_column
    
    def can_fulfill_order(self, inventory: InventoryState):
        """
        Check if the delivery can be fulfilled with the given inventory.
        
        Args:
            inventory (InventoryState): The truck's current inventory.

        Returns:
            bool: True if the inventory can fulfill the delivery order, False otherwise.
        """
        return inventory.can_fulfill_order(self.delivery_state)
    
    def has_fulfill_order(self):
        """
        Check if the delivery order has been fulfilled.

        Returns:
            bool: True if the delivery has been fulfilled, False otherwise.
        """
        return self.delivery_state.is_all_delivered()
    
    def fulfill_order(self, inventory: InventoryState):
        """
        Attempt to fulfill the delivery order using the truck's inventory.
        
        Args:
            inventory (InventoryState): The truck's current inventory.

        Returns:
            bool: True if the order was successfully fulfilled, False otherwise.
        """
        if self.can_fulfill_order(inventory):
            self.delivery_state.mark_delivered()
            return True 
        else:
            return False

# Object representing a farm that is available to be ordered from. This includes a display name for the farm, and the farm's location
class Farm:
    def __init__(self, farm_name: str, farm_row: int, farm_column: int):
        """
        Initialize the farm with its name and location (row, column).
        
        Args:
            farm_name (str): The name of the farm.
            farm_row (int): The row location of the farm.
            farm_column (int): The column location of the farm.
        """
        self.farm_name = farm_name
        self.farm_row = farm_row
        self.farm_column = farm_column
        self.farm_img = None
    
    def __hash__(self):
        """
        Generate a unique hash value for the farm based on its name and location.

        Returns:
            int: The hash value representing the farm.
        """
        return hash((self.farm_name, self.farm_row, self.farm_column))

    def __eq__(self, other):
        """
        Check if two Farm objects are equal by comparing their name and location.

        Args:
            other (Farm): The other object to compare with.

        Returns:
            bool: True if the farm name and location match, False otherwise.
        """
        return (self.farm_name, self.farm_row, self.farm_column) == (other.farm_name, other.farm_row, other.farm_column)
    
    def farm_image(self, cell_size):
        """
        Generate an image to represent the farm

        Returns:
            Surface : image for display for the farm
        """
        if getattr(self, "farm_img", None) is None:
            filename = "img/" + self.farm_name + ".png"
            file_path = path.join(path.dirname(__file__), filename)
            self.farm_img = pygame.transform.scale(
                pygame.image.load(file_path), cell_size
            )
        
        return self.farm_img
        
    def is_at_location(self, row, column):
        """
        Check if the farm is located at the specified row and column.
        
        Args:
            row (int): The row location to check.
            column (int): The column location to check.

        Returns:
            bool: True if the farm is at the given location, False otherwise.
        """
        return self.farm_row == row and self.farm_column == column
    
    def pickup_inventory(self, inventory: InventoryState):
        """
        Add the farm's inventory to the truck's inventory.
        
        Args:
            inventory (InventoryState): The truck's inventory state to update.
        
        Returns:
            None
        """
        inventory.pickup_inventory(self.farm_name)
    
    def has_picked_up_inventory(self, inventory: InventoryState):
        """
        Check if the truck has picked up inventory from this farm.
        
        Args:
            inventory (InventoryState): The truck's inventory state.

        Returns:
            bool: True if the truck has picked up inventory from this farm, False otherwise.
        """
        return inventory.has_inventory(self.farm_name)