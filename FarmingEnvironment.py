"""
FarmEnv: A custom OpenAI Gym environment simulating a farm delivery scenario.

The environment involves a delivery truck picking up food items from farms and delivering
to customers. It is grid-based and supports discrete actions like moving
in cardinal directions, picking up, and dropping off.
"""

from os import path
from typing import Optional
import pygame

import numpy as np
import random

from gym import Env, spaces
from gym.envs.toy_text.utils import categorical_sample
from StateObject import StateObject

from DeliveryState import DeliveryState, Delivery, InventoryState, Farm

MAP = [
    "+---------+",
    "| : | : : |",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "| | : | : |",
    "+---------+",
]
WINDOW_SIZE = (550, 400)
STATUS_BAR_SIZE = (550, 100)
ORDER_BAR_SIZE = (550, 200)

class FarmEnv(Env):
    """
    Custom Gym environment for farm delivery simulation.
    
    The agent (delivery truck) must pick up inventory from farms and deliver them to customers.
    """
    metadata = {
        "render_modes": ["human", "not_human"],
        "render_fps": 4,
    }

    def __init__(self, num_deliveries: int = 3, fix_location = False, fix_orders = False, render_mode: Optional[str] = None):
        """
        Initialize the environment.

        Args:
            num_deliveries (int): Number of delivery points to create.
            fix_location (bool): Whether to use fixed customer locations.
            fix_orders (bool): Whether to use fixed delivery requirements.
            render_mode (str, optional): Rendering mode for environment.
        """
        # Convert predefined MAP to a NumPy array for rendering
        self.desc = np.asarray(MAP, dtype="c")
        
        # Configuration flags for delivery setup
        self.fix_location = fix_location
        self.fix_orders = fix_orders
        
        # Toggle for reward shaping
        self.use_large_rewards = False

        # Define farm locations on the grid
        self.farm_locations = [Farm('currie_farm', 0, 0), Farm('union_farm', 0, 4), Farm('smith_farm', 4, 0)]
        self.number_of_farms = len(self.farm_locations)

        # Set up environment dimensions and delivery parameters
        inventory_states = 8  # Number of discrete inventory states
        self.num_deliveries = num_deliveries
        self.num_rows = 5
        self.num_columns = 5
        self.grid_size = self.num_rows * self.num_columns

        # Total number of possible discrete states in the environment
        self.num_states = (
            self.grid_size *            # Agent location
            inventory_states *          # Inventory state
            num_deliveries *            # Number of delivery orders
            (inventory_states - 1) *    # State of each delivery
            2 *                         # Whether or not the agent is carrying a delivery
            (self.grid_size - self.number_of_farms)  # Possible delivery destinations
        )

        # Optional cache for transition probabilities (if used)
        self.transition_table = {}

        # Set up the initial environment state
        self.initialize_state()

        # Define the number of possible actions
        num_actions = 6
        self.action_space = spaces.Discrete(num_actions)

        # Define the observation space size
        self.observation_space = spaces.Discrete(self.num_states)

        # Determine vectorized state size (for input to neural nets)
        self.state_vector_space = len(self.s.vector_representation())

        # Set rendering mode (e.g., human vs non-human)
        self.render_mode = render_mode

        # --------------------------
        # Pygame rendering utilities
        # --------------------------
        self.window = None
        self.clock = None
        self.cell_size = (
            WINDOW_SIZE[0] / self.desc.shape[1],  # Width of each grid cell
            WINDOW_SIZE[1] / self.desc.shape[0],  # Height of each grid cell
        )
        self.taxi_imgs = None                  # To store taxi image assets
        self.taxi_orientation = 0              # Direction the taxi is facing
        self.median_horiz = None               # Optional for visual centering
        self.median_vert = None                # Optional for visual centering
        self.background_img = None             # Background image for rendering
        self.destination_imgs = None           # Delivery target images (not yet picked up)
        self.destination_imgs_dropoff = None   # Delivery target images (picked up)

    def is_at_farm(self, location_row, location_column):
        """
        Checks if current location matches a farm's location.
        """
        for farm in self.farm_locations:
            if farm.is_at_location(location_row, location_column):
                return farm 
        return None
    
    def initialize_state(self):
        """
        Randomly initializes the inventory state and delivery states.
        """
        # Initialize an empty inventory state
        current_inventory_state = InventoryState()

        # Generate a random list of unique locations on the grid
        # Total needed: number of deliveries + number of farms
        location_list = random.sample(range(0, self.num_rows * self.num_columns), self.num_deliveries + self.number_of_farms)

        # List to store generated delivery state objects
        delivery_states = []

        # Fixed delivery locations (used only if fix_location=True)
        fixed_locations = [(2,1), (3,3), (4,2)]

        for _ in range(self.num_deliveries):
            # Randomly generate farm requirements for each delivery
            if self.fix_orders is False:
                while True:
                    # Randomly decide whether each farm is required for the delivery
                    union_farm = random.choice([True, False])
                    currie_farm = random.choice([True, False])
                    smith_farm = random.choice([True, False])

                    # At least one farm must be required (prevent all False)
                    if union_farm or currie_farm or smith_farm:
                        break
            else:
                # If fixed orders are enabled, require all farms
                union_farm = True
                currie_farm = True
                smith_farm = True

            # Determine delivery location
            if self.fix_location is True:
                # Use predefined fixed locations
                delivery_location = fixed_locations.pop()
                delivery_row = delivery_location[0]
                delivery_column = delivery_location[1]
            else:
                # Choose a random location from the list
                delivery_location = location_list.pop()
                delivery_row = int(delivery_location / self.num_columns)
                delivery_column = delivery_location % self.num_columns

                # Make sure the location is not at a farm
                while self.is_at_farm(delivery_row, delivery_column):
                    delivery_location = location_list.pop()
                    delivery_row = int(delivery_location / self.num_columns)
                    delivery_column = delivery_location % self.num_columns

            # Create the delivery object and add it to the list
            delivery_states.append(
                Delivery(
                    DeliveryState(union_farm, currie_farm, smith_farm),
                    delivery_row,
                    delivery_column
                )
            )
        # Set the agent's initial location (currently hardcoded at top-left corner)
        location = [0, 0]

        # Set the current state object with initialized inventory, deliveries, and location
        self.s = StateObject(current_inventory_state, delivery_states, location[0], location[1])

        
    def is_valid_position(self, col_position, row_position):
        """
        Check if the position is inside the grid bounds.
        """
        if row_position >= self.num_columns or row_position < 0:
            return False
        if col_position >= self.num_columns or col_position < 0:
            return False
        return True
    
    def get_transition(self, a):
        """
        Computes the state transition given an action a.
        Returns a list containing a tuple: (probability, next_state, reward, done)
        """
        # Check if transition is already cached in the transition table
        if self.s.__hash__() in self.transition_table and a in self.transition_table[self.s.__hash__()]:
            return self.transition_table[self.s.__hash__()][a]
        
        # Make a copy of the current state to preserve the original
        prev_state = self.s.copy

        # Extract current components of the state
        current_inventory = self.s.current_inventory_state
        deliveries = self.s.delivery_states
        row_position = self.s.row
        column_position = self.s.column
        reward = 0  # default reward

        # Action 0: Move South (up visually)
        if a == 0:
            if self.is_valid_position(row_position - 1, column_position):
                row_position -= 1
            reward = -1  # small penalty for movement

        # Action 1: Move North (down visually)
        elif a == 1:
            if self.is_valid_position(row_position + 1, column_position):
                row_position += 1
            reward = -1

        # Action 2: Move East (right)
        elif a == 2:
            if self.is_valid_position(row_position, column_position + 1) and self.desc[row_position + 1, 2 * column_position + 2] == b":":
                column_position += 1
            reward = -1

        # Action 3: Move West (left)
        elif a == 3:
            if self.is_valid_position(row_position, column_position - 1) and self.desc[row_position + 1, 2 * column_position] == b":":
                column_position -= 1
            reward = -1

        # Action 4: Pick up produce at farm
        elif a == 4:
            farm = self.is_at_farm(row_position, column_position)
            if farm is None or farm.has_picked_up_inventory(current_inventory):
                # Either not at a farm or already picked up
                reward = -1000 if getattr(self, "use_large_rewards", False) else -10
            else:
                # Pick up from farm and reward the action
                farm.pickup_inventory(current_inventory)
                reward = 10000 if getattr(self, "use_large_rewards", False) else 20

        # Action 5: Drop off produce to customer
        elif a == 5:
            target_customer = None
            # Find if there's a customer at the current location
            for customer in deliveries:
                if customer.is_at_location(row_position, column_position):
                    target_customer = customer
                    break
            
            if (
                target_customer is None or 
                not target_customer.can_fulfill_order(current_inventory) or 
                target_customer.has_fulfill_order()
            ):
                # Invalid drop-off
                reward = -1000 if getattr(self, "use_large_rewards", False) else -10
            else:
                # Successfully fulfill customer order
                target_customer.fulfill_order(current_inventory)
                reward = 10000 if getattr(self, "use_large_rewards", False) else 20

        # Check if all orders have been fulfilled (end condition)
        met_end_condition = all(customer.has_fulfill_order() for customer in deliveries)

        # Create the new state after performing the action
        state = StateObject(current_inventory, deliveries, row_position, column_position)

        # Cache this transition for future lookups
        if prev_state.__hash__() not in self.transition_table:
            self.transition_table[prev_state.__hash__()] = {}
        self.transition_table[prev_state.__hash__()][a] = [(1.0, state, reward, met_end_condition)]

        # Return the transition
        return self.transition_table[prev_state.__hash__()][a]

    def action_mask(self, state):
        """
        Computes which actions are currently valid given the state.
        Returns a mask where 1 indicates valid actions and 0 indicates invalid actions.
        """
        # Initialize mask for 6 actions (all invalid initially)
        mask = np.zeros(self.action_space.n, dtype=np.int8)

        # Extract components of the state
        current_inventory_state = state.current_inventory_state
        delivery_states = state.delivery_states
        row = state.row
        col = state.column

        # Check if the current location is at a farm
        farm = self.is_at_farm(row, col)

        # Find the customer located at the current position (if any)
        target_customer = None
        for customer in delivery_states:
            if customer.is_at_location(row, col):
                target_customer = customer

        # Check if moving south is valid (up visually)
        if self.is_valid_position(row - 1, col):
            mask[0] = 1  # South is valid, so mark the action as valid

        # Check if moving north is valid (down visually)
        if self.is_valid_position(row + 1, col):
            mask[1] = 1  # North is valid, so mark the action as valid

        # Check if moving east is valid (right)
        if self.is_valid_position(row, col + 1) and self.desc[row + 1, 2 * col + 2] == b":":
            mask[2] = 1  # East is valid, so mark the action as valid

        # Check if moving west is valid (left)
        if self.is_valid_position(row, col - 1) and self.desc[row + 1, 2 * col] == b":":
            mask[3] = 1  # West is valid, so mark the action as valid

        # Check if picking up inventory is valid (at a farm and not already picked up)
        if farm is not None and not farm.has_picked_up_inventory(current_inventory_state):
            mask[4] = 1  # Pickup action is valid, so mark the action as valid

        # Check if dropping off to customer is valid (customer at location and can fulfill order)
        if target_customer is not None and target_customer.can_fulfill_order(current_inventory_state) and not target_customer.has_fulfill_order():
            mask[5] = 1  # Drop-off action is valid, so mark the action as valid

        # Return the mask with valid actions (1 = valid, 0 = invalid)
        return mask

    def step(self, a):
        """
        Perform one step in the environment using action a.
        """
        transitions = self.get_transition(a)
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        (probability, new_state, reward, met_end_condition) = transitions[i]
        self.s = new_state
        self.lastaction = a
        self.turns += 1
        self.total_rewards += reward

        if self.render_mode == "human":
            self.render()
        return (self.s.copy(), reward, met_end_condition, False, {"prob": probability, "action_mask": self.action_mask(new_state)})

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """
        Reset the environment to an initial state.
        """
        super().reset(seed=seed)
        self.initialize_state()
        self.lastaction = None
        self.turns = 0
        self.total_rewards = 0
        self.taxi_orientation = 0

        if self.render_mode == "human":
            self.render()
        return self.s.copy(), {"prob": 1.0, "action_mask": self.action_mask(self.s)}

    def render(self):
        """
        Renders the environment using the GUI.
        """
        self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        """
        Renders the environment using pygame in human mode. It displays the taxi, 
        farms, deliveries, and status bar with real-time updates. This method also 
        handles loading and drawing of images for various elements such as the taxi, 
        farms, and grid cells.

        Parameters:
        mode (str): The mode in which to render the environment. If "human", it 
                    updates the pygame window for display.
        """
        
        # Initialize the pygame window if it hasn't been created yet
        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Farm Delivery Service")  # Set the window title
            window_size = (WINDOW_SIZE[0], WINDOW_SIZE[1] + STATUS_BAR_SIZE[1] + ORDER_BAR_SIZE[1])  # Adjust window size to include status bar
            if mode == "human":
                self.window = pygame.display.set_mode(window_size)  # Set up the display window

        # Set font for status bar text
        font = pygame.font.SysFont("Arial", 16)

        # Define colors for the status bar background and text
        background_color = (255, 255, 255)  # White background
        text_color = (0, 0, 0)  # Black text

        # Ensure the window has been properly initialized
        assert (self.window is not None), "Something went wrong with pygame. This should never happen."

        # Initialize the clock for controlling frame rate
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Load the taxi images if they haven't been loaded already
        if self.taxi_imgs is None:
            file_names = [
                path.join(path.dirname(__file__), "img/cab_front.png"),
                path.join(path.dirname(__file__), "img/cab_rear.png"),
                path.join(path.dirname(__file__), "img/cab_right.png"),
                path.join(path.dirname(__file__), "img/cab_left.png"),
            ]
            # Scale the images to the specified cell size
            self.taxi_imgs = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]

        # Load destination images if not already loaded
        if getattr(self, 'destination_imgs', None) is None:
            self.destination_imgs = []
            for i in range(self.num_deliveries):
                file_name = path.join(path.dirname(__file__), "img/location_" + str(i+1) + ".png")
                new_image = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                new_image.set_alpha(170)  # Set transparency for the image
                self.destination_imgs.append(new_image)

        if getattr(self, 'destination_imgs_dropoff', None) is None:
            self.destination_imgs_dropoff = []
            for i in range(self.num_deliveries):
                file_name = path.join(path.dirname(__file__), "img/location_" + str(i+1) + "_dropoff.png")
                new_image = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                new_image.set_alpha(170)  # Set transparency for the image
                self.destination_imgs_dropoff.append(new_image)

        # Load median (wall) images for the grid if not already loaded
        if self.median_horiz is None:
            file_names = [
                path.join(path.dirname(__file__), "img/gridworld_median_left.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_horiz.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_right.png"),
            ]
            self.median_horiz = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]

        if self.median_vert is None:
            file_names = [
                path.join(path.dirname(__file__), "img/gridworld_median_top.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_vert.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_bottom.png"),
            ]
            self.median_vert = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]

        # Load background image if not already loaded
        if self.background_img is None:
            file_name = path.join(path.dirname(__file__), "img/taxi_background.png")
            self.background_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)

        # Get the environment description (grid layout)
        desc = self.desc

        # Loop through each grid cell and render the corresponding image (background, wall, etc.)
        for y in range(0, desc.shape[0]):
            for x in range(0, desc.shape[1]):
                cell = (x * self.cell_size[0], y * self.cell_size[1])
                self.window.blit(self.background_img, cell)  # Draw background

                # Draw vertical walls
                if desc[y][x] == b"|" and (y == 0 or desc[y - 1][x] != b"|"):
                    self.window.blit(self.median_vert[0], cell)
                elif desc[y][x] == b"|" and (y == desc.shape[0] - 1 or desc[y + 1][x] != b"|"):
                    self.window.blit(self.median_vert[2], cell)
                elif desc[y][x] == b"|":
                    self.window.blit(self.median_vert[1], cell)

                # Draw horizontal walls
                elif desc[y][x] == b"-" and (x == 0 or desc[y][x - 1] != b"-"):
                    self.window.blit(self.median_horiz[0], cell)
                elif desc[y][x] == b"-" and (x == desc.shape[1] - 1 or desc[y][x + 1] != b"-"):
                    self.window.blit(self.median_horiz[2], cell)
                elif desc[y][x] == b"-":
                    self.window.blit(self.median_horiz[1], cell)

        # Render the farms
        for farm in self.farm_locations:
            self.window.blit(farm.farm_image(self.cell_size), self.get_surf_loc((farm.farm_row, farm.farm_column)))

        # Update taxi orientation based on the last action taken
        if self.lastaction in [0, 1, 2, 3]:
            self.taxi_orientation = self.lastaction

        # Draw the taxi at its current position
        taxi_row = self.s.row
        taxi_col = self.s.column
        taxi_location = self.get_surf_loc((taxi_row, taxi_col))

        # Render deliveries (pick up and drop off locations)
        deliveries = self.s.delivery_states
        delivery_images = []
        order_images = []
        for index, delivery in enumerate(deliveries):
            
            delivery_location = self.get_surf_loc((delivery.delivery_row, delivery.delivery_column))
            if delivery.has_fulfill_order():
                image = self.destination_imgs_dropoff[index]
            else:
                image = self.destination_imgs[index]
            
            delivery_images.append(image)
            
            # Now get the related farms we want a delivery from 
            delivery_state = delivery.delivery_state
            orders = []
            for farm in self.farm_locations:
                has_order = getattr(delivery_state, farm.farm_name, False)
                if has_order:
                    orders.append(farm.farm_image(self.cell_size))
            order_images.append(orders)
            self.window.blit(image, delivery_location)

        # Render the taxi image
        self.window.blit(self.taxi_imgs[self.taxi_orientation], taxi_location)

        # Render the status bar at the bottom of the window
        status_bar_height = STATUS_BAR_SIZE[1]
        status_bar_rect = pygame.Rect(0, WINDOW_SIZE[1], STATUS_BAR_SIZE[0], status_bar_height)
        pygame.draw.rect(self.window, background_color, status_bar_rect)

        # Prepare the status text to display
        status_text = []
        
        for farm in self.farm_locations:
            content = [f"Pick up from {farm.farm_name}: {getattr(self.s.current_inventory_state, farm.farm_name)}"]
            content.append(farm.farm_image(self.cell_size))
            status_text.append(content)
        
        status_text.append([f"Total steps: {self.turns}", None])
        status_text.append([f"Total rewards: {self.total_rewards}", None])
        
        # Space each status message evenly
        space_per_row = int(status_bar_height / len(status_text))
        current_position = WINDOW_SIZE[1]
        
        # Render each line of status text
        for content in status_text:
            text = content[0]
            image = content[1]
            x_position = 10
            if image:
                resize_image = pygame.transform.scale(image, (20, 20))
                # Position for the farm image
                farm_position = (x_position, current_position)
                x_position += resize_image.get_width() + 20
                
                # Blit the farm image
                self.window.blit(resize_image, farm_position)
            
            rendered_text = font.render(text, True, text_color)
            self.window.blit(rendered_text, (x_position, current_position))
            current_position += space_per_row
        
        # render the legend
        self._render_order_status(delivery_images, order_images)

        # Update the display if the mode is 'human'
        if mode == "human":
            pygame.display.update()  # Update the display window
            self.clock.tick(self.metadata["render_fps"])  # Control the frame rate

    def _render_order_status(self, deliveries, orders):
        """
        Renders the order status with 3 locations and associated farm images.
        Each location has up to 3 farm images displayed below it in a row.
        """
        # Set font for order status text
        font = pygame.font.SysFont("Arial", 16)

        # Define colors for the order bar background and text
        background_color = (255, 255, 255)  # White background
        text_color = (0, 0, 0)  # Black text
        
        # Fill the order bar area with a slightly darker background
        order_bar_height = ORDER_BAR_SIZE[1]
        order_bar_rect = pygame.Rect(0, WINDOW_SIZE[1] + STATUS_BAR_SIZE[1], ORDER_BAR_SIZE[0], order_bar_height)
        pygame.draw.rect(self.window, background_color, order_bar_rect)

        # Draw a subtle shadow for the order bar
        order_bar_shadow = pygame.Rect(0, WINDOW_SIZE[1] + STATUS_BAR_SIZE[1] + 3, ORDER_BAR_SIZE[0], order_bar_height)
        pygame.draw.rect(self.window, (180, 180, 180), order_bar_shadow)

        # Add a rounded border for the order bar
        pygame.draw.rect(self.window, (200, 200, 200), order_bar_rect, border_radius=10)

        # Define position and spacing for the legend
        starting_y = WINDOW_SIZE[1] + STATUS_BAR_SIZE[1]
        legend_x = 10  # Starting x position for the order_bar
        legend_y = starting_y + 10  # Starting y position for the order_bar
        spacing = 10    # Space between images
        farm_spacing = 10  # Space between farm images in the row  
        legend_spacing = 20
        
        # Make the background colour white
        order_bar_height = ORDER_BAR_SIZE[1]
        order_bar_rect = pygame.Rect(0, starting_y, ORDER_BAR_SIZE[0], order_bar_height)
        self.window.fill(background_color, order_bar_rect)  # Fill the order bar area with white
        
        rendered_text = font.render("Order Status", True, text_color)
        self.window.blit(rendered_text, (legend_x, legend_y))
        
        legend_y += legend_spacing
        for i, location_img in enumerate(deliveries):
            # Blit the location image
            resize_location_img = pygame.transform.scale(location_img, (30, 30))
            # Position for the location image
            location_position = (legend_x, legend_y + i * (resize_location_img.get_height() + spacing))
            self.window.blit(resize_location_img, location_position)
            
            order = orders[i]
            # Now, render up to 3 farm images beneath the location
            for j, farm_image in enumerate(order):
                if farm_image is not None:
                    # Position for the farm image
                    resize_farm_image = pygame.transform.scale(farm_image, (20, 20))
                    farm_position = (location_position[0] + resize_location_img.get_width() + legend_spacing + (j * (resize_farm_image.get_width() + farm_spacing)), 
                                    location_position[1])
                    
                    # Blit the farm image
                    self.window.blit(resize_farm_image, farm_position)


    def get_surf_loc(self, map_loc):
        """
        Get pixel surface location on screen from grid coordinates.
        """
        return (map_loc[1] * 2 + 1) * self.cell_size[0], (
            map_loc[0] + 1
        ) * self.cell_size[1]

    def close(self):
        """
        Close the rendering window (if any).
        """
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()