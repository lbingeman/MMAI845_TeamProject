from contextlib import closing
from io import StringIO
from os import path
from typing import Optional

import numpy as np
import random

from gym import Env, logger, spaces, utils
from gym.envs.toy_text.utils import categorical_sample
from gym.error import DependencyNotInstalled
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


class FarmEnv(Env):
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(self, num_deliveries: int = 3, render_mode: Optional[str] = None):
        self.desc = np.asarray(MAP, dtype="c")

        # Farm locations
        self.farm_locations = [Farm('currie_farm', 0,0), Farm('union_farm', 0,4), Farm('smith_farm', 4,0)]
        self.number_of_farms = len(self.farm_locations)
        
        inventory_states = 8
        self.num_deliveries = num_deliveries
        self.num_rows = 5
        self.num_columns = 5
        self.grid_size = self.num_rows * self.num_columns
        self.num_states = self.grid_size * inventory_states * num_deliveries * inventory_states * (self.grid_size - self.number_of_farms)

        self.transition_table = {}
        
        # Initialize the state
        self.initialize_state()
        
        num_actions = 6

        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(self.num_states)

        self.render_mode = render_mode

        # pygame utils
        self.window = None
        self.clock = None
        self.cell_size = (
            WINDOW_SIZE[0] / self.desc.shape[1],
            WINDOW_SIZE[1] / self.desc.shape[0],
        )
        self.taxi_imgs = None
        self.taxi_orientation = 0
        self.destination_img = None
        self.destination_img_pickedup = None
        self.farm_img = None
        self.destination_img = None
        self.median_horiz = None
        self.median_vert = None
        self.background_img = None

    def is_at_farm(self, location_row, location_column):
        for farm in self.farm_locations:
            if farm.is_at_location(location_row, location_column):
                return farm 
        return None
    
    def initialize_state(self):
        current_inventory_state = InventoryState()
        # Initialize peoples locations and delivery state
        location_list = random.sample(range(0, self.num_rows * self.num_columns), self.num_deliveries + self.number_of_farms)
        delivery_states = []
        
        for _ in range(self.num_deliveries):
            # Randomly initialize the DeliveryState object
            while True:
                union_farm = random.choice([True, False])
                currie_farm = random.choice([True, False])
                smith_farm = random.choice([True, False])
                
                # Ensure that not all states are False
                if union_farm or currie_farm or smith_farm:
                    break
            delivery_location = location_list.pop()
            delivery_row = int(delivery_location/self.num_columns)
            delivery_column = delivery_location % self.num_columns
            while self.is_at_farm(delivery_row, delivery_column):
                delivery_location = location_list.pop()
                delivery_row = int(delivery_location/self.num_columns)
                delivery_column = delivery_location % self.num_columns
            
            # Create a new DeliveryState object and append it to the list
            delivery_states.append(Delivery(DeliveryState(union_farm, currie_farm, smith_farm), delivery_row, delivery_column))
        
        location = random.sample(range(0, self.num_rows), 1)[0], random.sample(range(0, self.num_columns), 1)[0]

        # Print out the farm locations
        for farm in self.farm_locations:
            print("Farm: ", farm.farm_row, " ", farm.farm_column)
        
        for delivery in delivery_states:
            print("Delivery: ", delivery.delivery_row, " ", delivery.delivery_column)
        
        self.s = StateObject(current_inventory_state, delivery_states, location[0], location[1])
        
    def is_valid_position(self, col_position, row_position):
        if row_position >= self.num_columns or row_position < 0:
            return False
        if col_position >= self.num_columns or col_position < 0:
            return False
        return True
    
    def get_transition(self, a):        
        # Let's see if we have a transition for it
        if self.s in self.transition_table and a in self.transition_table[self.s]:
            return self.transition_table[self.s][a]
        
        current_inventory = self.s.current_inventory_state
        deliveries = self.s.delivery_states
        row_position = self.s.row
        column_position = self.s.column
        reward = 0
        
        if a == 0:
            # Move south
            if self.is_valid_position(row_position - 1, column_position):
                row_position = row_position - 1
            reward = -1
        elif a == 1:
            # Move north
            if self.is_valid_position(row_position + 1, column_position):
                row_position = row_position + 1
            reward = -1
        elif a == 2:
            # Move east
            if self.is_valid_position(row_position, column_position + 1) and self.desc[row_position + 1, 2 * column_position + 2] == b":":
                column_position = column_position + 1
            reward = -1
        elif a == 3:
            # Move west
            if self.is_valid_position(row_position, column_position - 1) and self.desc[row_position + 1, 2 * column_position] == b":":
                column_position = column_position - 1
            reward = -1
        elif a == 4:
            # pickup produce
            ## First check if we are at a farm
            farm = self.is_at_farm(row_position, column_position)
            if farm is None or farm.has_picked_up_inventory(current_inventory):
                reward = -10
            else:
                farm.pickup_inventory(current_inventory)
                reward = 20
        elif a == 5:
            # drop off with customer
            target_customer = None
            for customer in deliveries:
                if customer.is_at_location(row_position, column_position):
                    target_customer = customer
                    break 
            
            if target_customer is None or not target_customer.can_fulfill_order(current_inventory) or target_customer.has_fulfill_order():
                reward = -10
            else:
                target_customer.fulfill_order(current_inventory)
                reward = 20
        
        ## Check if we meet the end condition
        met_end_condition = True
        for customer in deliveries:
            if not customer.has_fulfill_order():
                met_end_condition = False 
                break
        
        state = StateObject(current_inventory, deliveries, row_position, column_position)
        # set the transition table
        if self.s not in self.transition_table:
            self.transition_table[self.s] = {}
            
        self.transition_table[self.s][a] = [(1.0, state, reward, met_end_condition)]
        return self.transition_table[self.s][a]

    def action_mask(self, state):
        """Computes an action mask for the action space using the state information."""
        mask = np.zeros(6, dtype=np.int8)
        current_inventory_state = state.current_inventory_state
        delivery_states = state.delivery_states
        row = state.row
        col = state.column
        farm = self.is_at_farm(row, col)
        target_customer = None
        for customer in delivery_states:
            if customer.is_at_location(row, col):
                target_customer = customer
        if self.is_valid_position(row - 1, col):
            mask[0] = 1
        if self.is_valid_position(row + 1, col):
            mask[1] = 1
        if self.is_valid_position(row, col + 1) and self.desc[row + 1, 2 * col + 2] == b":":
            mask[2] = 1
        if self.is_valid_position(row, col - 1) and self.desc[row + 1, 2 * col] == b":":
            mask[3] = 1
        if farm and not farm.has_picked_up_inventory(current_inventory_state):
            mask[4] = 1
        if target_customer and target_customer.can_fulfill_order(current_inventory_state):
            mask[5] = 1
        return mask

    def step(self, a):
        transitions = self.get_transition(a)
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        (probability, new_state, reward, met_end_condition) = transitions[i]
        self.s = new_state
        self.lastaction = a
        self.turns += 1

        if self.render_mode == "human":
            self.render()
        return (self.s, reward, met_end_condition, False, {"prob": probability, "action_mask": self.action_mask(new_state)})

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.initialize_state()
        self.lastaction = None
        self.turns = 0
        self.taxi_orientation = 0

        if self.render_mode == "human":
            self.render()
        return self.s, {"prob": 1.0, "action_mask": self.action_mask(self.s)}

    def render(self):
        self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame  # dependency to pygame only if rendering with human
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[toy_text]`"
            )

        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Farm Delivery Service")
            window_size = (WINDOW_SIZE[0], WINDOW_SIZE[1] + STATUS_BAR_SIZE[1])
            if mode == "human":
                self.window = pygame.display.set_mode(window_size)
            elif mode == "rgb_array":
                self.window = pygame.Surface(window_size)
        
        # Set the font for the status bar
        font = pygame.font.SysFont("Arial", 16)

        # Define the colors for the status bar
        background_color = (255, 255, 255)
        text_color = (0, 0, 0)

        assert (
            self.window is not None
        ), "Something went wrong with pygame. This should never happen."
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.taxi_imgs is None:
            file_names = [
                path.join(path.dirname(__file__), "img/cab_front.png"),
                path.join(path.dirname(__file__), "img/cab_rear.png"),
                path.join(path.dirname(__file__), "img/cab_right.png"),
                path.join(path.dirname(__file__), "img/cab_left.png"),
            ]
            self.taxi_imgs = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.farm_img is None:
            file_name = path.join(path.dirname(__file__), "img/farm.png")
            self.farm_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.destination_img is None:
            file_name = path.join(path.dirname(__file__), "img/hotel.png")
            self.destination_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
            self.destination_img.set_alpha(170)
        if self.destination_img_pickedup is None:
            file_name = path.join(path.dirname(__file__), "img/hotel_pickedup.png")
            self.destination_img_pickedup = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
            self.destination_img.set_alpha(170)
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
        if self.background_img is None:
            file_name = path.join(path.dirname(__file__), "img/taxi_background.png")
            self.background_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        desc = self.desc

        for y in range(0, desc.shape[0]):
            for x in range(0, desc.shape[1]):
                cell = (x * self.cell_size[0], y * self.cell_size[1])
                self.window.blit(self.background_img, cell)
                if desc[y][x] == b"|" and (y == 0 or desc[y - 1][x] != b"|"):
                    self.window.blit(self.median_vert[0], cell)
                elif desc[y][x] == b"|" and (
                    y == desc.shape[0] - 1 or desc[y + 1][x] != b"|"
                ):
                    self.window.blit(self.median_vert[2], cell)
                elif desc[y][x] == b"|":
                    self.window.blit(self.median_vert[1], cell)
                elif desc[y][x] == b"-" and (x == 0 or desc[y][x - 1] != b"-"):
                    self.window.blit(self.median_horiz[0], cell)
                elif desc[y][x] == b"-" and (
                    x == desc.shape[1] - 1 or desc[y][x + 1] != b"-"
                ):
                    self.window.blit(self.median_horiz[2], cell)
                elif desc[y][x] == b"-":
                    self.window.blit(self.median_horiz[1], cell)

        deliveries = self.s.delivery_states
        taxi_row = self.s.row
        taxi_col = self.s.column
        
        # draw the farms
        for farm in self.farm_locations:
            self.window.blit(self.farm_img, self.get_surf_loc((farm.farm_row, farm.farm_column)))

        if self.lastaction in [0, 1, 2, 3]:
            self.taxi_orientation = self.lastaction
        
        taxi_location = self.get_surf_loc((taxi_row, taxi_col))
        # draw the deliveries
        for delivery in deliveries:
            delivery_location = self.get_surf_loc((delivery.delivery_row, delivery.delivery_column))
            if delivery.has_fulfill_order():
                image = self.destination_img_pickedup
            else:
                image = self.destination_img
            self.window.blit(image, delivery_location)
        
        self.window.blit(self.taxi_imgs[self.taxi_orientation], taxi_location)

        ## Write update
        status_bar_height = STATUS_BAR_SIZE[1]
        status_bar_rect = pygame.Rect(0, WINDOW_SIZE[1], STATUS_BAR_SIZE[0], status_bar_height)
        self.window.fill(background_color, status_bar_rect)
        status_text = [f"Pick up from Currie Farm: {self.s.current_inventory_state.currie_farm}", 
                       f"Pick up from Union Farm: {self.s.current_inventory_state.union_farm}",
                       f"Pick up from Smith Farm: {self.s.current_inventory_state.smith_farm}",
                       f"Total steps: {self.turns}"]
        space_per_row = int(status_bar_height / len(status_text))
        current_position = WINDOW_SIZE[1]
        for text in status_text:
            rendered_text = font.render(text, True, text_color)
            self.window.blit(rendered_text, (10, current_position))
            current_position += space_per_row
        
        if mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def get_surf_loc(self, map_loc):
        return (map_loc[1] * 2 + 1) * self.cell_size[0], (
            map_loc[0] + 1
        ) * self.cell_size[1]

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()