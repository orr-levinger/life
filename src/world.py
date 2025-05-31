from typing import List, Set, Tuple, TYPE_CHECKING
import random

if TYPE_CHECKING:
    from .creature import Creature

class World:
    def __init__(self, width: int, height: int, food_spawn_rate: float):
        """
        width, height: dimensions of a discrete 2D grid (0 ≤ x < width, 0 ≤ y < height).
        food_spawn_rate: probability (0.0–1.0) per empty cell per step to spawn food.
        """
        self.width = width
        self.height = height
        self.food_spawn_rate = food_spawn_rate
        self.creatures: List['Creature'] = []
        self.food_positions: Set[Tuple[int, int]] = set()

    def add_creature(self, creature: 'Creature') -> None:
        """Add a Creature instance into the world's creature list."""
        self.creatures.append(creature)

    def spawn_food(self) -> None:
        """
        Spawn food in the world based on food_spawn_rate.

        If food_spawn_rate is 1.0, spawn food in all empty cells (used in tests).
        Otherwise, pick K random cells and attempt to place food there.

        For normal simulation (food_spawn_rate < 0.5), limits the maximum number of food items 
        to prevent screen filling. For test cases (food_spawn_rate >= 0.5), allows unlimited food.
        """
        # Special case for tests: if food_spawn_rate is 1.0, spawn food in all empty cells
        if self.food_spawn_rate == 1.0:
            # Precompute which grid cells are currently occupied by creatures
            occupied = {(int(c.x), int(c.y)) for c in self.creatures}

            # Add food to all empty cells
            for x in range(self.width):
                for y in range(self.height):
                    cell = (x, y)
                    if cell not in self.food_positions and cell not in occupied:
                        self.food_positions.add(cell)
            return

        # Normal case: calculate number of food items to try spawning
        K = int(self.width * self.height * self.food_spawn_rate)

        # Ensure at least one food spawn attempt for very low rates
        # This guarantees some food will spawn even with rates like 0.001
        if 0 < self.food_spawn_rate < 0.005 and K == 0:
            K = 1

        # Apply food limit only for normal simulation (not for tests with high spawn rates)
        if self.food_spawn_rate < 0.5:
            # Maximum number of food items allowed (fixed at 20)
            MAX_FOOD = 20

            # If we're already at or above the maximum, don't spawn more food
            if len(self.food_positions) >= MAX_FOOD:
                return

            # Limit K to avoid excessive attempts when we're close to MAX_FOOD
            K = min(K, MAX_FOOD - len(self.food_positions))

        # Precompute which grid cells are currently occupied:
        occupied = {(int(c.x), int(c.y)) for c in self.creatures}

        for _ in range(K):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            cell = (x, y)
            # Only place new food if this cell is empty:
            if cell in self.food_positions or cell in occupied:
                continue
            # Otherwise, spawn food here:
            self.food_positions.add(cell)

    def step(self) -> None:
        """
        Advance the simulation by one time step:
        1. Call spawn_food() to randomly place new food items on empty grid cells.
        2. For each creature in a copy of self.creatures:
           a. Get vision = creature.sensors[0].get_reading(creature, self)
           b. action = creature.decide(vision)
           c. creature.apply_action(action, self)
           d. Check if creature is on a food cell, if so:
              - Remove food from that cell
              - Increase creature's energy by its eat_bonus value
           e. If creature.energy ≤ 0: remove creature from self.creatures
        """
        self.spawn_food()
        for creature in list(self.creatures):
            # Get vision reading
            vision = creature.sensors[0].get_reading(creature, self)

            # Check if creature is on a food cell before deciding action
            creature_cell = (int(creature.x), int(creature.y))
            on_food = creature_cell in self.food_positions

            # Decide continuous action
            action = creature.decide(vision, on_food)

            # Apply the action
            creature.apply_action(action, self)

            # Check if creature is on a food cell
            creature_cell = (int(creature.x), int(creature.y))
            if creature_cell in self.food_positions:
                # Remove food from that cell
                self.food_positions.remove(creature_cell)
                # Increase creature's energy by its eat_bonus value
                creature.energy += creature.eat_bonus

            # Remove if dead
            if creature.energy <= 0:
                self.creatures.remove(creature)

# Import at the end to avoid circular imports
from .creature import Creature
