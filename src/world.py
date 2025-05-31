from typing import List, Set, Tuple, TYPE_CHECKING

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
        For each empty grid cell (no creature and no existing food), 
        spawn a new food point with probability self.food_spawn_rate.
        """
        # Compute the set of currently occupied grid-cells by creatures
        occupied_cells = {(int(creature.x), int(creature.y)) for creature in self.creatures}

        # Loop over every grid-cell
        for x in range(self.width):
            for y in range(self.height):
                # Skip if cell is already occupied by food or creature
                if (x, y) in self.food_positions or (x, y) in occupied_cells:
                    continue

                # Otherwise, generate a uniform random number in [0,1)
                # If it is less than food_spawn_rate, add (x, y) to food_positions
                import random
                if random.random() < self.food_spawn_rate:
                    self.food_positions.add((x, y))

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
