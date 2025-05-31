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
        In Stage 1: leave this method empty (no food spawning) so tests focus only on energy deduction.
        """
        pass

    def step(self) -> None:
        """
        Advance the simulation by one time step:
        1. Call spawn_food()  (currently does nothing in Stage 1).
        2. For each creature in a copy of self.creatures:
           a. Get vision = creature.sensors[0].get_reading(creature, self)
           b. action = creature.decide(vision)
           c. creature.apply_action(action, self)
           d. If creature.energy ≤ 0: remove creature from self.creatures
        """
        self.spawn_food()
        for creature in list(self.creatures):
            # Get vision reading
            vision = creature.sensors[0].get_reading(creature, self)
            # Decide continuous action
            action = creature.decide(vision)
            # Apply the action
            creature.apply_action(action, self)
            # Remove if dead
            if creature.energy <= 0:
                self.creatures.remove(creature)

# Import at the end to avoid circular imports
from .creature import Creature
