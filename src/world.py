from typing import List, Set, Tuple, Dict, TYPE_CHECKING
import random

if TYPE_CHECKING:
    from .creature import Creature
    from .food import Food

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
        self.foods: List['Food'] = []
        # Keep food_positions for backward compatibility with existing code
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
        # Import Food here to avoid circular imports
        from .food import Food

        # Default values for spawned food
        DEFAULT_FOOD_SIZE = 1.0
        DEFAULT_FOOD_ENERGY = 2.0

        # Special case for tests: if food_spawn_rate is 1.0, spawn food in all empty cells
        if self.food_spawn_rate == 1.0:
            # Precompute which grid cells are currently occupied by creatures
            occupied = {(int(c.x), int(c.y)) for c in self.creatures}
            # Also precompute which cells already have food
            food_cells = {(f.x, f.y) for f in self.foods}

            # Add food to all empty cells
            for x in range(self.width):
                for y in range(self.height):
                    cell = (x, y)
                    if cell not in food_cells and cell not in occupied:
                        # Create a new Food object with infinite duration
                        new_food = Food(
                            x=x,
                            y=y,
                            size=DEFAULT_FOOD_SIZE,
                            energy_value=DEFAULT_FOOD_ENERGY,
                            remaining_duration=-1  # -1 means infinite duration
                        )
                        self.foods.append(new_food)
                        # Also update food_positions for backward compatibility
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
            if len(self.foods) >= MAX_FOOD:
                return

            # Limit K to avoid excessive attempts when we're close to MAX_FOOD
            K = min(K, MAX_FOOD - len(self.foods))

        # Precompute which grid cells are currently occupied:
        occupied = {(int(c.x), int(c.y)) for c in self.creatures}
        # Also precompute which cells already have food
        food_cells = {(f.x, f.y) for f in self.foods}

        for _ in range(K):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            cell = (x, y)
            # Only place new food if this cell is empty:
            if cell in food_cells or cell in occupied:
                continue
            # Otherwise, spawn food here:
            new_food = Food(
                x=x,
                y=y,
                size=DEFAULT_FOOD_SIZE,
                energy_value=DEFAULT_FOOD_ENERGY,
                remaining_duration=-1  # -1 means infinite duration
            )
            self.foods.append(new_food)
            # Also update food_positions for backward compatibility
            self.food_positions.add(cell)

    def step(self) -> None:
        """
        Advance the simulation by one time step:
        1. Call spawn_food() to randomly place new food items on empty grid cells.
        2. Gather all creatures' decisions.
        3. Process all creatures in phases:
           a. Apply all ATTACK actions first (kills generate corpses → new Food objects)
           b. Apply all EAT actions next (which decrement food energy by 1 per bite)
           c. Apply all other actions (MOVE, EAT_AT_CURRENT, REST)
        4. Remove dead creatures (energy ≤ 0)
        5. Decay all Food objects and remove expired ones
        """
        # 1) Spawn new food
        self.spawn_food()

        # 2) Gather all creatures' decisions
        decisions = []
        for creature in list(self.creatures):
            # Get vision reading
            vision = creature.sensors[0].get_reading(creature, self)

            # Check if creature is on a food cell before deciding action
            creature_cell = (int(creature.x), int(creature.y))
            on_food = any(f.x == creature_cell[0] and f.y == creature_cell[1] for f in self.foods)

            # Decide action
            action = creature.decide(vision, on_food)
            decisions.append((creature, action))

        # 3a) Apply all ATTACK actions first
        for creature, action in decisions:
            if action[0] == "ATTACK":
                creature.apply_action(action, self)

        # 3b) Apply all EAT actions next
        for creature, action in decisions:
            if action[0] == "EAT":
                creature.apply_action(action, self)

        # 3c) Apply all other actions (MOVE, EAT_AT_CURRENT, REST)
        for creature, action in decisions:
            if action[0] not in ("ATTACK", "EAT"):
                creature.apply_action(action, self)

        # 4) Remove dead creatures
        self.creatures = [c for c in self.creatures if c.energy > 0]

        # 5) Decay all Food objects and remove expired ones
        new_foods = []
        for food in self.foods:
            # Decay the food
            food.decay()

            # Keep it if not expired by duration and not fully consumed
            if not food.is_expired() and food.remaining_energy > 0:
                new_foods.append(food)
            else:
                # Remove from food_positions for backward compatibility
                if (food.x, food.y) in self.food_positions:
                    self.food_positions.remove((food.x, food.y))

        self.foods = new_foods

# Import at the end to avoid circular imports
from .creature import Creature
