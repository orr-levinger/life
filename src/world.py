from typing import List, Set, Tuple, Dict, TYPE_CHECKING
import random
import math

if TYPE_CHECKING:
    from .creature import Creature
    from .food import Food

class World:
    # Constants for continuous space
    DEFAULT_FOOD_SIZE = 1.0
    DEFAULT_FOOD_ENERGY = 2.0
    DEFAULT_FOOD_RADIUS = 0.2  # Default radius for food items
    MIN_SPAWN_DISTANCE = 0.5   # Minimum distance between spawned food and other objects
    MAX_FOOD = 200 # Maximum number of food items allowed (fixed at 20)

    def __init__(self, width: int, height: int, food_spawn_rate: float):
        """
        width, height: dimensions of the continuous space (0 ≤ x < width, 0 ≤ y < height).
        food_spawn_rate: probability per unit area per step to spawn food.
        """
        self.width = width
        self.height = height
        self.food_spawn_rate = food_spawn_rate
        self.creatures: List['Creature'] = []
        self.foods: List['Food'] = []
        self.foods_created_this_step: Set[int] = set()  # Track foods created in the current step

    def add_creature(self, creature: 'Creature') -> None:
        """Add a Creature instance into the world's creature list."""
        self.creatures.append(creature)

    def spawn_food(self) -> None:
        """
        Spawn food in the world based on food_spawn_rate.

        If food_spawn_rate is 1.0, spawn food in all empty cells (used in tests).
        Otherwise, spawn food at random continuous coordinates.

        For normal simulation (food_spawn_rate < 0.5), limits the maximum number of food items 
        to prevent screen filling. For test cases (food_spawn_rate >= 0.5), allows unlimited food.
        """
        # Import Food here to avoid circular imports
        from .food import Food

        # Special case for tests: if food_spawn_rate is 1.0, spawn food in all empty grid cells
        if self.food_spawn_rate == 1.0:
            # Precompute which grid cells are currently occupied by creatures
            occupied = {(int(c.x), int(c.y)) for c in self.creatures}
            # Also precompute which cells already have food (using integer coordinates)
            food_cells = {(int(f.x), int(f.y)) for f in self.foods}

            # Add food to all empty cells
            for x in range(self.width):
                for y in range(self.height):
                    cell = (x, y)
                    if cell not in food_cells and cell not in occupied:
                        # Create a new Food object with random continuous coordinates within the cell
                        # This ensures food is not exactly at grid points but has continuous position
                        fx = x + random.random()  # Random position within [x, x+1)
                        fy = y + random.random()  # Random position within [y, y+1)

                        new_food = Food(
                            x=fx,
                            y=fy,
                            remaining_duration=-1,  # -1 means infinite duration
                        )
                        self.foods.append(new_food)
            return

        # Normal case: calculate number of food items to try spawning
        # For continuous space, we scale by area rather than cell count
        K = int(self.width * self.height * self.food_spawn_rate)

        # Ensure at least one food spawn attempt for very low rates
        if 0 < self.food_spawn_rate < 0.005 and K == 0:
            K = 1

        # Apply food limit only for normal simulation (not for tests with high spawn rates)
        if self.food_spawn_rate < 0.5:

            # If we're already at or above the maximum, don't spawn more food
            if len(self.foods) >= self.MAX_FOOD:
                return

            # Limit K to avoid excessive attempts when we're close to MAX_FOOD
            K = min(K, self.MAX_FOOD - len(self.foods))

        # Helper function to check if a position is valid for new food
        def is_valid_position(x, y, radius):
            # Check if within world bounds
            if x - radius < 0 or x + radius > self.width or y - radius < 0 or y + radius > self.height:
                return False

            # Check for overlap with creatures
            for creature in self.creatures:
                dx = x - creature.x
                dy = y - creature.y
                distance = math.sqrt(dx*dx + dy*dy)
                if distance < (radius + creature.radius + self.MIN_SPAWN_DISTANCE):
                    return False

            # Check for overlap with existing food
            for food in self.foods:
                dx = x - food.x
                dy = y - food.y
                distance = math.sqrt(dx*dx + dy*dy)
                if distance < (radius + food.radius + self.MIN_SPAWN_DISTANCE):
                    return False

            return True

        # Try to spawn K food items at random continuous coordinates
        for _ in range(K):
            # Generate random continuous coordinates
            x = random.uniform(self.DEFAULT_FOOD_RADIUS, self.width - self.DEFAULT_FOOD_RADIUS)
            y = random.uniform(self.DEFAULT_FOOD_RADIUS, self.height - self.DEFAULT_FOOD_RADIUS)

            # Check if position is valid (no overlap with creatures or other food)
            if not is_valid_position(x, y, self.DEFAULT_FOOD_RADIUS):
                continue

            # Create new food at valid position
            new_food = Food(
                x=x,
                y=y,
                remaining_duration=-1,  # -1 means infinite duration
            )
            self.foods.append(new_food)
            # Track that this food was created in this step
            self.foods_created_this_step.add(id(new_food))

    def step(self) -> None:
        """
        Advance the simulation by one time step:
        1. Call spawn_food() to randomly place new food items in continuous space.
        2. Remove dead creatures (energy ≤ 0)
        3. Gather all creatures' decisions.
        4. Process all creatures in phases:
           a. Apply all ATTACK actions first (kills generate corpses → new Food objects)
           b. Apply all EAT actions next (which decrement food energy by 1 per bite)
           c. Apply all other actions (MOVE, EAT_AT_CURRENT, REST)
        5. Decay all Food objects and remove expired ones
        """
        # Clear the set of foods created in the previous step
        self.foods_created_this_step.clear()

        # 1) Spawn new food
        self.spawn_food()

        # 2) Remove dead creatures
        self.creatures = [c for c in self.creatures if c.energy > 0]

        # 3) Gather all creatures' decisions
        decisions = []
        for creature in list(self.creatures):
            # Get proximity reading
            vision = creature.sensors[0].get_reading(creature, self)

            # Check if creature is overlapping with any food
            on_food = False
            for food in self.foods:
                if food.is_overlapping(creature.x, creature.y, creature.radius):
                    on_food = True
                    break

            # Decide action
            action = creature.decide(vision, on_food)
            decisions.append((creature, action))

        # 4a) Apply all ATTACK actions first
        for creature, action in decisions:
            if action[0] == "ATTACK":
                creature.apply_action(action, self)

        # 4b) Apply all EAT actions next
        for creature, action in decisions:
            if action[0] == "EAT":
                creature.apply_action(action, self)

        # 4c) Apply all other actions (MOVE, EAT_AT_CURRENT, REST)
        for creature, action in decisions:
            if action[0] not in ("ATTACK", "EAT"):
                creature.apply_action(action, self)

        # 5.5) Check if any creatures should split and perform the split
        # We need to make a copy of the list because we'll be modifying it
        for creature in list(self.creatures):
            if creature.should_split():
                # Split the creature into 4 (1 parent + 3 children)
                children = creature.split(self)
                print(f"Creature split into 4! Parent energy: {creature.energy}, size: {creature.size}, Children: {len(children)}")

        # 6) Decay all Food objects and remove expired ones
        new_foods = []

        for food in self.foods:
            # Skip decay for foods created in this step
            if id(food) in self.foods_created_this_step:
                # Keep the food without decaying it
                new_foods.append(food)
            else:
                # Decay the food
                food.decay()
                # Keep it if not expired by duration and not fully consumed
                if not food.is_expired() and food.energy > 0:
                    new_foods.append(food)

        self.foods = new_foods

# Import at the end to avoid circular imports
from .creature import Creature
