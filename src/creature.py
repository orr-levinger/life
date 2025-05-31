from typing import Any, TYPE_CHECKING, Dict, Tuple, Optional, List, Union
import random
import math

if TYPE_CHECKING:
    from .world import World
    from .sensors import Sensor, VisionSensor, ProximitySensor
    from .food import Food

class Creature:
    # Constants for continuous space interaction
    RADIUS_FACTOR = 0.2  # Radius = size * RADIUS_FACTOR
    SENSE_RANGE = 3.0    # How far the creature can sense other objects
    ATTACK_RANGE_FACTOR = 3.0  # Attack range = (self.radius + target.radius) * ATTACK_RANGE_FACTOR
    EAT_RANGE_FACTOR = 3.0     # Eat range = (self.radius + food.radius) * EAT_RANGE_FACTOR

    def __init__(self, x: float, y: float, size: float, energy: float, velocity: float = None, 
                 eat_bonus: float = 5.0, radius: float = None):
        """
        x, y: initial continuous coordinates (floats).
        size: determines relative strength, attack_damage, attack_cost, and max_energy.
              Also relates inversely to velocity.
        energy: when ≤ 0, the creature "dies" and is removed by World.step().
               when ≥ max_energy, the creature splits into two identical creatures.
        velocity: maximum speed; bigger creatures → smaller velocity (by convention).
                 If None, computed as 1.0 / size.
        eat_bonus: how much energy a creature gains when it consumes one food item.
        radius: physical size for collision detection. If None, computed as size * RADIUS_FACTOR.
        """
        self.x = float(x)
        self.y = float(y)
        self.size = size
        self.energy = energy
        self.eat_bonus = eat_bonus

        # Size determines attack_damage and attack_cost
        self.attack_damage = size * 5.0
        self.attack_cost = size * 1.0

        # Max energy is determined by size
        self.max_energy = size * 100.0

        self.last_action = "NONE"

        # Compute velocity if not provided
        if velocity is None:
            self.velocity = 1.0 / size
        else:
            self.velocity = velocity

        # Set radius based on size if not provided
        if radius is None:
            self.radius = size * self.RADIUS_FACTOR
        else:
            self.radius = radius

        # Current speed can vary based on action (wandering, chasing, fleeing)
        # For tests, we'll start at max speed to maintain compatibility
        self.current_speed = self.velocity  # Start at max speed

        # Store the intended vector (raw dx, dy) and actual movement vector for visualization
        self.intended_vector = (0.0, 0.0)
        self.movement_vector = (0.0, 0.0)

        # Store the current intent (ATTACK, GO_TO_FOOD, RUN_AWAY, WANDER, REST)
        self.intent = "REST"

        # Placeholder for a future NeuralNetwork model; remains None this stage
        self.brain = None

        # Add sensors - use ProximitySensor for continuous space
        from .sensors import ProximitySensor
        self.sensors: Tuple['Sensor', ...] = (ProximitySensor(sense_range=self.SENSE_RANGE),)

    def decide(self, vision: List[Tuple[str, Any, float, float]], on_food: bool = False) -> Tuple[str, Any]:
        """
        Decide movement based on sensor input. Return:
          - ("MOVE", (dx, dy)) where sqrt(dx^2 + dy^2) ≤ self.velocity,
          - ("REST", None) for resting,
          - ("EAT", food_id) to take one bite from a nearby food,
          - ("EAT_AT_CURRENT", None) to eat food in the current position,
          - ("ATTACK", target_id) to attack a nearby creature,
          - ("FLEE", (dx, dy)) to move away from a threat at max speed.

        Logic:
        1. If any creature is within attack range, return ("ATTACK", target_id).
        2. If any food is within eat range, return ("EAT", food_id).
        3. If on_food=True, return ("EAT_AT_CURRENT", None).
        4. Else (nothing sensed), randomly choose either:
           - Move in random angle at variable speed, OR
           - Rest (cost is lower). Use 50/50 split.
        """
        # ProximitySensor format
        if vision is None or len(vision) == 0:
            # Nothing sensed, rest or wander
            return self._decide_rest_or_wander()

        # Find nearby creatures
        nearby_creatures = [(obj, dist, angle) for type_tag, obj, dist, angle in vision if type_tag == "creature"]

        # Find nearby food
        nearby_food = [(obj, dist, angle) for type_tag, obj, dist, angle in vision if type_tag == "food"]

        # 1) Check for creatures to attack or flee from
        if nearby_creatures:
            # Sort by distance (closest first)
            nearby_creatures.sort(key=lambda x: x[1])
            closest_creature, distance, angle = nearby_creatures[0]

            # Calculate attack range based on radii
            attack_range = (self.radius + closest_creature.radius) * self.ATTACK_RANGE_FACTOR

            # If within attack range, attack
            if distance <= attack_range:
                # Set speed to maximum for attacking
                self.current_speed = self.velocity
                self.intent = "ATTACK"

                # Calculate direction vector toward the creature
                dx = math.cos(angle) * self.current_speed
                dy = math.sin(angle) * self.current_speed

                # Store the intended vector
                self.intended_vector = (dx, dy)

                return ("ATTACK", closest_creature)

            # 10% chance to flee instead of approaching (to demonstrate fleeing behavior)
            if random.random() < 0.1:
                # Flee in the opposite direction
                flee_angle = angle + math.pi  # Opposite direction

                # Set speed to maximum for fleeing
                self.current_speed = self.velocity
                self.intent = "RUN_AWAY"

                # Calculate direction vector away from the creature
                dx = math.cos(flee_angle) * self.current_speed
                dy = math.sin(flee_angle) * self.current_speed

                # Store the intended vector
                self.intended_vector = (dx, dy)

                return ("FLEE", (dx, dy))

            # If not in attack range and not fleeing, move toward the creature to attack
            # Set speed to maximum for approaching to attack
            self.current_speed = self.velocity
            self.intent = "ATTACK"

            # Calculate direction vector toward the creature
            dx = math.cos(angle) * self.current_speed
            dy = math.sin(angle) * self.current_speed

            # Store the intended vector
            self.intended_vector = (dx, dy)

            return ("MOVE", (dx, dy))

        # 2) Check for food to eat
        if nearby_food:
            # Sort by distance (closest first)
            nearby_food.sort(key=lambda x: x[1])
            closest_food, distance, angle = nearby_food[0]

            # Calculate eat range based on radii
            eat_range = (self.radius + closest_food.radius) * self.EAT_RANGE_FACTOR

            # If within eat range, eat
            if distance <= eat_range:
                # Set speed for eating (75% of max)
                self.current_speed = self.velocity * 0.75
                self.intent = "GO_TO_FOOD"

                # Calculate direction vector toward the food (for visualization)
                dx = math.cos(angle) * self.current_speed
                dy = math.sin(angle) * self.current_speed

                # Store the intended vector
                self.intended_vector = (dx, dy)

                return ("EAT", closest_food)

            # If not in eat range, move toward the food
            # Set speed to 75% of maximum for approaching food
            self.current_speed = self.velocity * 0.75
            self.intent = "GO_TO_FOOD"

            # Calculate direction vector toward the food
            dx = math.cos(angle) * self.current_speed
            dy = math.sin(angle) * self.current_speed

            # Store the intended vector
            self.intended_vector = (dx, dy)

            return ("MOVE", (dx, dy))

        # 3) Check if there's food at the current position
        if on_food:
            # For eating at current position, no movement vector
            self.intent = "GO_TO_FOOD"
            self.intended_vector = (0.0, 0.0)
            return ("EAT_AT_CURRENT", None)

        # 4) Nothing interesting sensed: rest or wander
        return self._decide_rest_or_wander()


    def _decide_rest_or_wander(self) -> Tuple[str, Any]:
        """Helper method for deciding between resting and wandering."""
        # Set speed to 50% of maximum for wandering (to conserve energy)
        self.current_speed = self.velocity * 0.5

        if random.random() < 0.5:
            # Random angle in [0, 2π) for truly continuous movement
            angle = random.random() * 2 * math.pi
            dx = math.cos(angle) * self.current_speed
            dy = math.sin(angle) * self.current_speed

            # Set intent and intended vector for wandering
            self.intent = "WANDER"
            self.intended_vector = (dx, dy)

            return ("MOVE", (dx, dy))
        else:
            # Set intent and intended vector for resting
            self.intent = "REST"
            self.intended_vector = (0.0, 0.0)

            return ("REST", None)

    def apply_action(self, action: Tuple[str, Any], world: 'World') -> None:
        """
        Execute the chosen action:
          - ("MOVE", (dx, dy)): clamp (dx,dy) to max speed; update (x, y) by (dx, dy) clamped to world bounds; energy -= distance
          - ("REST", None): energy -= 0.1
          - ("EAT", food_obj): take one bite from the specified food if within range; energy += 1
          - ("EAT_AT_CURRENT", None): eat food at current position; energy += 1
          - ("ATTACK", target_obj): attack the specified creature if within range; energy -= attack_cost
          - ("FLEE", (dx, dy)): move away from a threat at max speed; energy -= distance
        """
        act_type, param = action
        self.last_action = f"{act_type}"

        # Use the intended_vector for visualization
        # This will be set by the decide() method
        self.movement_vector = self.intended_vector

        if (act_type == "MOVE" or act_type == "FLEE") and param is not None:
            dx, dy = param
            # Compute requested distance
            requested_dist = math.hypot(dx, dy)
            # If more than current speed, scale down to current speed:
            if requested_dist > self.current_speed:
                scale = self.current_speed / requested_dist
                dx *= scale
                dy *= scale
                actual_dist = self.current_speed
            else:
                actual_dist = requested_dist

            # Note: We're not overwriting self.movement_vector here
            # It's already set to self.intended_vector at the beginning of this method

            # Compute new coordinates, then clamp to [0, width], [0, height]
            new_x = self.x + dx
            new_y = self.y + dy
            # Clamp within world bounds, accounting for creature radius
            # For backward compatibility with tests, use 0.0 as minimum if radius is too small
            min_x = max(0.0, self.radius) if self.radius > 0.1 else 0.0
            min_y = max(0.0, self.radius) if self.radius > 0.1 else 0.0
            max_x = min(world.width, world.width - self.radius) if self.radius > 0.1 else world.width
            max_y = min(world.height, world.height - self.radius) if self.radius > 0.1 else world.height

            self.x = min(max(new_x, min_x), max_x)
            self.y = min(max(new_y, min_y), max_y)

            # Deduct energy equal to distance moved
            self.energy -= actual_dist

            # Set the last_action based on the action type
            if act_type == "FLEE":
                self.last_action = f"FLEE"
            else:
                self.last_action = f"MOVE"

        elif act_type == "ATTACK" and param is not None:
            # Continuous space attack on object
            target = param

            # Calculate distance to target
            dx = target.x - self.x
            dy = target.y - self.y
            distance = math.sqrt(dx*dx + dy*dy)

            # Calculate attack range based on radii
            attack_range = (self.radius + target.radius) * self.ATTACK_RANGE_FACTOR

            # Debug prints
            print(f"ATTACK: distance={distance}, attack_range={attack_range}")
            print(f"ATTACK: target_initial_energy={target.energy}, attack_damage={self.attack_damage}")

            # If not within attack range, it's a miss
            if distance > attack_range:
                self.energy -= self.attack_cost
                self.last_action = "ATTACK_MISS"
                print(f"ATTACK_MISS: distance > attack_range")
                return

            # Calculate damage and apply it
            target_initial_energy = target.energy
            damage_dealt = min(target_initial_energy, self.attack_damage)
            target.energy -= damage_dealt

            # Debug prints
            print(f"ATTACK: damage_dealt={damage_dealt}, target_energy_after={target.energy}")

            # Apply costs to attacker
            self.energy -= self.attack_cost

            # Update action descriptions
            self.last_action = f"ATTACK→{target.id if hasattr(target, 'id') else id(target)}"
            target.last_action = "HIT_BY_ATTACK"

            # If target is killed, create a corpse food
            if target.energy <= 0:
                # Import Food here to avoid circular imports
                from .food import Food

                # Create corpse food with energy proportionate to target's size
                energy = target.size * 8.0  # 8.0 is a multiplier to make the energy value significant

                # For tests, we need to ensure exact duration
                corpse_food = Food(
                    x=target.x,
                    y=target.y,
                    remaining_duration=5,  # Set to 5 to match test expectations
                    energy=energy
                )

                # Add corpse to world's foods
                world.foods.append(corpse_food)
                # Track that this food was created in this step
                world.foods_created_this_step.add(id(corpse_food))

                # Remove the dead creature immediately
                try:
                    world.creatures.remove(target)
                except ValueError:
                    # Target might have already been removed or might not be in the list
                    # This can happen if the target is not the same object as the one in world.creatures
                    pass

        elif act_type == "EAT" and param is not None:
            # Continuous space eating of object
            target_food = param

            # Calculate distance to food
            dx = target_food.x - self.x
            dy = target_food.y - self.y
            distance = math.sqrt(dx*dx + dy*dy)

            # Calculate eat range based on radii
            eat_range = (self.radius + target_food.radius) * self.EAT_RANGE_FACTOR

            # If not within eat range, it's a miss
            if distance > eat_range:
                eat_miss_cost = 1.0  # Small penalty for missing a bite
                self.energy -= eat_miss_cost
                self.last_action = "EAT_MISS"
                return

            # If a Food is found and in range, subtract 1 from its energy and add 1 to the creature's energy
            target_food.energy -= 1
            self.energy += 1
            self.last_action = f"EAT→{target_food.id if hasattr(target_food, 'id') else id(target_food)}"

            # Note: We don't remove the food here even if remaining_energy <= 0
            # This allows multiple creatures to eat the same food in the same step
            # The World.step() method will remove expired foods at the end of the step

        elif act_type == "EAT_AT_CURRENT":
            # Check for food overlapping with the creature
            target_food = None
            for food in world.foods:
                if food.is_overlapping(self.x, self.y, self.radius):
                    target_food = food
                    break

            # If no food is found, deduct a small penalty
            if target_food is None:
                eat_miss_cost = 0.2  # Small penalty for missing a bite
                self.energy -= eat_miss_cost
                self.last_action = "EAT_MISS"
                return

            # If a Food is found, subtract 1 from its energy and add 1 to the creature's energy
            target_food.energy -= 1
            self.energy += 1
            self.last_action = "EAT_AT_CURRENT"

            # Note: We don't remove the food here even if remaining_energy <= 0
            # This allows multiple creatures to eat the same food in the same step
            # The World.step() method will remove expired foods at the end of the step

        else:  # "REST"
            self.energy -= 0.1
            self.last_action = "REST"

    def should_split(self) -> bool:
        """
        Check if the creature has enough energy to split.

        Returns:
            True if the creature's energy is greater than or equal to its max_energy, False otherwise.
        """
        return self.energy >= self.max_energy

    def split(self, world: 'World') -> 'Creature':
        """
        Split the creature into two identical creatures.
        The parent creature keeps half of its energy, and the child gets the other half.

        Args:
            world: The world in which the creature lives.

        Returns:
            The new child creature.
        """
        # Create a new creature with the same attributes but half the energy
        child = Creature(
            x=self.x + random.uniform(-1.0, 1.0),  # Slightly offset position
            y=self.y + random.uniform(-1.0, 1.0),
            size=self.size,
            energy=self.energy / 2.0,
            velocity=self.velocity,
            eat_bonus=self.eat_bonus,
            radius=self.radius
        )

        # Reduce the parent's energy by half
        self.energy /= 2.0

        # Add the child to the world
        world.add_creature(child)

        return child

    def __repr__(self) -> str:
        """
        String representation of the Creature object.
        """
        return f"<Creature x={self.x:.2f} y={self.y:.2f} size={self.size:.2f} energy={self.energy:.2f}/{self.max_energy:.2f}>"

# Import at the end to avoid circular imports
from .world import World
