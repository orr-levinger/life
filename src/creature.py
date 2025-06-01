# src/creature.py

from typing import Any, TYPE_CHECKING, Dict, Tuple, Optional, List, Union
import random               # Used for random number generation (e.g., wandering direction, mutation)
import math                 # Used for trigonometric functions and distance calculations
import os                   # Used for filesystem operations when saving brains
import pickle               # Used for serializing (saving) neural network objects to disk

if TYPE_CHECKING:
    # These imports are only for type checking and do not affect runtime behavior.
    from .world import World
    from .sensors import Sensor, ProximitySensor
    from .food import Food
    from .neural_network import NeuralNetwork

class Creature:
    # ============================================================
    # Creature class: represents one autonomous agent in the world.
    # Each Creature has position, energy, physics-related properties,
    # and a neural network "brain" to make decisions.
    # ============================================================

    # === Constants for continuous space interaction ===
    RADIUS_FACTOR = 0.2            # Multiplier to compute physical radius = size * RADIUS_FACTOR
    SENSE_RANGE = 3.0              # Maximum distance at which a creature can sense others
    ATTACK_RANGE_FACTOR = 3.0      # Attack range = (self.radius + target.radius) * ATTACK_RANGE_FACTOR
    EAT_RANGE_FACTOR = 3.0         # Eat range = (self.radius + food.radius) * EAT_RANGE_FACTOR

    # === Constants for neural network architecture ===
    INPUT_SIZE = 12                # Number of inputs to the neural network (state + sensory features)
    HIDDEN_SIZES = [8, 8]          # Two hidden layers, each with 8 neurons
    OUTPUT_SIZE = 8                # Number of outputs: 6 action types + 2 continuous movement parameters

    # === Constants for brain-saving mechanism ===
    BRAIN_SAVE_THRESHOLD = 0.8     # Fraction of max_energy above which to save the brain
    SAVE_DIR = "saved_brains"      # Directory where high-performing brains will be saved

    def __init__(
            self,
            x: float,
            y: float,
            size: float,
            energy: float,
            velocity: float = None,
            eat_bonus: float = 5.0,
            radius: float = None,
            brain: Optional['NeuralNetwork'] = None,
            create_brain: bool = False,
            parent_id: int = None
    ):
        """
        Initialize a Creature.

        Args:
            x (float): Initial x-coordinate in continuous space.
            y (float): Initial y-coordinate in continuous space.
            size (float): Determines strength, attack cost, and maximum energy.
                           Larger size → more energy capacity and damage, but slower.
            energy (float): Starting energy. Energy ≤ 0 means death; ≥ max_energy means split.
            velocity (float, optional): Max speed. If None, computed as 1.0 / size.
            eat_bonus (float): Energy gained when consuming one unit of food.
            radius (float, optional): Physical collision radius. If None, computed as size * RADIUS_FACTOR.
            brain (NeuralNetwork, optional): Pre-existing neural network to control decisions.
            create_brain (bool): If True and brain is None, instantiates a new NeuralNetwork.
            parent_id (int, optional): ID of the parent creature (used to avoid attacking close kin).
        """
        # --- Position and physical attributes ---
        self.x = float(x)                   # Current x-coordinate
        self.y = float(y)                   # Current y-coordinate
        self.size = size                    # Creature "size" scalar
        self.energy = energy                # Current energy level
        self.eat_bonus = eat_bonus          # Energy reward per food unit
        self.id = id(self)                  # Unique identifier (Python's built-in id)
        self.parent_id = parent_id or 0     # Parent ID (0 if no parent)

        # --- Tracking whether this creature's last decision used its brain ---
        self.using_brain = False            # True if last call to decide() used the neural network
        self.brain_saved = False            # Prevent saving the same brain more than once

        # --- Counter for steps without positive reward ---
        self.steps_without_reward = 0       # If ≥10, forces a random/legacy decision to encourage exploration

        # --- Combat and splitting parameters derived from "size" ---
        self.attack_damage = size * 5.0     # Damage dealt on successful attack
        self.attack_cost = size * 1.0       # Energy cost to attempt an attack
        self.max_energy = size * 100.0      # Energy threshold for splitting

        # --- Last action label, used for tracking/logging ---
        self.last_action = "NONE"

        # --- Velocity logic: larger creatures move more slowly (1/size) if not specified ---
        if velocity is None:
            self.velocity = 1.0 / size      # Base max speed
        else:
            self.velocity = velocity         # Use provided speed

        # --- Physical radius for collisions: size * RADIUS_FACTOR unless overridden ---
        if radius is None:
            self.radius = size * self.RADIUS_FACTOR
        else:
            self.radius = radius

        # --- Current speed can vary based on action (e.g., chasing or fleeing) ---
        self.current_speed = self.velocity  # Start at max speed by default

        # --- Movement vectors for visualization/tracking ---
        self.intended_vector = (0.0, 0.0)    # Raw dx/dy from decide()
        self.movement_vector = (0.0, 0.0)    # Actual dx/dy after clamping to speed/bounds

        # --- Current intent (used for rendering or analysis) ---
        self.intent = "REST"                # Possible values: "ATTACK", "GO_TO_FOOD", "RUN_AWAY", "WANDER", "REST"

        # --- Neural network initialization ---
        if brain is not None:
            # Use a pre-existing NeuralNetwork instance
            self.brain = brain
        elif create_brain:
            # Dynamically import and instantiate a new network if requested
            from .neural_network import NeuralNetwork
            self.brain = NeuralNetwork(
                input_size=self.INPUT_SIZE,
                hidden_sizes=self.HIDDEN_SIZES,
                output_size=self.OUTPUT_SIZE
            )
        else:
            # No neural network (legacy mode)
            self.brain = None

        # --- Sensors setup: currently only a single ProximitySensor is attached ---
        from .sensors import ProximitySensor
        self.sensors: Tuple['Sensor', ...] = (ProximitySensor(sense_range=self.SENSE_RANGE),)

    def decide(self, vision: List[Tuple[str, Any, float, float]], on_food: bool = False) -> Tuple[str, Any]:
        """
        Determine the next action for this creature based on sensory inputs.

        Args:
            vision (List[Tuple[str, Any, float, float]]): A list of sensed objects in the format
                (type_tag, object_reference, distance, angle). type_tag is "creature" or "food".
            on_food (bool): True if the creature is currently on a food source.

        Returns:
            Tuple[action_type (str), action_params (Any)]:
                - "MOVE": (dx, dy) movement vector, |(dx,dy)| ≤ self.velocity
                - "REST": None
                - "EAT": food_object reference to attempt to eat
                - "EAT_AT_CURRENT": None (eat food underneath)
                - "ATTACK": target_creature reference to attempt attack
                - "FLEE": (dx, dy) movement vector for fleeing
        """
        # If the creature hasn't gained energy (no reward) for 10 steps,
        # force a random legacy action (no brain) to encourage exploration
        if self.steps_without_reward >= 10:
            action_type, action_params = self._decide_without_brain(vision, on_food)
            self.using_brain = False
            # Set internal intent/velocity vectors for visualization
            self._set_intent_and_vector(action_type, action_params, vision)
            return action_type, action_params

        # If no brain is installed, fallback to legacy decision logic
        if self.brain is None:
            self.using_brain = False           # Mark that we did not use the neural network
            return self._decide_without_brain(vision, on_food)

        # Otherwise, use the neural network to decide
        self.using_brain = True                # Mark that we are using the brain

        # Package sensory data into a dictionary for the neural network
        sensory_inputs = {
            'vision': vision,                  # Nearby creatures/food information
            'on_food': on_food,                # Are we standing on food now?
            'creature_state': {
                'energy': self.energy,
                'size': self.size,
                'velocity': self.velocity,
                'max_energy': self.max_energy
            }
        }

        # Let the neural network choose an action
        action_type, action_params = self.brain.decide(sensory_inputs)

        # After deciding, set intent and movement vector for rendering/tracking
        self._set_intent_and_vector(action_type, action_params, vision)

        return action_type, action_params

    def _set_intent_and_vector(self, action_type: str, action_params: Any, vision: List[Tuple[str, Any, float, float]]) -> None:
        """
        Based on the action chosen, set the creature's intent label and intended_vector.

        Args:
            action_type (str): One of "MOVE", "REST", "EAT", "EAT_AT_CURRENT", "ATTACK", "FLEE".
            action_params (Any): Parameters corresponding to the action, e.g. (dx, dy) or object reference.
            vision: Same vision list passed to decide(), used to calculate direction toward/away from targets.
        """
        if action_type == "MOVE":
            # Unpack movement vector
            dx, dy = action_params
            # Clamp current speed to velocity if the magnitude is too big
            self.current_speed = min(math.hypot(dx, dy), self.velocity)
            self.intent = "WANDER"             # Intent is wandering around
            self.intended_vector = (dx, dy)    # Raw intended dx/dy

        elif action_type == "REST":
            # No movement, minimal energy drain
            self.current_speed = 0.0
            self.intent = "REST"
            self.intended_vector = (0.0, 0.0)

        elif action_type == "EAT":
            # The parameter is a Food object to move toward
            food = action_params
            # Compute direction vector toward the food for visualization
            dx = food.x - self.x
            dy = food.y - self.y
            dist = math.hypot(dx, dy)
            if dist > 0:
                # Normalize and scale to 75% of max velocity for eating approach
                dx = (dx / dist) * self.velocity * 0.75
                dy = (dy / dist) * self.velocity * 0.75
            self.current_speed = self.velocity * 0.75
            self.intent = "GO_TO_FOOD"
            self.intended_vector = (dx, dy)

        elif action_type == "EAT_AT_CURRENT":
            # No movement; creature eats directly beneath itself
            self.current_speed = 0.0
            self.intent = "GO_TO_FOOD"         # Still considered "going to food" intent
            self.intended_vector = (0.0, 0.0)

        elif action_type == "ATTACK":
            # Attack parameter is a target Creature object
            target = action_params
            # Compute vector toward the target for visualization
            dx = target.x - self.x
            dy = target.y - self.y
            dist = math.hypot(dx, dy)
            if dist > 0:
                # Normalize and scale to max velocity to rush the target
                dx = (dx / dist) * self.velocity
                dy = (dy / dist) * self.velocity
            self.current_speed = self.velocity
            self.intent = "ATTACK"
            self.intended_vector = (dx, dy)

        elif action_type == "FLEE":
            # Flee parameter is a (dx, dy) vector away from the threat
            dx, dy = action_params
            # Clamp speed to velocity
            self.current_speed = min(math.hypot(dx, dy), self.velocity)
            self.intent = "RUN_AWAY"
            self.intended_vector = (dx, dy)

    def _decide_without_brain(self, vision: List[Tuple[str, Any, float, float]], on_food: bool = False) -> Tuple[str, Any]:
        """
        Legacy (hard-coded) decision logic for creatures without a neural network.

        Steps:
          1) Look for nearby creatures to attack or flee from
          2) Otherwise look for nearby food to eat
          3) Otherwise if on food, eat at current location
          4) Otherwise, randomly rest or wander

        Args:
            vision: List of (type_tag, object, distance, angle)
            on_food: True if creature is directly on a food source

        Returns:
            Tuple[action_type (str), action_params (Any)] as in decide().
        """
        # If nothing is sensed, go to resting/wandering logic
        if vision is None or len(vision) == 0:
            return self._decide_rest_or_wander()

        # --- 1) Check for nearby creatures (enemies) ---
        nearby_creatures = [
            (obj, dist, angle)
            for type_tag, obj, dist, angle in vision
            # Ignore same-family creatures (parent/child) to prevent immediate sibling cannibalism
            if type_tag == "creature" and obj.parent_id != self.parent_id
        ]
        # --- 2) Check for nearby food ---
        nearby_food = [
            (obj, dist, angle)
            for type_tag, obj, dist, angle in vision
            if type_tag == "food"
        ]

        # --- If there are creatures nearby, handle attack/flee logic ---
        if nearby_creatures:
            # Sort by distance so the closest creature is first
            nearby_creatures.sort(key=lambda x: x[1])
            closest_creature, distance, angle = nearby_creatures[0]

            # Compute dynamic attack range (sum of radii * factor)
            attack_range = (self.radius + closest_creature.radius) * self.ATTACK_RANGE_FACTOR

            # If in attack range, attempt to attack
            if distance <= attack_range and closest_creature.parent_id != self.parent_id:
                # Rush at full velocity to strike
                self.current_speed = self.velocity
                self.intent = "ATTACK"
                # Compute movement vector (using angle from sensor)
                dx = math.cos(angle) * self.current_speed
                dy = math.sin(angle) * self.current_speed
                self.intended_vector = (dx, dy)
                return ("ATTACK", closest_creature)

            # Otherwise, 10% chance to flee instead of closing in
            if random.random() < 0.1:
                flee_angle = angle + math.pi   # Directly opposite direction
                self.current_speed = self.velocity
                self.intent = "RUN_AWAY"
                dx = math.cos(flee_angle) * self.current_speed
                dy = math.sin(flee_angle) * self.current_speed
                self.intended_vector = (dx, dy)
                return ("FLEE", (dx, dy))

            # If not fleeing and not yet in range, move toward the creature for a future attack
            self.current_speed = self.velocity
            self.intent = "ATTACK"
            dx = math.cos(angle) * self.current_speed
            dy = math.sin(angle) * self.current_speed
            self.intended_vector = (dx, dy)
            return ("MOVE", (dx, dy))

        # --- If no creatures to fight, check for food ---
        if nearby_food:
            nearby_food.sort(key=lambda x: x[1])
            closest_food, distance, angle = nearby_food[0]
            eat_range = (self.radius + closest_food.radius) * self.EAT_RANGE_FACTOR

            # If within eating distance, attempt to eat
            if distance <= eat_range:
                self.current_speed = self.velocity * 0.75  # Approach at 75% speed for eating
                self.intent = "GO_TO_FOOD"
                dx = math.cos(angle) * self.current_speed
                dy = math.sin(angle) * self.current_speed
                self.intended_vector = (dx, dy)
                return ("EAT", closest_food)

            # Otherwise, move toward the food at 75% speed
            self.current_speed = self.velocity * 0.75
            self.intent = "GO_TO_FOOD"
            dx = math.cos(angle) * self.current_speed
            dy = math.sin(angle) * self.current_speed
            self.intended_vector = (dx, dy)
            return ("MOVE", (dx, dy))

        # --- If there's food at the current location, eat without moving ---
        if on_food:
            self.intent = "GO_TO_FOOD"
            self.intended_vector = (0.0, 0.0)
            return ("EAT_AT_CURRENT", None)

        # --- Otherwise, nothing interesting: rest or wander ---
        return self._decide_rest_or_wander()

    def _decide_rest_or_wander(self) -> Tuple[str, Any]:
        """
        Helper function: randomly choose between resting (no movement) or wandering (random direction).
        Resting conserves a bit of energy; wandering expends some energy but may discover food/targets.
        """
        # Wander at 50% of maximum speed when moving
        self.current_speed = self.velocity * 0.5

        if random.random() < 0.5:
            # Choose a random direction in [0, 2π)
            angle = random.random() * 2 * math.pi
            dx = math.cos(angle) * self.current_speed
            dy = math.sin(angle) * self.current_speed
            self.intent = "WANDER"
            self.intended_vector = (dx, dy)
            return ("MOVE", (dx, dy))
        else:
            # Rest: no movement
            self.intent = "REST"
            self.intended_vector = (0.0, 0.0)
            return ("REST", None)

    def apply_action(self, action: Tuple[str, Any], world: 'World') -> None:
        """
        Execute the chosen action and adjust energy accordingly. Then provide feedback (reward) to the brain.

        Action types:
          - ("MOVE", (dx, dy)): Move and deduct energy equal to distance traveled.
          - ("FLEE", (dx, dy)): Move (similar to MOVE) and deduct energy equal to distance traveled.
          - ("REST", None): Deduct a small flat energy cost (0.1).
          - ("EAT", Food): Attempt to eat the specified food object if in range; energy += 1 if successful.
          - ("EAT_AT_CURRENT", None): Eat any food overlapping current position; energy += 1 if successful.
          - ("ATTACK", Creature): Attempt to attack the specified creature; energy -= attack_cost, damage dealt.

        After the action, compute the energy change (new_energy - initial_energy) as the "reward"
        and call self.brain.apply_reward(reward) if a brain is installed.
        """
        act_type, param = action
        self.last_action = f"{act_type}"         # Store action label for logging or debugging

        initial_energy = self.energy             # Record pre-action energy
        self.movement_vector = self.intended_vector  # For visualization: record how we moved

        # --- Handle MOVE or FLEE (movement) ---
        if (act_type == "MOVE" or act_type == "FLEE") and param is not None:
            dx, dy = param
            requested_dist = math.hypot(dx, dy)  # Euclidean distance of intended move

            # If requested movement exceeds current_speed, scale it down
            if requested_dist > self.current_speed:
                scale = self.current_speed / requested_dist
                dx *= scale
                dy *= scale
                actual_dist = self.current_speed
            else:
                actual_dist = requested_dist

            # Compute new coordinates before clamping to world bounds
            new_x = self.x + dx
            new_y = self.y + dy

            # Compute boundaries accounting for creature radius
            min_x = max(0.0, self.radius) if self.radius > 0.1 else 0.0
            min_y = max(0.0, self.radius) if self.radius > 0.1 else 0.0
            max_x = min(world.width, world.width - self.radius) if self.radius > 0.1 else world.width
            max_y = min(world.height, world.height - self.radius) if self.radius > 0.1 else world.height

            # Clamp position so the creature remains entirely inside world boundaries
            self.x = min(max(new_x, min_x), max_x)
            self.y = min(max(new_y, min_y), max_y)

            # Deduct energy equal to distance actually traveled
            self.energy -= actual_dist

            # Update last_action string for clarity
            if act_type == "FLEE":
                self.last_action = "FLEE"
            else:
                self.last_action = "MOVE"

        # --- Handle ATTACK action ---
        elif act_type == "ATTACK" and param is not None:
            target = param  # Target creature object

            # Compute distance to target
            dx = target.x - self.x
            dy = target.y - self.y
            distance = math.sqrt(dx * dx + dy * dy)

            # Compute attack range same as in decide()
            attack_range = (self.radius + target.radius) * self.ATTACK_RANGE_FACTOR

            # Debug prints for tracing attacks (can be removed in production)
            print(f"ATTACK: distance={distance}, attack_range={attack_range}")
            print(f"ATTACK: target_initial_energy={target.energy}, attack_damage={self.attack_damage}")

            # If the target is out of range, the attack misses
            if distance > attack_range:
                self.energy -= self.attack_cost  # Pay energy cost for attempted attack
                self.last_action = "ATTACK_MISS"
            else:
                # Otherwise, deal damage
                target_initial_energy = target.energy
                damage_dealt = min(target_initial_energy, self.attack_damage)
                target.energy -= damage_dealt

                # Debug print: how much damage was dealt
                print(f"ATTACK: damage_dealt={damage_dealt}, target_energy_after={target.energy}")

                # Deduct attack cost from attacker’s energy
                self.energy -= self.attack_cost

                # Update action labels for logging
                self.last_action = f"ATTACK→{target.id if hasattr(target, 'id') else id(target)}"
                target.last_action = "HIT_BY_ATTACK"

                # If target’s energy falls to 0 or below, create a corpse Food
                if target.energy <= 0:
                    from .food import Food
                    # Corpse energy is proportional to target’s size * 8.0
                    energy = target.size * 8.0
                    corpse_food = Food(
                        x=target.x,
                        y=target.y,
                        remaining_duration=5,  # Food lasts 5 steps
                        energy=energy
                    )
                    # Add corpse to the world’s food list
                    world.foods.append(corpse_food)
                    world.foods_created_this_step.add(id(corpse_food))

                    # Remove the dead creature from the world if present
                    try:
                        world.creatures.remove(target)
                    except ValueError:
                        # If it’s already removed or not exactly the same object, ignore
                        pass

        # --- Handle EAT action when pursuing a food object ---
        elif act_type == "EAT" and param is not None:
            target_food = param  # Food object to attempt to eat

            # Compute distance to that food
            dx = target_food.x - self.x
            dy = target_food.y - self.y
            distance = math.sqrt(dx * dx + dy * dy)

            # Eat range based on radii
            eat_range = (self.radius + target_food.radius) * self.EAT_RANGE_FACTOR

            # If out of range, the eating attempt misses
            if distance > eat_range:
                eat_miss_cost = 1.0
                self.energy -= eat_miss_cost
                self.last_action = "EAT_MISS"
            else:
                # If in range, subtract one unit of energy from the food and add one to the creature
                target_food.energy -= 1
                self.energy += 1
                self.last_action = f"EAT→{target_food.id if hasattr(target_food, 'id') else id(target_food)}"

        # --- Handle EAT_AT_CURRENT action when creature is exactly on a food source ---
        elif act_type == "EAT_AT_CURRENT":
            target_food = None
            # Find any food overlapping the creature’s current position
            for food in world.foods:
                if food.is_overlapping(self.x, self.y, self.radius):
                    target_food = food
                    break

            # If no overlapping food is found, small energy penalty
            if target_food is None:
                eat_miss_cost = 0.2
                self.energy -= eat_miss_cost
                self.last_action = "EAT_MISS"
            else:
                # Otherwise, subtract one energy unit from that food and add one to creature
                target_food.energy -= 1
                self.energy += 1
                self.last_action = "EAT_AT_CURRENT"

        # --- Handle REST action (no parameters) ---
        else:  # "REST"
            self.energy -= 0.1     # Flat energy penalty for resting
            self.last_action = "REST"

        # --- After action, compute energy change as reward signal ---
        energy_change = self.energy - initial_energy
        if energy_change > 0:
            # Reset no-reward counter if we gained energy
            self.using_brain = True
            self.steps_without_reward = 0
        else:
            # Increment counter if we did not gain energy
            self.steps_without_reward += 1

        # If a brain exists, pass the reward (positive or negative) to it
        if self.brain is not None:
            self.brain.apply_reward(energy_change)

        # If creature’s energy reaches threshold and its brain hasn’t been saved yet, save it
        if (
                self.brain is not None
                and not self.brain_saved
                and self.energy >= Creature.BRAIN_SAVE_THRESHOLD * self.max_energy
        ):
            self._save_brain()

    def _save_brain(self) -> None:
        """
        Serialize (pickle) the creature’s neural network to disk if it hasn’t been saved before.
        Filename format: brain_<creature_id>_<rounded_energy>.pkl
        """
        if self.brain is None:
            return

        # Ensure save directory exists
        os.makedirs(Creature.SAVE_DIR, exist_ok=True)
        energy_int = int(round(self.energy))
        filename = f"brain_{self.id}_{energy_int}.pkl"
        filepath = os.path.join(Creature.SAVE_DIR, filename)

        try:
            # Write the brain object to disk
            with open(filepath, "wb") as f:
                pickle.dump(self.brain, f)
            self.brain_saved = True
        except Exception:
            # If saving fails (e.g., disk error), silently skip
            pass

    def get_color(self) -> str:
        """
        Return a color string based on whether the creature used its brain on the last decision.
        - "purple" if using the neural network.
        - "green" otherwise (legacy or fallback logic).
        """
        return "purple" if self.using_brain else "green"

    def should_split(self) -> bool:
        """
        Check if this creature’s energy has reached its maximum threshold.
        Returns True if energy ≥ max_energy, otherwise False.
        """
        return self.energy >= self.max_energy

    def split(self, world: 'World') -> List['Creature']:
        """
        When a creature has enough energy to divide, split it into four:
          1) A clone with the same brain.
          2) Three mutated offspring with variations of the parent brain.

        Each new creature (including the clone) receives 1/4 of the parent’s energy,
        and the parent’s energy is set to 0. All children are placed near the parent’s position.

        Args:
            world (World): Reference to the world, so new creatures can be added.

        Returns:
            List of newly created child Creature instances.
        """
        # Each child (and the clone) gets 1/4 of parent’s current energy
        energy_per_creature = self.energy / 4.0

        # Parent expends all energy in the process of splitting
        self.energy = 0

        children: List['Creature'] = []

        # --- Create a clone with identical brain (deep copy via clone()) ---
        clone = Creature(
            x=self.x + random.uniform(-1.0, 1.0),  # Slight random offset so children don’t overlap exactly
            y=self.y + random.uniform(-1.0, 1.0),
            size=self.size,
            energy=energy_per_creature,
            velocity=self.velocity,
            eat_bonus=self.eat_bonus,
            radius=self.radius,
            brain=self.brain.clone() if self.brain else None,  # Clone the neural network weights
            parent_id=self.id                                  # Set parent_id to this creature’s ID
        )
        world.add_creature(clone)
        children.append(clone)

        # --- Create three mutated children with incremental mutation rates ---
        for i in range(3):
            mutated_brain = None
            if self.brain:
                # Mutation rate increases for each of the three children: 0.1, 0.2, 0.3
                mutation_rate = 0.1 * (i + 1)
                mutation_scale = 0.2
                mutated_brain = self.brain.mutate(mutation_rate, mutation_scale)

            mutated_child = Creature(
                x=self.x + random.uniform(-1.0, 1.0),
                y=self.y + random.uniform(-1.0, 1.0),
                size=self.size,
                energy=energy_per_creature,
                velocity=self.velocity,
                eat_bonus=self.eat_bonus,
                radius=self.radius,
                brain=mutated_brain,
                parent_id=self.id
            )
            world.add_creature(mutated_child)
            children.append(mutated_child)

        return children

    def __repr__(self) -> str:
        """
        String representation for debugging: shows position, size, and energy status.
        Format: <Creature x=XX.XX y=YY.YY size=ZZ.ZZ energy=EE.EE/MM.MM>
        """
        return f"<Creature x={self.x:.2f} y={self.y:.2f} size={self.size:.2f} energy={self.energy:.2f}/{self.max_energy:.2f}>"

# Import World at the end to avoid circular imports (Creature → World → Creature etc.)
from .world import World
