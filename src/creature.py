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
            brain: Optional['NeuralNetwork'] = None,
            create_brain: bool = False,
            parent_id: int = None,
            generation: int = 0
        ):
        """
        Initialize a Creature.

        Args:
            x (float): Initial x-coordinate in continuous space.
            y (float): Initial y-coordinate in continuous space.
            size (float): Determines strength, attack cost, and maximum energy.
                           Larger size → more energy capacity and damage, but slower.
            energy (float): Starting energy. Energy ≤ 0 means death; ≥ max_energy means split.
            brain (NeuralNetwork, optional): Pre-existing neural network to control decisions.
            create_brain (bool): If True and brain is None, instantiates a new NeuralNetwork.
            parent_id (int, optional): ID of the parent creature (used to avoid attacking close kin).
            generation (int): Which generation this creature belongs to (0 for the original seed).
        """
        # --- Position and physical attributes ---
        self.x = float(x)                   # Current x-coordinate
        self.y = float(y)                   # Current y-coordinate
        self.size = size                    # Creature "size" scalar
        self.energy = energy                # Current energy level
        self.id = id(self)                  # Unique identifier (Python's built-in id)
        self.parent_id = parent_id or 0     # Parent ID (0 if no parent)

        # --- Generation counter: we will not use the brain until generation ≥ 10 ---
        self.generation = generation        # Generation number (0 for starting creatures)

        # --- Tracking whether this creature's last decision used its brain ---
        self.using_brain = False            # True if last call to decide() used the neural network
        self.brain_saved = False            # Prevent saving the same brain more than once

        # --- Counter for steps without positive reward ---
        self.steps_without_reward = 0       # If ≥10, forces a random/legacy decision to encourage exploration

        # --- Cumulative statistics for NN score calculation ---
        self.nn_reward_count = 0            # Count of times energy_change > 0 when using brain
        self.nn_fallback_count = 0          # Count of times forced to use legacy logic
        self.nn_total_steps = 0             # Count of total calls to decide()

        # --- Combat and splitting parameters derived from "size" ---
        self.attack_damage = size * 5.0     # Damage dealt on successful attack
        self.attack_cost = size * 1.0       # Energy cost to attempt an attack
        self.max_energy = size * 100.0      # Energy threshold for splitting

        # --- Last action label, used for tracking/logging ---
        self.last_action = "NONE"

        # --- Velocity logic: larger creatures move more slowly (1/size) if not specified ---
        self.velocity = 1.0 / size          # Base max speed

        # --- Physical radius for collisions: size * RADIUS_FACTOR unless overridden ---
        self.radius = size * self.RADIUS_FACTOR

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

        If generation < 10, ALWAYS use legacy logic (no brain) but count it as a fallback.
        Once generation >= 10, revert to the old logic (including step‐without‐reward fallback).

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
        # --- Increment total step counter for NN score calculation ---
        self.nn_total_steps += 1

        # ----------------------------------------------------
        # 1) If generation < 10 → force legacy logic, no brain.
        #    (Count as a fallback if a brain exists, so we know these
        #     decisions are “observations” for generations 0–9.)
        # ----------------------------------------------------
        if self.generation < 10:
            if self.brain is not None:
                # We “fell back” to legacy purely for generation < 10
                self.nn_fallback_count += 1

            action_type, action_params = self._decide_without_brain(vision, on_food)
            self.using_brain = False
            # Set intent & vector for visualization
            self._set_intent_and_vector(action_type, action_params, vision)
            return action_type, action_params

        # ----------------------------------------------------
        # 2) Now that generation >= 10, resume the existing logic:
        #    (a) If no brain installed → legacy
        #    (b) If too many steps_without_reward → legacy
        #    (c) Otherwise → use brain
        # ----------------------------------------------------

        # If the creature hasn't gained energy (no reward) for 10 steps,
        # force a random legacy action (no brain) to encourage exploration
        if self.steps_without_reward >= 10:
            if self.brain is not None:
                self.nn_fallback_count += 1

            action_type, action_params = self._decide_without_brain(vision, on_food)
            self.using_brain = False
            self._set_intent_and_vector(action_type, action_params, vision)
            return action_type, action_params

        # If no brain is installed, fallback to legacy decision logic
        if self.brain is None:
            self.using_brain = False
            return self._decide_without_brain(vision, on_food)

        # Otherwise, use the neural network to decide
        self.using_brain = True

        # Package sensory data into a dictionary for the neural network
        sensory_inputs = {
            'vision': vision,
            'on_food': on_food,
            'creature_state': {
                'energy': self.energy,
                'size': self.size,
                'velocity': self.velocity,
                'max_energy': self.max_energy,
                'position': (self.x, self.y),
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
            dx, dy = action_params
            self.current_speed = min(math.hypot(dx, dy), self.velocity)
            self.intent = "WANDER"
            self.intended_vector = (dx, dy)

        elif action_type == "REST":
            self.current_speed = 0.0
            self.intent = "REST"
            self.intended_vector = (0.0, 0.0)

        elif action_type == "EAT":
            food = action_params
            dx = food.x - self.x
            dy = food.y - self.y
            dist = math.hypot(dx, dy)
            if dist > 0:
                dx = (dx / dist) * self.velocity * 0.75
                dy = (dy / dist) * self.velocity * 0.75
            self.current_speed = self.velocity * 0.75
            self.intent = "GO_TO_FOOD"
            self.intended_vector = (dx, dy)

        elif action_type == "EAT_AT_CURRENT":
            self.current_speed = 0.0
            self.intent = "GO_TO_FOOD"
            self.intended_vector = (0.0, 0.0)

        elif action_type == "ATTACK":
            target = action_params
            dx = target.x - self.x
            dy = target.y - self.y
            dist = math.hypot(dx, dy)
            if dist > 0:
                dx = (dx / dist) * self.velocity
                dy = (dy / dist) * self.velocity
            self.current_speed = self.velocity
            self.intent = "ATTACK"
            self.intended_vector = (dx, dy)

        elif action_type == "FLEE":
            dx, dy = action_params
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
        if vision is None or len(vision) == 0:
            return self._decide_rest_or_wander()

        # --- 1) Check for nearby creatures (enemies) ---
        nearby_creatures = [
            (obj, dist, angle)
            for type_tag, obj, dist, angle in vision
            if (
                type_tag == "creature"
                and obj.id != self.parent_id
                and obj.parent_id != self.id
                and not (
                    self.parent_id != 0 and obj.parent_id == self.parent_id
                )
            )
        ]
        # --- 2) Check for nearby food ---
        nearby_food = [
            (obj, dist, angle)
            for type_tag, obj, dist, angle in vision
            if type_tag == "food"
        ]

        if nearby_creatures:
            nearby_creatures.sort(key=lambda x: x[1])
            closest_creature, distance, angle = nearby_creatures[0]
            attack_range = (self.radius + closest_creature.radius) * self.ATTACK_RANGE_FACTOR

            if distance <= attack_range:
                self.current_speed = self.velocity
                self.intent = "ATTACK"
                dx = math.cos(angle) * self.current_speed
                dy = math.sin(angle) * self.current_speed
                self.intended_vector = (dx, dy)
                return ("ATTACK", closest_creature)

            if random.random() < 0.4:
                flee_angle = angle + math.pi
                self.current_speed = self.velocity
                self.intent = "RUN_AWAY"
                dx = math.cos(flee_angle) * self.current_speed
                dy = math.sin(flee_angle) * self.current_speed
                self.intended_vector = (dx, dy)
                return ("FLEE", (dx, dy))

            self.current_speed = self.velocity
            self.intent = "ATTACK"
            dx = math.cos(angle) * self.current_speed
            dy = math.sin(angle) * self.current_speed
            self.intended_vector = (dx, dy)
            return ("MOVE", (dx, dy))

        if nearby_food:
            nearby_food.sort(key=lambda x: x[1])
            closest_food, distance, angle = nearby_food[0]
            eat_range = (self.radius + closest_food.radius) * self.EAT_RANGE_FACTOR

            if distance <= eat_range:
                self.current_speed = self.velocity * 0.75
                self.intent = "GO_TO_FOOD"
                dx = math.cos(angle) * self.current_speed
                dy = math.sin(angle) * self.current_speed
                self.intended_vector = (dx, dy)
                return ("EAT", closest_food)

            self.current_speed = self.velocity * 0.75
            self.intent = "GO_TO_FOOD"
            dx = math.cos(angle) * self.current_speed
            dy = math.sin(angle) * self.current_speed
            self.intended_vector = (dx, dy)
            return ("MOVE", (dx, dy))

        if on_food:
            self.intent = "GO_TO_FOOD"
            self.intended_vector = (0.0, 0.0)
            return ("EAT_AT_CURRENT", None)

        return self._decide_rest_or_wander()

    def _decide_rest_or_wander(self) -> Tuple[str, Any]:
        """
        Helper function: randomly choose between resting (no movement) or wandering (random direction).
        Resting conserves a bit of energy; wandering expends some energy but may discover food/targets.
        """
        self.current_speed = self.velocity * 0.5

        if random.random() < 0.5:
            angle = random.random() * 2 * math.pi
            dx = math.cos(angle) * self.current_speed
            dy = math.sin(angle) * self.current_speed
            self.intent = "WANDER"
            self.intended_vector = (dx, dy)
            return ("MOVE", (dx, dy))
        else:
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
          - ("EAT", Food): Attempt to eat the specified food object if in range; energy += up to 5 if successful.
          - ("EAT_AT_CURRENT", None): Eat any food overlapping current position; energy += up to 5 if successful.
          - ("ATTACK", Creature): Attempt to attack the specified creature; energy -= attack_cost, damage dealt.

        After the action, compute the energy change (new_energy - initial_energy) as the “reward” and
        call `self.brain.apply_reward(reward, terminal)` so that multi‐step (episode‐based) REINFORCE works.
        """
        act_type, param = action
        self.last_action = f"{act_type}"

        initial_energy = self.energy
        self.movement_vector = self.intended_vector

        # --- Handle MOVE or FLEE (movement) ---
        if (act_type == "MOVE" or act_type == "FLEE") and param is not None:
            dx, dy = param
            requested_dist = math.hypot(dx, dy)

            if requested_dist > self.current_speed:
                scale = self.current_speed / requested_dist
                dx *= scale
                dy *= scale
                actual_dist = self.current_speed
            else:
                actual_dist = requested_dist

            new_x = self.x + dx
            new_y = self.y + dy

            min_x = max(0.0, self.radius) if self.radius > 0.1 else 0.0
            min_y = max(0.0, self.radius) if self.radius > 0.1 else 0.0
            max_x = min(world.width, world.width - self.radius) if self.radius > 0.1 else world.width
            max_y = min(world.height, world.height - self.radius) if self.radius > 0.1 else world.height

            self.x = min(max(new_x, min_x), max_x)
            self.y = min(max(new_y, min_y), max_y)
            self.energy -= actual_dist

            if act_type == "FLEE":
                self.last_action = "FLEE"
            else:
                self.last_action = "MOVE"

        # --- Handle ATTACK action ---
        elif act_type == "ATTACK" and param is not None:
            target = param
            dx = target.x - self.x
            dy = target.y - self.y
            distance = math.sqrt(dx * dx + dy * dy)
            attack_range = (self.radius + target.radius) * self.ATTACK_RANGE_FACTOR


            if distance > attack_range:
                self.energy -= self.attack_cost
                self.last_action = "ATTACK_MISS"
            else:
                target_initial_energy = target.energy
                damage_dealt = min(target_initial_energy, self.attack_damage)
                target.energy -= damage_dealt


                self.energy -= self.attack_cost
                self.last_action = f"ATTACK→{target.id if hasattr(target, 'id') else id(target)}"
                target.last_action = "HIT_BY_ATTACK"

                if target.energy <= 0:
                    from .food import Food
                    energy = target.size * 8.0
                    corpse_food = Food(
                        x=target.x,
                        y=target.y,
                        remaining_duration=5,
                        energy=energy
                    )
                    world.foods.append(corpse_food)
                    world.foods_created_this_step.add(id(corpse_food))
                    try:
                        world.creatures.remove(target)
                    except ValueError:
                        pass

        # --- Handle EAT action when pursuing a food object ---
        elif act_type == "EAT" and param is not None:
            target_food = param
            dx = target_food.x - self.x
            dy = target_food.y - self.y
            distance = math.sqrt(dx * dx + dy * dy)
            eat_range = (self.radius + target_food.radius) * self.EAT_RANGE_FACTOR

            if distance > eat_range:
                eat_miss_cost = 1.0
                self.energy -= eat_miss_cost
                self.last_action = "EAT_MISS"
            else:
                if target_food.energy <= 0:
                    amt = 1.0
                else:
                    amt = min(1.0, target_food.energy)
                target_food.energy -= amt
                self.energy += amt
                if target_food.energy <= 0:
                    try:
                        world.foods.remove(target_food)
                    except ValueError:
                        pass
                self.last_action = f"EAT→{target_food.id if hasattr(target_food, 'id') else id(target_food)}"

        # --- Handle EAT_AT_CURRENT action when creature is exactly on a food source ---
        elif act_type == "EAT_AT_CURRENT":
            target_food = None
            for food in world.foods:
                if food.is_overlapping(self.x, self.y, self.radius):
                    target_food = food
                    break

            if target_food is None:
                eat_miss_cost = 0.2
                self.energy -= eat_miss_cost
                self.last_action = "EAT_MISS"
            else:
                if target_food.energy <= 0:
                    amt = 1.0
                else:
                    amt = min(1.0, target_food.energy)
                target_food.energy -= amt
                self.energy += amt
                if target_food.energy <= 0:
                    try:
                        world.foods.remove(target_food)
                    except ValueError:
                        pass
                self.last_action = "EAT_AT_CURRENT"

        # --- Handle REST action (no parameters) ---
        else:  # "REST"
            self.energy -= 0.1
            self.last_action = "REST"

        # --- After action, compute energy change as reward signal ---
        energy_change = self.energy - initial_energy

        # --- Determine terminal condition for reward ---
        terminal = False
        if act_type in ("EAT", "EAT_AT_CURRENT") and energy_change > 0:
            # Successfully ate food → end of short episode
            terminal = True
        elif self.energy <= 0:
            # Creature died → end of episode
            terminal = True

        # --- If the creature gained energy and used brain, count it for score ---
        if energy_change > 0 and self.using_brain:
            self.nn_reward_count += energy_change

        # --- Update steps_without_reward counter ---
        if energy_change > 0:
            self.steps_without_reward = 0
        else:
            self.steps_without_reward += 1

        # --- Pass the reward into the neural network as episodic (REINFORCE) ---
        if self.brain is not None:
            # If terminal, pass actual energy_change; otherwise pass zero
            self.brain.apply_reward(energy_change if terminal else 0.0, terminal)

        # --- If creature’s energy reaches threshold and its brain hasn’t been saved yet, save it ---
        if (
                self.brain is not None
                and self.get_nn_score() >= 80
                and not self.brain_saved
        ):
            self._save_brain()

    def _save_brain(self) -> None:
        """
        Serialize (pickle) the creature’s neural network to disk if it hasn’t been saved before.
        Filename format: brain_<creature_id>_<rounded_energy>.pkl
        """
        if self.brain is None:
            return

        os.makedirs(Creature.SAVE_DIR, exist_ok=True)
        energy_int = int(round(self.energy))
        filename = f"brain_{self.id}_{energy_int}.pkl"
        filepath = os.path.join(Creature.SAVE_DIR, filename)

        try:
            with open(filepath, "wb") as f:
                pickle.dump(self.brain, f)
            self.brain_saved = True
        except Exception:
            pass

    def get_nn_score(self) -> float:
        """
        Compute a normalized NN score between 0 and 100 for this creature:
            score = ((num_positive_rewards - num_fallbacks) / total_steps) * 100

        If the result is negative, clamp to 0. If it exceeds 100, clamp to 100.
        """
        if self.nn_total_steps == 0:
            return 0.0

        raw = (self.nn_reward_count - self.nn_fallback_count) / self.nn_total_steps * 100.0
        if raw < 0.0:
            return 0.0
        if raw > 100.0:
            return 100.0
        return raw

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
        and the parent’s energy is set to 0. All children are placed near the parent’s position,
        and their generation = parent.generation + 1.

        Args:
            world (World): Reference to the world, so new creatures can be added.

        Returns:
            List of newly created child Creature instances.
        """
        energy_per_creature = self.energy / 4.0
        self.energy = 0
        children: List['Creature'] = []

        # --- Create a clone with identical brain (deep copy via clone()) ---
        clone = Creature(
            x=self.x + random.uniform(-1.0, 1.0),
            y=self.y + random.uniform(-1.0, 1.0),
            size=self.size,
            energy=energy_per_creature,
            brain=self.brain.clone() if self.brain else None,
            parent_id=self.id,
            generation=self.generation + 1
        )
        world.add_creature(clone)
        children.append(clone)

        # --- Create three mutated children with incremental mutation rates and slight size variation ---
        for i in range(3):
            mutated_brain = None
            if self.brain:
                mutation_rate = 0.1 * (i + 1)
                mutation_scale = 0.2
                mutated_brain = self.brain.mutate(mutation_rate, mutation_scale)

            size_mutation_factor = 1.0 + random.uniform(-0.1, 0.1)
            mutated_size = max(self.size * size_mutation_factor, 0.1)

            mutated_child = Creature(
                x=self.x + random.uniform(-1.0, 1.0),
                y=self.y + random.uniform(-1.0, 1.0),
                size=mutated_size,
                energy=energy_per_creature,
                brain=mutated_brain,
                parent_id=self.id,
                generation=self.generation + 1
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
