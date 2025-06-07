from typing import Any, TYPE_CHECKING, Dict, Tuple, Optional, List, Union
import random               # For random number generation
import math                 # For trigonometric functions and distance calculations
import os                   # For filesystem operations when saving brains
import pickle               # For serializing (saving) neural network objects to disk

if TYPE_CHECKING:
    from .world import World
    from .sensors import Sensor, ProximitySensor
    from .food import Food
    from .neural_network import NeuralNetwork

class Creature:
    RADIUS_FACTOR = 0.2
    SENSE_RANGE = 3.0
    ATTACK_RANGE_FACTOR = 3.0
    EAT_RANGE_FACTOR = 3.0
    MIN_STEPS = 500
    INPUT_SIZE = 12
    HIDDEN_SIZES = [8, 8]
    OUTPUT_SIZE = 8
    BRAIN_SAVE_THRESHOLD = 0.8
    SAVE_DIR = "saved_brains"

    ACTION_TYPES = ["MOVE", "REST", "EAT", "EAT_AT_CURRENT", "ATTACK", "FLEE"]

    def __init__(self, x: float, y: float, size: float, energy: float,
                 brain: Optional['NeuralNetwork'] = None, create_brain: bool = False,
                 parent_id: int = None, generation: int = 0):
        self.x = float(x)
        self.y = float(y)
        self.size = size
        self.energy = energy
        self.id = id(self)
        self.parent_id = parent_id or 0
        self.generation = generation

        self.using_brain = False
        self.brain_saved = False
        self.steps_without_reward = 0
        self.nn_reward_count = 0
        self.nn_fallback_count = 0
        self.nn_total_steps = 0

        self.attack_damage = size * 5.0
        self.attack_cost = size * 1.0
        self.max_energy = size * 100.0
        self.last_action = "NONE"

        self.velocity = 1.0 / size
        self.radius = size * self.RADIUS_FACTOR
        self.current_speed = self.velocity
        self.intended_vector = (0.0, 0.0)
        self.movement_vector = (0.0, 0.0)
        self.intent = "REST"

        if brain is not None:
            self.brain = brain
        elif create_brain:
            from .neural_network import NeuralNetwork
            self.brain = NeuralNetwork(self.INPUT_SIZE, self.HIDDEN_SIZES, self.OUTPUT_SIZE)
        else:
            self.brain = None

        from .sensors import ProximitySensor
        self.sensors: Tuple['Sensor', ...] = (ProximitySensor(sense_range=self.SENSE_RANGE),)

    def _action_to_index(self, action_type: str) -> Optional[int]:
        if action_type in self.ACTION_TYPES:
            return self.ACTION_TYPES.index(action_type)
        return None

    def decide(self, vision: List[Tuple[str, Any, float, float]], on_food: bool = False, step: int = 0) -> Tuple[str, Any]:
        self.nn_total_steps += 1

        if step < self.MIN_STEPS:
            if self.brain is not None:
                self.nn_fallback_count += 1

            action_type, action_params = self._decide_without_brain(vision, on_food)
            self.using_brain = False
            self._set_intent_and_vector(action_type, action_params, vision)

            if self.brain is not None:
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
                input_vec = self.brain._process_sensory_inputs(sensory_inputs)
                idx = self._action_to_index(action_type)

                # -------- NEW: build targets for the 2 continuous heads --------
                cont_targets = None
                if action_type in ("MOVE", "FLEE"):
                    dx, dy = action_params
                    angle_01 = (math.atan2(dy, dx) % (2 * math.pi)) / (2 * math.pi)
                    speed_01 = min(math.hypot(dx, dy) / self.velocity, 1.0)
                    cont_targets = (angle_01, speed_01)
                # ----------------------------------------------------------------

                if idx is not None:
                    self.brain.train_supervised_full(input_vec, idx, cont_targets)

            return action_type, action_params

        if self.brain is None:
            self.using_brain = False
            return self._decide_without_brain(vision, on_food)

        self.using_brain = True
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
        action_type, action_params = self.brain.decide(sensory_inputs)
        self._set_intent_and_vector(action_type, action_params, vision)
        return action_type, action_params

    def _set_intent_and_vector(self, action_type: str, action_params: Any,
                               vision: List[Tuple[str, Any, float, float]]) -> None:
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

    def _decide_without_brain(self, vision: List[Tuple[str, Any, float, float]],
                              on_food: bool = False) -> Tuple[str, Any]:
        if not vision:
            return self._decide_rest_or_wander()

        nearby_creatures = []
        for type_tag, obj, dist, angle in vision:
            if type_tag != "creature":
                continue
            if obj.id == self.parent_id or obj.parent_id == self.id:
                continue
            if self.parent_id != 0 and obj.parent_id == self.parent_id:
                continue
            if 0.67 < obj.size / self.size < 1.5:
                continue
            nearby_creatures.append((obj, dist, angle))

        nearby_food = [(obj, dist, angle) for type_tag, obj, dist, angle in vision if type_tag == "food"]

        if nearby_creatures:
            nearby_creatures.sort(key=lambda x: x[1])
            closest_creature, distance, angle = nearby_creatures[0]
            attack_range = (self.radius + closest_creature.radius) * self.ATTACK_RANGE_FACTOR

            if closest_creature.size >= self.size * 1.5:
                flee_angle = angle + math.pi
                self.current_speed = self.velocity
                self.intent = "RUN_AWAY"
                dx = math.cos(flee_angle) * self.current_speed
                dy = math.sin(flee_angle) * self.current_speed
                self.intended_vector = (dx, dy)
                return ("FLEE", (dx, dy))

            if self.size >= closest_creature.size * 1.5 and distance <= attack_range:
                self.current_speed = self.velocity
                self.intent = "ATTACK"
                dx = math.cos(angle) * self.current_speed
                dy = math.sin(angle) * self.current_speed
                self.intended_vector = (dx, dy)
                return ("ATTACK", closest_creature)

            self.current_speed = self.velocity
            self.intent = "MOVE"
            dx = math.cos(angle) * self.current_speed
            dy = math.sin(angle) * self.current_speed
            self.intended_vector = (dx, dy)
            return ("MOVE", (dx, dy))

        if nearby_food:
            nearby_food.sort(key=lambda x: x[1])
            closest_food, distance, angle = nearby_food[0]
            eat_range = (self.radius + closest_food.radius) * self.EAT_RANGE_FACTOR

            self.current_speed = self.velocity * 0.75
            self.intent = "GO_TO_FOOD"
            dx = math.cos(angle) * self.current_speed
            dy = math.sin(angle) * self.current_speed
            self.intended_vector = (dx, dy)

            if distance <= eat_range:
                return ("EAT", closest_food)
            return ("MOVE", (dx, dy))

        if on_food:
            self.intent = "GO_TO_FOOD"
            self.intended_vector = (0.0, 0.0)
            return ("EAT_AT_CURRENT", None)

        return self._decide_rest_or_wander()

    def _decide_rest_or_wander(self) -> Tuple[str, Any]:
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
        act_type, param = action
        self.last_action = f"{act_type}"
        initial_energy = self.energy
        self.movement_vector = self.intended_vector

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

        elif act_type == "ATTACK" and param is not None:
            target = param
            dx = target.x - self.x
            dy = target.y - self.y
            distance = math.hypot(dx, dy)
            attack_range = (self.radius + target.radius) * self.ATTACK_RANGE_FACTOR

            if distance > attack_range:
                self.energy -= self.attack_cost
                self.last_action = "ATTACK_MISS"
            else:
                from .food import Food
                damage_dealt = min(target.energy, self.attack_damage)
                target.energy -= damage_dealt
                self.energy -= self.attack_cost
                self.last_action = f"ATTACK→{target.id}"
                target.last_action = "HIT_BY_ATTACK"

                if target.energy <= 0:
                    energy = target.size * 8.0
                    corpse_food = Food(x=target.x, y=target.y, remaining_duration=5, energy=energy)
                    world.foods.append(corpse_food)
                    world.foods_created_this_step.add(id(corpse_food))
                    try:
                        world.creatures.remove(target)
                    except ValueError:
                        pass

        elif act_type == "EAT" and param is not None:
            target_food = param
            dx = target_food.x - self.x
            dy = target_food.y - self.y
            distance = math.hypot(dx, dy)
            eat_range = (self.radius + target_food.radius) * self.EAT_RANGE_FACTOR

            if distance > eat_range:
                self.energy -= 1.0
                self.last_action = "EAT_MISS"
            else:
                amt = min(1.0, target_food.energy) if target_food.energy > 0 else 1.0
                target_food.energy -= amt
                self.energy += amt
                if target_food.energy <= 0:
                    try:
                        world.foods.remove(target_food)
                    except ValueError:
                        pass
                self.last_action = f"EAT→{target_food.id if hasattr(target_food, 'id') else id(target_food)}"

        elif act_type == "EAT_AT_CURRENT":
            target_food = None
            for food in world.foods:
                if food.is_overlapping(self.x, self.y, self.radius):
                    target_food = food
                    break

            if target_food is None:
                self.energy -= 0.2
                self.last_action = "EAT_MISS"
            else:
                amt = min(1.0, target_food.energy) if target_food.energy > 0 else 1.0
                target_food.energy -= amt
                self.energy += amt
                if target_food.energy <= 0:
                    try:
                        world.foods.remove(target_food)
                    except ValueError:
                        pass
                self.last_action = "EAT_AT_CURRENT"

        else:  # REST
            self.energy -= 0.1
            self.last_action = "REST"

        energy_change = self.energy - initial_energy
        terminal = (act_type in ("EAT", "EAT_AT_CURRENT") and energy_change > 0) or (self.energy <= 0)

        if energy_change > 0 and self.using_brain:
            self.nn_reward_count += energy_change

        self.steps_without_reward = 0 if energy_change > 0 else self.steps_without_reward + 1

        if self.brain is not None:
            self.brain.apply_reward(energy_change if terminal else 0.0, terminal)

        if self.brain is not None and self.get_nn_score() >= 80 and not self.brain_saved:
            self._save_brain()

    def _save_brain(self) -> None:
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
        if self.nn_total_steps == 0:
            return 0.0
        raw = (self.nn_reward_count - self.nn_fallback_count) / self.nn_total_steps * 100.0
        return max(0.0, min(100.0, raw))

    def get_color(self) -> str:
        return "purple" if self.using_brain else "green"

    def should_split(self) -> bool:
        return self.energy >= self.max_energy

    def split(self, world: 'World') -> List['Creature']:
        energy_per_creature = self.energy / 4.0
        self.energy = 0
        children: List['Creature'] = []

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

        for i in range(2):
            mutated_brain = None
            if self.brain:
                mutation_rate = 0.1 * (i + 1)
                mutation_scale = 0.2
                mutated_brain = self.brain.mutate(mutation_rate, mutation_scale)

            size_mutation_factor = 1.0 + random.uniform(-0.5, 0.5)
            mutated_size = min(max(self.size * size_mutation_factor, 0.5), 2.0)

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
        return f"<Creature x={self.x:.2f} y={self.y:.2f} size={self.size:.2f} energy={self.energy:.2f}/{self.max_energy:.2f}>"

from .world import World
