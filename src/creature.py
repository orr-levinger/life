from typing import Any, TYPE_CHECKING, Dict, Tuple, Optional
import random
import math

if TYPE_CHECKING:
    from .world import World
    from .sensors import Sensor, VisionSensor

class Creature:
    def __init__(self, x: float, y: float, size: float, energy: float, velocity: float = None, eat_bonus: float = 5.0):
        """
        x, y: initial continuous coordinates (floats).
        size: determines relative strength and relates inversely to velocity.
        energy: when ≤ 0, the creature "dies" and is removed by World.step().
        velocity: maximum speed; bigger creatures → smaller velocity (by convention).
                 If None, computed as 1.0 / size.
        eat_bonus: how much energy a creature gains when it consumes one food item.
        """
        self.x = float(x)
        self.y = float(y)
        self.size = size
        self.energy = energy
        self.eat_bonus = eat_bonus
        # Compute velocity if not provided
        if velocity is None:
            self.velocity = 1.0 / size
        else:
            self.velocity = velocity
        # Placeholder for a future NeuralNetwork model; remains None this stage
        self.brain = None

        # Add sensors
        from .sensors import VisionSensor
        self.sensors: Tuple['Sensor', ...] = (VisionSensor(),)

    def decide(self, vision: Dict[str, str], on_food: bool = False) -> Tuple[str, Optional[Tuple[float, float]]]:
        """
        Decide movement based on VisionSensor. Return:
          - ("MOVE", (dx, dy)) where sqrt(dx^2 + dy^2) ≤ self.velocity,
          - ("REST", None) for resting,
          - ("EAT", direction) to move toward and eat food in an adjacent cell,
          - ("EAT_AT_CURRENT", None) to eat food in the current cell.

        Logic:
        1. If any direction maps to "food", return ("EAT", direction) to move toward and eat that food.
        2. If the creature's current cell has food (on_food=True), return ("EAT_AT_CURRENT", None).
        3. Else if any direction maps to "creature", move directly away at max speed.
        4. Else (nothing sensed), randomly choose either:
           - Move in random angle at max speed, OR
           - Rest (cost is lower). Use 50/50 split.
        """
        if vision is None:
            return ("REST", None)

        # 1) Food adjacent? Return EAT action with direction
        for direction, content in vision.items():
            if content == "food":
                if direction == "north":
                    return ("EAT", "north")
                if direction == "south":
                    return ("EAT", "south")
                if direction == "east":
                    return ("EAT", "east")
                if direction == "west":
                    return ("EAT", "west")

        # 2) Check if there's food in the current cell
        if on_food:
            return ("EAT_AT_CURRENT", None)

        # 3) Creature adjacent: flee
        for direction, content in vision.items():
            if content == "creature":
                if direction == "north":
                    return ("MOVE", (0.0, -self.velocity))
                if direction == "south":
                    return ("MOVE", (0.0, self.velocity))
                if direction == "east":
                    return ("MOVE", (-self.velocity, 0.0))
                if direction == "west":
                    return ("MOVE", (self.velocity, 0.0))

        # 4) Nothing sensed: random
        if random.random() < 0.5:
            # Random angle in [0, 2π)
            angle = random.random() * 2 * math.pi
            dx = math.cos(angle) * self.velocity
            dy = math.sin(angle) * self.velocity
            return ("MOVE", (dx, dy))
        else:
            return ("REST", None)

    def apply_action(self, action: Tuple[str, Any], world: 'World') -> None:
        """
        Execute the chosen action:
          - ("MOVE", (dx, dy)): clamp (dx,dy) to max speed; update (x, y) by (dx, dy) clamped to world bounds; energy -= distance
          - ("REST", None): energy -= 0.1
          - ("EAT", direction): move into adjacent food cell at max speed; energy -= distance
          - ("EAT_AT_CURRENT", None): eat food in current cell; energy -= 0.2
        """
        act_type, param = action

        if act_type == "MOVE" and param is not None:
            dx, dy = param
            # Compute requested distance
            requested_dist = math.hypot(dx, dy)
            # If more than max, scale down to max:
            if requested_dist > self.velocity:
                scale = self.velocity / requested_dist
                dx *= scale
                dy *= scale
                actual_dist = self.velocity
            else:
                actual_dist = requested_dist

            # Compute new coordinates, then clamp to [0, width], [0, height]
            new_x = self.x + dx
            new_y = self.y + dy
            # Clamp within world bounds, accounting for creature size and visualization offset:
            # The visualization adds 0.5 to center creatures in grid cells, so we need to adjust the bounds
            # to ensure creatures stay fully within the screen
            self.x = min(max(new_x, 0.0), world.width - 1.0)
            self.y = min(max(new_y, 0.0), world.height - 1.0)

            # Deduct energy equal to distance moved
            self.energy -= actual_dist

        elif act_type == "EAT" and param is not None:
            # Move into adjacent food cell at a fixed speed (1.0) regardless of creature size
            # This ensures all creatures can eat food regardless of their size
            direction = param
            dx, dy = 0.0, 0.0
            fixed_speed = 1.0  # Fixed speed for eating, independent of creature size

            if direction == "north":
                dx, dy = 0.0, fixed_speed
            elif direction == "south":
                dx, dy = 0.0, -fixed_speed
            elif direction == "east":
                dx, dy = fixed_speed, 0.0
            elif direction == "west":
                dx, dy = -fixed_speed, 0.0

            # Compute new coordinates, then clamp to [0, width], [0, height]
            new_x = self.x + dx
            new_y = self.y + dy
            # Clamp within world bounds
            self.x = min(max(new_x, 0.0), world.width - 1.0)
            self.y = min(max(new_y, 0.0), world.height - 1.0)

            # Deduct energy equal to distance moved
            actual_dist = math.hypot(dx, dy)
            self.energy -= actual_dist

        elif act_type == "EAT_AT_CURRENT":
            # Eat food in current cell
            # Deduct a small "eat-in-place" cost
            # The energy bonus will be applied in world.step()
            self.energy -= 0.2

        else:  # "REST"
            self.energy -= 0.1

# Import at the end to avoid circular imports
from .world import World
