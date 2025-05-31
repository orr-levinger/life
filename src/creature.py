from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .world import World

class Creature:
    def __init__(self, x: int, y: int, size: float, energy: float, velocity: float):
        """
        x, y: initial grid coordinates (integers).
        size: determines relative strength (unused in Stage 1) and relates inversely to velocity.
        energy: when ≤ 0, the creature "dies" and is removed by World.step().
        velocity: number of grid cells per step; bigger creatures → smaller velocity (by convention).
        """
        self.x = x
        self.y = y
        self.size = size
        self.energy = energy
        self.velocity = velocity
        # Placeholder for a future NeuralNetwork model; remains None this stage
        self.brain = None

    def decide(self, sensor_inputs: Any) -> str:
        """
        sensor_inputs: always None in Stage 1.
        Return one of: 'MOVE', 'EAT', 'ATTACK', 'REST'. 
        In Stage 1, we ignore sensors and always do 'REST'.
        """
        return 'REST'

    def apply_action(self, action: str, world: 'World') -> None:
        """
        Execute the chosen action. In Stage 1:
          - Regardless of action type ('MOVE', 'EAT', 'ATTACK', 'REST'),
            we only deduct 1 energy as a minimal placeholder implementation.
          - No actual movement, eating, or attacking is performed yet.
        """
        # Deduct fixed energy cost for any action:
        self.energy -= 1
        # TODO (future): implement movement, eating, attacking logic

# Import at the end to avoid circular imports
from .world import World
