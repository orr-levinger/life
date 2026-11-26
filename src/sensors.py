# src/sensors.py

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List, Tuple, Union

if TYPE_CHECKING:  # type hints only: importing these at run time would make an import cycle
    from .creature import Creature
    from .food import Food
    from .world import World


class Sensor:
    """
    Base Sensor interface. Every concrete sensor must implement get_reading().
    """

    def get_reading(self, creature: Creature, world: World) -> Any:
        raise NotImplementedError("Must implement get_reading() in subclasses")


class ProximitySensor(Sensor):
    """
    A continuous-space sensor that detects nearby objects within a certain radius.

    It returns a list of sightings, where each sighting is a tuple:
    (type: str, object: Union[Creature, Food], distance: float, angle: float)

    - type: "creature" or "food"
    - object: reference to the actual Creature or Food object
    - distance: along the ground from the sensing creature to the object
    - angle: in radians, in the sensing creature's own view (0 = ahead, π/2 = to its left)
    """

    def __init__(self, sense_range: float = 3.0):
        """
        Initialize a ProximitySensor with a given sensing radius.

        Args:
            sense_range: Maximum distance at which objects can be detected
        """
        self.sense_range = sense_range

    def get_reading(self, creature: Creature, world: World) -> List[Tuple[str, Union[Creature, Food], float, float]]:
        """
        Get a list of all objects (creatures and food) within sensing range.

        Args:
            creature: The creature doing the sensing
            world: The world containing all objects

        Returns:
            List of tuples (type, object, distance, angle) for all objects within sense_range
        """
        sightings = []

        # Loop over all creatures (except self)
        for other in world.creatures:
            if other is creature:  # Skip self
                continue

            distance, angle = _where(creature, other.p, world)
            if distance <= self.sense_range:
                sightings.append(("creature", other, distance, angle))

        # Loop over all food
        for food in world.foods:
            distance, angle = _where(creature, food.p, world)
            if distance <= self.sense_range:
                sightings.append(("food", food, distance, angle))

        # Sort by distance (closest first)
        sightings.sort(key=lambda x: x[2])

        return sightings


def _where(creature: Creature, point, world: World) -> Tuple[float, float]:
    """How far along the ground, and at which angle of the creature's own view (0 ahead, π/2 to its left)."""
    from .sphere import arc, bearing
    ahead, left = bearing(creature.p, creature.facing, point)
    return float(arc(creature.p, point, world.rules["planet_radius"])), math.atan2(float(left), float(ahead))
