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
    - distance: Euclidean distance from the sensing creature to the object
    - angle: angle in radians from the sensing creature to the object (0 = east, π/2 = north)
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

            # Calculate distance
            dx = other.x - creature.x
            dy = other.y - creature.y
            distance = math.sqrt(dx*dx + dy*dy)

            # If within sensing range, add to sightings
            if distance <= self.sense_range:
                # Calculate angle (0 = east, π/2 = north)
                angle = math.atan2(dy, dx)
                sightings.append(("creature", other, distance, angle))

        # Loop over all food
        for food in world.foods:
            # Calculate distance
            dx = food.x - creature.x
            dy = food.y - creature.y
            distance = math.sqrt(dx*dx + dy*dy)

            # If within sensing range, add to sightings
            if distance <= self.sense_range:
                # Calculate angle (0 = east, π/2 = north)
                angle = math.atan2(dy, dx)
                sightings.append(("food", food, distance, angle))

        # Sort by distance (closest first)
        sightings.sort(key=lambda x: x[2])

        return sightings
