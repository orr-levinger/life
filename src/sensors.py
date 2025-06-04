# src/sensors.py

from typing import Any, Dict, List, Tuple, Union
import sys
import os
import math

# Handle both being imported as a module and being run directly
if __name__ == "__main__":
    # When run directly, use absolute imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.world import World
    from src.creature import Creature
    from src.food import Food
else:
    # When imported as a module, use relative imports
    from .world import World
    from .creature import Creature
    from .food import Food

class Sensor:
    """
    Base Sensor interface. Every concrete sensor must implement get_reading().
    """

    def get_reading(self, creature: Creature, world: World) -> Any:
        raise NotImplementedError("Must implement get_reading() in subclasses")


class VisionSensor(Sensor):
    """
    A simple vision sensor that looks one cell in each of the four directions (grid-based).
    It returns a dictionary mapping directions to:
      - 'empty' if nothing is in that adjacent cell,
      - 'creature' if a Creature occupies that cell,
      - 'food' if a food item occupies that cell.
    Format: {'north': str, 'south': str, 'east': str, 'west': str}

    Note: This is the legacy grid-based sensor. For continuous space, use ProximitySensor instead.
    """

    def get_reading(self, creature: Creature, world: World) -> Dict[str, str]:
        x, y = int(creature.x), int(creature.y)
        readings: Dict[str, str] = {
            "north": "empty",
            "south": "empty",
            "east":  "empty",
            "west":  "empty"
        }
        # Check one cell away in each cardinal direction, within bounds:
        # North
        if y + 1 < world.height:
            if any(int(c.x) == x and int(c.y) == y + 1 for c in world.creatures):
                readings["north"] = "creature"
            elif any(f.x == x and f.y == y + 1 for f in world.foods):
                readings["north"] = "food"
        # South
        if y - 1 >= 0:
            if any(int(c.x) == x and int(c.y) == y - 1 for c in world.creatures):
                readings["south"] = "creature"
            elif any(f.x == x and f.y == y - 1 for f in world.foods):
                readings["south"] = "food"
        # East
        if x + 1 < world.width:
            if any(int(c.x) == x + 1 and int(c.y) == y for c in world.creatures):
                readings["east"] = "creature"
            elif any(f.x == x + 1 and f.y == y for f in world.foods):
                readings["east"] = "food"
        # West
        if x - 1 >= 0:
            if any(int(c.x) == x - 1 and int(c.y) == y for c in world.creatures):
                readings["west"] = "creature"
            elif any(f.x == x - 1 and f.y == y for f in world.foods):
                readings["west"] = "food"
        return readings


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
