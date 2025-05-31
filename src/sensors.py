# src/sensors.py

from typing import Any, Dict
import sys
import os

# Handle both being imported as a module and being run directly
if __name__ == "__main__":
    # When run directly, use absolute imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.world import World
    from src.creature import Creature
else:
    # When imported as a module, use relative imports
    from .world import World
    from .creature import Creature

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
            # Check for food in world.foods (new way)
            elif any(f.x == x and f.y == y + 1 for f in world.foods):
                readings["north"] = "food"
            # Fallback to food_positions for backward compatibility
            elif (x, y + 1) in world.food_positions:
                readings["north"] = "food"
        # South
        if y - 1 >= 0:
            if any(int(c.x) == x and int(c.y) == y - 1 for c in world.creatures):
                readings["south"] = "creature"
            # Check for food in world.foods (new way)
            elif any(f.x == x and f.y == y - 1 for f in world.foods):
                readings["south"] = "food"
            # Fallback to food_positions for backward compatibility
            elif (x, y - 1) in world.food_positions:
                readings["south"] = "food"
        # East
        if x + 1 < world.width:
            if any(int(c.x) == x + 1 and int(c.y) == y for c in world.creatures):
                readings["east"] = "creature"
            # Check for food in world.foods (new way)
            elif any(f.x == x + 1 and f.y == y for f in world.foods):
                readings["east"] = "food"
            # Fallback to food_positions for backward compatibility
            elif (x + 1, y) in world.food_positions:
                readings["east"] = "food"
        # West
        if x - 1 >= 0:
            if any(int(c.x) == x - 1 and int(c.y) == y for c in world.creatures):
                readings["west"] = "creature"
            # Check for food in world.foods (new way)
            elif any(f.x == x - 1 and f.y == y for f in world.foods):
                readings["west"] = "food"
            # Fallback to food_positions for backward compatibility
            elif (x - 1, y) in world.food_positions:
                readings["west"] = "food"
        return readings
