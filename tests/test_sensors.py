import unittest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.world import World
from src.creature import Creature
from src.food import Food
from src.sensors import VisionSensor, ProximitySensor

class TestSensors(unittest.TestCase):
    def test_vision_sensor_detects_creatures_and_food(self):
        world = World(5, 5, food_spawn_rate=0.0)
        c = Creature(2.0, 2.0, size=1.0, energy=5.0)
        world.add_creature(c)

        north_creature = Creature(2.0, 3.0, size=1.0, energy=5.0)
        world.add_creature(north_creature)

        south_food = Food(x=2.4, y=1.3, remaining_duration=-1, energy=1.0)
        west_food = Food(x=1.8, y=2.5, remaining_duration=-1, energy=1.0)
        world.foods.extend([south_food, west_food])

        sensor = VisionSensor()
        reading = sensor.get_reading(c, world)

        self.assertEqual(reading["north"], "creature")
        self.assertEqual(reading["south"], "food")
        self.assertEqual(reading["west"], "food")
        self.assertEqual(reading["east"], "empty")

    def test_proximity_sensor_sorting_and_range(self):
        world = World(10, 10, food_spawn_rate=0.0)
        c = Creature(5.0, 5.0, size=1.0, energy=5.0)
        world.add_creature(c)

        close_food = Food(x=6.0, y=5.0, remaining_duration=-1, energy=1.0)
        far_food = Food(x=5.0, y=8.0, remaining_duration=-1, energy=1.0)
        world.foods.extend([close_food, far_food])

        creature_a = Creature(7.0, 5.0, size=1.0, energy=5.0)
        creature_b = Creature(3.0, 5.0, size=1.0, energy=5.0)
        world.add_creature(creature_a)
        world.add_creature(creature_b)

        sensor = ProximitySensor(sense_range=3.0)
        sightings = sensor.get_reading(c, world)

        # Should detect close_food first
        self.assertEqual(sightings[0][1], close_food)
        # Should only include objects within sense_range
        self.assertTrue(all(s[2] <= 3.0 for s in sightings))
        # Should detect exactly four objects
        self.assertEqual(len(sightings), 4)

if __name__ == "__main__":
    unittest.main()
