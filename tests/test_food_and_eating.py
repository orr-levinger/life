import sys, os
import unittest
import random

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.world import World
from src.creature import Creature
from src.sensors import VisionSensor

class TestFoodAndEating(unittest.TestCase):
    def test_spawn_food(self):
        """Test that spawn_food() adds food to all empty cells when food_spawn_rate=1.0"""
        # Create a small world (3×2), set food_spawn_rate = 1.0
        world = World(3, 2, food_spawn_rate=1.0)

        # Set a fixed seed for reproducibility
        random.seed(42)

        # Call spawn_food() directly (no creatures present)
        world.spawn_food()

        # Afterward, every grid cell (x, y) should be in food_positions
        expected_food_positions = {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)}
        self.assertEqual(world.food_positions, expected_food_positions)

    def test_food_does_not_spawn_under_creatures(self):
        """Test that food does not spawn under creatures"""
        # Create a world (3×3) with food_spawn_rate = 1.0
        world = World(3, 3, food_spawn_rate=1.0)

        # Add a creature at (1.0, 1.0)
        creature = Creature(1.0, 1.0, size=1.0, energy=10.0)
        world.add_creature(creature)

        # Set a fixed seed for reproducibility
        random.seed(42)

        # Call spawn_food()
        world.spawn_food()

        # Ensure (1,1) is not in food_positions, but all other cells are
        expected_food_positions = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}
        self.assertEqual(world.food_positions, expected_food_positions)
        self.assertNotIn((1, 1), world.food_positions)

    def test_eating_adjacent_food(self):
        """Test that creatures can eat food in adjacent cells"""
        # Create a world (5×5) with food_spawn_rate = 0.0
        world = World(5, 5, food_spawn_rate=0.0)

        # Add a creature at (2.0, 2.0) with eat_bonus=5.0
        creature = Creature(2.0, 2.0, size=1.0, energy=10.0, eat_bonus=5.0)
        world.add_creature(creature)

        # Manually place food at (2,3) (one cell north)
        from src.food import Food
        food = Food(x=2, y=3, size=1.0, energy_value=creature.eat_bonus, remaining_duration=-1)
        world.foods.append(food)
        world.food_positions.add((2, 3))  # For backward compatibility

        # Check that VisionSensor.get_reading(creature, world) reports "food" for "north"
        vision = VisionSensor().get_reading(creature, world)
        self.assertEqual(vision["north"], "food")

        # Call action = creature.decide(reading), which should be ("EAT","north")
        action = creature.decide(vision)
        self.assertEqual(action[0], "EAT")
        self.assertEqual(action[1], "north")

        # Initial energy
        initial_energy = creature.energy

        # Call creature.apply_action(action, world) (movement cost only)
        creature.apply_action(action, world)

        # Verify (creature.x, creature.y) moved north
        self.assertEqual(creature.x, 2.0)
        self.assertEqual(creature.y, 3.0)

        # Verify energy decreased by distance
        self.assertEqual(creature.energy, initial_energy - 1.0)

        # Then call world.step()
        world.step()

        # Assert that (2,3) is no longer in world.food_positions
        self.assertNotIn((2, 3), world.food_positions)

        # Assert that creature.energy has netted (- movementCost + eat_bonus - cost from next step)
        # Initial energy - 1.0 (movement cost) + 5.0 (eat_bonus) - 0.2 (eat cost in next step)
        self.assertAlmostEqual(creature.energy, initial_energy - 1.0 + 5.0 - 0.2, places=5)

    def test_eating_in_current_cell(self):
        """Test that creatures can eat food in their current cell"""
        # Create a world (4×4) with food_spawn_rate = 0.0
        world = World(4, 4, food_spawn_rate=0.0)

        # Add a creature at (1.0, 1.0) with eat_bonus=5.0
        creature = Creature(1.0, 1.0, size=1.0, energy=10.0, eat_bonus=5.0)
        world.add_creature(creature)

        # Manually place food at (1,1)
        from src.food import Food
        food = Food(x=1, y=1, size=1.0, energy_value=creature.eat_bonus, remaining_duration=-1)
        world.foods.append(food)
        world.food_positions.add((1, 1))  # For backward compatibility

        # Initial energy
        initial_energy = creature.energy

        # Let reading = VisionSensor.get_reading(creature, world) (all directions return "empty")
        vision = VisionSensor().get_reading(creature, world)

        # Check if creature is on a food cell
        on_food = (int(creature.x), int(creature.y)) in world.food_positions
        self.assertTrue(on_food)

        # When you call action = creature.decide(reading, on_food), it should yield ("EAT_AT_CURRENT", None)
        action = creature.decide(vision, on_food)
        self.assertEqual(action[0], "EAT_AT_CURRENT")
        self.assertIsNone(action[1])

        # Call creature.apply_action(action, world)
        creature.apply_action(action, world)

        # Verify creature.energy decreased by the "eat-in-place" cost (0.2)
        self.assertAlmostEqual(creature.energy, initial_energy - 0.2, places=5)

        # Now call world.step()
        world.step()

        # Verify that (1,1) is removed from food_positions
        self.assertNotIn((1, 1), world.food_positions)

        # Verify creature.energy increased by eat_bonus, netting (initialEnergy - 0.2 + eatBonus - cost from next step)
        self.assertAlmostEqual(creature.energy, initial_energy - 0.2 + 5.0 - 0.2, places=5)

    def test_spawn_and_eat_in_one_step(self):
        """Test that food can spawn and be eaten in one step"""
        # Create a world (2×2) with food_spawn_rate = 1.0
        world = World(2, 2, food_spawn_rate=1.0)

        # Add one creature at (0.0, 0.0) with energy = 5.0
        creature = Creature(0.0, 0.0, size=1.0, energy=5.0, eat_bonus=5.0)
        world.add_creature(creature)

        # Set a fixed seed for reproducibility
        random.seed(42)

        # Initial energy
        initial_energy = creature.energy

        # Override creature's decide method to always rest
        # This ensures it doesn't move and just eats in place
        def always_rest_or_eat(vision, on_food=False):
            if on_food:
                return ("EAT_AT_CURRENT", None)
            return ("REST", None)

        creature.decide = always_rest_or_eat

        # Call world.step()
        world.step()

        # In the new implementation, we spawn food in all empty cells
        # With a 2x2 grid and one creature, there are 3 empty cells
        # The creature eats the food in its cell, so there should be 3 foods left
        # (the creature doesn't eat any food because it's not on a food cell to begin with)
        self.assertEqual(len(world.foods), 3)

        # Assert creature.energy has netted initial_energy - 0.2 (eat cost) + 2.0 (default food energy)
        # The default energy value for spawned food is 2.0 (defined in World.spawn_food)
        self.assertAlmostEqual(creature.energy, initial_energy - 0.2 + 2.0, places=5)

if __name__ == "__main__":
    unittest.main()
