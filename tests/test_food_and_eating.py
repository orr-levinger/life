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
        """Test that creatures can eat food in adjacent cells without moving"""
        # Create a world (5×5) with food_spawn_rate = 0.0
        world = World(5, 5, food_spawn_rate=0.0)

        # Add a creature at (2.0, 2.0)
        creature = Creature(2.0, 2.0, size=1.0, energy=10.0)
        world.add_creature(creature)

        # Manually place food at (2,3) (one cell north) with remaining_energy = 5
        from src.food import Food
        food = Food(x=2, y=3, size=1.0, energy_value=5.0, remaining_duration=-1)
        world.foods.append(food)
        world.food_positions.add((2, 3))  # For backward compatibility

        # Check that VisionSensor.get_reading(creature, world) reports "food" for "north"
        vision = VisionSensor().get_reading(creature, world)
        self.assertEqual(vision["north"], "food")

        # Call action = creature.decide(reading), which should be ("EAT","north")
        action = creature.decide(vision)
        self.assertEqual(action[0], "EAT")
        self.assertEqual(action[1], "north")

        # Initial energy and food remaining energy
        initial_energy = creature.energy
        initial_food_energy = food.remaining_energy

        # Call creature.apply_action(action, world)
        creature.apply_action(action, world)

        # Verify creature position hasn't changed
        self.assertEqual(creature.x, 2.0)
        self.assertEqual(creature.y, 2.0)

        # Verify creature energy increased by 1
        self.assertEqual(creature.energy, initial_energy + 1.0)

        # Verify food remaining energy decreased by 1
        self.assertEqual(food.remaining_energy, initial_food_energy - 1.0)

        # Verify food is still in world.foods (not fully consumed)
        self.assertIn(food, world.foods)
        self.assertIn((2, 3), world.food_positions)

        # Take 4 more bites to fully consume the food
        for _ in range(4):
            # Get vision again
            vision = VisionSensor().get_reading(creature, world)
            action = creature.decide(vision)
            creature.apply_action(action, world)

        # Monkey-patch decide method to always rest
        # This ensures the creature doesn't try to eat something else in the next step
        def always_rest(vision, on_food=False):
            return ("REST", None)

        creature.decide = always_rest

        # Call world.step() to trigger the removal of expired foods
        world.step()

        # Verify food is now gone (fully consumed)
        self.assertNotIn(food, world.foods)
        self.assertNotIn((2, 3), world.food_positions)

        # Verify creature energy increased by a total of 5 and then decreased by 0.1 for REST
        self.assertEqual(creature.energy, initial_energy + 5.0 - 0.1)

    def test_eating_in_current_cell(self):
        """Test that creatures can eat food in their current cell incrementally"""
        # Create a world (4×4) with food_spawn_rate = 0.0
        world = World(4, 4, food_spawn_rate=0.0)

        # Add a creature at (1.0, 1.0)
        creature = Creature(1.0, 1.0, size=1.0, energy=10.0)
        world.add_creature(creature)

        # Manually place food at (1,1) with remaining_energy = 5
        from src.food import Food
        food = Food(x=1, y=1, size=1.0, energy_value=5.0, remaining_duration=-1)
        world.foods.append(food)
        world.food_positions.add((1, 1))  # For backward compatibility

        # Initial energy and food remaining energy
        initial_energy = creature.energy
        initial_food_energy = food.remaining_energy

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

        # Verify creature energy increased by 1
        self.assertEqual(creature.energy, initial_energy + 1.0)

        # Verify food remaining energy decreased by 1
        self.assertEqual(food.remaining_energy, initial_food_energy - 1.0)

        # Verify food is still in world.foods (not fully consumed)
        self.assertIn(food, world.foods)
        self.assertIn((1, 1), world.food_positions)

        # Take 4 more bites to fully consume the food
        for _ in range(4):
            # Get vision again
            vision = VisionSensor().get_reading(creature, world)
            action = creature.decide(vision, on_food=True)
            creature.apply_action(action, world)

        # Monkey-patch decide method to always rest
        # This ensures the creature doesn't try to eat something else in the next step
        def always_rest(vision, on_food=False):
            return ("REST", None)

        creature.decide = always_rest

        # Call world.step() to trigger the removal of expired foods
        world.step()

        # Verify food is now gone (fully consumed)
        self.assertNotIn(food, world.foods)
        self.assertNotIn((1, 1), world.food_positions)

        # Verify creature energy increased by a total of 5 and then decreased by 0.1 for REST
        self.assertEqual(creature.energy, initial_energy + 5.0 - 0.1)

    def test_spawn_and_eat_in_one_step(self):
        """Test that food can spawn and be eaten incrementally"""
        # Create a world (2×2) with food_spawn_rate = 1.0
        world = World(2, 2, food_spawn_rate=1.0)

        # Add one creature at (0.0, 0.0) with energy = 5.0
        creature = Creature(0.0, 0.0, size=1.0, energy=5.0)
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

        # Call world.step() to spawn food
        world.step()

        # In the new implementation, we spawn food in all empty cells
        # With a 2x2 grid and one creature, there are 3 empty cells
        # The creature doesn't eat any food because it's not on a food cell to begin with
        self.assertEqual(len(world.foods), 3)

        # Manually place food at (0,0) with remaining_energy = 2
        from src.food import Food
        food = Food(x=0, y=0, size=1.0, energy_value=2.0, remaining_duration=-1)
        world.foods.append(food)
        world.food_positions.add((0, 0))  # For backward compatibility

        # Initial food remaining energy
        initial_food_energy = food.remaining_energy

        # Call world.step() again
        world.step()

        # Verify creature energy increased by 1
        self.assertEqual(creature.energy, initial_energy - 0.1 + 1.0)  # -0.1 for REST in first step, +1 for one bite

        # Verify food remaining energy decreased by 1
        self.assertEqual(food.remaining_energy, initial_food_energy - 1.0)

        # Verify food is still in world.foods (not fully consumed)
        self.assertIn(food, world.foods)
        self.assertIn((0, 0), world.food_positions)

        # Call world.step() one more time to fully consume the food
        world.step()

        # Verify food is now gone (fully consumed)
        self.assertNotIn(food, world.foods)
        self.assertNotIn((0, 0), world.food_positions)

        # Verify creature energy increased by a total of 2
        self.assertEqual(creature.energy, initial_energy - 0.1 + 2.0)

    def test_simultaneous_eating(self):
        """Test that multiple creatures can eat the same food simultaneously"""
        # Create a world (5×5) with food_spawn_rate = 0.0
        world = World(5, 5, food_spawn_rate=0.0)

        # Add two creatures on opposite sides of a food
        creature_a = Creature(1.0, 2.0, size=1.0, energy=10.0)  # Left of food
        creature_b = Creature(3.0, 2.0, size=1.0, energy=10.0)  # Right of food
        world.add_creature(creature_a)
        world.add_creature(creature_b)

        # Manually place food at (2,2) with remaining_energy = 5
        from src.food import Food
        food = Food(x=2, y=2, size=1.0, energy_value=5.0, remaining_duration=-1)
        world.foods.append(food)
        world.food_positions.add((2, 2))  # For backward compatibility

        # Initial energy and food remaining energy
        initial_energy_a = creature_a.energy
        initial_energy_b = creature_b.energy
        initial_food_energy = food.remaining_energy

        # Monkey-patch decide methods to always eat toward the food
        def always_eat_east(vision, on_food=False):
            return ("EAT", "east")

        def always_eat_west(vision, on_food=False):
            return ("EAT", "west")

        creature_a.decide = always_eat_east
        creature_b.decide = always_eat_west

        # Call world.step()
        world.step()

        # Verify both creatures' energy increased by 1
        self.assertEqual(creature_a.energy, initial_energy_a + 1.0)
        self.assertEqual(creature_b.energy, initial_energy_b + 1.0)

        # Verify food remaining energy decreased by 2 (1 per creature)
        self.assertEqual(food.remaining_energy, initial_food_energy - 2.0)

        # Verify food is still in world.foods (not fully consumed)
        self.assertIn(food, world.foods)
        self.assertIn((2, 2), world.food_positions)

        # Call world.step() again
        world.step()

        # Verify both creatures' energy increased by another 1
        self.assertEqual(creature_a.energy, initial_energy_a + 2.0)
        self.assertEqual(creature_b.energy, initial_energy_b + 2.0)

        # Verify food remaining energy decreased by another 2
        self.assertEqual(food.remaining_energy, initial_food_energy - 4.0)

        # Verify food is still in world.foods (not fully consumed)
        self.assertIn(food, world.foods)
        self.assertIn((2, 2), world.food_positions)

        # Call world.step() one more time to fully consume the food
        world.step()

        # Verify both creatures' energy increased by another 1 (total of 3 each)
        # Note: Since food had 5 energy and 2 creatures took 2 per step,
        # on the 3rd step only 1 energy was left, so they each got 0.5 (or one got 1 and the other got 0)
        # But our implementation gives 1 energy to each creature that bites, even if the food runs out
        self.assertEqual(creature_a.energy, initial_energy_a + 3.0)
        self.assertEqual(creature_b.energy, initial_energy_b + 3.0)

        # Verify food is now gone (fully consumed)
        self.assertNotIn(food, world.foods)
        self.assertNotIn((2, 2), world.food_positions)


    def test_corpse_decay(self):
        """Test that a corpse's remaining_energy is used up before it disappears or before its remaining_duration expires"""
        # Create a world (5×5) with food_spawn_rate = 0.0
        world = World(5, 5, food_spawn_rate=0.0)

        # Test case 1: Corpse decays due to remaining_duration reaching 0
        # Manually place a corpse at (1,1) with remaining_energy = 4 and remaining_duration = 2
        from src.food import Food
        corpse1 = Food(x=1, y=1, size=1.0, energy_value=4.0, remaining_duration=2)
        world.foods.append(corpse1)
        world.food_positions.add((1, 1))  # For backward compatibility

        # Call world.step() twice to let the corpse decay
        world.step()
        self.assertEqual(corpse1.remaining_duration, 1)
        self.assertEqual(corpse1.remaining_energy, 4.0)  # Energy unchanged
        self.assertIn(corpse1, world.foods)  # Corpse still exists

        world.step()
        # Corpse should be gone due to remaining_duration reaching 0
        self.assertNotIn(corpse1, world.foods)
        self.assertNotIn((1, 1), world.food_positions)

        # Test case 2: Corpse disappears when remaining_energy reaches 0 before remaining_duration
        # Manually place a corpse at (2,2) with remaining_energy = 2 and remaining_duration = 5
        corpse2 = Food(x=2, y=2, size=1.0, energy_value=2.0, remaining_duration=5)
        world.foods.append(corpse2)
        world.food_positions.add((2, 2))  # For backward compatibility

        # Add a creature at (1.0, 2.0) to eat the corpse
        creature = Creature(1.0, 2.0, size=1.0, energy=10.0)
        world.add_creature(creature)

        # Monkey-patch decide method to always eat toward the corpse
        def always_eat_east(vision, on_food=False):
            return ("EAT", "east")

        creature.decide = always_eat_east

        # Call world.step() to let the creature take one bite
        world.step()
        self.assertEqual(corpse2.remaining_duration, 4)  # Duration decreased
        self.assertEqual(corpse2.remaining_energy, 1.0)  # Energy decreased by 1
        self.assertIn(corpse2, world.foods)  # Corpse still exists

        # Call world.step() again to let the creature take another bite
        world.step()
        # Corpse should be gone due to remaining_energy reaching 0
        self.assertNotIn(corpse2, world.foods)
        self.assertNotIn((2, 2), world.food_positions)


if __name__ == "__main__":
    unittest.main()
