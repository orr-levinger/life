import sys, os
import unittest
import random
import math

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.world import World
from src.creature import Creature
from src.food import Food
from src.sensors import ProximitySensor

class TestFoodAndEating(unittest.TestCase):
    def test_spawn_food(self):
        """Test that spawn_food() adds food to all empty cells when food_spawn_rate=1.0"""
        # Create a small world (3×2), set food_spawn_rate = 1.0
        world = World(3, 2, food_spawn_rate=1.0)

        # Set a fixed seed for reproducibility
        random.seed(42)

        # Call spawn_food() directly (no creatures present)
        world.spawn_food()

        # Afterward, every grid cell should have a food item
        # Count the number of food items in each grid cell
        food_cells = {(int(f.x), int(f.y)) for f in world.foods}
        expected_food_cells = {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)}
        self.assertEqual(food_cells, expected_food_cells)
        self.assertEqual(len(world.foods), 6)  # 3x2 = 6 cells

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

        # Ensure no food is at the creature's position
        food_cells = {(int(f.x), int(f.y)) for f in world.foods}
        expected_food_cells = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}
        self.assertEqual(food_cells, expected_food_cells)
        self.assertNotIn((1, 1), food_cells)

    def test_eating_nearby_food(self):
        """Test that creatures can eat food that's within eat range"""
        # Create a world (5×5) with food_spawn_rate = 0.0
        world = World(5, 5, food_spawn_rate=0.0)

        # Add a creature at (2.0, 2.0)
        creature = Creature(2.0, 2.0, size=1.0, energy=10.0)
        world.add_creature(creature)

        # Manually place food at (2.5, 2.5) with remaining_energy = 5
        # This is within eat range of the creature
        food = Food(x=2.5, y=2.5, size=1.0, energy_value=5.0, remaining_duration=-1)
        world.foods.append(food)

        # Initial energy and food remaining energy
        initial_energy = creature.energy
        initial_food_energy = food.remaining_energy

        # Override creature's decide method to always eat the food
        def always_eat_food(vision, on_food=False):
            # Find nearby food
            nearby_food = [(obj, dist, angle) for type_tag, obj, dist, angle in vision if type_tag == "food"]

            if nearby_food:
                # Sort by distance (closest first)
                nearby_food.sort(key=lambda x: x[1])
                closest_food, distance, angle = nearby_food[0]

                # Calculate eat range based on radii
                eat_range = (creature.radius + closest_food.radius) * creature.EAT_RANGE_FACTOR

                # If within eat range, eat
                if distance <= eat_range:
                    # Set intent for visualization
                    creature.intent = "GO_TO_FOOD"
                    creature.intended_vector = (math.cos(angle) * creature.velocity * 0.75, 
                                               math.sin(angle) * creature.velocity * 0.75)
                    return ("EAT", closest_food)

            # If no food in range, rest
            return ("REST", None)

        creature.decide = always_eat_food

        # Call world.step()
        world.step()

        # Verify creature position hasn't changed
        self.assertAlmostEqual(creature.x, 2.0, places=5)
        self.assertAlmostEqual(creature.y, 2.0, places=5)

        # Verify creature energy increased by 1
        self.assertAlmostEqual(creature.energy, initial_energy + 1.0, places=5)

        # Verify food remaining energy decreased by 1
        self.assertAlmostEqual(food.remaining_energy, initial_food_energy - 1.0, places=5)

        # Verify food is still in world.foods (not fully consumed)
        self.assertIn(food, world.foods)

        # Take 4 more bites to fully consume the food
        for _ in range(4):
            world.step()

        # Verify food is now gone (fully consumed)
        self.assertNotIn(food, world.foods)

        # Verify creature energy increased by a total of 5
        self.assertAlmostEqual(creature.energy, initial_energy + 5.0, places=5)

    def test_eating_at_current_position(self):
        """Test that creatures can eat food at their current position incrementally"""
        # Create a world (4×4) with food_spawn_rate = 0.0
        world = World(4, 4, food_spawn_rate=0.0)

        # Add a creature at (1.0, 1.0)
        creature = Creature(1.0, 1.0, size=1.0, energy=10.0)
        world.add_creature(creature)

        # Manually place food at the same position (1.0, 1.0) with remaining_energy = 5
        food = Food(x=1.0, y=1.0, size=1.0, energy_value=5.0, remaining_duration=-1)
        world.foods.append(food)

        # Initial energy and food remaining energy
        initial_energy = creature.energy
        initial_food_energy = food.remaining_energy

        # Override creature's decide method to always eat at current position if on food
        def always_eat_at_current(vision, on_food=False):
            if on_food:
                # Set intent for visualization
                creature.intent = "GO_TO_FOOD"
                creature.intended_vector = (0.0, 0.0)
                return ("EAT_AT_CURRENT", None)
            return ("REST", None)

        creature.decide = always_eat_at_current

        # Call world.step()
        world.step()

        # Verify creature energy increased by 1
        self.assertAlmostEqual(creature.energy, initial_energy + 1.0, places=5)

        # Verify food remaining energy decreased by 1
        self.assertAlmostEqual(food.remaining_energy, initial_food_energy - 1.0, places=5)

        # Verify food is still in world.foods (not fully consumed)
        self.assertIn(food, world.foods)

        # Take 4 more bites to fully consume the food
        for _ in range(4):
            world.step()

        # Verify food is now gone (fully consumed)
        self.assertNotIn(food, world.foods)

        # Verify creature energy increased by a total of 5
        self.assertAlmostEqual(creature.energy, initial_energy + 5.0, places=5)

    def test_simultaneous_eating(self):
        """Test that multiple creatures can eat the same food simultaneously"""
        # Create a world (5×5) with food_spawn_rate = 0.0
        world = World(5, 5, food_spawn_rate=0.0)

        # Add two creatures on opposite sides of a food
        creature_a = Creature(1.5, 2.0, size=1.0, energy=10.0)  # Left of food
        creature_b = Creature(2.5, 2.0, size=1.0, energy=10.0)  # Right of food
        world.add_creature(creature_a)
        world.add_creature(creature_b)

        # Manually place food at (2.0, 2.0) with remaining_energy = 5
        food = Food(x=2.0, y=2.0, size=1.0, energy_value=5.0, remaining_duration=-1)
        world.foods.append(food)

        # Initial energy and food remaining energy
        initial_energy_a = creature_a.energy
        initial_energy_b = creature_b.energy
        initial_food_energy = food.remaining_energy

        # Override creatures' decide methods to always eat the food
        def always_eat_food(vision, on_food=False):
            # Find nearby food
            nearby_food = [(obj, dist, angle) for type_tag, obj, dist, angle in vision if type_tag == "food"]

            if nearby_food:
                # Sort by distance (closest first)
                nearby_food.sort(key=lambda x: x[1])
                closest_food, distance, angle = nearby_food[0]

                # Calculate eat range based on radii
                eat_range = (self.radius + closest_food.radius) * self.EAT_RANGE_FACTOR

                # If within eat range, eat
                if distance <= eat_range:
                    # Set intent for visualization
                    self.intent = "GO_TO_FOOD"
                    self.intended_vector = (math.cos(angle) * self.velocity * 0.75, 
                                           math.sin(angle) * self.velocity * 0.75)
                    return ("EAT", closest_food)

            # If no food in range, rest
            return ("REST", None)

        # Bind the method to each creature
        creature_a.decide = lambda vision, on_food=False: always_eat_food(creature_a, vision, on_food)
        creature_b.decide = lambda vision, on_food=False: always_eat_food(creature_b, vision, on_food)

        # Call world.step()
        world.step()

        # Verify both creatures' energy increased by 1
        self.assertAlmostEqual(creature_a.energy, initial_energy_a + 1.0, places=5)
        self.assertAlmostEqual(creature_b.energy, initial_energy_b + 1.0, places=5)

        # Verify food remaining energy decreased by 2 (1 per creature)
        self.assertAlmostEqual(food.remaining_energy, initial_food_energy - 2.0, places=5)

        # Verify food is still in world.foods (not fully consumed)
        self.assertIn(food, world.foods)

        # Take 2 more bites to fully consume the food (5 energy / 2 creatures = 3 steps)
        for _ in range(2):
            world.step()

        # Verify food is now gone (fully consumed)
        self.assertNotIn(food, world.foods)

        # Verify both creatures' energy increased by a total of 3 (5 energy / 2 creatures = 2.5, rounded up to 3)
        self.assertAlmostEqual(creature_a.energy, initial_energy_a + 3.0, places=5)
        self.assertAlmostEqual(creature_b.energy, initial_energy_b + 3.0, places=5)

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
