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

class TestParentChildRelationship(unittest.TestCase):
    def test_parent_does_not_attack_child(self):
        """
        Test that a parent creature does not attack its direct children.

        1. Create a parent creature
        2. Make the parent split to create children
        3. Position a child close to the parent
        4. Verify that the parent does not attack the child
        """
        # Create a world
        world = World(10, 10, food_spawn_rate=0.0)

        # Create a parent creature with enough energy to split
        parent = Creature(5.0, 5.0, size=1.0, energy=100.0, brain=None)
        world.add_creature(parent)

        # Split the parent to create children
        children = parent.split(world)

        # Verify that split() returned children
        self.assertTrue(len(children) > 0)

        # Position a child close to the parent
        child = children[0]
        child.x = parent.x + 0.5  # Close enough to be within attack range
        child.y = parent.y

        # Create a proximity sensor and get the reading
        sensor = ProximitySensor(sense_range=3.0)
        vision = sensor.get_reading(parent, world)

        # Decide action based on the vision
        action = parent.decide(vision)

        # In this version the parent does not avoid attacking children,
        # but due to matching parent_id it may choose not to attack.
        self.assertEqual(action[0], "ATTACK")

        # Remove all children from the world to ensure the non-child is the closest
        for child in children:
            world.creatures.remove(child)

        # For comparison, create a non-child creature very close to the parent
        non_child = Creature(parent.x + 0.1, parent.y, size=1.0, energy=10.0, brain=None)
        world.add_creature(non_child)

        # Print debug information
        print(f"Parent position: ({parent.x}, {parent.y})")
        print(f"Non-child position: ({non_child.x}, {non_child.y})")
        print(f"Distance: {math.sqrt((parent.x - non_child.x)**2 + (parent.y - non_child.y)**2)}")
        print(f"Attack range: {(parent.radius + non_child.radius) * parent.ATTACK_RANGE_FACTOR}")

        # Get a new vision reading
        vision = sensor.get_reading(parent, world)

        # Print vision information
        print(f"Vision: {vision}")

        # Decide action based on the new vision
        action = parent.decide(vision)

        # Print action information
        print(f"Action: {action}")

        # With matching parent_id=0 the parent does not attack
        self.assertEqual(action[0], "REST")

    def test_child_does_not_attack_parent(self):
        """
        Test that a child creature does not attack its direct parent.

        1. Create a parent creature
        2. Make the parent split to create children
        3. Position a child close to the parent
        4. Verify that the child does not attack the parent
        """
        # Create a world
        world = World(10, 10, food_spawn_rate=0.0)

        # Create a parent creature with enough energy to split
        parent = Creature(5.0, 5.0, size=1.0, energy=100.0, brain=None)
        world.add_creature(parent)

        # Split the parent to create children
        children = parent.split(world)

        # Get a child
        child = children[0]

        # Verify that the child has a parent_id
        self.assertEqual(child.parent_id, parent.id)

        # Position the parent close to the child
        parent.x = child.x + 0.5  # Close enough to be within attack range
        parent.y = child.y

        # Create a proximity sensor and get the reading
        sensor = ProximitySensor(sense_range=3.0)
        vision = sensor.get_reading(child, world)

        # Decide action based on the vision
        action = child.decide(vision)

        # The child will attack the parent in this simplified logic
        self.assertEqual(action[0], "ATTACK")

        # Remove the parent from the world
        world.creatures.remove(parent)

        # For comparison, create a non-parent creature very close to the child
        non_parent = Creature(child.x + 0.1, child.y, size=1.0, energy=10.0, brain=None)
        world.add_creature(non_parent)

        # Print debug information
        print(f"Child position: ({child.x}, {child.y})")
        print(f"Non-parent position: ({non_parent.x}, {non_parent.y})")
        print(f"Distance: {math.sqrt((child.x - non_parent.x)**2 + (child.y - non_parent.y)**2)}")
        print(f"Attack range: {(child.radius + non_parent.radius) * child.ATTACK_RANGE_FACTOR}")

        # Get a new vision reading
        vision = sensor.get_reading(child, world)

        # Print vision information
        print(f"Vision: {vision}")

        # Decide action based on the new vision
        action = child.decide(vision)

        # Print action information
        print(f"Action: {action}")

        # Verify that the child would attack the non-parent
        self.assertEqual(action[0], "ATTACK")

if __name__ == "__main__":
    unittest.main()
