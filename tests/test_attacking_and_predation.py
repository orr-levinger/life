# tests/test_attacking_and_predation.py

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

class TestAttackingAndPredation(unittest.TestCase):
    def test_successful_attack_leaves_corpse(self):
        """
        Test that a successful attack that kills a creature leaves a corpse with correct energy_value.

        Create two creatures: A at (2,2) with enough energy to strike, and B at (2,3) with energy 4.0.
        Set A.attack_damage = 5.0, A.attack_cost = 1.0, A.attack_bonus = 2.0.
        Monkey-patch A.decide() to always attack B.
        Monkey-patch B.decide() to always rest.
        Call world.step().
        Verify that B's energy goes from 4.0 → 4.0 − 5.0 = −1.0 (dead).
        Confirm that a new Food object appears at B's position with energy_value = 2 × 4.0 = 8.0.
        Confirm that A.energy changed by −1.0 (attack cost) + 2.0 (attack bonus) = +1.0 net.
        Confirm that B is removed from world.creatures.
        """
        # Create a world
        world = World(5, 5, food_spawn_rate=0.0)

        # Create attacker (A) and target (B)
        attacker = Creature(2.0, 2.0, size=1.0, energy=10.0, attack_damage=5.0, attack_cost=1.0, attack_bonus=2.0)
        target = Creature(2.0, 3.0, size=1.0, energy=4.0)

        # Add creatures to the world (target first so attacker moves after it in the step)
        world.add_creature(target)
        world.add_creature(attacker)

        # Print initial state
        print(f"Initial state:")
        print(f"  Attacker: x={attacker.x}, y={attacker.y}, energy={attacker.energy}, attack_damage={attacker.attack_damage}")
        print(f"  Target: x={target.x}, y={target.y}, energy={target.energy}")
        print(f"  Distance: {math.sqrt((attacker.x - target.x)**2 + (attacker.y - target.y)**2)}")
        print(f"  Attack range: {(attacker.radius + target.radius) * attacker.ATTACK_RANGE_FACTOR}")

        # Monkey-patch decide methods
        def always_attack_target(vision, on_food=False):
            # Print vision
            print(f"Vision for attacker:")
            for type_tag, obj, dist, angle in vision:
                print(f"  {type_tag}: dist={dist}, angle={angle}")

            # Find nearby creatures
            nearby_creatures = [(obj, dist, angle) for type_tag, obj, dist, angle in vision if type_tag == "creature"]

            if nearby_creatures:
                # Sort by distance (closest first)
                nearby_creatures.sort(key=lambda x: x[1])
                closest_creature, distance, angle = nearby_creatures[0]

                # Set intent for visualization
                attacker.intent = "ATTACK"
                attacker.intended_vector = (math.cos(angle) * attacker.velocity, 
                                           math.sin(angle) * attacker.velocity)

                print(f"Attacker decides to attack: {closest_creature}")
                return ("ATTACK", closest_creature)

            # If no creatures nearby, rest
            print("No creatures nearby, attacker decides to rest")
            return ("REST", None)

        def always_rest(vision, on_food=False):
            return ("REST", None)

        attacker.decide = always_attack_target
        target.decide = always_rest

        # Initial energy values
        attacker_initial_energy = attacker.energy
        target_initial_energy = target.energy

        # Call world.step()
        world.step()

        # Print state after step
        print(f"State after step:")
        print(f"  Attacker: x={attacker.x}, y={attacker.y}, energy={attacker.energy}")
        print(f"  Target: x={target.x}, y={target.y}, energy={target.energy}")
        print(f"  Target in world.creatures: {target in world.creatures}")
        print(f"  Number of foods: {len(world.foods)}")
        if world.foods:
            print(f"  Food: x={world.foods[0].x}, y={world.foods[0].y}, energy_value={world.foods[0].energy_value}")

        # Verify target is dead and removed from world.creatures
        self.assertNotIn(target, world.creatures)

        # Verify attacker's energy changed correctly
        # -1.0 (attack cost) + 2.0 (attack bonus) = +1.0 net
        self.assertAlmostEqual(attacker.energy, attacker_initial_energy + 1.0, places=5)

        # Verify a corpse was created at target's position
        self.assertEqual(len(world.foods), 1)
        corpse = world.foods[0]
        self.assertAlmostEqual(corpse.x, 2.0, places=5)
        self.assertAlmostEqual(corpse.y, 3.0, places=5)

        # Verify corpse has correct energy_value (2 * damage_dealt)
        # damage_dealt = min(target_initial_energy, attacker.attack_damage) = min(4.0, 5.0) = 4.0
        # energy_value = 2 * 4.0 = 8.0
        self.assertAlmostEqual(corpse.energy_value, 8.0, places=5)

        # Verify corpse has correct remaining_duration
        # The corpse duration should be 5 as set in Creature.apply_action
        self.assertEqual(corpse.remaining_duration, 5)

    def test_missed_attack(self):
        """
        Test that a missed attack costs the attacker energy but produces no corpse.

        Place a single creature A at (2,2) with energy 10.0. No target in attack range.
        Monkey-patch A.decide() to attack a non-existent target.
        Call world.step().
        Confirm that A.energy decreases by attack_cost (e.g. 1.0).
        Confirm that no new Food object is created.
        Confirm that A remains in world.creatures.
        """
        # Create a world
        world = World(5, 5, food_spawn_rate=0.0)

        # Create attacker
        attacker = Creature(2.0, 2.0, size=1.0, energy=10.0, attack_damage=5.0, attack_cost=1.0, attack_bonus=2.0)

        # Add creature to the world
        world.add_creature(attacker)

        # Create a target that's not in the world (for the attack to miss)
        target = Creature(10.0, 10.0, size=1.0, energy=4.0)  # Far away, not in world

        # Monkey-patch decide method to always attack the non-existent target
        def always_attack_missing_target(vision, on_food=False):
            # Set intent for visualization
            attacker.intent = "ATTACK"
            attacker.intended_vector = (1.0, 0.0)  # Arbitrary direction

            return ("ATTACK", target)

        attacker.decide = always_attack_missing_target

        # Initial energy value
        attacker_initial_energy = attacker.energy

        # Call world.step()
        world.step()

        # Verify attacker's energy decreased by attack_cost
        self.assertAlmostEqual(attacker.energy, attacker_initial_energy - attacker.attack_cost, places=5)

        # Verify no corpse was created
        self.assertEqual(len(world.foods), 0)

        # Verify attacker remains in world.creatures
        self.assertIn(attacker, world.creatures)

    def test_corpse_decay(self):
        """
        Test that a corpse decays after X steps.

        Create world 5×5, two creatures A and B adjacent. Let B have energy small enough to die in one hit.
        Let A kill B in step 1.
        Confirm a corpse Food appears at B's position with remaining_duration = D.
        Advance D additional steps by calling world.step() in a loop without any creatures attacking that corpse.
        After exactly D calls to world.step(), assert that the corpse no longer exists in world.foods.
        If you call one more world.step(), confirm the corpse is still gone and does not re-appear.
        """
        # Create a world
        world = World(5, 5, food_spawn_rate=0.0)

        # Create attacker (A) and target (B)
        attacker = Creature(2.0, 2.0, size=1.0, energy=10.0, attack_damage=5.0, attack_cost=1.0, attack_bonus=2.0)
        target = Creature(2.0, 3.0, size=1.0, energy=4.0)

        # Add creatures to the world (target first so attacker moves after it in the step)
        world.add_creature(target)
        world.add_creature(attacker)

        # Monkey-patch decide methods
        def always_attack_target(vision, on_food=False):
            # Find nearby creatures
            nearby_creatures = [(obj, dist, angle) for type_tag, obj, dist, angle in vision if type_tag == "creature"]

            if nearby_creatures:
                # Sort by distance (closest first)
                nearby_creatures.sort(key=lambda x: x[1])
                closest_creature, distance, angle = nearby_creatures[0]

                # Set intent for visualization
                attacker.intent = "ATTACK"
                attacker.intended_vector = (math.cos(angle) * attacker.velocity, 
                                           math.sin(angle) * attacker.velocity)

                return ("ATTACK", closest_creature)

            # If no creatures nearby, rest
            return ("REST", None)

        def always_rest(vision, on_food=False):
            return ("REST", None)

        attacker.decide = always_attack_target
        target.decide = always_rest

        # Call world.step() to kill the target
        world.step()

        # Verify a corpse was created
        self.assertEqual(len(world.foods), 1)
        corpse = world.foods[0]

        # Get the corpse's remaining_duration
        # The corpse duration should be 5 as set in Creature.apply_action
        corpse_duration = corpse.remaining_duration
        self.assertEqual(corpse_duration, 5)

        # Monkey-patch attacker to rest so it doesn't attack the corpse
        attacker.decide = always_rest

        # Advance corpse_duration - 1 steps
        for _ in range(corpse_duration - 1):
            world.step()
            # Corpse should still exist
            self.assertEqual(len(world.foods), 1)

        # Advance one more step (total = corpse_duration)
        world.step()

        # Verify corpse no longer exists
        self.assertEqual(len(world.foods), 0)

        # Advance one more step
        world.step()

        # Verify corpse is still gone
        self.assertEqual(len(world.foods), 0)

    def test_creature_can_eat_corpse(self):
        """
        Test that a creature can eat a corpse and gain correct energy.

        Create a world and two creatures A (at (2,2), energy 10.0) and B (at (2,3), energy 4.0).
        A attacks and kills B in step 1; corpse appears with energy_value = 8.0. A's energy is now 11.0.
        In step 2, monkey-patch A's decide() so that it always returns ("MOVE", (0.0, +A.velocity)) (i.e. move north).
        Call world.step().
        Confirm that A's new position is (2.0, 3.0) (the corpse's position).

        With incremental eating, the creature needs to take multiple bites to consume the entire corpse.
        After moving onto the corpse, the creature will call EAT_AT_CURRENT to take bites.
        Each bite gives +1 energy, and the corpse is removed when its remaining_energy reaches 0.

        Confirm the corpse is removed from world.foods after being fully consumed.
        """
        # Create a world
        world = World(5, 5, food_spawn_rate=0.0)

        # Create attacker (A) and target (B)
        attacker = Creature(2.0, 2.0, size=1.0, energy=10.0, attack_damage=5.0, attack_cost=1.0, attack_bonus=2.0)
        target = Creature(2.0, 3.0, size=1.0, energy=4.0)

        # Add creatures to the world (target first so attacker moves after it in the step)
        world.add_creature(target)
        world.add_creature(attacker)

        # Monkey-patch decide methods for step 1
        def always_attack_target(vision, on_food=False):
            # Find nearby creatures
            nearby_creatures = [(obj, dist, angle) for type_tag, obj, dist, angle in vision if type_tag == "creature"]

            if nearby_creatures:
                # Sort by distance (closest first)
                nearby_creatures.sort(key=lambda x: x[1])
                closest_creature, distance, angle = nearby_creatures[0]

                # Set intent for visualization
                attacker.intent = "ATTACK"
                attacker.intended_vector = (math.cos(angle) * attacker.velocity, 
                                           math.sin(angle) * attacker.velocity)

                return ("ATTACK", closest_creature)

            # If no creatures nearby, rest
            return ("REST", None)

        def always_rest(vision, on_food=False):
            return ("REST", None)

        attacker.decide = always_attack_target
        target.decide = always_rest

        # Call world.step() to kill the target
        world.step()

        # Verify attacker's energy is now 11.0 (10.0 - 1.0 + 2.0)
        self.assertAlmostEqual(attacker.energy, 11.0, places=5)

        # Verify a corpse was created with energy_value = 8.0
        self.assertEqual(len(world.foods), 1)
        corpse = world.foods[0]
        self.assertAlmostEqual(corpse.energy_value, 8.0, places=5)
        self.assertAlmostEqual(corpse.remaining_energy, 8.0, places=5)

        # Monkey-patch attacker to move north in step 2
        def move_north(vision, on_food=False):
            # Set intent for visualization
            attacker.intent = "GO_TO_FOOD"
            attacker.intended_vector = (0.0, attacker.velocity)

            return ("MOVE", (0.0, attacker.velocity))

        attacker.decide = move_north

        # Call world.step() again to move onto the corpse
        world.step()

        # Verify attacker's new position is (2.0, 3.0)
        self.assertAlmostEqual(attacker.x, 2.0, places=5)
        self.assertAlmostEqual(attacker.y, 3.0, places=5)

        # Verify attacker's energy decreased by movement cost
        # Starting from 11.0, new energy should be 10.0
        self.assertAlmostEqual(attacker.energy, 10.0, places=5)

        # Now monkey-patch attacker to eat at current position
        def eat_at_current(vision, on_food=False):
            # Set intent for visualization
            attacker.intent = "GO_TO_FOOD"
            attacker.intended_vector = (0.0, 0.0)

            return ("EAT_AT_CURRENT", None)

        attacker.decide = eat_at_current

        # Take bites until the corpse is fully consumed
        # We'll need to take 8 bites total, but we need to check if the corpse exists after each bite
        for i in range(8):
            # If the corpse is gone, we're done
            if len(world.foods) == 0:
                break

            # Call world.step() to eat one bite
            world.step()

            # Verify attacker's energy increased by 1 each step
            self.assertAlmostEqual(attacker.energy, 10.0 + (i + 1), places=5)

        # Verify the corpse is gone after all bites
        self.assertEqual(len(world.foods), 0)

        # Verify attacker's final energy is 18.0 (10.0 + 8.0)
        self.assertAlmostEqual(attacker.energy, 18.0, places=5)

    def test_mixed_interactions(self):
        """
        Test mixed interactions between food and predators.

        Create a world with one spawned food at (4,4).
        Place predator A at (2,2), herbivore B at (3,2).
        In step N, check that B will either go for the spawned food or get attacked by A 
        based on the decision logic's priority ("ATTACK" supersedes "EAT").
        Confirm that if B is within attack range of A, it gets attacked first.
        """
        # Create a world
        world = World(5, 5, food_spawn_rate=0.0)

        # Create a spawned food at (4,4)
        spawned_food = Food(x=4.0, y=4.0, size=1.0, energy_value=2.0, remaining_duration=-1)
        world.foods.append(spawned_food)

        # Create predator (A) and herbivore (B)
        predator = Creature(2.0, 2.0, size=1.0, energy=10.0, attack_damage=5.0, attack_cost=1.0, attack_bonus=2.0)
        herbivore = Creature(3.0, 2.0, size=1.0, energy=4.0)  # Reduced energy so it dies in one hit

        # Add creatures to the world (target first so attacker moves after it in the step)
        world.add_creature(herbivore)
        world.add_creature(predator)

        # Monkey-patch predator to always attack nearby creatures
        def predator_attack_nearby(vision, on_food=False):
            # Find nearby creatures
            nearby_creatures = [(obj, dist, angle) for type_tag, obj, dist, angle in vision if type_tag == "creature"]

            if nearby_creatures:
                # Sort by distance (closest first)
                nearby_creatures.sort(key=lambda x: x[1])
                closest_creature, distance, angle = nearby_creatures[0]

                # Calculate attack range based on radii
                attack_range = (predator.radius + closest_creature.radius) * predator.ATTACK_RANGE_FACTOR

                # If within attack range, attack
                if distance <= attack_range:
                    # Set intent for visualization
                    predator.intent = "ATTACK"
                    predator.intended_vector = (math.cos(angle) * predator.velocity, 
                                               math.sin(angle) * predator.velocity)

                    return ("ATTACK", closest_creature)

            # If no creatures in attack range, rest
            return ("REST", None)

        # Monkey-patch herbivore to go for food if not in danger
        def herbivore_eat_or_flee(vision, on_food=False):
            # Find nearby creatures (potential threats)
            nearby_creatures = [(obj, dist, angle) for type_tag, obj, dist, angle in vision if type_tag == "creature"]

            # Find nearby food
            nearby_food = [(obj, dist, angle) for type_tag, obj, dist, angle in vision if type_tag == "food"]

            # If there are creatures nearby, flee
            if nearby_creatures:
                # Sort by distance (closest first)
                nearby_creatures.sort(key=lambda x: x[1])
                closest_creature, distance, angle = nearby_creatures[0]

                # Calculate flee direction (opposite of creature)
                flee_angle = angle + math.pi

                # Set intent for visualization
                herbivore.intent = "RUN_AWAY"
                herbivore.intended_vector = (math.cos(flee_angle) * herbivore.velocity, 
                                            math.sin(flee_angle) * herbivore.velocity)

                return ("FLEE", (math.cos(flee_angle) * herbivore.velocity, 
                                math.sin(flee_angle) * herbivore.velocity))

            # If there's food nearby, go for it
            if nearby_food:
                # Sort by distance (closest first)
                nearby_food.sort(key=lambda x: x[1])
                closest_food, distance, angle = nearby_food[0]

                # Calculate eat range based on radii
                eat_range = (herbivore.radius + closest_food.radius) * herbivore.EAT_RANGE_FACTOR

                # If within eat range, eat
                if distance <= eat_range:
                    # Set intent for visualization
                    herbivore.intent = "GO_TO_FOOD"
                    herbivore.intended_vector = (math.cos(angle) * herbivore.velocity * 0.75, 
                                               math.sin(angle) * herbivore.velocity * 0.75)

                    return ("EAT", closest_food)

                # If not in eat range, move toward the food
                # Set intent for visualization
                herbivore.intent = "GO_TO_FOOD"
                herbivore.intended_vector = (math.cos(angle) * herbivore.velocity * 0.75, 
                                           math.sin(angle) * herbivore.velocity * 0.75)

                return ("MOVE", (math.cos(angle) * herbivore.velocity * 0.75, 
                                math.sin(angle) * herbivore.velocity * 0.75))

            # If nothing interesting, rest
            return ("REST", None)

        predator.decide = predator_attack_nearby
        herbivore.decide = herbivore_eat_or_flee

        # Call world.step()
        world.step()

        # Verify herbivore is dead (attacked by predator)
        self.assertNotIn(herbivore, world.creatures)

        # Verify a corpse was created at herbivore's position
        self.assertEqual(len(world.foods), 2)  # Original food + corpse

        # Find the corpse (the one at herbivore's position)
        corpse = None
        for food in world.foods:
            if math.isclose(food.x, 3.0, abs_tol=0.01) and math.isclose(food.y, 2.0, abs_tol=0.01):
                corpse = food
                break

        self.assertIsNotNone(corpse)

        # Verify corpse has correct energy_value
        expected_energy = 2.0 * min(4.0, predator.attack_damage)  # 2 * min(4.0, 5.0) = 8.0
        self.assertAlmostEqual(corpse.energy_value, expected_energy, places=5)

if __name__ == "__main__":
    unittest.main()
