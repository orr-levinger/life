import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

from src.food import Food
from src.learner import LEARNING_RATE, Learner, NetWeights
from src.stages import BABY_SHARE, MAX_AGE, MAX_CREATURES, STAGES
from tests.helpers import GARDEN_NO_FOOD, NO_FOOD, NO_FOOD_ATTACKS, add, fixed_brain, make_world, populate

STAY = [0, 0, -1]


def food(x: float, y: float, energy: float = 10.0) -> Food:
    return Food(x, y, remaining_duration=-1, energy=energy)


class TestCosts(unittest.TestCase):
    def test_step_cost(self):
        w = make_world()
        c = add(w, 10, 10, policy="brain", brain=fixed_brain(STAY))
        w.step()
        self.assertAlmostEqual(c.energy, 50.0 - NO_FOOD.step_cost)

    def test_move_cost_and_speed_cap(self):
        w = make_world()
        c = add(w, 10, 10, size=1.0)
        c.apply_action(("MOVE", (3.0, 4.0)), w)
        self.assertAlmostEqual(c.x, 10.6)
        self.assertAlmostEqual(c.y, 10.8)
        self.assertAlmostEqual(c.energy, 50.0 - 1.0 * NO_FOOD.move_cost - NO_FOOD.step_cost)

    def test_move_clamped_in_corner(self):
        w = make_world()
        c = add(w, 0.3, 0.3, size=1.0)
        c.apply_action(("MOVE", (-1.0, -1.0)), w)
        self.assertAlmostEqual(c.x, c.radius)
        self.assertAlmostEqual(c.y, c.radius)


class TestEating(unittest.TestCase):
    def test_eating_on_contact(self):
        w = make_world()
        c = add(w, 10, 10, policy="brain", brain=fixed_brain(STAY))
        f = food(10, 10)
        w.foods.append(f)
        w.step()
        self.assertAlmostEqual(c.energy, 51.0 - NO_FOOD.step_cost)
        self.assertAlmostEqual(f.energy, 9.0)
        self.assertAlmostEqual(c.food_eaten, 1.0)
        self.assertEqual(w.total_bites, 1)

    def test_last_bite_goes_to_one(self):
        w = make_world()
        eaters = [add(w, 10, 10, policy="brain", brain=fixed_brain(STAY)),
                  add(w, 10.2, 10, policy="brain", brain=fixed_brain(STAY))]
        f = food(10.1, 10, energy=1.0)
        w.foods.append(f)
        w.step()
        self.assertNotIn(f, w.foods)
        self.assertEqual(sorted(c.food_eaten for c in eaters), [0.0, 1.0])
        self.assertEqual(w.total_bites, 1)

    def test_energy_capped_at_max(self):
        w = make_world(evolving=False)
        c = add(w, 10, 10, energy=99.99, policy="brain", brain=fixed_brain(STAY))
        w.foods.append(food(10, 10))
        w.step()
        self.assertEqual(c.energy, c.max_energy)


class TestHunting(unittest.TestCase):
    def test_hunter_gains_eighty_percent(self):
        w = make_world(NO_FOOD_ATTACKS)
        hunter = add(w, 10, 10, size=2.0, energy=100.0)
        prey = add(w, 11, 10, size=1.0, energy=50.0)
        hunter.apply_action(("ATTACK", prey), w)
        self.assertAlmostEqual(prey.energy, 40.0)
        self.assertAlmostEqual(hunter.energy, 100.0 + 0.8 * 10.0 - 2.0 - NO_FOOD_ATTACKS.step_cost)

    def test_kill_leaves_corpse_and_counts_attack_death(self):
        w = make_world(NO_FOOD_ATTACKS)
        hunter = add(w, 10, 10, size=2.0, energy=100.0)
        prey = add(w, 11, 10, size=1.0, energy=5.0)
        hunter.apply_action(("ATTACK", prey), w)
        self.assertNotIn(prey, w.creatures)
        self.assertEqual(w.deaths["attack"], 1)
        self.assertIn(prey, w.dead_this_step)
        corpse = w.foods[-1]
        self.assertEqual((corpse.x, corpse.y, corpse.energy, corpse.remaining_duration), (11, 10, 8.0, 5))

    def test_dead_target_is_a_miss(self):
        w = make_world(NO_FOOD_ATTACKS)
        hunter = add(w, 10, 10, size=2.0, energy=100.0)
        prey = add(w, 11, 10, size=1.0, energy=50.0)
        w.creatures.remove(prey)
        hunter.apply_action(("ATTACK", prey), w)
        self.assertEqual(hunter.last_action, "ATTACK_MISS")
        self.assertEqual(prey.energy, 50.0)

    def test_family_cannot_be_hurt(self):
        w = make_world(NO_FOOD_ATTACKS)
        parent = add(w, 10, 10, size=2.0, energy=100.0)
        child = add(w, 11, 10, size=1.0, energy=50.0, parent_id=parent.id)
        parent.apply_action(("ATTACK", child), w)
        self.assertEqual(child.energy, 50.0)
        self.assertEqual(parent.last_action, "ATTACK_MISS")


class TestLifeAndDeath(unittest.TestCase):
    def test_garden_never_kills(self):
        w = make_world(GARDEN_NO_FOOD)
        c = add(w, 10, 10, energy=0.01, policy="brain", brain=fixed_brain(STAY))
        c.age = MAX_AGE + 10
        for _ in range(50):
            w.step()
        self.assertEqual(w.creatures, [c])
        self.assertEqual(c.energy, 0.0)
        self.assertEqual(sum(w.deaths.values()), 0)

    def test_hunger_kills_in_mortal_stage(self):
        w = make_world(NO_FOOD)
        c = add(w, 10, 10, energy=0.01, policy="brain", brain=fixed_brain(STAY))
        w.step()
        self.assertNotIn(c, w.creatures)
        self.assertEqual(w.deaths["hunger"], 1)

    def test_old_age_kills_in_mortal_stage(self):
        w = make_world(NO_FOOD)
        c = add(w, 10, 10, policy="brain", brain=fixed_brain(STAY))
        c.age = MAX_AGE - 1
        w.step()
        self.assertNotIn(c, w.creatures)
        self.assertEqual(w.deaths["age"], 1)

    def test_birth_at_seventy_percent(self):
        w = make_world(GARDEN_NO_FOOD)
        parent_brain = fixed_brain(STAY)
        parent = add(w, 10, 10, size=1.0, energy=80.0, policy="brain", brain=parent_brain)
        w.step()
        self.assertEqual(len(w.creatures), 2)
        baby = w.creatures[1]
        before_birth = 80.0 - GARDEN_NO_FOOD.step_cost
        self.assertAlmostEqual(baby.energy, BABY_SHARE * before_birth)
        self.assertAlmostEqual(parent.energy, (1 - BABY_SHARE) * before_birth)
        self.assertEqual((baby.parent_id, baby.generation, baby.policy), (parent.id, 1, "brain"))
        self.assertTrue(0.8 <= baby.size <= 1.2)
        self.assertLessEqual(abs(baby.x - parent.x), 1.0)
        changed = np.mean(baby.brain.w1 != parent_brain.w1)
        self.assertTrue(0.0 < changed < 0.3)  # a mutated copy of the parent's brain
        self.assertEqual(w.births, 1)

    def test_no_birth_below_seventy_percent(self):
        w = make_world(GARDEN_NO_FOOD)
        add(w, 10, 10, size=1.0, energy=60.0, policy="brain", brain=fixed_brain(STAY))
        w.step()
        self.assertEqual(len(w.creatures), 1)

    def test_births_pause_at_limit(self):
        w = make_world(GARDEN_NO_FOOD, species="random")
        for i in range(MAX_CREATURES):
            add(w, 1 + (i % 10) * 4.5, 1 + (i // 10) * 3.4, energy=90.0)
        w.step()
        self.assertEqual((len(w.creatures), w.births), (MAX_CREATURES, 0))
        w.creatures.pop()
        w.step()
        self.assertEqual((len(w.creatures), w.births), (MAX_CREATURES, 1))

    def test_soup_refills_a_dying_world(self):
        w = make_world(NO_FOOD)
        for i in range(3):
            add(w, 10 + i * 5, 10, policy="brain", brain=fixed_brain(STAY))
        w.step()
        self.assertEqual((len(w.creatures), w.newcomers), (4, 1))
        self.assertEqual(w.creatures[-1].policy, "brain")
        self.assertIsNotNone(w.creatures[-1].brain)

    def test_no_soup_in_garden(self):
        w = make_world(GARDEN_NO_FOOD)
        for i in range(3):
            add(w, 10 + i * 5, 10, policy="brain", brain=fixed_brain(STAY))
        w.step()
        self.assertEqual((len(w.creatures), w.newcomers), (3, 0))

    def test_newcomer_is_of_the_world_species(self):
        self.assertEqual(make_world(species="random").add_newcomer().policy, "random")
        newcomer = make_world(species="brain").add_newcomer()
        self.assertEqual(newcomer.policy, "brain")
        self.assertIsNotNone(newcomer.brain)

    def test_exam_world_has_no_births_or_deaths(self):
        w = make_world(NO_FOOD, evolving=False)
        rich = add(w, 10, 10, energy=90.0, policy="brain", brain=fixed_brain(STAY))
        poor = add(w, 30, 20, energy=0.01, policy="brain", brain=fixed_brain(STAY))
        poor.age = MAX_AGE
        for _ in range(10):
            w.step()
        self.assertEqual(w.creatures, [rich, poor])
        self.assertEqual((w.births, w.newcomers, sum(w.deaths.values())), (0, 0, 0))

    def test_same_seed_same_world(self):
        worlds = [make_world(STAGES[1], seed=5), make_world(STAGES[1], seed=5)]
        for w in worlds:
            populate(w)
            for _ in range(200):
                w.step()
        positions = [[(c.id, c.x, c.y) for c in w.creatures] for w in worlds]
        self.assertEqual(positions[0], positions[1])

    def test_brain_creatures_never_call_rules(self):
        w = make_world(STAGES[1], seed=2)
        populate(w)
        with mock.patch("src.policies.rules_action", side_effect=AssertionError("rules called for a brain")):
            for _ in range(500):
                w.step()


class TestLearnersInWorld(unittest.TestCase):
    def learner(self, slots: int) -> Learner:
        learner = Learner(slots, 0)
        rng = np.random.default_rng(0)
        for slot in range(slots):
            learner.install(slot, NetWeights.random(rng), LEARNING_RATE)
        return learner

    def test_learner_creatures_act_and_record(self):
        learner = self.learner(2)
        w = make_world(learner=learner)
        creatures = [add(w, 10, 10, policy="learner", slot=0), add(w, 30, 20, policy="learner", slot=1)]
        before = {c.slot: c.energy for c in creatures}
        with mock.patch.object(learner, "record", wraps=learner.record) as record:
            w.step()
        calls = {call.args[0]: call.args for call in record.call_args_list}
        self.assertEqual(sorted(calls), [0, 1])
        for creature in creatures:
            _, reward, done = calls[creature.slot]
            self.assertAlmostEqual(reward, creature.energy - before[creature.slot])
            self.assertFalse(done)

    def test_death_marks_done(self):
        learner = self.learner(1)
        w = make_world(replace(NO_FOOD, name="one dying learner"), learner=learner)
        for i in range(10):  # ten others keep the soup away
            add(w, 3 + i * 4, 25)
        dying = add(w, 10, 10, energy=0.01, policy="learner", slot=0)
        with mock.patch.object(learner, "record", wraps=learner.record) as record:
            w.step()
        record.assert_called_once()
        self.assertTrue(record.call_args.args[2])
        self.assertNotIn(dying, w.creatures)


if __name__ == "__main__":
    unittest.main()
