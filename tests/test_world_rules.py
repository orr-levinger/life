import math
import unittest
from unittest import mock

import numpy as np

from src.food import Food
from src.learner import LEARNING_RATE, Learner, NetWeights
from src.stages import BABY_SHARE, EAT_MISS_COST, MAX_AGE, MAX_CREATURES, STAGES
from tests.helpers import GARDEN_NO_FOOD, NO_FOOD, NO_FOOD_ATTACKS, add, fixed_brain, make_world, populate
from tests.test_learner import live, same

STAY = [0, 0, -1, -1]
EAT = [0, 0, 1, -1]


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
    def test_eat_action_takes_a_bite(self):
        w = make_world()
        c = add(w, 10, 10, policy="brain", brain=fixed_brain(EAT))
        f = food(10, 10)
        w.foods.append(f)
        w.step()
        self.assertAlmostEqual(c.energy, 51.0 - NO_FOOD.step_cost)
        self.assertAlmostEqual(f.energy, 9.0)
        self.assertAlmostEqual(c.food_eaten, 1.0)
        self.assertEqual((w.total_bites, c.last_action), (1, "EAT"))

    def test_no_bite_without_choosing_to_eat(self):
        w = make_world()
        c = add(w, 10, 10, policy="brain", brain=fixed_brain(STAY))
        f = food(10, 10)
        w.foods.append(f)
        w.step()
        self.assertEqual((f.energy, c.food_eaten, w.total_bites), (10.0, 0.0, 0))

    def test_eat_miss_costs_energy(self):
        w = make_world()
        c = add(w, 10, 10, policy="brain", brain=fixed_brain(EAT))
        w.step()
        self.assertAlmostEqual(c.energy, 50.0 - EAT_MISS_COST - NO_FOOD.step_cost)
        self.assertEqual(c.last_action, "EAT_MISS")

    def test_second_eater_misses_last_bite(self):
        w = make_world()
        eaters = [add(w, 10, 10, policy="brain", brain=fixed_brain(EAT)),
                  add(w, 10.2, 10, policy="brain", brain=fixed_brain(EAT))]
        f = food(10.1, 10, energy=1.0)
        w.foods.append(f)
        w.step()
        self.assertNotIn(f, w.foods)
        self.assertEqual(sorted(c.food_eaten for c in eaters), [0.0, 1.0])
        self.assertEqual(sorted(c.last_action for c in eaters), ["EAT", "EAT_MISS"])
        self.assertEqual(w.total_bites, 1)

    def test_energy_capped_at_max(self):
        w = make_world(evolving=False)
        c = add(w, 10, 10, energy=99.99, policy="brain", brain=fixed_brain(EAT))
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
    def world(self, stage=NO_FOOD, inherit="learned"):
        learner = Learner(MAX_CREATURES, 0)
        return make_world(stage, learner=learner, species="learner", inherit=inherit), learner

    def born(self, w, learner, x, y, energy=50.0):
        slot = learner.allocate(NetWeights.random(w.np_rng), LEARNING_RATE)
        return add(w, x, y, energy=energy, policy="learner", slot=slot)

    def crowd(self, w):
        for i in range(10):  # ten wanderers keep the soup away
            add(w, 3 + i * 4, 25)

    def test_newcomer_gets_a_slot_and_a_random_network(self):
        w, learner = self.world()
        first, second = w.add_newcomer(), w.add_newcomer()
        self.assertEqual((first.policy, first.slot, second.slot), ("learner", 0, 1))
        self.assertFalse(same(learner.current(0), learner.current(1)))
        self.assertAlmostEqual(learner.lr(0), LEARNING_RATE)

    def test_learner_creatures_act_and_record(self):
        w, learner = self.world()
        self.crowd(w)
        creatures = [self.born(w, learner, 10, 10), self.born(w, learner, 30, 15)]
        before = {c.slot: c.energy for c in creatures}
        with mock.patch.object(learner, "record", wraps=learner.record) as record:
            w.step()
        calls = {call.args[0]: call.args for call in record.call_args_list}
        self.assertEqual(sorted(calls), [0, 1])
        for creature in creatures:
            _, reward, done = calls[creature.slot]
            self.assertAlmostEqual(reward, creature.energy - before[creature.slot])
            self.assertFalse(done)

    def test_dead_learner_frees_its_slot(self):
        w, learner = self.world()
        self.crowd(w)
        dying = self.born(w, learner, 10, 10, energy=0.01)
        with mock.patch.object(learner, "record", wraps=learner.record) as record:
            w.step()
        record.assert_called_once()
        self.assertTrue(record.call_args.args[2])  # its last step is marked as the end of its life
        self.assertNotIn(dying, w.creatures)
        self.assertEqual(learner.allocate(NetWeights.random(w.np_rng), LEARNING_RATE), dying.slot)

    def parent_that_has_learned(self, inherit):
        w, learner = self.world(GARDEN_NO_FOOD, inherit)
        parent = self.born(w, learner, 10, 10, energy=80.0)
        live(learner, parent.slot, 65)
        learner.update()
        self.assertFalse(same(learner.current(parent.slot), learner.inborn(parent.slot)))
        learned, born_with = learner.current(parent.slot), learner.inborn(parent.slot)
        w.step()
        baby = w.creatures[1]
        self.assertEqual((baby.policy, baby.parent_id, baby.generation), ("learner", parent.id, 1))
        return learner, parent, baby, learned, born_with

    def test_learner_baby_inherits_learned_brain(self):
        learner, parent, baby, learned, born_with = self.parent_that_has_learned("learned")
        genes = learner.inborn(baby.slot)
        self.assertGreater(np.mean(genes.w1 == learned.w1), 0.7)
        self.assertLess(np.mean(np.isclose(genes.w1, born_with.w1)), 0.3)
        self.assertNotEqual(learner.lr(baby.slot), learner.lr(parent.slot))
        self.assertLess(abs(math.log(learner.lr(baby.slot) / learner.lr(parent.slot))), 1.5)

    def test_learner_baby_inherits_born_brain_when_inborn(self):
        learner, _, baby, learned, born_with = self.parent_that_has_learned("inborn")
        genes = learner.inborn(baby.slot)
        self.assertGreater(np.mean(np.isclose(genes.w1, born_with.w1)), 0.7)
        self.assertLess(np.mean(genes.w1 == learned.w1), 0.3)

    def test_birth_is_not_a_negative_reward(self):
        w, learner = self.world(GARDEN_NO_FOOD)
        parent = self.born(w, learner, 10, 10, energy=80.0)
        with mock.patch.object(learner, "record", wraps=learner.record) as record:
            w.step()
        self.assertEqual(len(w.creatures), 2)
        self.assertLess(parent.energy, 55.0)  # it gave its baby 40 % of its energy
        _, reward, _ = record.call_args_list[0].args
        self.assertGreater(reward, -2.0)  # but its reward only shows the cost of the step itself

    def test_garden_learner_keeps_learning_at_zero_energy(self):
        w, learner = self.world(GARDEN_NO_FOOD)
        empty = self.born(w, learner, 10, 10, energy=0.0)
        with mock.patch.object(learner, "record", wraps=learner.record) as record:
            for _ in range(3):
                w.step()
        self.assertEqual(w.creatures, [empty])
        self.assertEqual([(call.args[0], call.args[2]) for call in record.call_args_list], [(empty.slot, False)] * 3)


if __name__ == "__main__":
    unittest.main()
