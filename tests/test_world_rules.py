import math
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

from src.food import Food
from src.learner import LEARNING_RATE, Learner, NetWeights
from src.settings import BABY_SHARE, EAT_MISS_COST, MAX_CREATURES, START
from tests.helpers import NO_FOOD, NO_FOOD_ATTACKS, add, fixed_brain, make_world, populate
from tests.test_learner import live, same

STAY = [0, 0, -1, -1]
EAT = [0, 0, 1, -1]


def food(x: float, y: float, energy: float = 10.0) -> Food:
    return Food(x, y, remaining_duration=-1, energy=energy)


def crowd(w) -> None:
    """Ten wanderers with too little energy for a baby: they keep the primordial soup away."""
    for i in range(10):
        add(w, 3 + i * 4, 25)


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
        self.assertAlmostEqual(c.x, 10.15)  # at most a quarter of a unit a step for size 1
        self.assertAlmostEqual(c.y, 10.2)
        self.assertAlmostEqual(c.energy, 50.0 - 0.25 * NO_FOOD.move_cost - NO_FOOD.step_cost)

    def test_speed_is_a_quarter_of_one_over_size(self):
        w = make_world()
        self.assertEqual([add(w, 5 + 10 * i, 10, size=size).velocity for i, size in enumerate((0.5, 1.0, 2.0))],
                         [0.5, 0.25, 0.125])

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

    def test_world_counts_energy_earned_and_spent(self):
        w = make_world()
        add(w, 10, 10, policy="brain", brain=fixed_brain(EAT))
        add(w, 30, 20, policy="brain", brain=fixed_brain(EAT))  # eats at nothing
        w.foods.append(food(10, 10))
        w.step()
        self.assertAlmostEqual(w.energy_earned, 1.0)
        self.assertAlmostEqual(w.energy_spent, EAT_MISS_COST + 2 * NO_FOOD.step_cost)
        mover = add(w, 20, 5, size=1.0)
        mover.apply_action(("MOVE", (3.0, 4.0)), w)
        self.assertAlmostEqual(w.energy_spent, EAT_MISS_COST + 3 * NO_FOOD.step_cost + 0.25 * NO_FOOD.move_cost)  # a quarter unit at most

    def test_each_creature_counts_its_own_energy(self):
        w = make_world()
        eater = add(w, 10, 10, policy="brain", brain=fixed_brain(EAT))
        misser = add(w, 30, 20, policy="brain", brain=fixed_brain(EAT))
        w.foods.append(food(10, 10))
        for _ in range(3):
            w.step()
        mover = add(w, 20, 5, size=1.0)
        mover.apply_action(("MOVE", (3.0, 4.0)), w)
        self.assertAlmostEqual(eater.earned, 3.0)
        self.assertAlmostEqual(misser.earned, 0.0)
        self.assertAlmostEqual(misser.spent, 3 * (EAT_MISS_COST + NO_FOOD.step_cost))
        self.assertAlmostEqual(sum(c.earned for c in w.creatures), w.energy_earned)
        self.assertAlmostEqual(sum(c.spent for c in w.creatures), w.energy_spent)

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
        self.assertAlmostEqual(w.energy_earned, 0.8 * 10.0)
        self.assertAlmostEqual(w.energy_spent, 2.0 + NO_FOOD_ATTACKS.step_cost)

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
    def test_energy_out_is_death_from_the_start(self):
        w = make_world(START)
        crowd(w)
        c = add(w, 10, 10, energy=0.01, policy="brain", brain=fixed_brain(STAY))
        w.step()
        self.assertNotIn(c, w.creatures)
        self.assertEqual((w.deaths["hunger"], w.deaths["age"]), (1, 0))

    def test_hungry_elder_dies_of_old_age(self):
        w = make_world()
        crowd(w)
        c = add(w, 10, 10, energy=29.0, policy="brain", brain=fixed_brain(STAY))  # below 30 % of its maximum
        c.lifespan, c.age = 100, 99
        w.step()
        self.assertNotIn(c, w.creatures)
        self.assertEqual((w.deaths["age"], w.deaths["hunger"]), (1, 0))

    def test_well_fed_elder_lives_on(self):
        w = make_world()
        crowd(w)
        c = add(w, 10, 10, energy=31.0, policy="brain", brain=fixed_brain(STAY))  # above 30 % of its maximum
        c.lifespan, c.age = 100, 99
        for _ in range(10):
            w.step()
        self.assertIn(c, w.creatures)
        self.assertEqual(sum(w.deaths.values()), 0)
        for _ in range(20):  # living costs energy: once it is poorly fed, old age takes it
            w.step()
        self.assertNotIn(c, w.creatures)
        self.assertEqual((w.deaths["age"], w.deaths["hunger"]), (1, 0))

    def test_every_creature_has_its_own_lifespan(self):
        w = make_world(START, species="random")
        populate(w, 30)
        spans = [c.lifespan for c in w.creatures]
        shortest, longest = START.lifespan
        self.assertTrue(all(shortest <= span <= longest for span in spans))
        self.assertGreater(len(set(spans)), 20)
        for creature in w.creatures:
            creature.energy = 90.0
        w.step()
        babies = [c for c in w.creatures if c.generation == 1]
        self.assertTrue(babies)
        self.assertTrue(all(shortest <= baby.lifespan <= longest for baby in babies))

    def test_lifespans_follow_the_settings_in_force(self):
        w = make_world(START, species="random")
        w.set_settings(replace(START, lifespan=(100, 200)), 1)
        populate(w, 30)
        self.assertTrue(all(100 <= c.lifespan <= 200 for c in w.creatures))

    def test_a_creature_dies_at_its_own_lifespan(self):
        w = make_world()
        crowd(w)
        young = add(w, 10, 10, energy=20.0, policy="brain", brain=fixed_brain(STAY))
        old = add(w, 20, 10, energy=20.0, policy="brain", brain=fixed_brain(STAY))
        young.lifespan, old.lifespan = 200, 50
        young.age = old.age = 49
        w.step()
        self.assertIn(young, w.creatures)
        self.assertNotIn(old, w.creatures)

    def test_a_death_of_old_age_makes_room_for_a_baby(self):
        w = make_world(species="random")
        for i in range(MAX_CREATURES):
            add(w, 1 + (i % 10) * 4.5, 1 + (i // 10) * 3.4, energy=90.0)
        elder = w.creatures[0]
        elder.lifespan, elder.age, elder.energy = 100, 99, 20.0
        w.step()
        self.assertEqual((len(w.creatures), w.deaths["age"], w.births), (MAX_CREATURES, 1, 1))

    def test_food_per_step_can_be_a_fraction(self):
        w = make_world(replace(START, food_per_step=0.4, max_food=10 ** 6), evolving=False)
        with mock.patch.object(type(w), "_fits", return_value=True):  # count every candidate, crowded or not
            for _ in range(2000):
                w.spawn_food()
        self.assertTrue(700 <= len(w.foods) <= 900)  # about 0.4 a step

    def test_birth_at_seventy_percent(self):
        w = make_world()
        crowd(w)
        parent_brain = fixed_brain(STAY)
        parent = add(w, 10, 10, size=1.0, energy=80.0, policy="brain", brain=parent_brain)
        w.step()
        self.assertEqual(len(w.creatures), 12)
        baby = w.creatures[-1]
        before_birth = 80.0 - NO_FOOD.step_cost
        self.assertAlmostEqual(baby.energy, BABY_SHARE * before_birth)
        self.assertAlmostEqual(parent.energy, (1 - BABY_SHARE) * before_birth)
        self.assertEqual((baby.parent_id, baby.generation, baby.policy), (parent.id, 1, "brain"))
        self.assertTrue(0.8 <= baby.size <= 1.2)
        self.assertLessEqual(abs(baby.x - parent.x), 1.0)
        changed = np.mean(baby.brain.w1 != parent_brain.w1)
        self.assertTrue(0.0 < changed < 0.3)  # a mutated copy of the parent's brain
        self.assertEqual(w.births, 1)

    def test_no_birth_below_seventy_percent(self):
        w = make_world()
        crowd(w)
        add(w, 10, 10, size=1.0, energy=60.0, policy="brain", brain=fixed_brain(STAY))
        w.step()
        self.assertEqual((len(w.creatures), w.births), (11, 0))

    def test_births_pause_at_limit(self):
        w = make_world(species="random")
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

    def test_newcomer_is_of_the_world_species(self):
        self.assertEqual(make_world(species="random").add_newcomer().policy, "random")
        newcomer = make_world(species="brain").add_newcomer()
        self.assertEqual(newcomer.policy, "brain")
        self.assertIsNotNone(newcomer.brain)

    def test_exam_world_has_no_births_or_deaths(self):
        w = make_world(NO_FOOD, evolving=False)
        rich = add(w, 10, 10, energy=90.0, policy="brain", brain=fixed_brain(STAY))
        poor = add(w, 30, 20, energy=0.01, policy="brain", brain=fixed_brain(STAY))
        poor.lifespan, poor.age = 100, 100
        for _ in range(10):
            w.step()
        self.assertEqual(w.creatures, [rich, poor])
        self.assertEqual((w.births, w.newcomers, sum(w.deaths.values())), (0, 0, 0))

    def test_same_seed_same_world(self):
        worlds = [make_world(START, seed=5), make_world(START, seed=5)]
        for w in worlds:
            populate(w)
            for _ in range(200):
                w.step()
        positions = [[(c.id, c.x, c.y) for c in w.creatures] for w in worlds]
        self.assertEqual(positions[0], positions[1])

    def test_brain_creatures_never_call_rules(self):
        w = make_world(START, seed=2)
        populate(w)
        with mock.patch("src.policies.rules_action", side_effect=AssertionError("rules called for a brain")):
            for _ in range(500):
                w.step()


class TestLearnersInWorld(unittest.TestCase):
    def world(self, inherit="learned"):
        learner = Learner(MAX_CREATURES, 0)
        return make_world(learner=learner, species="learner", inherit=inherit), learner

    def born(self, w, learner, x, y, energy=50.0):
        slot = learner.allocate(NetWeights.random(w.np_rng), LEARNING_RATE)
        return add(w, x, y, energy=energy, policy="learner", slot=slot)

    def test_newcomer_gets_a_slot_and_a_random_network(self):
        w, learner = self.world()
        first, second = w.add_newcomer(), w.add_newcomer()
        self.assertEqual((first.policy, first.slot, second.slot), ("learner", 0, 1))
        self.assertFalse(same(learner.current(0), learner.current(1)))
        self.assertAlmostEqual(learner.lr(0), LEARNING_RATE)

    def test_learner_creatures_act_and_record(self):
        w, learner = self.world()
        crowd(w)
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
        crowd(w)
        dying = self.born(w, learner, 10, 10, energy=0.01)
        with mock.patch.object(learner, "record", wraps=learner.record) as record:
            w.step()
        record.assert_called_once()
        self.assertTrue(record.call_args.args[2])  # its last step is marked as the end of its life
        self.assertNotIn(dying, w.creatures)
        self.assertEqual(learner.allocate(NetWeights.random(w.np_rng), LEARNING_RATE), dying.slot)

    def parent_that_has_learned(self, inherit):
        w, learner = self.world(inherit)
        crowd(w)
        parent = self.born(w, learner, 10, 10, energy=80.0)
        live(learner, parent.slot, 65)
        learner.update()
        self.assertFalse(same(learner.current(parent.slot), learner.inborn(parent.slot)))
        learned, born_with = learner.current(parent.slot), learner.inborn(parent.slot)
        w.step()
        baby = w.creatures[-1]
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
        w, learner = self.world()
        crowd(w)
        parent = self.born(w, learner, 10, 10, energy=80.0)
        with mock.patch.object(learner, "record", wraps=learner.record) as record:
            w.step()
        self.assertEqual(w.births, 1)
        self.assertLess(parent.energy, 55.0)  # it gave its baby 40 % of its energy
        _, reward, _ = record.call_args_list[0].args
        self.assertGreater(reward, -2.0)  # but its reward only shows the cost of the step itself


if __name__ == "__main__":
    unittest.main()
