import math
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

from src.brain import N_SENSES
from src.creature import Creature, full_tank
from src.food import Food
from src.policies import decide_all
from src.learner import LEARNING_RATE, Learner, NetWeights
from src.rules import ROOM_SLOTS as MAX_CREATURES, gene_limits, rule, starting_rules
from src.settings import START
BABY_SHARE, EAT_MISS_COST = rule("baby_share").start, rule("eat_miss").start
from tests.helpers import LIVING, NO_FOOD, NO_FOOD_ATTACKS, add, fixed_brain, living, make_world, populate
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
        self.assertAlmostEqual(c.energy, 50.0 - LIVING)

    def test_move_cost_and_speed_cap(self):
        w = make_world()
        c = add(w, 10, 10, size=1.0)
        c.apply_action(("MOVE", (3.0, 4.0)), w)
        self.assertAlmostEqual(c.x, 10.48)  # at most its top speed, 0.8 a step for size 1
        self.assertAlmostEqual(c.y, 10.64)
        sprint = 0.8 * NO_FOOD.move_cost * (0.8 / 0.25) ** 2  # faster than cruising costs the square of the speed
        self.assertAlmostEqual(c.energy, 50.0 - sprint - LIVING)

    def test_sprinting_is_faster_than_cruising_at_every_size(self):
        w = make_world()
        for i, size in enumerate((0.5, 0.56, 0.62, 1.0, 2.0)):
            c = add(w, 5 + 8 * i, 10, size=size)
            self.assertGreater(c.top_speed, 1.1 * c.velocity, size)  # a burst is clearly faster than its normal speed

    def test_one_reach_rules_eating_and_attacking(self):
        w = make_world(NO_FOOD_ATTACKS)
        me = add(w, 10, 10, size=1.0)
        other = add(w, 10.5, 10, size=1.0)  # 0.5 apart: within 3 × (0.2 + 0.2), not within 1 × it
        piece = food(10, 10.5)
        self.assertTrue(me.can_reach(piece))
        w.rules["reach"] = 1.0
        self.assertFalse(me.can_reach(piece))
        before = other.energy
        me.apply_action(("ATTACK", other), w)
        self.assertEqual((me.last_action, other.energy), ("ATTACK_MISS", before))

    def test_moving_makes_a_sound_and_anything_else_does_not(self):
        w = make_world()
        c = add(w, 10, 10, size=1.0)
        self.assertFalse(c.moved)  # a newborn has made no sound yet
        c.apply_action(("MOVE", (0.1, 0.0)), w)
        self.assertTrue(c.moved)
        c.apply_action(("EAT", None), w)
        self.assertFalse(c.moved)
        c.apply_action(("MOVE", (0.0, 0.0)), w)  # standing still
        self.assertFalse(c.moved)

    def test_top_speed_grows_with_size(self):
        w = make_world()
        tops = [add(w, 5 + 10 * i, 10, size=size).top_speed for i, size in enumerate((0.5, 1.0, 2.0))]
        self.assertEqual([round(t, 3) for t in tops], [0.566, 0.8, 1.131])

    def test_cruising_costs_as_before(self):
        w = make_world()
        c = add(w, 10, 10, size=1.0)
        c.apply_action(("MOVE", (0.2, 0.0)), w)
        self.assertAlmostEqual(c.energy, 50.0 - 0.2 * NO_FOOD.move_cost - LIVING)

    def test_sprinting_costs_the_square_of_the_speed(self):
        w = make_world()
        c = add(w, 10, 10, size=2.0, energy=150.0)
        c.apply_action(("MOVE", (c.top_speed, 0.0)), w)
        cost = c.top_speed * NO_FOOD.move_cost * (c.top_speed / c.velocity) ** 2
        self.assertAlmostEqual(c.energy, 150.0 - 2.0 * cost - living(2.0))  # a size-2 body pays twice
        self.assertGreater(cost / (c.top_speed * NO_FOOD.move_cost), 20)  # far more than cruising

    def test_a_big_sprinter_catches_a_small_cruiser(self):
        w = make_world()
        big, small = add(w, 10, 10, size=2.0, energy=150.0), add(w, 12, 10, size=0.5)
        gap = small.x - big.x
        big.apply_action(("MOVE", (big.top_speed, 0.0)), w)
        small.apply_action(("MOVE", (small.top_speed, 0.0)), w)  # running away as fast as it can
        self.assertLess(small.x - big.x, gap)

    def test_speed_is_a_quarter_over_the_root_of_size(self):
        w = make_world()
        speeds = [add(w, 5 + 10 * i, 10, size=size).velocity for i, size in enumerate((0.5, 1.0, 2.0))]
        np.testing.assert_allclose(speeds, [0.25 / math.sqrt(0.5), 0.25, 0.25 / math.sqrt(2.0)])

    def test_move_clamped_in_corner(self):
        w = make_world()
        c = add(w, 0.3, 0.3, size=1.0)
        c.apply_action(("MOVE", (-1.0, -1.0)), w)
        self.assertAlmostEqual(c.x, c.radius)
        self.assertAlmostEqual(c.y, c.radius)


class TestSenseGenes(unittest.TestCase):
    def test_senses_cost_by_area(self):
        SENSE_COST = rule("vision_cost").start
        w = make_world()
        today = add(w, 10, 10)
        today.vision_width, today.vision_reach, today.hearing = 360.0, 3.0, 0.0
        before = today.energy
        today.apply_action(("MOVE", (0.0, 0.0)), w)
        self.assertAlmostEqual(before - today.energy, NO_FOOD.step_cost + 0.05, places=4)  # today's senses: 0.05
        sharp = add(w, 20, 10)
        sharp.vision_width, sharp.vision_reach, sharp.hearing = 90.0, 6.0, 2.0
        before = sharp.energy
        sharp.apply_action(("MOVE", (0.0, 0.0)), w)
        area = 90.0 / 360.0 * math.pi * 36.0 + math.pi * 4.0
        self.assertAlmostEqual(before - sharp.energy, NO_FOOD.step_cost + SENSE_COST * area)

    def test_a_bigger_body_pays_more_to_live_and_move(self):
        w = make_world()
        small, big = add(w, 10, 10, size=1.0), add(w, 20, 10, size=2.0, energy=150.0)
        for c in (small, big):
            c.vision_width, c.vision_reach, c.hearing = 30.0, 1.0, 0.0
        senses = lambda c: w.rules["vision_cost"] * c.sense_area()
        spent = []
        for c in (small, big):
            before = c.energy
            c.apply_action(("MOVE", (0.1, 0.0)), w)
            spent.append(before - c.energy - senses(c))
        self.assertAlmostEqual(spent[0], NO_FOOD.step_cost * 1.0 + 0.1 * NO_FOOD.move_cost * 1.0)
        self.assertAlmostEqual(spent[1], NO_FOOD.step_cost * 2.0 ** w.rules["living_power"]  # more, less than twice
                               + 0.1 * NO_FOOD.move_cost * 2.0)  # moving: twice

    def test_vision_and_hearing_are_priced_apart(self):
        w = make_world()
        c = add(w, 10, 10)
        c.vision_width, c.vision_reach, c.hearing = 90.0, 4.0, 2.0
        w.rules["vision_cost"], w.rules["hearing_cost"] = 0.0, 0.01
        before = c.energy
        c.apply_action(("MOVE", (0.0, 0.0)), w)
        self.assertAlmostEqual(before - c.energy, NO_FOOD.step_cost + 0.01 * math.pi * 2.0 ** 2)  # hearing alone

    def test_the_sprint_power_is_a_rule(self):
        w = make_world()
        c = add(w, 10, 10, size=1.0)
        c.vision_width, c.vision_reach, c.hearing = 30.0, 1.0, 0.0
        w.rules["sprint_power"] = 1.0
        before = c.energy
        c.apply_action(("MOVE", (0.4, 0.0)), w)
        sprint = 0.4 * NO_FOOD.move_cost * (0.4 / 0.25) ** 1.0
        self.assertAlmostEqual(before - c.energy, NO_FOOD.step_cost + sprint + w.rules["vision_cost"] * c.sense_area())

    def test_a_baby_changes_one_gene(self):
        genes = ("size", "vision_width", "vision_reach", "hearing", "fear_of_death", "love_of_babies", "learns_from_others",
                 "mutation_size", "mutations")
        picked = set()
        for seed in range(120):
            w = make_world(seed=seed)
            crowd(w)
            parent = add(w, 10, 10, energy=80.0, policy="brain", brain=fixed_brain(STAY))
            parent.vision_width, parent.vision_reach, parent.hearing = 100.0, 5.0, 2.0
            w.step()
            baby = next(c for c in w.creatures if c.parent_id == parent.id)
            changed = [g for g in genes if getattr(baby, g) != getattr(parent, g)]
            self.assertLessEqual(len(changed), 1, changed)  # one gene at most (none when learning speed was picked)
            for g in changed:
                if g == "mutations":  # a count: one more or one fewer
                    self.assertEqual(abs(baby.mutations - parent.mutations), 1)
                else:  # by up to its parent's share of the gene's range
                    low, high = gene_limits(w.rules, {"fear_of_death": "fear", "love_of_babies": "love",
                                                      "learns_from_others": "learns"}.get(g, g))
                    step = max(parent.mutation_size, w.rules["gene_min_change"]) * (high - low)
                    self.assertLessEqual(abs(getattr(baby, g) - getattr(parent, g)), step + 1e-9)
            picked.update(changed)
        self.assertEqual(picked, set(genes))  # every gene is picked sometime

    def test_a_baby_changes_as_many_genes_as_its_parents_gene_says(self):
        w = make_world(seed=2)
        crowd(w)
        parent = add(w, 10, 10, energy=80.0, policy="brain", brain=fixed_brain(STAY))
        parent.mutations = 3
        with mock.patch.object(w.rng, "sample", wraps=w.rng.sample) as sample:
            w.step()
        self.assertEqual(sample.call_args.args[1], 3)

    def test_a_changed_gene_moves_by_up_to_its_parents_mutation_size(self):
        moves = []
        for seed in range(30):
            w = make_world(seed=seed)
            crowd(w)
            parent = add(w, 10, 10, size=1.0, energy=80.0, policy="brain", brain=fixed_brain(STAY))
            parent.mutation_size = 0.4
            with mock.patch.object(w.rng, "sample", lambda genes, k: ["size"]):
                w.step()
            baby = next(c for c in w.creatures if c.parent_id == parent.id)
            moves.append(abs(baby.size - 1.0))
        self.assertLessEqual(max(moves), 0.4 * 1.5 + 1e-9)  # 40 % of its range, 0.5-2
        self.assertGreater(max(moves), 0.1)

    def test_a_gene_changes_by_a_share_of_its_range_never_of_itself(self):
        def moves(hearing, mutation_size):
            out = []
            for seed in range(30):
                w = make_world(seed=seed)
                crowd(w)
                parent = add(w, 10, 10, size=1.0, energy=80.0, policy="brain", brain=fixed_brain(STAY))
                parent.hearing, parent.mutation_size = hearing, mutation_size
                with mock.patch.object(w.rng, "sample", lambda genes, k: ["hearing"]):
                    w.step()
                out.append(next(c for c in w.creatures if c.parent_id == parent.id).hearing - hearing)
            return out
        low, high = gene_limits(starting_rules(), "hearing")  # 0-4
        for hearing in (3.0, 0.5):  # 5 % of the range, 0.2, whatever the hearing is now
            changed = moves(hearing, 0.05)
            self.assertTrue(all(abs(m) <= 0.05 * (high - low) + 1e-9 for m in changed))
            self.assertGreater(max(map(abs, changed)), 0.1)
        changed = moves(0.5, 0.001)  # below the smallest change: 1 % of the range, 0.04
        self.assertEqual(starting_rules()["gene_min_change"], 0.01)
        self.assertTrue(all(abs(m) <= 0.01 * (high - low) + 1e-9 for m in changed))
        self.assertGreater(max(map(abs, changed)), 0.02)

    def test_bigger_brain_costs_more_energy(self):
        learner = Learner(MAX_CREATURES, 0)
        w = make_world(learner=learner, species="learner")
        small = add(w, 10, 10, policy="learner", slot=learner.allocate(NetWeights.random(w.np_rng), LEARNING_RATE))
        big_brain = NetWeights.random(w.np_rng).with_layer(on=True)
        big = add(w, 20, 10, policy="learner", slot=learner.allocate(big_brain, LEARNING_RATE))
        w.count_neurons()
        self.assertEqual((small.neurons, big.neurons), (32, 64))
        spent = []
        for c in (small, big):
            before = c.energy
            c.apply_action(("MOVE", (0.0, 0.0)), w)
            spent.append(before - c.energy)
        self.assertAlmostEqual(spent[1] - spent[0], 32 * w.rules["neuron_cost"])
        self.assertEqual(w.rules["neuron_cost"], 0.0005)

    def test_babies_inherit_senses_in_the_limits(self):
        limits = lambda gene: (rule(f"gene.{gene}.low").start, rule(f"gene.{gene}.high").start)
        VISION_WIDTH_LIMITS, VISION_REACH_LIMITS, HEARING_LIMITS = (limits(g) for g in ("vision_width", "vision_reach", "hearing"))
        w = make_world(seed=4)
        crowd(w)
        parent = add(w, 10, 10, energy=80.0)
        parent.vision_width, parent.vision_reach, parent.hearing = 110.0, 9.9, 2.0
        w.step()
        baby = next(c for c in w.creatures if c.parent_id == parent.id)
        self.assertTrue(110.0 * 0.8 <= baby.vision_width <= 120.0)  # never past its limit
        self.assertTrue(9.9 * 0.8 <= baby.vision_reach <= 10.0)
        self.assertTrue(2.0 * 0.8 <= baby.hearing <= 2.0 * 1.2)
        for _ in range(30):
            c = w.add_newcomer()
            self.assertTrue(VISION_WIDTH_LIMITS[0] <= c.vision_width <= VISION_WIDTH_LIMITS[1])
            self.assertTrue(VISION_REACH_LIMITS[0] <= c.vision_reach <= VISION_REACH_LIMITS[1])
            self.assertTrue(HEARING_LIMITS[0] <= c.hearing <= HEARING_LIMITS[1])


class TestEating(unittest.TestCase):
    def test_eat_action_takes_a_bite(self):
        w = make_world()
        c = add(w, 10, 10, policy="brain", brain=fixed_brain(EAT))
        f = food(10, 10)
        w.foods.append(f)
        w.step()
        self.assertAlmostEqual(c.energy, 51.0 - LIVING)
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
        self.assertAlmostEqual(w.energy_spent, EAT_MISS_COST + 2 * LIVING)
        mover = add(w, 20, 5, size=1.0)
        mover.apply_action(("MOVE", (3.0, 4.0)), w)
        self.assertAlmostEqual(w.energy_spent, EAT_MISS_COST + 3 * LIVING + 0.8 * NO_FOOD.move_cost * (0.8 / 0.25) ** 2)  # a full sprint

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
        self.assertAlmostEqual(misser.spent, 3 * (EAT_MISS_COST + LIVING))
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
        self.assertAlmostEqual(c.energy, 50.0 - EAT_MISS_COST - LIVING)
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
        hunter.diet = 1.0  # a meat eater: a hit feeds it in full
        hunter.rules["diet_bite"] = 0.0  # this test: the share it keeps, not the harder bite
        prey = add(w, 11, 10, size=1.0, energy=50.0)
        hunter.apply_action(("ATTACK", prey), w)
        self.assertAlmostEqual(prey.energy, 40.0)
        self.assertAlmostEqual(hunter.energy, 100.0 + 0.8 * 10.0 - 2.0 - living(2.0))
        self.assertAlmostEqual(w.energy_earned, 0.8 * 10.0)
        self.assertAlmostEqual(w.energy_spent, 2.0 + living(2.0))

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

    def test_a_creature_hit_feels_it(self):
        from src.senses import sense
        w = make_world(NO_FOOD_ATTACKS)
        hunter = add(w, 11, 10, size=2.0, energy=100.0)
        prey = add(w, 10, 10, size=1.0, energy=50.0)
        self.assertTrue(all(sense(prey, w).vector[16:20] == 0.0))  # nothing felt yet
        hunter.apply_action(("ATTACK", prey), w)
        v = sense(prey, w).vector
        self.assertEqual(v.shape, (N_SENSES,))
        self.assertEqual(v[16], 1.0)  # hit
        self.assertAlmostEqual(v[17], 0.1)  # 10 of its 100 taken
        self.assertAlmostEqual(v[18], 1.0)  # from the east, where the hunter is
        self.assertAlmostEqual(v[19], 0.0)

    def test_pain_lasts_one_step(self):
        w = make_world(NO_FOOD_ATTACKS)
        crowd(w)
        hunter = add(w, 11, 10, size=2.0, energy=100.0)
        prey = add(w, 10, 10, size=1.0, energy=50.0)
        hunter.apply_action(("ATTACK", prey), w)
        felt = []
        def nobody_attacks(world, acting, senses):
            return [("MOVE", (0.0, 0.0)) for _ in acting]

        with mock.patch("src.world.decide_all", nobody_attacks):
            w.step(lambda c, senses, action: felt.append(senses.vector[16]) if c is prey else None)
            w.step(lambda c, senses, action: felt.append(senses.vector[16]) if c is prey else None)
        self.assertEqual(felt, [1.0, 0.0])  # felt in the next step, gone the step after

    def test_dead_target_is_a_miss(self):
        w = make_world(NO_FOOD_ATTACKS)
        hunter = add(w, 10, 10, size=2.0, energy=100.0)
        prey = add(w, 11, 10, size=1.0, energy=50.0)
        w.creatures.remove(prey)
        hunter.apply_action(("ATTACK", prey), w)
        self.assertEqual(hunter.last_action, "ATTACK_MISS")
        self.assertEqual(prey.energy, 50.0)


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
        c = add(w, 10, 10, energy=32.0, policy="brain", brain=fixed_brain(STAY))  # above 30 % of its maximum
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
        within = lambda c: shortest * c.size ** w.rules["lifespan_power"] - 1 <= c.lifespan <= longest * c.size ** w.rules["lifespan_power"] + 1  # noqa: E731
        self.assertTrue(all(within(c) for c in w.creatures))  # its own, drawn for its size
        self.assertGreater(len(set(spans)), 20)
        for creature in w.creatures:
            creature.energy = 90.0
        w.step()
        babies = [c for c in w.creatures if c.generation == 1]
        self.assertTrue(babies)
        self.assertTrue(all(within(baby) for baby in babies))

    def test_lifespans_follow_the_settings_in_force(self):
        w = make_world(START, species="random")
        w.set_settings(replace(START, lifespan=(100, 200)), 1)
        populate(w, 30)
        self.assertTrue(all(100 * c.size ** w.rules["lifespan_power"] - 1 <= c.lifespan <= 200 * c.size ** w.rules["lifespan_power"] + 1
                            for c in w.creatures))  # for its size

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
        room = round(w.rules["room"])  # the test world's room, full
        for i in range(room):
            add(w, 1 + (i % 10) * 4.5, 1 + (i // 10) * 3.4, energy=90.0)
        elder = w.creatures[0]
        elder.lifespan, elder.age, elder.energy = 100, 99, 20.0
        w.step()
        self.assertEqual((len(w.creatures), w.deaths["age"], w.births), (room, 1, 1))

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
        before_birth = 80.0 - LIVING
        self.assertAlmostEqual(baby.energy, BABY_SHARE * before_birth)
        self.assertAlmostEqual(parent.energy, (1 - BABY_SHARE) * before_birth)
        self.assertEqual((baby.parent_id, baby.generation, baby.policy), (parent.id, 1, "brain"))
        self.assertTrue(0.8 <= baby.size <= 1.2)
        self.assertLessEqual(abs(baby.x - parent.x), 1.0)
        changed = np.mean(baby.brain.w1 != parent_brain.w1)
        self.assertTrue(0.0 <= changed < 0.05)  # the parent's brain: a copy, or a few weights changed when they are the gene
        self.assertEqual(w.births, 1)

    def test_no_birth_below_seventy_percent(self):
        w = make_world()
        crowd(w)
        add(w, 10, 10, size=1.0, energy=60.0, policy="brain", brain=fixed_brain(STAY))
        w.step()
        self.assertEqual((len(w.creatures), w.births), (11, 0))

    def test_births_pause_at_limit(self):
        w = make_world(species="random")
        room = round(w.rules["room"])  # the test world's room, full
        for i in range(room):
            add(w, 1 + (i % 10) * 4.5, 1 + (i // 10) * 3.4, energy=90.0)
        w.step()
        self.assertEqual((len(w.creatures), w.births), (room, 0))
        w.creatures.pop()
        w.step()
        self.assertEqual((len(w.creatures), w.births), (room, 1))

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

    def test_every_gene_of_a_newcomer_is_random(self):
        w, learner = self.world()
        newcomers = [w.add_newcomer() for _ in range(30)]
        r = w.rules
        drawn = {"fear": [c.fear_of_death for c in newcomers], "love": [c.love_of_babies for c in newcomers],
                 "learns": [c.learns_from_others for c in newcomers], "mutations": [c.mutations for c in newcomers],
                 "mutation_size": [c.mutation_size for c in newcomers],
                 "learning_speed": [learner.lr(c.slot) for c in newcomers],
                 "layers": [learner.layers(c.slot) for c in newcomers],
                 "neurons": [n for c in newcomers for n in learner.brain_size(c.slot)]}
        for gene, values in drawn.items():
            self.assertGreater(len(set(values)), 1, gene)  # not one value for all
            low, high = r[f"gene.{gene}.new_low"], r[f"gene.{gene}.new_high"]
            self.assertTrue(all(low <= v <= high for v in values), (gene, low, high, values))
        self.assertTrue(all(isinstance(v, int) for v in drawn["mutations"] + drawn["layers"] + drawn["neurons"]))
        self.assertEqual([c.neurons for c in newcomers], [sum(learner.brain_size(c.slot)) for c in newcomers])

    def test_the_owner_changes_a_creatures_genes(self):
        w, learner = self.world()
        c = self.born(w, learner, 10, 10)
        w.count_neurons()
        w.edit_genes(c.id, {"size": 1.5, "vision_width": 60.0, "vision_reach": 4.0, "hearing": 2.0, "fear_of_death": 3.0,
                            "love_of_babies": 4.0, "learns_from_others": 0.2, "learning_speed": 0.001, "layers": 2,
                            "neurons": [10, 5]})
        self.assertEqual((c.size, c.vision_width, c.vision_reach, c.hearing), (1.5, 60.0, 4.0, 2.0))
        self.assertEqual((c.fear_of_death, c.love_of_babies, c.learns_from_others), (3.0, 4.0, 0.2))
        self.assertAlmostEqual(c.radius, 1.5 * w.rules["body_radius"])  # its body follows its size
        self.assertAlmostEqual(c.max_energy, full_tank(1.5, w.rules))
        self.assertAlmostEqual(learner.lr(c.slot), 0.001)
        self.assertEqual((learner.layers(c.slot), learner.active_neurons(c.slot)), (2, 15))
        self.assertEqual((c.layers, c.neurons), (2, 15))  # what it pays for each step
        self.assertEqual(list(learner.inborn(c.slot).size), [2, 10, 5, 0])  # its babies inherit the change
        w.edit_genes(c.id, {"mutations": 3, "mutation_size": 0.1})
        self.assertEqual((c.mutations, c.mutation_size), (3, 0.1))

    def test_a_smaller_tank_keeps_its_energy_inside(self):
        w, learner = self.world()
        c = self.born(w, learner, 10, 10, energy=90.0)
        c.size, c.max_energy = 1.0, w.rules["tank"]
        c.energy = 90.0
        w.edit_genes(c.id, {"size": 0.5})
        self.assertAlmostEqual(c.energy, full_tank(0.5, w.rules))

    def test_a_gene_change_for_a_creature_gone_changes_nothing(self):
        w, learner = self.world()
        c = self.born(w, learner, 10, 10)
        w.edit_genes(c.id + 1000, {"size": 1.9})
        self.assertNotEqual(c.size, 1.9)

    def test_newcomer_gets_a_slot_and_a_random_network(self):
        w, learner = self.world()
        first, second = w.add_newcomer(), w.add_newcomer()
        self.assertEqual((first.policy, first.slot, second.slot), ("learner", 0, 1))
        self.assertFalse(same(learner.current(0), learner.current(1)))
        self.assertTrue(w.rules["gene.learning_speed.new_low"] <= learner.lr(0) <= w.rules["gene.learning_speed.new_high"])

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

    def rewards(self, w, learner, steps=1):
        """Each step's rewards, by slot."""
        out = []
        for _ in range(steps):
            with mock.patch.object(learner, "record", wraps=learner.record) as record:
                w.step()
            out.append({call.args[0]: call.args[1:] for call in record.call_args_list})
        return out

    def test_a_dying_learner_pays_its_own_fear_of_death(self):
        w, learner = self.world()
        crowd(w)
        starving = self.born(w, learner, 10, 10, energy=0.01)
        starving.fear_of_death = 7.0
        [step] = self.rewards(w, learner)
        reward, done = step[starving.slot]
        self.assertTrue(done)
        self.assertAlmostEqual(reward, starving.energy - 0.01 - 7.0)  # hunger: its energy change, minus its own fear

    def test_a_learner_killed_by_an_attack_pays_its_own_fear_of_death(self):
        learner = Learner(MAX_CREATURES, 0)
        w = make_world(NO_FOOD_ATTACKS, learner=learner, species="learner")
        crowd(w)
        prey = self.born(w, learner, 10, 10, energy=5.0)
        prey.fear_of_death = 3.0
        hunter = add(w, 11, 10, size=2.0, energy=100.0)
        real = decide_all

        def hunter_attacks(world, acting, senses):
            actions = real(world, acting, senses)
            return [("ATTACK", prey) if c is hunter else a for c, a in zip(acting, actions)]

        with mock.patch("src.world.decide_all", hunter_attacks):
            [step] = self.rewards(w, learner)
        self.assertEqual(w.deaths["attack"], 1)
        reward, done = step[prey.slot]
        self.assertTrue(done)
        self.assertAlmostEqual(reward, prey.energy - 5.0 - 3.0)

    def test_a_learner_is_paid_its_own_love_of_babies(self):
        w, learner = self.world()
        crowd(w)
        parent = self.born(w, learner, 10, 10, energy=80.0)
        parent.love_of_babies = 12.0
        with mock.patch.object(learner, "record", wraps=learner.record) as record:
            w.step()
        self.assertEqual(w.births, 1)
        after_gift = parent.energy
        before_gift = after_gift / (1.0 - BABY_SHARE)
        first = {call.args[0]: call.args[1] for call in record.call_args_list}[parent.slot]
        self.assertAlmostEqual(first, before_gift - 80.0)  # the step it has the baby: its energy change, the gift no penalty
        [step] = self.rewards(w, learner)
        reward, done = step[parent.slot]
        self.assertFalse(done)
        self.assertAlmostEqual(reward, parent.energy - after_gift + 12.0)  # the next step: plus its own love of babies

    def test_a_living_learner_is_paid_its_energy_change(self):
        w, learner = self.world()
        crowd(w)
        parent = self.born(w, learner, 10, 10, energy=80.0)
        self.rewards(w, learner, steps=2)  # a baby, then its pay
        before = parent.energy
        [step] = self.rewards(w, learner)
        self.assertAlmostEqual(step[parent.slot][0], parent.energy - before)  # paid once, never again

    def test_learners_in_sight_get_the_lesson(self):
        w, learner = self.world()
        crowd(w)
        actor = self.born(w, learner, 10, 10)
        near = self.born(w, learner, 12, 10)  # 2 units: in sight (3)
        far = self.born(w, learner, 20, 10)  # 8 units from the nearest
        [step] = self.rewards(w, learner)
        self.assertEqual(len(learner._lessons[near.slot]), 1)  # the actor's step
        self.assertAlmostEqual(learner._lessons[near.slot][0]["reward"], step[actor.slot][0])
        self.assertEqual(len(learner._lessons[actor.slot]), 1)  # the near one's step, never its own
        self.assertAlmostEqual(learner._lessons[actor.slot][0]["reward"], step[near.slot][0])
        self.assertEqual(learner._lessons[far.slot], [])

    def test_a_watcher_values_a_lesson_by_its_own_genes(self):
        w, learner = self.world()
        crowd(w)
        dying = self.born(w, learner, 10, 10, energy=0.01)
        dying.fear_of_death = 50.0  # its own fear: not the watcher's
        watcher = self.born(w, learner, 11, 10)
        watcher.fear_of_death = 4.0
        [step] = self.rewards(w, learner)
        reward, done = step[dying.slot]
        [lesson] = learner._lessons[watcher.slot]
        self.assertAlmostEqual(lesson["reward"], reward + 50.0 - 4.0)  # its energy change, minus the watcher's fear
        self.assertEqual(lesson["step"]["done"], 1.0)

        w, learner = self.world()
        crowd(w)
        parent = self.born(w, learner, 10, 10, energy=80.0)
        parent.love_of_babies = 30.0
        watcher = self.born(w, learner, 11, 10, energy=40.0)
        watcher.love_of_babies = 2.0
        steps = self.rewards(w, learner, steps=2)  # a baby, then the step that pays for it
        lesson = next(l for l in learner._lessons[watcher.slot] if l["step"] is learner._last[parent.slot])
        self.assertAlmostEqual(lesson["reward"], steps[1][parent.slot][0] - 30.0 + 2.0)  # plus the watcher's love

    def test_a_lesson_weighs_the_watchers_learning_from_others(self):
        w, learner = self.world()
        crowd(w)
        actor = self.born(w, learner, 10, 10)
        watcher = self.born(w, learner, 12, 10)
        actor.learns_from_others, watcher.learns_from_others = 0.9, 0.3
        self.rewards(w, learner)
        self.assertAlmostEqual(learner._lessons[watcher.slot][0]["weight"], 0.3)
        self.assertAlmostEqual(learner._lessons[actor.slot][0]["weight"], 0.9)

    def test_babies_inherit_values_in_the_limits(self):
        FEAR_LIMITS, LOVE_LIMITS, LEARN_LIMITS = ((rule(f"gene.{g}.low").start, rule(f"gene.{g}.high").start)
                                                for g in ("fear", "love", "learns"))
        w, learner = self.world()
        crowd(w)
        parent = self.born(w, learner, 10, 10, energy=80.0)
        parent.fear_of_death, parent.love_of_babies, parent.learns_from_others = 95.0, 5.0, 0.95
        w.step()
        baby = next(c for c in w.creatures if c.parent_id == parent.id)
        self.assertTrue(95.0 * 0.8 <= baby.fear_of_death <= FEAR_LIMITS[1])
        self.assertTrue(5.0 * 0.8 <= baby.love_of_babies <= 5.0 * 1.2)
        self.assertTrue(0.95 * 0.8 <= baby.learns_from_others <= LEARN_LIMITS[1])
        self.assertEqual((FEAR_LIMITS, LOVE_LIMITS, LEARN_LIMITS), ((0.1, 100.0), (0.1, 100.0), (0.01, 1.0)))

    def test_watchers_who_hear_get_the_lesson(self):
        w, learner = self.world()
        crowd(w)
        actor = self.born(w, learner, 10, 10)
        hearer = self.born(w, learner, 12, 10)
        deaf = self.born(w, learner, 10, 13)
        for c in (actor, hearer, deaf):
            c.vision_width, c.vision_reach = 30.0, 3.0
        hearer.facing, hearer.hearing = (1.0, 0.0), 3.0  # the actor behind it: unseen, heard
        deaf.facing, deaf.hearing = (0.0, 1.0), 0.0  # the actor behind it: unseen, unheard
        actor.facing, actor.hearing = (0.0, -1.0), 0.0
        act = Creature.apply_action

        def acts_and_sounds(self, action, world):  # whatever the actor does, it makes a sound this step
            act(self, action, world)
            self.moved = self.moved or self is actor
        with mock.patch.object(Creature, "_move", lambda self, vector, world: None), \
                mock.patch.object(Creature, "apply_action", acts_and_sounds):  # nobody turns
            self.rewards(w, learner)
        self.assertEqual(len(learner._lessons[hearer.slot]), 1)
        self.assertEqual(learner._lessons[deaf.slot], [])

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
        self.assertGreater(np.mean(genes.w1[:, :32] == learned.w1[:, :32]), 0.7)  # its 32 neurons in use
        self.assertLess(np.mean(np.isclose(genes.w1[:, :32], born_with.w1[:, :32])), 0.3)
        ratio = learner.lr(baby.slot) / learner.lr(parent.slot)  # changed only when learning speed is the baby's one gene
        self.assertLessEqual(abs(ratio - 1.0), 0.05 + 1e-9)

    def test_learner_baby_inherits_born_brain_when_inborn(self):
        learner, _, baby, learned, born_with = self.parent_that_has_learned("inborn")
        genes = learner.inborn(baby.slot)
        self.assertGreater(np.mean(np.isclose(genes.w1[:, :32], born_with.w1[:, :32])), 0.7)
        self.assertLess(np.mean(genes.w1[:, :32] == learned.w1[:, :32]), 0.3)

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
