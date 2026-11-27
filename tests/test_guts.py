"""Two guts, gills and lungs (the owner, 2026-11-27: "pull the genes to edges", with no trap on the slow way between):
each organ's worth is a straight line, and each one costs upkeep."""
import json
import unittest
from pathlib import Path

import numpy as np

from src.brain import N_SENSES, Brain
from src.food import Food
from src.saves import load_world
from src.terrain import Obstacle, Terrain
from src.world import GENES
from tests.helpers import NO_FOOD, NO_FOOD_ATTACKS, add, at, cap, make_world

BEFORE = Path(__file__).parent / "data" / "before_guts"  # written by the code of before the two guts


def quiet(world):
    """No price for living, senses or neurons: only what eating and hitting give is left."""
    world.rules.update(step_cost=0.0, vision_cost=0.0, hearing_cost=0.0, neuron_cost=0.0, pixel_cost=0.0,
                       attack_cost=0.0)
    return world


def bite(grass_gut, meat_gut, **food):
    """What a creature with these guts gains from one bite of a food."""
    w = quiet(make_world(NO_FOOD))
    c = add(w, 10, 10, energy=20.0)
    c.grass_gut, c.meat_gut = grass_gut, meat_gut
    piece = Food(at(10.0, 10.0), remaining_duration=-1, energy=10.0, **food)
    w.foods.append(piece)
    c.apply_action(("EAT", piece), w)
    return c.energy - 20.0


class TestGuts(unittest.TestCase):
    def test_each_gut_gets_its_share_of_a_bite(self):
        full = make_world().rules["bite"]
        self.assertAlmostEqual(bite(0.5, 0.0), 0.5 * full)
        self.assertAlmostEqual(bite(0.5, 0.0, fruit=True), 0.5 * full)
        self.assertAlmostEqual(bite(0.0, 0.5, meat=True), 0.5 * full)
        self.assertAlmostEqual(bite(1.0, 1.0), full)  # both guts: the full bite of each
        self.assertAlmostEqual(bite(1.0, 1.0, meat=True), full)
        self.assertAlmostEqual(bite(1.0, 0.0, meat=True), 0.0)  # a grass eater gets nothing from meat

    def test_a_hit_takes_and_feeds_by_the_meat_gut(self):
        w = quiet(make_world(NO_FOOD_ATTACKS))
        hunter, prey = add(w, 10, 10, energy=20.0), add(w, 11, 10, energy=50.0)
        hunter.grass_gut, hunter.meat_gut = 1.0, 0.5
        damage = hunter.attack_damage(w)
        self.assertAlmostEqual(damage, hunter.size * w.rules["attack_damage"] * (1 + 0.5 * w.rules["diet_bite"]))
        hunter.apply_action(("ATTACK", prey), w)
        self.assertAlmostEqual(prey.energy, 50.0 - damage)
        self.assertAlmostEqual(hunter.energy, 20.0 + w.rules["hunter_share"] * damage * 0.5)

    def test_gills_in_a_pond_lungs_on_land(self):
        w = make_world(NO_FOOD)
        w.terrain = Terrain([Obstacle("pond", (cap(20.0, 10.0, 3.0),))], [], w.radius)
        r = w.rules
        swimmer, walker = add(w, 20, 10), add(w, 5, 5)
        for c in (swimmer, walker):
            c.gills, c.lungs = 0.8, 0.3
        self.assertAlmostEqual(swimmer._pace(w), 0.8)
        self.assertAlmostEqual(swimmer._home_factor(w), 1 - 0.8 * r["swim_saving"])
        self.assertAlmostEqual(walker._pace(w), 1 - 0.7 * r["land_penalty"])
        self.assertAlmostEqual(walker._home_factor(w), 1 + 0.7 * r["land_cost"])
        walker.lungs = 1.0  # a full walker: no land price, whatever its gills
        self.assertEqual((walker._pace(w), walker._home_factor(w)), (1.0, 1.0))
        shore = add(w, 16.5, 10, size=0.5)
        shore.gills, shore.lungs = 0.0, 1.0
        start = shore.p.copy()
        shore.apply_action(("MOVE", (0.5, 0.0)), w)
        np.testing.assert_allclose(shore.p, start)  # no gills: a pond is a wall
        self.assertTrue(shore.bumped)

    def test_organs_cost_upkeep(self):
        w = quiet(make_world(NO_FOOD))
        w.rules.update(step_cost=0.1, gut_cost=0.15, breath_cost=0.15)
        c = add(w, 10, 10, energy=40.0)
        c.grass_gut, c.meat_gut, c.gills, c.lungs = 1.0, 0.5, 0.2, 1.0
        c.stealth = c.armour = 0.0
        c.apply_action(("MOVE", (0.0, 0.0)), w)
        body = 0.1 * c.size ** w.rules["living_power"]
        self.assertAlmostEqual(40.0 - c.energy, body * (1 + 1.5 * 0.15) * (1 + 1.2 * 0.15))

    def test_a_baby_inherits_the_four_genes(self):
        self.assertTrue({"grass_gut", "meat_gut", "gills", "lungs"} <= set(GENES))
        self.assertFalse({"diet", "water"} & set(GENES))
        w = make_world(NO_FOOD)
        w.rules["soup"] = 0.0
        parent = add(w, 10, 10, energy=80.0)
        parent.grass_gut, parent.meat_gut, parent.gills, parent.lungs = 0.6, 0.4, 0.3, 0.7
        parent.mutations, parent.mutation_size = len(GENES), 0.1  # every gene changes
        w._birth(parent)
        baby = next(c for c in w.creatures if c.parent_id == parent.id)
        for gene in ("grass_gut", "meat_gut", "gills", "lungs"):
            mine, theirs = getattr(baby, gene), getattr(parent, gene)
            self.assertNotEqual(mine, theirs, gene)
            self.assertTrue(0.0 <= mine <= 1.0 and abs(mine - theirs) <= 0.1 + 1e-9, gene)

    def test_a_save_of_before_goes_on_the_same(self):
        w, _, _ = load_world(BEFORE / "save.npz", learner_seed=0)
        genes = {c.id: (c.meat_gut, c.grass_gut, c.gills, c.lungs) for c in w.creatures}
        np.testing.assert_allclose(genes[1], (0.3, 0.7, 0.6, 0.4))
        np.testing.assert_allclose(genes[2], (0.8, 0.2, 0.1, 0.9))
        self.assertEqual((w.rules["gut_cost"], w.rules["breath_cost"]), (0.0, 0.0))
        for _ in range(3):
            w.step()
        before = json.loads((BEFORE / "after_3_steps.json").read_text())
        self.assertEqual({str(c.id): c.last_action for c in w.creatures}, before["actions"])
        for c in w.creatures:
            self.assertAlmostEqual(c.energy, before["energies"][str(c.id)], places=9)

    def test_a_brain_of_before_chooses_the_same(self):
        self.assertEqual(N_SENSES, 200)
        rng = np.random.default_rng(3)
        old = (rng.normal(size=(196, Brain.N_HIDDEN)), rng.normal(size=Brain.N_HIDDEN),
               rng.normal(size=(Brain.N_HIDDEN, Brain.N_OUT)), rng.normal(size=Brain.N_OUT))
        brain = Brain(*old)
        np.testing.assert_array_equal(brain.w1[196:], 0.0)  # the 4 new senses weigh nothing
        senses = rng.normal(size=200)
        hidden = np.tanh(senses[:196] @ old[0] + old[1])
        np.testing.assert_allclose(brain.forward(senses), np.tanh(hidden @ old[2] + old[3]))


if __name__ == "__main__":
    unittest.main()
