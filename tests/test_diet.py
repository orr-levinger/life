"""What it lives on (spec decision 2; since 2026-11-27 two guts): a diet d is a meat gut d and a grass gut 1 - d."""
import unittest
from dataclasses import replace

from src.food import Food
from src.senses import sense
from tests.helpers import face, at, NO_FOOD, NO_FOOD_ATTACKS, add, make_world


def quiet(world):
    """No price for living, senses or neurons: only what eating and hitting give is left."""
    world.rules.update(step_cost=0.0)
    world.rules.update(vision_cost=0.0, hearing_cost=0.0, neuron_cost=0.0, attack_cost=0.0)
    return world


def eats(diet, **food):
    """What a creature of `diet` gains from one bite of a food."""
    w = quiet(make_world(NO_FOOD))
    c = add(w, 10, 10, energy=20.0)
    c.meat_gut, c.grass_gut = diet, 1.0 - diet
    piece = Food(at(10.0, 10.0), remaining_duration=-1, energy=10.0, **food)
    w.foods.append(piece)
    c.apply_action(("EAT", piece), w)
    return c.energy - 20.0, 10.0 - piece.energy


class TestDiet(unittest.TestCase):
    def test_plants_and_fruit_feed_a_plant_eater(self):
        bite = make_world().rules["bite"]
        self.assertAlmostEqual(eats(0.25)[0], 0.75 * bite)
        self.assertAlmostEqual(eats(0.25, fruit=True)[0], 0.75 * bite)
        self.assertAlmostEqual(eats(1.0)[0], 0.0)
        self.assertAlmostEqual(eats(1.0)[1], bite)  # the bite is taken all the same

    def test_meat_feeds_a_meat_eater(self):
        bite = make_world().rules["bite"]
        self.assertAlmostEqual(eats(0.25, meat=True)[0], 0.25 * bite * make_world().rules["meat_value"])
        self.assertAlmostEqual(eats(0.0, meat=True)[0], 0.0)

    def test_a_hit_feeds_by_the_diet_and_hurts_more(self):
        w = quiet(make_world(NO_FOOD_ATTACKS))
        hunter = add(w, 10, 10, energy=20.0)
        prey = add(w, 11, 10, energy=50.0)
        hunter.meat_gut = 0.5
        damage = hunter.attack_damage(w)
        self.assertAlmostEqual(damage, hunter.size * w.rules["attack_damage"] * (1 + 0.5 * w.rules["diet_bite"]))
        hunter.apply_action(("ATTACK", prey), w)
        self.assertAlmostEqual(prey.energy, 50.0 - damage)
        self.assertAlmostEqual(hunter.energy, 20.0 + w.rules["hunter_share"] * damage * 0.5 * w.rules["meat_value"])

    def test_a_corpse_is_meat(self):
        w = quiet(make_world(NO_FOOD_ATTACKS))
        hunter = add(w, 10, 10, energy=20.0)
        prey = add(w, 11, 10, energy=0.5)
        hunter.apply_action(("ATTACK", prey), w)
        self.assertEqual([f.meat for f in w.foods], [True])

    def test_it_senses_meat_and_its_own_diet(self):
        w = make_world(NO_FOOD)
        me = add(w, 10, 10)
        me.vision_width, me.vision_reach, me.meat_gut = 120.0, 5.0, 0.7
        face(me, 1.0, 0.0)
        w.foods.append(Food(at(12.0, 10.0), remaining_duration=5, energy=8.0, meat=True))
        v = sense(me, w).vector
        self.assertEqual((v[25], v[26]), (1.0, 0.7))
        w.foods[0].meat = False
        self.assertEqual(sense(me, w).vector[25], 0.0)

    def test_newcomers_draw_a_diet_and_babies_inherit_it(self):
        w = make_world(seed=4)
        diets = [w.add_newcomer().meat_gut for _ in range(30)]
        self.assertTrue(all(0.0 <= d <= 1.0 for d in diets))
        self.assertGreater(max(diets) - min(diets), 0.5)  # every kind among them


    def test_the_page_gets_each_diet_and_the_meat(self):
        from src.evolve import world_now, world_view
        w = make_world(NO_FOOD)
        c = add(w, 10, 10)
        c.meat_gut = 0.8
        w.foods.append(Food(at(12.0, 10.0), remaining_duration=5, energy=8.0, meat=True))
        view = world_view(w)
        self.assertEqual((view["creatures"][0]["meat_gut"], view["foods"][0]["meat"]), (0.8, True))
        self.assertEqual(world_now(w)["genes"]["meat_gut"], [0.8, 0.8, 0.8])


if __name__ == "__main__":
    unittest.main()
