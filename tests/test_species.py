"""Species: a baby is in its parent's; a group that drifts apart splits off; a species with no one left ends (the owner,
2026-11-27: "a tree of creatures and how they split into different creatures")."""
import unittest

from src.species import facts
from src.terrain import Obstacle, Terrain
from tests.helpers import NO_FOOD, add, cap, make_world


def herd(w, n=20):
    """`n` creatures alike, side by side, all in species 1."""
    return [add(w, 3.0 * k, 0.0) for k in range(n)]


def drift(creatures):
    """Three genes moved across their whole range (a meat eater, armoured): 3/20 of the way apart, more than 10 %."""
    for c in creatures:
        c.meat_gut, c.grass_gut, c.armour = 1.0, 0.0, 1.0


class TestSpecies(unittest.TestCase):
    def test_the_book_counts_species_alive(self):
        from src.species import SpeciesBook
        self.assertIsNone(SpeciesBook().alive)  # no look yet
        w = make_world(NO_FOOD)
        for i, number in enumerate((1, 2, 3)):
            add(w, 10 + i * 5, 10).species = number
        w.species_book.look(w, 250)
        self.assertEqual(w.species_book.alive, 3)

    def test_a_creature_counts_its_steps_in_water(self):
        """Where it really lives (the owner, 2026-11-28: "Actually also activity")."""
        w = make_world(NO_FOOD)
        w.terrain = Terrain([Obstacle("pond", (cap(20.0, 10.0, 3.0),))], [], w.radius)
        swimmer, walker = add(w, 20, 10, energy=80.0), add(w, 5, 5, energy=80.0)
        swimmer.gills = 1.0
        for _ in range(5):
            w.step()
        self.assertEqual((swimmer.in_water, walker.in_water), (5, 0))

    def test_a_species_says_its_share_in_water(self):
        w = make_world(NO_FOOD)
        a, b = add(w, 1, 1), add(w, 3, 3)
        (a.age, a.in_water), (b.age, b.in_water) = (100, 80), (300, 20)
        self.assertAlmostEqual(facts([a, b], w)["in_water"], 0.25)  # its members' steps in water over all their steps

    def test_a_baby_is_in_its_parents_species(self):
        w = make_world(NO_FOOD)
        parent = add(w, 0.0, 0.0, energy=100.0)
        parent.species = 7
        w.species_book.species[7] = w.species_book.new_entry(7, parent=1, began=0)
        w._birth(parent)
        baby = w.creatures[-1]
        self.assertEqual((baby.parent_id, baby.species), (parent.id, 7))
        self.assertEqual(w.species_book.species[7]["births"], 1)

    def test_a_drifted_group_splits_off(self):
        w = make_world(NO_FOOD)
        everyone = herd(w)
        drift(everyone[:6])
        w.species_book.look(w, 250)
        self.assertEqual({c.species for c in everyone[:6]}, {2})
        self.assertEqual({c.species for c in everyone[6:]}, {1})
        new = w.species_book.species[2]
        self.assertEqual((new["parent"], new["began"], new["ended"]), (1, 250, None))
        self.assertEqual(new["counts"][-1], [250, 6])

    def test_one_odd_creature_does_not_split(self):
        w = make_world(NO_FOOD)
        everyone = herd(w)
        drift(everyone[:1])
        w.species_book.look(w, 250)
        self.assertEqual({c.species for c in everyone}, {1})

    def test_a_species_drifting_together_stays_one(self):
        w = make_world(NO_FOOD)
        everyone = herd(w)
        w.species_book.look(w, 250)
        drift(everyone)
        w.species_book.look(w, 500)
        self.assertEqual({c.species for c in everyone}, {1})
        self.assertEqual(len(w.species_book.species), 1)

    def test_a_species_with_no_one_left_ends(self):
        w = make_world(NO_FOOD)
        everyone = herd(w)
        drift(everyone[:6])
        w.species_book.look(w, 250)
        for c in everyone[:6]:
            w.kill(c, "hunger")
        w.species_book.look(w, 500)
        self.assertEqual(w.species_book.species[2]["ended"], 500)
        self.assertEqual(w.species_book.species[2]["deaths"]["hunger"], 6)
        self.assertIsNone(w.species_book.species[1]["ended"])

    def test_a_species_knows_its_facts(self):
        w = make_world(NO_FOOD)
        everyone = herd(w, 10)
        for k, c in enumerate(everyone):
            c.earned, c.spent, c.age, c.size = 20.0, 10.0, 100 * k, 1.0 + 0.1 * k
        w.kill(everyone[-1], "attack")
        w.species_book.look(w, 250)
        entry = w.species_book.species[1]
        facts = entry["facts"]
        self.assertEqual(facts["alive"], 9)
        self.assertAlmostEqual(facts["smart"], 2.0)  # earned ÷ spent
        self.assertEqual((facts["size"][0], facts["size"][2]), (1.0, 1.8))  # smallest, largest
        self.assertEqual(facts["oldest"], 800)
        self.assertEqual(entry["deaths"]["attack"], 1)
        self.assertIn("meat_gut", entry["now"])  # its typical member, gene by gene
        self.assertEqual(entry["now"]["eye_pixels"], 16.0)  # its eye's sharpness too
        self.assertEqual(entry["start"], entry["now"])  # its first look: it began like this


if __name__ == "__main__":
    unittest.main()
