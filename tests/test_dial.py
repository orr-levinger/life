import unittest
from dataclasses import fields, replace

from src.dial import FEW, HUNGRY, HUNTERS_AT, KNOBS, MANY, WELL_FED, Dial, difficulty, value, with_value
from src.settings import START, Settings

EASY_LIFE = (MANY, WELL_FED + 0.1, 0)  # many creatures, well fed, no newcomers
HARD_LIFE = (FEW - 1, WELL_FED + 0.1, 0)  # too few
IN_BETWEEN = ((MANY + FEW) // 2, (WELL_FED + HUNGRY) / 2, 0)


def changed_fields(a: Settings, b: Settings) -> list:
    return [f.name for f in fields(Settings) if getattr(a, f.name) != getattr(b, f.name)]


def knob(name: str):
    return next(k for k in KNOBS if k.name == name)


def is_harder(old: Settings, new: Settings, name: str) -> bool:
    if name == "attacks":
        return new.attacks and not old.attacks
    return (value(new, name) - value(old, name)) * knob(name).harder > 0


class TestDial(unittest.TestCase):
    def test_starts_at_the_easy_start(self):
        dial = Dial()
        self.assertEqual((dial.level, dial.settings), (0, START))

    def test_many_and_well_fed_nudges_one_knob_harder(self):
        dial = Dial()
        self.assertEqual(dial.review(*EASY_LIFE), "harder")
        changed = changed_fields(START, dial.settings)
        self.assertEqual(len(changed), 1)
        self.assertTrue(is_harder(START, dial.settings, changed[0]))
        ratio = value(dial.settings, changed[0]) / value(START, changed[0])
        self.assertAlmostEqual(ratio, 0.95 if knob(changed[0]).harder < 0 else 1.05)
        self.assertAlmostEqual(dial.level, 1.0)

    def test_nudges_take_turns(self):
        dial = Dial()
        seen = []
        for _ in range(len(KNOBS)):
            before = dial.settings
            dial.review(*EASY_LIFE)
            seen.extend(changed_fields(before, dial.settings))
        self.assertEqual(sorted(seen), sorted(k.name for k in KNOBS))  # each knob once per turn

    def test_few_takes_the_last_nudge_back(self):
        dial = Dial()
        dial.review(*EASY_LIFE)
        after_one = dial.settings
        dial.review(*EASY_LIFE)
        self.assertEqual(dial.review(*HARD_LIFE), "easier")
        self.assertEqual(dial.settings, after_one)
        self.assertEqual(dial.review(*HARD_LIFE), "easier")
        self.assertEqual(dial.settings, START)

    def test_hungry_takes_the_last_nudge_back(self):
        dial = Dial()
        dial.review(*EASY_LIFE)
        self.assertEqual(dial.review(MANY + 10, HUNGRY - 0.01, 0), "easier")
        self.assertEqual(dial.settings, START)

    def test_newcomers_take_the_last_nudge_back(self):
        dial = Dial()
        dial.review(*EASY_LIFE)
        self.assertEqual(dial.review(MANY, WELL_FED + 0.1, 2), "easier")
        self.assertEqual(dial.settings, START)

    def test_in_between_holds(self):
        dial = Dial()
        dial.review(*EASY_LIFE)
        settings = dial.settings
        self.assertEqual(dial.review(*IN_BETWEEN), "hold")
        self.assertEqual(dial.review(MANY, None, 0), "hold")  # nobody measured: no change
        self.assertEqual(dial.settings, settings)

    def test_auto_can_go_easier_than_the_start(self):
        dial = Dial()
        self.assertEqual(dial.review(*HARD_LIFE), "easier")
        changed = changed_fields(START, dial.settings)
        self.assertEqual(len(changed), 1)
        self.assertFalse(is_harder(START, dial.settings, changed[0]))
        self.assertAlmostEqual(dial.level, -1.0)
        dial.review(*HARD_LIFE)
        self.assertAlmostEqual(dial.level, -2.0)
        self.assertEqual(len(changed_fields(START, dial.settings)), 2)  # easier nudges take turns too

    def test_knobs_stay_within_their_ends(self):
        hard, easy = Dial(), Dial()
        for _ in range(500):
            hard.review(*EASY_LIFE)
            easy.review(*HARD_LIFE)
        for k in KNOBS:
            low, high = sorted((k.easiest, k.hardest))
            self.assertAlmostEqual(value(hard.settings, k.name), k.hardest, msg=k.name)
            self.assertAlmostEqual(value(easy.settings, k.name), k.easiest, msg=k.name)
            for dial in (hard, easy):
                self.assertTrue(low - 1e-9 <= value(dial.settings, k.name) <= high + 1e-9, k.name)
        self.assertTrue(hard.settings.attacks)
        self.assertEqual(hard.review(*EASY_LIFE), "hold")  # nothing left to make harder

    def test_hunters_come_as_their_own_nudge(self):
        dial = Dial()
        while dial.level < HUNTERS_AT - 1e-9:
            dial.review(*EASY_LIFE)
            self.assertFalse(dial.settings.attacks)
        before = dial.settings
        dial.review(*EASY_LIFE)
        self.assertEqual(changed_fields(before, dial.settings), ["attacks"])
        self.assertAlmostEqual(dial.level, HUNTERS_AT + 1)

    def test_difficulty_counts_the_nudges(self):
        dial = Dial()
        for n in range(1, 13):
            dial.review(*EASY_LIFE)
            self.assertAlmostEqual(difficulty(dial.settings), n)
        self.assertAlmostEqual(difficulty(replace(START, attacks=True)), 1.0)

    def test_difficulty_of_the_owner_s_settings(self):
        settings = with_value(START, "food_energy", 30 * 0.95 ** 3)  # three nudges' worth less energy per food
        self.assertAlmostEqual(difficulty(settings), 3.0)
        settings = with_value(settings, "lifespan", 5000 / 0.95 ** 2)  # two nudges' worth longer lives
        self.assertAlmostEqual(difficulty(settings), 1.0)

    def test_set_forgets_the_nudges_to_take_back(self):
        dial = Dial()
        dial.review(*EASY_LIFE)
        mine = with_value(START, "max_food", 150)
        dial.set(mine)
        self.assertEqual(dial.settings, mine)
        dial.review(*HARD_LIFE)  # nothing to take back: an easier nudge from the owner's settings
        changed = changed_fields(mine, dial.settings)
        self.assertEqual(len(changed), 1)
        self.assertFalse(is_harder(mine, dial.settings, changed[0]))

    def test_a_range_keeps_its_shape(self):
        settings = with_value(START, "food_energy", 15)
        self.assertEqual(settings.food_energy, (10.0, 20.0))
        self.assertEqual(value(settings, "food_energy"), 15)
        self.assertEqual(with_value(START, "max_food", 10 ** 6).max_food, knob("max_food").easiest)  # held at its end

    def test_max_level_stops_the_dial(self):
        dial = Dial(max_level=2)
        for _ in range(5):
            dial.review(*EASY_LIFE)
        self.assertAlmostEqual(dial.level, 2)

    def test_turn_to_is_the_same_as_that_many_nudges(self):
        by_reviews, turned = Dial(), Dial()
        for _ in range(7):
            by_reviews.review(*EASY_LIFE)
        turned.turn_to(7)
        self.assertEqual(turned.settings, by_reviews.settings)


if __name__ == "__main__":
    unittest.main()
