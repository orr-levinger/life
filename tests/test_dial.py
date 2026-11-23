import unittest
from dataclasses import fields

from src.dial import HUNTERS_AT, KNOBS, Dial, nudged
from src.settings import START, Settings


def changed_fields(a: Settings, b: Settings) -> list:
    return [f.name for f in fields(Settings) if getattr(a, f.name) != getattr(b, f.name)]


def harder(old: Settings, new: Settings, field: str) -> bool:
    """Whether `field` moved in the direction that makes survival harder."""
    before, after = getattr(old, field), getattr(new, field)
    if field in ("move_cost", "step_cost"):
        return after > before
    if field == "attacks":
        return after and not before
    return after < before  # less food, poorer food, shorter lives


class TestDial(unittest.TestCase):
    def test_starts_at_the_easy_start(self):
        dial = Dial()
        self.assertEqual((dial.level, dial.settings), (0, START))

    def test_no_deaths_nudges_exactly_one_setting(self):
        dial = Dial()
        self.assertEqual(dial.review(deaths=0, alive=40, newcomers=0), "harder")
        self.assertEqual(dial.level, 1)
        changed = changed_fields(START, dial.settings)
        self.assertEqual(len(changed), 1)
        self.assertTrue(harder(START, dial.settings, changed[0]))
        old, new = getattr(START, changed[0]), getattr(dial.settings, changed[0])
        self.assertAlmostEqual(new / old if not isinstance(old, tuple) else new[0] / old[0], 0.95, delta=0.11)

    def test_nudges_take_turns(self):
        dial = Dial()
        seen = []
        for _ in range(len(KNOBS)):
            before = dial.settings
            dial.review(0, 40, 0)
            seen.extend(changed_fields(before, dial.settings))
        self.assertEqual(sorted(seen), sorted(name for name, _, _ in KNOBS))  # each setting once per turn

    def test_too_many_deaths_take_the_last_nudge_back(self):
        dial = Dial()
        dial.review(0, 40, 0)
        after_one = dial.settings
        dial.review(0, 40, 0)
        self.assertEqual(dial.review(deaths=11, alive=40, newcomers=0), "easier")  # more than a quarter died
        self.assertEqual((dial.level, dial.settings), (1, after_one))
        self.assertEqual(dial.review(deaths=11, alive=40, newcomers=0), "easier")
        self.assertEqual((dial.level, dial.settings), (0, START))

    def test_newcomers_take_the_last_nudge_back(self):
        dial = Dial()
        dial.review(0, 40, 0)
        self.assertEqual(dial.review(deaths=2, alive=9, newcomers=3), "easier")
        self.assertEqual(dial.level, 0)

    def test_some_deaths_hold(self):
        dial = Dial()
        dial.review(0, 40, 0)
        settings = dial.settings
        self.assertEqual(dial.review(deaths=5, alive=40, newcomers=0), "hold")  # some died, not too many
        self.assertEqual((dial.level, dial.settings), (1, settings))

    def test_never_easier_than_the_start(self):
        dial = Dial()
        self.assertEqual(dial.review(deaths=30, alive=40, newcomers=5), "hold")
        self.assertEqual((dial.level, dial.settings), (0, START))

    def test_settings_stay_within_limits(self):
        settings = START
        for n in range(400):
            settings = nudged(settings, n)
        self.assertGreaterEqual(settings.max_food, 20)
        self.assertGreater(settings.food_energy[0], 0)
        self.assertGreater(settings.food_per_step, 0)
        self.assertGreaterEqual(settings.lifespan[0], 500)
        self.assertLessEqual(settings.move_cost, 3.0)
        self.assertLessEqual(settings.step_cost, 0.5)

    def test_hunters_come_as_their_own_nudge(self):
        settings = START
        for n in range(HUNTERS_AT):
            settings = nudged(settings, n)
            self.assertFalse(settings.attacks)
        with_hunters = nudged(settings, HUNTERS_AT)
        self.assertEqual(changed_fields(settings, with_hunters), ["attacks"])
        self.assertTrue(with_hunters.attacks)

    def test_max_level_stops_the_dial(self):
        dial = Dial(max_level=2)
        for _ in range(5):
            dial.review(0, 40, 0)
        self.assertEqual(dial.level, 2)

    def test_turn_to_is_the_same_as_that_many_nudges(self):
        nudged_by_reviews, turned = Dial(), Dial()
        for _ in range(7):
            nudged_by_reviews.review(0, 40, 0)
        turned.turn_to(7)
        self.assertEqual((turned.level, turned.settings), (7, nudged_by_reviews.settings))


if __name__ == "__main__":
    unittest.main()
