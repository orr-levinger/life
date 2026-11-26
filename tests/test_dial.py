import unittest
from dataclasses import fields, replace

from src.dial import (KNOBS, Dial, difficulty, rates, slope, step_for, value,
                      with_value)
from src.settings import START, Settings

from src.rules import rule  # noqa: E402
from src.rules import starting_rules  # noqa: E402
from unittest import mock  # noqa: E402

from tests.helpers import SMALL_WORLD  # noqa: E402

RULES = {**starting_rules(), **SMALL_WORLD}  # Auto's arithmetic is checked around a target of 64
_RULES = mock.patch("src.dial.starting_rules", lambda: dict(RULES))  # a new Dial starts on them too


def setUpModule():
    _RULES.start()


def tearDownModule():
    _RULES.stop()


TARGET_ALIVE, HUNTERS_AT, MAX_NUDGES = RULES["auto_target"], RULES["auto_hunters_at"], round(RULES["auto_max_nudges"])
EASY = (TARGET_ALIVE + 8, 0.0, 0.0, 0)  # 8 above the target, steady: one nudge harder
HARD = (TARGET_ALIVE - 8, 0.0, 0.0, 0)  # 8 below the target, steady: one nudge easier


def changed_fields(a: Settings, b: Settings) -> list:
    return [f.name for f in fields(Settings) if getattr(a, f.name) != getattr(b, f.name)]


def knob(name: str):
    return next(k for k in KNOBS if k.name == name)


def is_harder(old: Settings, new: Settings, name: str) -> bool:
    if name == "attacks":
        return new.attacks and not old.attacks
    return (value(new, name) - value(old, name)) * knob(name).harder > 0


def rows(alive, energy, start=250):
    """Scoreboard points 250 steps apart."""
    return [{"step": start + 250 * i, "population": a, "energy": e} for i, (a, e) in enumerate(zip(alive, energy))]


class TestRates(unittest.TestCase):
    def test_slope_of_a_straight_line(self):
        self.assertAlmostEqual(slope([(250, 10.0), (500, 12.0), (750, 14.0), (1000, 16.0)]), 8.0)  # per 1,000 steps
        self.assertIsNone(slope([(250, 10.0)]))

    def test_one_odd_point_does_not_swing_it(self):
        steady = slope([(250, 60.0), (500, 60.0), (750, 60.0), (1000, 60.0)])
        odd = slope([(250, 60.0), (500, 70.0), (750, 60.0), (1000, 60.0)])
        self.assertEqual(steady, 0.0)
        self.assertLess(abs(odd), 10.0)  # a jump of 10 in one point moves the slope by less than that

    def test_rates_are_shares_and_points_per_1000_steps(self):
        alive_rate, energy_rate = rates(rows([60, 63, 66, 69], [0.5, 0.45, 0.4, 0.35]))
        self.assertAlmostEqual(alive_rate, 12.0 / 64.5)  # 12 more a 1,000 steps, out of about 64
        self.assertAlmostEqual(energy_rate, -0.2)  # 20 points less a 1,000 steps
        self.assertEqual(rates(rows([60], [0.5])), (None, None))


class TestStep(unittest.TestCase):
    def test_at_the_target_and_steady_holds(self):
        self.assertEqual(step_for(64, 0.0, 0.0, 0, RULES), 0)

    def test_full_and_steady_two_harder(self):
        self.assertEqual(step_for(80, 0.0, 0.0, 0, RULES), 2)

    def test_forty_and_steady_three_easier(self):
        self.assertEqual(step_for(40, 0.0, 0.0, 0, RULES), -3)

    def test_recovering_world_is_left_alone(self):
        self.assertEqual(step_for(50, 0.20, 0.0, 0, RULES), 0)  # below the target but rising 20 % a 1,000 steps

    def test_the_owner_s_crash_eases_four(self):
        crash = rows([80, 68, 55, 45], [0.82, 0.65, 0.48, 0.36], start=16250)
        alive_rate, energy_rate = rates(crash)
        self.assertEqual(step_for(45, alive_rate, energy_rate, 0, RULES), -4)

    def test_falling_energy_eases(self):
        self.assertEqual(step_for(64, 0.0, -0.10, 0, RULES), -2)  # 10 points a 1,000 steps

    def test_rising_energy_alone_never_eases(self):
        self.assertEqual(step_for(64, 0.0, 0.30, 0, RULES), 0)

    def test_newcomers_ease_four(self):
        self.assertEqual(step_for(80, 0.5, 0.2, 2, RULES), -MAX_NUDGES)

    def test_never_more_than_four(self):
        self.assertEqual(step_for(80, 1.0, 0.0, 0, RULES), MAX_NUDGES)
        self.assertEqual(step_for(10, -1.0, -1.0, 0, RULES), -MAX_NUDGES)

    def test_nothing_measured_counts_as_steady(self):
        self.assertEqual(step_for(72, None, None, 0, RULES), 1)


class TestDial(unittest.TestCase):
    def test_food_sliders_reach_further(self):  # the owner, 2026-11-26: tops 800 pieces, 20 a step, 100 energy
        ends = {k.name: k.easiest for k in KNOBS}
        self.assertEqual((ends["max_food"], ends["food_per_step"], ends["food_energy"]), (800.0, 20.0, 100.0))
        self.assertEqual((START.max_food, START.food_per_step, START.food_energy), (400, 6.0, (20, 40)))  # starts as before

    def test_fruit_slider_reaches_one(self):  # the owner, 2026-11-26
        self.assertEqual({k.name: k.easiest for k in KNOBS}["fruit_rate"], 1.0)
        self.assertEqual(START.fruit_rate, 0.02)  # a new world starts as before

    def test_starts_at_the_easy_start(self):
        dial = Dial()
        self.assertEqual((dial.level, dial.settings), (0, START))

    def test_a_step_of_two_nudges_two_knobs(self):
        dial = Dial()
        self.assertEqual(dial.review(80, 0.0, 0.0, 0), 2)
        changed = changed_fields(START, dial.settings)
        self.assertEqual(len(changed), 2)
        self.assertTrue(all(is_harder(START, dial.settings, name) for name in changed))
        self.assertAlmostEqual(dial.level, 2.0)

    def test_harder_nudges_take_turns(self):
        dial = Dial()
        seen = []
        for _ in range(len(KNOBS)):
            before = dial.settings
            dial.review(*EASY)
            seen.extend(changed_fields(before, dial.settings))
        self.assertEqual(sorted(seen), sorted(k.name for k in KNOBS))  # each knob once per turn

    def test_easier_takes_back_the_newest_harder_nudges_first(self):
        dial = Dial()
        dial.review(*EASY)
        after_one = dial.settings
        dial.review(80, 0.0, 0.0, 0)  # two more
        self.assertEqual(dial.review(48, 0.0, 0.0, 0), -2)  # 16 below: two back
        self.assertEqual(dial.settings, after_one)
        dial.review(*HARD)
        self.assertEqual(dial.settings, START)

    def test_holds_at_the_target(self):
        dial = Dial()
        dial.review(*EASY)
        settings = dial.settings
        self.assertEqual(dial.review(64, 0.0, 0.0, 0), 0)
        self.assertEqual(dial.settings, settings)

    def test_auto_can_go_easier_than_the_start(self):
        dial = Dial()
        self.assertEqual(dial.review(*HARD), -1)
        changed = changed_fields(START, dial.settings)
        self.assertEqual(len(changed), 1)
        self.assertFalse(is_harder(START, dial.settings, changed[0]))
        self.assertAlmostEqual(dial.level, -1.0)

    def test_knobs_stay_within_their_ends(self):
        hard, easy = Dial(), Dial()
        for _ in range(200):
            hard.review(80, 1.0, 0.0, 0)
            easy.review(10, -1.0, -1.0, 0)
        for k in KNOBS:
            low, high = sorted((k.easiest, k.hardest))
            self.assertAlmostEqual(value(hard.settings, k.name), k.hardest, msg=k.name)
            self.assertAlmostEqual(value(easy.settings, k.name), k.easiest, msg=k.name)
            for dial in (hard, easy):
                self.assertTrue(low - 1e-9 <= value(dial.settings, k.name) <= high + 1e-9, k.name)
        self.assertTrue(hard.settings.attacks)
        self.assertEqual(hard.review(80, 1.0, 0.0, 0), 0)  # nothing left to make harder

    def test_hunters_come_as_their_own_nudge(self):
        dial = Dial()
        while dial.level < HUNTERS_AT - 1e-9:
            dial.review(*EASY)
            self.assertFalse(dial.settings.attacks)
        before = dial.settings
        dial.review(*EASY)
        self.assertEqual(changed_fields(before, dial.settings), ["attacks"])

    def test_difficulty_counts_the_nudges(self):
        dial = Dial()
        for n in range(1, 13):
            dial.review(*EASY)
            self.assertAlmostEqual(difficulty(dial.settings, 0.05), n)
        self.assertAlmostEqual(difficulty(replace(START, attacks=True), 0.05), 1.0)

    def test_difficulty_of_the_owner_s_settings(self):
        settings = with_value(START, "food_energy", 30 * 0.95 ** 3)
        self.assertAlmostEqual(difficulty(settings, 0.05), 3.0)
        settings = with_value(settings, "lifespan", 5000 / 0.95 ** 2)
        self.assertAlmostEqual(difficulty(settings, 0.05), 1.0)

    def test_set_forgets_the_nudges_to_take_back(self):
        dial = Dial()
        dial.review(*EASY)
        mine = with_value(START, "max_food", 150)
        dial.set(mine)
        self.assertEqual(dial.settings, mine)
        dial.review(*HARD)  # nothing to take back: an easier nudge from the owner's settings
        changed = changed_fields(mine, dial.settings)
        self.assertEqual(len(changed), 1)
        self.assertFalse(is_harder(mine, dial.settings, changed[0]))

    def test_a_range_keeps_its_shape(self):
        settings = with_value(START, "food_energy", 15)
        self.assertEqual(settings.food_energy, (10.0, 20.0))
        self.assertEqual(with_value(START, "max_food", 10 ** 6).max_food, knob("max_food").easiest)

    def test_max_level_stops_the_dial(self):
        dial = Dial(max_level=2)
        for _ in range(5):
            dial.review(80, 0.0, 0.0, 0)
        self.assertAlmostEqual(dial.level, 2)

    def test_turn_to_is_the_same_as_that_many_nudges(self):
        by_reviews, turned = Dial(), Dial()
        for _ in range(7):
            by_reviews.review(*EASY)
        turned.turn_to(7)
        self.assertEqual(turned.settings, by_reviews.settings)


    def test_owner_s_hunters_stay_as_set(self):
        dial = Dial()
        dial.set_hunters(True)
        self.assertTrue(dial.settings.attacks)
        for _ in range(6):
            dial.review(*HARD)  # easing never takes the owner's hunters away
        self.assertTrue(dial.settings.attacks)
        off = Dial()
        off.set_hunters(False)
        while off.level < HUNTERS_AT + 2:
            off.review(*EASY)  # nor does Auto switch them on at difficulty 24
        self.assertFalse(off.settings.attacks)

    def test_the_owner_s_hunters_are_kept_in_a_save(self):
        dial = Dial()
        dial.set_hunters(True)
        dial.review(*EASY)
        again = Dial.from_state(dial.state())
        again.review(*HARD)
        again.review(*HARD)
        self.assertTrue(again.settings.attacks)

if __name__ == "__main__":
    unittest.main()
