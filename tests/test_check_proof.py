import csv
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

from src.check_proof import check
from src.scoreboard import ROW_KEYS

ROWS = 400  # a row every 250 steps: 100,000 steps


def brain_rows(intelligence: float = 2.5, first_intelligence: float = 0.5, wanderers: float = 1.0,
               toward: float = 0.8, flee: float = 0.8, hunters_at: int = 200, level_end: float = 30,
               generation_end: int = 40, late_newcomers: int = 0, alive_at_end: bool = True) -> List[Dict]:
    """A run's rows: the dial climbs evenly to `level_end`; hunters exist from row `hunters_at`; an exam every second
    row, the first five at `first_intelligence`."""
    rows = []
    for i in range(ROWS):
        step = (i + 1) * 250
        exam = step % 2000 == 0
        rows.append({
            "step": step, "level": round(level_end * (i + 1) / ROWS, 2), "population": 30, "energy": 0.6, "births": 5,
            "newcomers": late_newcomers if i >= ROWS - 20 else 0,
            "generation": round(generation_end * (i + 1) / ROWS),
            "bites_per_creature_step": 0.3, "toward_food": toward, "flee": flee, "direction_concentration": 0.1,
            "efficiency": 1.2,
            "intelligence": (first_intelligence if step <= 10000 else intelligence) if exam else "",
            "wanderers": wanderers if exam else "",
            "deaths_hunger": 3, "deaths_attack": 0, "deaths_age": 1,
            "max_food": 200, "food_energy": 30, "food_per_step": 3, "move_cost": 0.5, "step_cost": 0.05,
            "lifespan": 5000, "hunters": int(i >= hunters_at), "manual": 0,
        })
    if not alive_at_end:
        rows[-1]["population"] = 0
    return rows


def write(folder: Path, name: str, rows: List[Dict]) -> Path:
    path = folder / f"{name}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_KEYS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def verdicts(seed_rows: List[List[Dict]]) -> Dict[str, object]:
    with tempfile.TemporaryDirectory() as folder:
        brains = [write(Path(folder), f"brain{i}", rows) for i, rows in enumerate(seed_rows)]
        rules = [write(Path(folder), "rules1", brain_rows())]
        return {name: ok for name, ok, _ in check(brains, rules)}


def five(**changes) -> List[List[Dict]]:
    return [brain_rows(**changes) for _ in range(5)]


class TestCheckProof(unittest.TestCase):
    def test_all_pass(self):
        result = verdicts(five())
        self.assertEqual(sorted(result), ["1 survival got harder", "2 sustains itself", "3 generations",
                                          "4 intelligence", "5 steers to food", "6 flees hunters",
                                          "7 rules yardstick"])
        self.assertEqual([result[k] for k in sorted(result)][:6], [True] * 6)

    def test_a_world_that_stayed_easy_fails(self):
        self.assertFalse(verdicts(five(level_end=9.9))["1 survival got harder"])
        self.assertTrue(verdicts(five(level_end=10.0))["1 survival got harder"])

    def test_late_newcomers_fail(self):
        self.assertFalse(verdicts(five(late_newcomers=2))["2 sustains itself"])

    def test_extinct_at_end_fails(self):
        self.assertFalse(verdicts(five(alive_at_end=False))["2 sustains itself"])

    def test_few_generations_fail(self):
        self.assertFalse(verdicts(five(generation_end=12))["3 generations"])

    def test_low_intelligence_fails(self):
        self.assertFalse(verdicts(five(intelligence=1.8))["4 intelligence"])

    def test_intelligence_is_read_against_the_wanderers(self):
        self.assertFalse(verdicts(five(intelligence=2.5, wanderers=1.5))["4 intelligence"])
        self.assertTrue(verdicts(five(intelligence=1.0, first_intelligence=0.2, wanderers=0.4))["4 intelligence"])

    def test_flat_intelligence_fails(self):
        self.assertFalse(verdicts(five(intelligence=2.5, first_intelligence=2.5))["4 intelligence"])

    def test_low_steering_fails(self):
        self.assertFalse(verdicts(five(toward=0.6))["5 steers to food"])

    def test_windows_are_counted_in_steps(self):
        rows = brain_rows()
        for row in rows[-40:]:  # the last 10,000 steps steer badly; the 10,000 before them steer well
            row["toward_food"] = 0.1
        self.assertFalse(verdicts([rows] * 5)["5 steers to food"])
        late = brain_rows()
        late[-81]["newcomers"] = 3  # 20,250 steps before the end: outside the last 20,000
        self.assertTrue(verdicts([late] * 5)["2 sustains itself"])

    def test_three_seeds_reaching_hunters_fail(self):
        seeds = five()
        seeds[0] = brain_rows(hunters_at=ROWS)
        seeds[1] = brain_rows(hunters_at=ROWS)
        self.assertFalse(verdicts(seeds)["6 flees hunters"])

    def test_low_flee_fails(self):
        self.assertFalse(verdicts(five(flee=0.6))["6 flees hunters"])

    def test_rules_reported_not_judged(self):
        self.assertIsNone(verdicts(five())["7 rules yardstick"])


if __name__ == "__main__":
    unittest.main()
