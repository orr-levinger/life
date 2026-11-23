"""Check the design's proof criteria against the scoreboards of a species' runs (and report the rules' yardstick).

    python -m src.check_proof --brain runs/1-learning/scoreboard.csv ... --rules runs/1-rules-L0/scoreboard.csv ...
"""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

from .scoreboard import numbers, read_rows

LEVEL_MIN = 10  # the dial must have turned at least this far by the end
SUSTAIN_ROWS = 20  # the last 20,000 steps
GENERATION_MIN = 20
INTELLIGENCE_MIN = 2.0  # times the random wanderers' energy earned ÷ energy spent in the same exam
EXAMS_COMPARED = 5
TOWARD_FOOD_MIN = 0.7
TOWARD_FOOD_ROWS = 10  # the last 10,000 steps
FLEE_MIN = 0.70
SEEDS_WITH_HUNTERS = 4

Verdict = Tuple[str, Optional[bool], str]
Runs = Dict[str, List[Dict[str, str]]]


def _harder(runs: Runs) -> Verdict:
    levels = {name: int(rows[-1]["level"]) for name, rows in runs.items()}
    return ("1 survival got harder", all(level >= LEVEL_MIN for level in levels.values()),
            "; ".join(f"{name}: level {level} at the end" for name, level in levels.items()))


def _sustains(runs: Runs) -> Verdict:
    ok, details = True, []
    for name, rows in runs.items():
        late = sum(int(r["newcomers"]) for r in rows[-SUSTAIN_ROWS:])
        alive = int(rows[-1]["population"])
        ok &= late == 0 and alive > 0
        details.append(f"{name}: {late} newcomers in the last rows, {alive} alive at the end")
    return ("2 sustains itself", ok, "; ".join(details))


def _generations(runs: Runs) -> Verdict:
    ok, details = True, []
    for name, rows in runs.items():
        deepest = numbers(rows[-1:], "generation")
        ok &= bool(deepest) and deepest[0] >= GENERATION_MIN
        details.append(f"{name}: generation {int(deepest[0]) if deepest else '-'}")
    return ("3 generations", ok, "; ".join(details))


def _exam_score(runs: Runs) -> Verdict:
    ok, details = True, []
    for name, rows in runs.items():
        exams, wanderers = numbers(rows, "intelligence"), numbers(rows, "wanderers")
        if len(exams) < 2 * EXAMS_COMPARED or not wanderers:
            ok = False
            details.append(f"{name}: only {len(exams)} exams")
            continue
        first, last, yardstick = mean(exams[:EXAMS_COMPARED]), mean(exams[-EXAMS_COMPARED:]), mean(wanderers)
        ok &= last >= INTELLIGENCE_MIN * yardstick and last > first
        details.append(f"{name}: first {first:.2f} -> last {last:.2f} (random wanderers {yardstick:.2f})")
    return ("4 intelligence", ok, "; ".join(details))


def _steers(runs: Runs) -> Verdict:
    ok, details = True, []
    for name, rows in runs.items():
        values = numbers(rows[-TOWARD_FOOD_ROWS:], "toward_food")
        score = mean(values) if values else None
        ok &= score is not None and score >= TOWARD_FOOD_MIN
        details.append(f"{name}: {'-' if score is None else f'{score:.2f}'}")
    return ("5 steers to food", ok, "; ".join(details))


def _flees(runs: Runs) -> Verdict:
    reached, ok, details = 0, True, []
    for name, rows in runs.items():
        hunted = [r for r in rows if int(r["hunters"]) == 1]
        if not hunted:
            details.append(f"{name}: hunters never came")
            continue
        reached += 1
        values = numbers(hunted, "flee")
        score = mean(values) if values else None
        ok &= score is not None and score >= FLEE_MIN
        details.append(f"{name}: {'-' if score is None else f'{score:.0%}'}")
    ok &= reached >= SEEDS_WITH_HUNTERS
    return ("6 flees hunters", ok, f"{reached} of {len(runs)} lived with hunters; " + "; ".join(details))


def _rules(paths: Sequence[Path]) -> Verdict:
    details = []
    for path in paths:
        rows = read_rows(path)
        bites = numbers(rows, "bites_per_creature_step")
        details.append(f"level {rows[-1]['level']}: {mean(bites):.3f} bites per creature-step" if bites
                       else f"{path}: no rows")
    return ("7 rules yardstick", None, "; ".join(details))


def check(brain_csvs: Sequence[Path], rules_csvs: Sequence[Path]) -> List[Verdict]:
    runs = {str(path): read_rows(path) for path in brain_csvs}
    return [_harder(runs), _sustains(runs), _generations(runs), _exam_score(runs), _steers(runs), _flees(runs),
            _rules(rules_csvs)]


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Check the proof criteria.")
    parser.add_argument("--brain", nargs="+", type=Path, required=True)
    parser.add_argument("--rules", nargs="*", type=Path, default=[])
    args = parser.parse_args(argv)
    for name, ok, detail in check(args.brain, args.rules):
        label = "INFO" if ok is None else "PASS" if ok else "FAIL"
        print(f"{label}  {name}: {detail}")


if __name__ == "__main__":
    main()
