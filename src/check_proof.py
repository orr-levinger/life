"""Check the spec's proof criteria against the scoreboards of the brain runs (and report the rules' yardstick).

    python -m src.check_proof --brain runs/1-brain/scoreboard.csv ... --rules runs/1-rules-L1/scoreboard.csv ...
"""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

from .scoreboard import numbers, read_rows

EXAM_RATIO_MIN = 2.0
EXAMS_COMPARED = 5
TOWARD_FOOD_MIN = 0.7
TOWARD_FOOD_ROWS = 10  # the last 10,000 steps
FLEE_LEVEL = 4
FLEE_MIN = 0.70
SEEDS_AT_FLEE_LEVEL = 4

Verdict = Tuple[str, Optional[bool], str]


def _exam_ratio(runs: Dict[str, List[Dict[str, str]]]) -> Verdict:
    ok, details = True, []
    for name, rows in runs.items():
        exams = numbers(rows, "exam_ratio")
        if len(exams) < 2 * EXAMS_COMPARED:
            ok = False
            details.append(f"{name}: only {len(exams)} exams")
            continue
        first, last = mean(exams[:EXAMS_COMPARED]), mean(exams[-EXAMS_COMPARED:])
        ok &= last >= EXAM_RATIO_MIN and last > first
        details.append(f"{name}: first {first:.2f} -> last {last:.2f}")
    return ("1 exam ratio", ok, "; ".join(details))


def _toward_food(runs: Dict[str, List[Dict[str, str]]]) -> Verdict:
    ok, details = True, []
    for name, rows in runs.items():
        values = numbers(rows[-TOWARD_FOOD_ROWS:], "toward_food")
        score = mean(values) if values else None
        ok &= score is not None and score >= TOWARD_FOOD_MIN
        details.append(f"{name}: {'-' if score is None else f'{score:.2f}'}")
    return ("2 toward food", ok, "; ".join(details))


def _flee(runs: Dict[str, List[Dict[str, str]]]) -> Verdict:
    reached, ok, details = 0, True, []
    for name, rows in runs.items():
        at_level = [r for r in rows if int(r["level"]) >= FLEE_LEVEL]
        if not at_level:
            details.append(f"{name}: never reached level {FLEE_LEVEL}")
            continue
        reached += 1
        values = numbers(at_level, "flee")
        score = mean(values) if values else None
        ok &= score is not None and score >= FLEE_MIN
        details.append(f"{name}: {'-' if score is None else f'{score:.0%}'}")
    ok &= reached >= SEEDS_AT_FLEE_LEVEL
    return ("3 flee at level 4+", ok, f"{reached} of {len(runs)} reached level {FLEE_LEVEL}; " + "; ".join(details))


def _survives(runs: Dict[str, List[Dict[str, str]]]) -> Verdict:
    ok, details = True, []
    for name, rows in runs.items():
        ok &= int(rows[-1]["population"]) > 0
        details.append(f"{name}: population {rows[-1]['population']} at the end")
    return ("4 survives", ok, "; ".join(details))


def _rules(paths: Sequence[Path]) -> Verdict:
    details = []
    for path in paths:
        rows = read_rows(path)
        bites = numbers(rows, "bites_per_creature_step")
        details.append(f"level {rows[-1]['level']}: {mean(bites):.3f} bites per creature-step" if bites
                       else f"{path}: no rows")
    return ("5 rules yardstick", None, "; ".join(details))


def check(brain_csvs: Sequence[Path], rules_csvs: Sequence[Path]) -> List[Verdict]:
    runs = {str(path): read_rows(path) for path in brain_csvs}
    return [_exam_ratio(runs), _toward_food(runs), _flee(runs), _survives(runs), _rules(rules_csvs)]


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
