"""Compare kinds of runs: one row per name, averaged over its seeds, from the runs' scoreboards.

    python -m src.compare --runs learning runs/1-learning/scoreboard.csv ... --runs instinct runs/1-instinct/scoreboard.csv ...
"""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Union

from .check_proof import EXAMS_COMPARED, TOWARD_FOOD_STEPS
from .scoreboard import last_steps, numbers, read_rows

Cell = Union[str, int, float, None]
COLUMNS = ("name", "seeds", "intelligence", "wanderers", "toward_food", "flee", "level", "generation", "minutes")


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    known = [value for value in values if value is not None]
    return mean(known) if known else None


def _run_minutes(csv_path: Path) -> Optional[float]:
    timing = csv_path.parent / "time.txt"
    return float(timing.read_text()) / 60.0 if timing.exists() else None


def compare(runs: Dict[str, Sequence[Path]]) -> List[Dict[str, Cell]]:
    """Per name, the mean over its seeds of: intelligence (energy earned ÷ energy spent in the exam) over the last 5
    exams and the random wanderers' in the same exams, the steering score over the last
    10,000 steps, the flee score over rows with hunters, the dial's level and the deepest generation at the end; plus
    the minutes a run took. A score no seed could measure is None."""
    table = []
    for name, paths in runs.items():
        exam, wanderers, toward, flee, levels, generation, minutes = [], [], [], [], [], [], []
        for path in paths:
            rows = read_rows(path)
            exam.append(_mean(numbers(rows, "intelligence")[-EXAMS_COMPARED:]))
            wanderers.append(_mean(numbers(rows, "wanderers")[-EXAMS_COMPARED:]))
            toward.append(_mean(numbers(last_steps(rows, TOWARD_FOOD_STEPS), "toward_food")))
            flee.append(_mean(numbers([r for r in rows if int(r["hunters"]) == 1], "flee")))
            levels.append(float(rows[-1]["level"]))
            generation.append(_mean(numbers(rows[-1:], "generation")))
            minutes.append(_run_minutes(path))
        table.append({"name": name, "seeds": len(paths), "intelligence": _mean(exam), "wanderers": _mean(wanderers),
                      "toward_food": _mean(toward),
                      "flee": _mean(flee), "level": _mean(levels), "generation": _mean(generation),
                      "minutes": _mean(minutes)})
    return table


def format_table(table: List[Dict[str, Cell]]) -> str:
    def show(value: Cell) -> str:
        return "-" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)

    lines = [" | ".join(f"{column:>13}" for column in COLUMNS)]
    lines += [" | ".join(f"{show(row[column]):>13}" for column in COLUMNS) for row in table]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compare kinds of runs from their scoreboards.")
    parser.add_argument("--runs", nargs="+", action="append", required=True, metavar=("NAME", "CSV"),
                        help="a name followed by its runs' scoreboard.csv files; repeat per name")
    args = parser.parse_args(argv)
    print(format_table(compare({name: [Path(p) for p in paths] for name, *paths in args.runs})))


if __name__ == "__main__":
    main()
