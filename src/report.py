"""A world's report (the owner, 2026-11-27: "every 500 steps, report the state of the game ... any errors, any important
events, just report back"): a small `report/` folder in its run's folder, rewritten in place and trimmed so it never
grows, for the owner's Mac to copy to their S3 bucket (src/upload.py) and Claude to read.

- `state.jsonl`: a line every `report_every` steps: how many alive, the species, the deepest generation, how smart,
  the deaths since the last line, Auto's level, every knob and cost, the brain sizes (the last `STATE_KEEP`).
- `events.jsonl`: starts, resumes, updates and stops; species splitting off and ending; Auto's nudges (the last
  `EVENTS_KEEP`).
- `errors.jsonl`: every error in the world with its traceback (the last `ERRORS_KEEP`).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from .rules import COST_NAMES

STATE_KEEP, EVENTS_KEEP, ERRORS_KEEP = 500, 1000, 100
MIN_SPENT = 5.0  # a creature's earned ÷ spent counts once it has spent this much (as the page shows it)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def knob_changes(before: Mapping[str, float], after: Mapping[str, float]) -> Dict[str, list]:
    """The knobs and costs that moved: name → [from, to]."""
    return {name: [before[name], after[name]] for name in after if name in before and before[name] != after[name]}


class Reporter:
    def __init__(self, run_folder: Path) -> None:
        self.folder = Path(run_folder) / "report"
        self._deaths_seen: Dict[str, int] = {}

    def _append(self, name: str, record: Dict[str, Any], keep: int) -> None:
        """One line more in `name`, the oldest dropped past `keep` (written whole to a temporary file, then renamed)."""
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / name
        old = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        kept = (old + [json.dumps(record)])[-keep:]
        temporary = path.with_suffix(".tmp")
        temporary.write_text("\n".join(kept) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def event(self, kind: str, **what: Any) -> None:
        self._append("events.jsonl", {"kind": kind, **what, "time": _now()}, EVENTS_KEEP)

    def error(self, step: int, traceback: str) -> None:
        self._append("errors.jsonl", {"step": step, "traceback": traceback, "time": _now()}, ERRORS_KEEP)

    def state(self, world: Any, level: float, knobs: Mapping[str, float], step: int) -> None:
        """The world now, as one line."""
        alive = world.creatures
        smart = [c.earned / c.spent for c in alive if c.spent >= MIN_SPENT]
        moving = [c for c in alive if c.intent != "REST"]
        deaths = {cause: count - self._deaths_seen.get(cause, 0) for cause, count in world.deaths.items()}
        self._deaths_seen = dict(world.deaths)
        book = world.species_book
        thinking = [c for c in alive if c.layers]
        self._append("state.jsonl", {
            "step": step, "time": _now(), "alive": len(alive),
            "species_alive": len({c.species for c in alive}), "species_ever": len(book.species),
            "generation": max((c.generation for c in alive), default=0),
            "smart": round(float(np.mean(smart)), 4) if smart else None,
            "to_food": round(sum(c.intent == "GO_TO_FOOD" for c in moving) / len(moving), 4) if moving else None,
            "deaths": deaths, "level": round(float(level), 2),
            "knobs": {name: round(float(v), 6) for name, v in knobs.items()},
            "costs": {name: round(float(world.rules[name]), 6) for name in COST_NAMES},
            "neurons": round(float(np.mean([c.neurons for c in thinking])), 1) if thinking else None,
            "layers": round(float(np.mean([c.layers for c in thinking])), 2) if thinking else None,
        }, STATE_KEEP)
