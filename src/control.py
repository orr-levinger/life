"""How the live page talks to a running world: one small JSON file in the run's folder.

The page (through the viewer's server) writes it; the world reads it every few steps. It says the mode (Auto turns the
knobs itself, Manual leaves them to the owner), the knobs the owner set, whether to stop or pause, and whether the
owner has jumped into the world to watch it slowly (until when, and how many steps a second).
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Mapping, Optional

from .dial import KNOBS, knob_named, value, with_value
from .settings import START, Settings

MODES = ("auto", "manual")
CONTROL_FILE = "control.json"
HUNTERS = "hunters"  # the on/off switch beside the knobs
REPLACE_TRIES = 50  # the run may be reading the file at the moment it is replaced (Windows refuses then): try again
REPLACE_WAIT = 0.01
WATCH_SPEED = 10.0  # steps a second while the owner watches
WATCH_SPEEDS = (1.0, 30.0)  # the speed slider's ends
WATCH_LEASE = 10.0  # seconds a jump-in lasts unless the page renews it: a closed tab never leaves a world slow


@dataclass(frozen=True)
class Control:
    mode: str  # one of MODES
    settings: Optional[Settings]  # the owner's knobs; None: keep what the world has
    stop: bool
    watch_until: Optional[float] = None  # a time (time.time()): watched slowly until then; None: full speed
    watch_speed: float = WATCH_SPEED  # steps a second while watched
    paused: bool = False  # paused: the world takes no steps until it is continued
    hunters: Optional[bool] = None  # the owner's hunters switch while in Auto (in Manual it is part of `settings`)
    rules: Optional[Dict[str, float]] = None  # the owner's rule changes (only the changed rules); None: none

    def watched(self, now: float) -> bool:
        return self.watch_until is not None and now < self.watch_until


def knob_values(settings: Settings) -> dict:
    """The settings as the page shows them: each knob's value and the hunters switch."""
    return {**{k.name: value(settings, k.name) for k in KNOBS}, HUNTERS: settings.attacks}


def settings_from(values: Mapping[str, object], base: Settings) -> Settings:
    """`base` with the knobs named in `values` set. Refuses, in plain words, an unknown knob, a value that is not a
    number, and one outside the knob's ends."""
    settings = base
    for name, given in values.items():
        if name == HUNTERS:
            if not isinstance(given, bool):
                raise ValueError(f"hunters must be true or false, not {given!r}")
            settings = replace(settings, attacks=given)
            continue
        knob = knob_named(name)
        if isinstance(given, bool) or not isinstance(given, (int, float)) or not math.isfinite(given):
            raise ValueError(f"{name} must be a number, not {given!r}")
        low, high = sorted((knob.easiest, knob.hardest))
        if not low <= given <= high:
            raise ValueError(f"{name} must be between {low:g} and {high:g}, not {given:g}")
        settings = with_value(settings, name, float(given))
    return settings


def watch_speed_from(given: object) -> float:
    """A watching speed from the page, refused in plain words when it is not a number within the slider's ends."""
    if isinstance(given, bool) or not isinstance(given, (int, float)) or not math.isfinite(given):
        raise ValueError(f"watch speed must be a number, not {given!r}")
    low, high = WATCH_SPEEDS
    if not low <= given <= high:
        raise ValueError(f"watch speed must be between {low:g} and {high:g} steps a second, not {given:g}")
    return float(given)


def read_control(folder: Path) -> Optional[Control]:
    """The run's control file, or None when it has none."""
    path = folder / CONTROL_FILE
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if raw["mode"] not in MODES:
        raise ValueError(f"{path}: mode must be one of {MODES}, not {raw['mode']!r}")
    settings = None if raw["settings"] is None else settings_from(raw["settings"], START)
    if raw["hunters"] not in (None, True, False):
        raise ValueError(f"{path}: hunters must be true, false or null, not {raw['hunters']!r}")
    rules = raw.get("rules")  # optional: a control file without it holds no rule changes
    if rules is not None and not (isinstance(rules, dict) and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                                  for v in rules.values())):
        raise ValueError(f"{path}: rules must map each rule to a number, not {rules!r}")
    return Control(raw["mode"], settings, bool(raw["stop"]), raw["watch_until"], watch_speed_from(raw["watch_speed"]),
                   bool(raw["paused"]), raw["hunters"], None if rules is None else {k: float(v) for k, v in rules.items()})


def write_control(folder: Path, control: Control) -> None:
    """Written whole to a temporary file and renamed, so the run never reads half a file."""
    if control.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {control.mode!r}")
    settings = None if control.settings is None else knob_values(control.settings)
    write_json(folder / CONTROL_FILE, {"mode": control.mode, "settings": settings, "stop": control.stop,
                                       "watch_until": control.watch_until, "watch_speed": control.watch_speed,
                                       "paused": control.paused, "hunters": control.hunters, "rules": control.rules})


def write_json(path: Path, data: object) -> None:
    """Written whole to a temporary file and renamed, so a reader never sees half a file."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    for attempt in range(REPLACE_TRIES):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == REPLACE_TRIES - 1:
                raise
            time.sleep(REPLACE_WAIT)


def new_run_folder(runs: Path, seed: int, kind: str) -> Path:
    """Where a new run keeps its files: dated, so no run ever overwrites another."""
    return runs / f"{time.strftime('%Y%m%d-%H%M%S')}-{seed}-{kind}"
