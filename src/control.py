"""How the live page talks to a running world: one small JSON file in the run's folder.

The page (through the viewer's server) writes it; the world reads it every few steps. It says the mode (Auto turns the
knobs itself, Manual leaves them to the owner), the knobs the owner set, and whether to stop.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Optional

from .dial import KNOBS, knob_named, value, with_value
from .settings import START, Settings

MODES = ("auto", "manual")
CONTROL_FILE = "control.json"
HUNTERS = "hunters"  # the on/off switch beside the knobs
REPLACE_TRIES = 50  # the run may be reading the file at the moment it is replaced (Windows refuses then): try again
REPLACE_WAIT = 0.01


@dataclass(frozen=True)
class Control:
    mode: str  # one of MODES
    settings: Optional[Settings]  # the owner's knobs; None: keep what the world has
    stop: bool


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
    return Control(raw["mode"], settings, bool(raw["stop"]))


def write_control(folder: Path, control: Control) -> None:
    """Written whole to a temporary file and renamed, so the run never reads half a file."""
    if control.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {control.mode!r}")
    path, temporary = folder / CONTROL_FILE, folder / (CONTROL_FILE + ".tmp")
    settings = None if control.settings is None else knob_values(control.settings)
    temporary.write_text(json.dumps({"mode": control.mode, "settings": settings, "stop": control.stop}),
                         encoding="utf-8")
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
