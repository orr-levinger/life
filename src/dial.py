"""The difficulty dial: the hardness of survival drives intelligence, and it is tuned delicately.

Every window of steps the dial looks at who died. Nobody died: survival is too easy, so one setting is nudged a little
harder, one setting at a time. Too many died (or random newcomers had to be added): the last nudge is taken back.
Some died, not too many: nothing changes; that is where survival is hard enough to select.
"""
from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

from .settings import START, Settings

NUDGE = 0.05  # how much one nudge changes one setting
TOO_MANY = 0.25  # more than this share of the living dying in one window takes the last nudge back
HUNTERS_AT = 24  # the nudge after this many switches hunters on

# The settings a nudge can turn, in turn: (name, direction that makes survival harder, the limit it stops at).
KNOBS = (
    ("max_food", -1, 20.0),
    ("food_energy", -1, (2.0, 4.0)),
    ("move_cost", +1, 3.0),
    ("step_cost", +1, 0.5),
    ("lifespan", -1, (500.0, 750.0)),
    ("food_per_step", -1, 0.2),
)


def _turn(value: float, direction: int, limit: float) -> float:
    return max(value * (1.0 - NUDGE), limit) if direction < 0 else min(value * (1.0 + NUDGE), limit)


def nudged(settings: Settings, n: int) -> Settings:
    """The settings after nudge number `n` (0 is the first): one setting, a little harder, within its limit.
    Nudge number HUNTERS_AT switches hunters on."""
    if n == HUNTERS_AT:
        return replace(settings, attacks=True)
    name, direction, limit = KNOBS[(n if n < HUNTERS_AT else n - 1) % len(KNOBS)]
    value = getattr(settings, name)
    if isinstance(value, tuple):
        return replace(settings, **{name: tuple(_turn(v, direction, lim) for v, lim in zip(value, limit))})
    return replace(settings, **{name: _turn(value, direction, limit)})


class Dial:
    def __init__(self, start: Settings = START, max_level: Optional[int] = None) -> None:
        self._levels: List[Settings] = [start]  # the settings at each level; the last is in force
        self.max_level = max_level

    @property
    def level(self) -> int:
        """How many nudges are in force."""
        return len(self._levels) - 1

    @property
    def settings(self) -> Settings:
        return self._levels[-1]

    def review(self, deaths: int, alive: int, newcomers: int) -> str:
        """Look at one window: `deaths` in it, `alive` at its end, `newcomers` the primordial soup had to add.
        Returns what the dial did: "harder", "easier" or "hold"."""
        if newcomers > 0 or deaths > TOO_MANY * alive:
            if self.level == 0:
                return "hold"  # never easier than the easy start
            self._levels.pop()
            return "easier"
        if deaths == 0 and (self.max_level is None or self.level < self.max_level):
            self._levels.append(nudged(self.settings, self.level))
            return "harder"
        return "hold"

    def turn_to(self, level: int) -> None:
        """Start the dial at `level` nudges (for a world that begins harder than the easy start)."""
        while self.level < level:
            self._levels.append(nudged(self.settings, self.level))
