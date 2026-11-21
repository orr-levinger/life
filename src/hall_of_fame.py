"""The best brains so far, by how much their creatures ate in their lifetime."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from .brain import Brain


class HallOfFame:
    def __init__(self, size: int = 10) -> None:
        self.size = size
        self._entries: List[Tuple[float, Brain]] = []  # best first; on a tie the earlier entry stays ahead

    def offer(self, brain: Brain, food_eaten: float) -> None:
        position = next((i for i, (score, _) in enumerate(self._entries) if food_eaten > score), len(self._entries))
        self._entries.insert(position, (food_eaten, brain))
        del self._entries[self.size:]

    def brains(self) -> List[Brain]:
        return [brain for _, brain in self._entries]

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        for rank, (_, brain) in enumerate(self._entries):
            brain.save(folder / f"best_{rank:02d}.npz")
