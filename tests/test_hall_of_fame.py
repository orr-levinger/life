import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.brain import Brain
from src.hall_of_fame import HallOfFame


class TestHallOfFame(unittest.TestCase):
    def test_keeps_best_ten_of_fifteen(self):
        rng = np.random.default_rng(0)
        entries = [(float(score), Brain.random(rng)) for score in range(15)]
        random.Random(3).shuffle(entries)
        hall = HallOfFame(size=10)
        for score, brain in entries:
            hall.offer(brain, score)
        by_score = {id(brain): score for score, brain in entries}
        self.assertEqual([by_score[id(b)] for b in hall.brains()], [14.0 - i for i in range(10)])

    def test_empty_is_empty_list(self):
        self.assertEqual(HallOfFame().brains(), [])

    def test_save_writes_files(self):
        rng = np.random.default_rng(1)
        hall = HallOfFame()
        for score in (3.0, 1.0, 2.0):
            hall.offer(Brain.random(rng), score)
        with tempfile.TemporaryDirectory() as folder:
            hall.save(Path(folder) / "best")
            self.assertEqual(sorted(p.name for p in (Path(folder) / "best").iterdir()),
                             ["best_00.npz", "best_01.npz", "best_02.npz"])


if __name__ == "__main__":
    unittest.main()
