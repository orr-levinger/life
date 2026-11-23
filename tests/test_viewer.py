import json
import os
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

from src.viewer import make_server, runs
from tests.test_check_proof import brain_rows, write


def run_folder(root: Path, name: str, rows, record=None, seconds=None, last_written_ago: float = 0.0) -> Path:
    folder = root / name
    folder.mkdir()
    scoreboard = write(folder, "scoreboard", rows)
    if record is not None:
        (folder / "run.json").write_text(json.dumps(record))
    if seconds is not None:
        (folder / "time.txt").write_text(f"{seconds}\n")
    if last_written_ago:
        then = time.time() - last_written_ago
        os.utime(scoreboard, (then, then))
    return folder


def three_runs(root: Path) -> None:
    run_folder(root, "20261123-100000-1-learning", brain_rows(), seconds=120.0,
               record={"seed": 1, "kind": "learning", "steps": 100000, "started": "2026-11-23T10:00:00"})
    run_folder(root, "20261123-110000-2-learning", brain_rows()[:3],
               record={"seed": 2, "kind": "learning", "steps": 30000, "started": "2026-11-23T11:00:00"})
    run_folder(root, "20261123-090000-3-instinct", brain_rows()[:5], last_written_ago=3600,
               record={"seed": 3, "kind": "instinct", "steps": 30000, "started": "2026-11-23T09:00:00"})
    (root / "charts").mkdir()  # no scoreboard: not a run
    old = root / "1-evolution"  # a scoreboard of an older design, with other columns: not shown
    old.mkdir()
    (old / "scoreboard.csv").write_text("step,level,round\n1000,1,1\n")


class TestViewer(unittest.TestCase):
    def test_runs_lists_scoreboards_newest_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            three_runs(Path(tmp))
            found = runs(Path(tmp))
        self.assertEqual([(r["seed"], r["kind"]) for r in found], [(2, "learning"), (1, "learning"), (3, "instinct")])
        self.assertEqual([len(r["rows"]) for r in found], [3, 400, 5])
        self.assertEqual([r["steps"] for r in found], [30000, 100000, 30000])
        first = found[1]["rows"][0]
        self.assertEqual((first["step"], first["level"], first["population"]), (250, 0.07, 30))
        self.assertIsNone(first["intelligence"])  # no exam in that row
        self.assertEqual((found[1]["rows"][7]["intelligence"], found[1]["rows"][7]["wanderers"]), (0.5, 1))
        self.assertEqual(first["energy"], 0.6)

    def test_a_run_is_running_only_while_it_keeps_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            three_runs(Path(tmp))
            found = {(r["seed"], r["kind"]): (r["status"], r["seconds"]) for r in runs(Path(tmp))}
        self.assertEqual(found[(1, "learning")], ("finished", 120.0))
        self.assertEqual(found[(2, "learning")], ("running", None))  # no end yet, and written just now
        self.assertEqual(found[(3, "instinct")], ("stopped", None))  # no end, and silent for an hour

    def test_server_answers_page_and_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            three_runs(Path(tmp))
            server = make_server(Path(tmp), port=0)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with urllib.request.urlopen(base + "/") as response:
                    page = response.read().decode("utf-8")
                with urllib.request.urlopen(base + "/data") as response:
                    data = json.loads(response.read())
            finally:
                server.shutdown()
                server.server_close()
        self.assertIn("<title>Life", page)
        for words in ("Nothing is running", "Live", "History", "Average energy per creature"):
            self.assertIn(words, page)
        self.assertEqual(len(data), 3)


if __name__ == "__main__":
    unittest.main()
