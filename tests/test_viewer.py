import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

from src.viewer import make_server, runs
from tests.test_check_proof import brain_rows, write


def runs_folder(root: Path) -> None:
    finished = root / "1-learning"
    finished.mkdir()
    write(finished, "scoreboard", brain_rows())
    (finished / "time.txt").write_text("120.0\n")
    running = root / "2-learning"
    running.mkdir()
    write(running, "scoreboard", brain_rows()[:3])
    (root / "charts").mkdir()  # no scoreboard: not a run
    old = root / "1-evolution"  # a scoreboard of an older design, with other columns: not shown
    old.mkdir()
    (old / "scoreboard.csv").write_text("step,level,round\n1000,1,1\n")


class TestViewer(unittest.TestCase):
    def test_runs_lists_scoreboards(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_folder(Path(tmp))
            found = runs(Path(tmp))
        self.assertEqual([r["name"] for r in found], ["1-learning", "2-learning"])
        self.assertEqual([(r["seed"], r["kind"]) for r in found], [(1, "learning"), (2, "learning")])
        self.assertEqual([len(r["rows"]) for r in found], [100, 3])
        first = found[0]["rows"][0]
        self.assertEqual((first["step"], first["stage"], first["population"]), (1000, 1, 30))
        self.assertIsNone(first["exam_ratio"])  # no exam in that row
        self.assertEqual(found[0]["rows"][1]["exam_ratio"], 0.5)

    def test_unfinished_run_has_no_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_folder(Path(tmp))
            found = {r["name"]: r["seconds"] for r in runs(Path(tmp))}
        self.assertEqual(found, {"1-learning": 120.0, "2-learning": None})

    def test_server_answers_page_and_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_folder(Path(tmp))
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
        self.assertEqual([r["name"] for r in data], ["1-learning", "2-learning"])


if __name__ == "__main__":
    unittest.main()
