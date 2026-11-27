"""A world reports its state, events and errors to a small report/ folder (the owner, 2026-11-27: "every 500 steps,
report the state of the game ... any errors, any important events, just report back")."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.evolve import parse_args, run
from src.report import Reporter, knob_changes
from tests.helpers import NO_FOOD, SMALL_WORLD, add, make_world, small_world


def lines(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def quiet_run(argv):
    with small_world(), contextlib.redirect_stdout(io.StringIO()):
        return run(parse_args(argv))


class TestReport(unittest.TestCase):
    def test_a_run_writes_its_state_every_500_steps(self):  # every report_every steps (the test world: 100)
        with tempfile.TemporaryDirectory() as folder:
            quiet_run(["--seed", "2", "--steps", "200", "--species", "instinct", "--out", folder])
            state = lines(Path(folder) / "report" / "state.jsonl")
            events = lines(Path(folder) / "report" / "events.jsonl")
        self.assertEqual([s["step"] for s in state], [100, 200])
        self.assertEqual(SMALL_WORLD["report_every"], 100.0)
        for name in ("alive", "species_alive", "species_ever", "generation", "smart", "to_food", "deaths", "level",
                     "knobs", "costs", "neurons", "layers", "time"):
            self.assertIn(name, state[-1], name)
        self.assertEqual([e["kind"] for e in events][:1] + [e["kind"] for e in events][-1:], ["start", "stop"])

    def test_a_split_and_a_nudge_are_events(self):
        w = make_world(NO_FOOD)
        everyone = [add(w, 3.0 * k, 0.0) for k in range(20)]
        for c in everyone[:6]:
            c.diet, c.armour = 1.0, 1.0
        self.assertEqual(w.species_book.look(w, 250), [{"kind": "species_split", "species": 2, "parent": 1, "size": 6}])
        for c in everyone[:6]:
            w.kill(c, "hunger")
        self.assertEqual(w.species_book.look(w, 500), [{"kind": "species_ended", "species": 2}])
        self.assertEqual(knob_changes({"max_food": 400.0, "move_cost": 0.04}, {"max_food": 380.0, "move_cost": 0.04}),
                         {"max_food": [400.0, 380.0]})

    def test_an_error_is_written_with_its_traceback(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch("src.world.World.step", side_effect=RuntimeError("a bug in the world")):
                with self.assertRaises(RuntimeError):
                    quiet_run(["--seed", "2", "--steps", "50", "--species", "instinct", "--out", folder])
            errors = lines(Path(folder) / "report" / "errors.jsonl")
        self.assertIn("RuntimeError: a bug in the world", errors[-1]["traceback"])
        self.assertIn("Traceback", errors[-1]["traceback"])

    def test_the_files_are_trimmed(self):
        with tempfile.TemporaryDirectory() as folder:
            reporter = Reporter(Path(folder))
            for k in range(1005):
                reporter.event("auto", step=k)
            events = lines(Path(folder) / "report" / "events.jsonl")
        self.assertEqual((len(events), events[0]["step"], events[-1]["step"]), (1000, 5, 1004))


if __name__ == "__main__":
    unittest.main()
