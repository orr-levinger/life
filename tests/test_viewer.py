import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock
import urllib.error
import urllib.request
from pathlib import Path

from src.control import WATCH_LEASE, Control, read_control, write_control
from src.dial import KNOBS
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

    def test_history_shows_older_scoreboards(self):
        import csv
        from src.scoreboard import BEFORE_HUNTING
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "20261123-080000-4-learning"
            folder.mkdir()
            (folder / "run.json").write_text(json.dumps({"seed": 4, "kind": "learning", "steps": 1000,
                                                         "started": "2026-11-23T08:00:00"}))
            with open(folder / "scoreboard.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(BEFORE_HUNTING))
                writer.writeheader()
                writer.writerow({k: "250" if k == "step" else "1" for k in BEFORE_HUNTING})
            found = runs(Path(tmp))
        self.assertEqual([r["seed"] for r in found], [4])  # shown, not dropped
        row = found[0]["rows"][0]
        self.assertEqual((row["step"], row["population"], row["chase"], row["hearing"]), (250, 1, None, None))

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
        for words in ("Nothing is running", "Live", "History", "Average energy per creature", "Start a world"):
            self.assertIn(words, page)
        self.assertEqual(len(data), 3)


class Recorder:
    """Stands in for launching a world: records each command and folder, and the world never ends."""

    def __init__(self) -> None:
        self.launched = []

    def __call__(self, command, folder):
        self.launched.append((command, folder))
        return self

    def poll(self):
        return None


class Server:
    def __init__(self, root: Path, launch=None) -> None:
        self.server = make_server(root, port=0, launch=launch or Recorder())
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def get(self, path: str):
        with urllib.request.urlopen(self.base + path) as response:
            return json.loads(response.read())

    def status(self, path: str) -> int:
        try:
            with urllib.request.urlopen(self.base + path) as response:
                return response.status
        except urllib.error.HTTPError as refused:
            return refused.code

    def post(self, path: str, body, kind: str = "application/json"):
        request = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": kind})
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as refused:
            return refused.code, json.loads(refused.read())

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class TestControlRoom(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.recorder = Recorder()
        self.server = Server(self.root, self.recorder)

    def tearDown(self):
        self.server.close()
        self.tmp.cleanup()

    def running_world(self, name="20261123-120000-5-learning"):
        run_folder(self.root, name, brain_rows()[:8],
                   record={"seed": 5, "kind": "learning", "steps": None, "started": "2026-11-23T12:00:00"})
        write_control(self.root / name, Control("auto", None, False))
        return name

    def test_a_gene_change_is_written_for_the_world(self):
        name = self.running_world()
        genes = {"size": 1.2, "vision_width": 90.0, "layers": 2, "neurons": [8, 4], "learning_speed": 0.001}
        status, _ = self.server.post("/control", {"run": name, "edit": {"id": 42, "genes": genes}})
        self.assertEqual(status, 200)
        change = read_control(self.root / name).edit
        self.assertEqual((change["id"], change["genes"]), (42, genes))
        self.assertIsInstance(change["seq"], int)
        self.assertEqual(self.server.post("/control", {"run": name, "paused": True})[0], 200)
        self.assertEqual(read_control(self.root / name).edit, change)  # another change keeps it

    def test_a_gene_change_outside_its_limits_is_refused(self):
        name = self.running_world()
        for genes, words in (({"size": 9.0}, "Size"), ({"layers": 2, "neurons": [8]}, "one neuron count for each layer"),
                             ({"neurons": [8]}, "layers"), ({"wings": 2.0}, "wings")):
            status, answer = self.server.post("/control", {"run": name, "edit": {"id": 42, "genes": genes}})
            self.assertEqual(status, 400, genes)
            self.assertIn(words, answer["error"])

    def test_rules_endpoint_lists_every_rule(self):
        from src.rules import RULES
        name = self.running_world()
        answer = self.server.get(f"/rules?run={name}")
        self.assertEqual([r["name"] for r in answer["rules"]], [r.name for r in RULES])
        eat = next(r for r in answer["rules"] if r["name"] == "diet_bite")
        self.assertEqual((eat["group"], eat["value"], eat["start"], eat["low"], eat["high"]), ("senses", 1.0, 1.0, 0.0, 5.0))
        (self.root / name / "now.json").write_text(json.dumps({"step": 9, "rules": {**{r.name: r.start for r in RULES},
                                                                                     "diet_bite": 0.4}}))
        answer = self.server.get(f"/rules?run={name}")
        self.assertEqual(next(r for r in answer["rules"] if r["name"] == "diet_bite")["value"], 0.4)  # the world's own

    def test_a_rule_change_out_of_range_is_refused(self):
        name = self.running_world()
        status, answer = self.server.post("/control", {"run": name, "rules": {"diet_bite": 9}})
        self.assertEqual((status, answer["error"]), (400, "A meat eater's hit takes more: × (1 + meat gut × this) can be at most 5"))
        status, answer = self.server.post("/control", {"run": name, "rules": {"gene.size.low": 3}})
        self.assertEqual(status, 400)
        self.assertIsNone(read_control(self.root / name).rules)
        status, answer = self.server.post("/control", {"run": name, "rules": {"diet_bite": 0.5}})
        self.assertEqual(status, 200)
        status, answer = self.server.post("/control", {"run": name, "rules": {"tank": 150}})
        self.assertEqual(read_control(self.root / name).rules, {"diet_bite": 0.5, "tank": 150.0})  # changes add up

    def test_page_has_costs_values_and_genes(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ('id="tab-rules"', 'id="rules-page"', "function drawRules", "RULE_GROUPS", "answer.groups",
                      "/rules?run="):
            self.assertIn(words, page)

    def test_charts_are_smoothed_over_a_chosen_span(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ("CHART_SPANS = [[10000, \"10k\"], [50000, \"50k\"], [0, \"all\"]]", "CHART_POINTS", "CHART_SMOOTH",
                      "span: 10000", "function chartRows", "function smoothed", "marks.filter((m) => m.byYou)"):
            self.assertIn(words, page)

    def test_page_shows_the_groups(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ("RULE_GROUPS", "/rules?run="):
            self.assertIn(words, page)
        answer = self.server.get(f"/rules?run={self.running_world()}")
        self.assertEqual([g["name"] for g in answer["groups"]],
                         ["costs", "knobs", "world", "body", "senses", "birth", "genes", "learning", "auto", "exam", "runner"])
        self.assertEqual(answer["groups"][0]["title"], "Costs")

    def test_page_has_no_switches_and_no_read_only_rows(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        start = page.index("// ---- Rules:")
        body = page[start:page.index("function render()", start)]  # the Rules page's own code
        for word in ("checkbox", "readonly", "disabled"):
            self.assertNotIn(word, body)
        self.assertIn('type="number"', body)

    def test_knobs_list_the_ends_of_every_slider(self):
        answer = self.server.get("/knobs")
        self.assertEqual([k["name"] for k in answer["knobs"]], [k.name for k in KNOBS if not k.cost])  # costs: below
        energy = next(k for k in answer["knobs"] if k["name"] == "food_energy")
        self.assertEqual((energy["label"], energy["easiest"], energy["hardest"], energy["start"]),
                         ("Energy per food", 100.0, 3.0, 30.0))
        self.assertIs(answer["hunters"], False)

    def test_the_next_worlds_rules_are_changed_with_nothing_running(self):
        answer = self.server.get("/rules")
        first = next(r for r in answer["rules"] if r["name"] == "first_creatures")
        self.assertEqual((answer["next"], first["value"]), (True, first["start"]))
        status, _ = self.server.post("/next-rules", {"rules": {"first_creatures": 12, "planet_radius": 40}})
        self.assertEqual(status, 200)
        values = {r["name"]: r["value"] for r in self.server.get("/rules")["rules"]}
        self.assertEqual((values["first_creatures"], values["planet_radius"]), (12.0, 40.0))  # kept on disk
        status, answer = self.server.post("/next-rules", {"rules": {"first_creatures": 9999}})
        self.assertEqual(status, 400)
        self.assertIn("at most", answer["error"])

    def test_an_old_next_world_rule_that_is_gone_is_dropped(self):  # a cost's knob end went (2026-11-27)
        (self.root / "next_world.json").write_text(json.dumps({"rules": {"knob.eat_miss.easy": 0.5, "eat_miss": 0.15,
                                                                         "tank": 120.0}}))
        status, answer = self.server.post("/next-rules", {"rules": {"tank": 130.0}})
        self.assertEqual((status, answer["rules"]), (200, {"eat_miss": 0.15, "tank": 130.0}))  # a cost is a rule again

    def test_a_new_world_starts_on_the_next_worlds_rules(self):
        self.server.post("/next-rules", {"rules": {"first_creatures": 12}})
        status, answer = self.server.post("/start", {"species": "learning", "seed": 3, "mode": "auto"})
        self.assertEqual(status, 200)
        self.assertEqual(read_control(self.root / answer["name"]).rules, {"first_creatures": 12.0})

    def test_start_launches_a_world(self):
        status, answer = self.server.post("/start", {"species": "learning", "seed": 3, "mode": "manual",
                                                      "settings": {"lifespan": 8000}})
        self.assertEqual(status, 200)
        command, folder = self.recorder.launched[0]
        self.assertEqual(folder.name, answer["name"])
        self.assertRegex(answer["name"], r"^\d{8}-\d{6}-3-learning$")
        self.assertEqual(command[1:9], ["-m", "src.evolve", "--seed", "3", "--species", "learning", "--forever",
                                        "--mode"])
        control = read_control(folder)
        self.assertEqual((control.mode, control.stop), ("manual", False))
        self.assertEqual(control.settings.lifespan, (6400.0, 9600.0))

    def test_start_refuses_a_third_world(self):
        self.running_world()
        self.assertEqual(self.server.post("/start", {"species": "instinct", "seed": 1, "mode": "auto"})[0], 200)
        status, answer = self.server.post("/start", {"species": "instinct", "seed": 2, "mode": "auto"})
        self.assertEqual((status, answer["error"]), (409, "2 worlds are running already: stop one first"))
        self.assertEqual(len(self.recorder.launched), 1)

    def test_start_refuses_a_bad_species(self):
        status, answer = self.server.post("/start", {"species": "dragons", "seed": 1, "mode": "auto"})
        self.assertEqual((status, answer["error"]), (400, "species must be one of learning, instinct"))
        status, answer = self.server.post("/start", {"species": "learning", "seed": 1, "mode": "auto",
                                                     "settings": {"food_energy": 0.5}})
        self.assertEqual((status, answer["error"]), (400, "food_energy must be between 3 and 100, not 0.5"))
        self.assertEqual(self.recorder.launched, [])

    def test_control_writes_the_run_s_file(self):
        name = self.running_world()
        self.assertEqual(self.server.post("/control", {"run": name, "mode": "manual"})[0], 200)
        self.assertEqual(read_control(self.root / name), Control("manual", None, False))  # keeps the world's knobs
        status, answer = self.server.post("/control", {"run": name, "settings": {"food_energy": 24}})
        self.assertEqual(status, 200)
        control = read_control(self.root / name)
        self.assertEqual(control.settings.food_energy, (16.0, 24.0 * 4 / 3))
        self.assertEqual(control.settings.max_food, 200)  # the other knobs as the world's last row shows them
        self.server.post("/control", {"run": name, "settings": {"lifespan": 7000}})
        both = read_control(self.root / name).settings
        self.assertEqual((both.food_energy[0], both.lifespan), (16.0, (5600.0, 8400.0)))  # changes add up
        self.assertEqual(self.server.post("/control", {"run": name, "mode": "auto"})[0], 200)
        self.assertEqual(read_control(self.root / name), Control("auto", None, False))

    def test_knobs_stay_still_in_auto(self):
        name = self.running_world()
        status, answer = self.server.post("/control", {"run": name, "settings": {"food_energy": 24}})
        self.assertEqual((status, answer["error"]), (409, "switch to Manual to turn the knobs"))

    def test_control_refuses_a_value_outside_the_ends(self):
        name = self.running_world()
        status, answer = self.server.post("/control", {"run": name, "mode": "manual", "settings": {"lifespan": 50}})
        self.assertEqual((status, answer["error"]), (400, "lifespan must be between 625 and 10000, not 50"))
        self.assertEqual(read_control(self.root / name), Control("auto", None, False))

    def test_control_refuses_a_run_that_is_not_running(self):
        three_runs(self.root)
        status, answer = self.server.post("/control", {"run": "20261123-100000-1-learning", "stop": True})
        self.assertEqual(status, 404)  # finished
        status, _ = self.server.post("/control", {"run": "../../elsewhere", "stop": True})
        self.assertEqual(status, 404)

    def test_posts_must_be_json(self):
        name = self.running_world()
        status, answer = self.server.post("/control", {"run": name, "stop": True}, kind="text/plain")
        self.assertEqual(status, 415)
        self.assertEqual(read_control(self.root / name).stop, False)

    def test_stop_is_written_to_the_run(self):
        name = self.running_world()
        self.assertEqual(self.server.post("/control", {"run": name, "stop": True})[0], 200)
        self.assertEqual(read_control(self.root / name), Control("auto", None, True))

    def test_runs_say_their_mode(self):
        name = self.running_world()
        write_control(self.root / name, Control("manual", None, False))
        self.assertEqual([(run["name"], run["mode"]) for run in runs(self.root)], [(name, "manual")])

    def test_a_world_just_started_is_running_before_its_first_row(self):
        folder = self.root / "20261123-130000-9-instinct"
        folder.mkdir()
        (folder / "run.json").write_text(json.dumps({"seed": 9, "kind": "instinct", "steps": None,
                                                     "started": "2026-11-23T13:00:00"}))
        found = runs(self.root)
        self.assertEqual([(run["status"], run["rows"]) for run in found], [("running", [])])

    def test_page_has_the_control_room(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ("Start", "Stop", "Auto", "Manual", "Difficulty"):
            self.assertIn(words, page)

    def test_jump_in_sets_the_lease(self):
        name = self.running_world()
        before = time.time()
        self.assertEqual(self.server.post("/control", {"run": name, "watch": True, "watch_speed": 5})[0], 200)
        control = read_control(self.root / name)
        self.assertTrue(before + WATCH_LEASE - 1 <= control.watch_until <= time.time() + WATCH_LEASE + 1)
        self.assertEqual((control.watch_speed, control.mode), (5, "auto"))
        status, answer = self.server.post("/control", {"run": name, "watch": True, "watch_speed": 99})
        self.assertEqual((status, answer["error"]), (400, "watch speed must be between 1 and 30 steps a second, not 99"))
        self.server.post("/control", {"run": name, "mode": "manual"})
        self.assertIsNotNone(read_control(self.root / name).watch_until)  # other changes keep the world watched

    def test_jump_out_clears_it(self):
        name = self.running_world()
        self.server.post("/control", {"run": name, "watch": True})
        self.assertEqual(self.server.post("/control", {"run": name, "watch": False})[0], 200)
        self.assertIsNone(read_control(self.root / name).watch_until)

    def test_view_of_a_watched_world(self):
        name = self.running_world()
        self.server.post("/control", {"run": name, "watch": True})
        picture = {"step": 7, "creatures": [], "foods": []}
        (self.root / name / "view.json").write_text(json.dumps(picture))
        self.assertEqual(self.server.get(f"/view?run={name}"), picture)

    def test_no_view_when_not_watched(self):
        name = self.running_world()
        (self.root / name / "view.json").write_text(json.dumps({"step": 7, "creatures": [], "foods": []}))
        self.assertEqual(self.server.status(f"/view?run={name}"), 404)
        self.assertEqual(self.server.status("/view?run=..%2F..%2Felsewhere"), 404)
        self.assertEqual(self.server.status("/view"), 404)

    def saved_world(self):
        name = self.running_world()
        (self.root / name / "saves").mkdir()
        for step in (2000, 4000):
            (self.root / name / "saves" / f"step-{step:07d}.npz").write_bytes(b"")
        return name

    def test_saves_are_listed(self):
        name = self.saved_world()
        self.assertEqual(self.server.get(f"/saves?run={name}"), [2000, 4000])
        self.assertEqual(self.server.status("/saves?run=..%2F..%2Felsewhere"), 404)

    def test_resume_from_a_save_launches_a_world(self):
        name = self.saved_world()
        with mock.patch("src.viewer.save_header", return_value={"seed": 5, "kind": "learning", "step": 4000}):
            status, answer = self.server.post("/start", {"from": {"run": name, "step": 4000}, "mode": "manual"})
        self.assertEqual(status, 200)
        command, folder = self.recorder.launched[0]
        self.assertRegex(answer["name"], r"^\d{8}-\d{6}-5-learning$")
        self.assertEqual(command[1:4], ["-m", "src.evolve", "--resume"])
        self.assertEqual(Path(command[4]), (self.root / name / "saves" / "step-0004000.npz").resolve())
        self.assertIn("--forever", command)
        self.assertEqual(read_control(folder).mode, "manual")

    def test_back_to_a_save_stops_the_world_then_starts_from_it(self):
        """The owner, 2026-11-27: "It's stops and saved and we resume from the past"."""
        name = self.saved_world()
        folder = self.root / name
        self.assertEqual(self.server.post("/control", {"run": name, "mode": "manual"})[0], 200)
        with mock.patch("src.viewer.save_header", return_value={"seed": 5, "kind": "learning", "step": 2000}):
            status, answer = self.server.post("/control", {"run": name, "back_to": 2000})
            self.assertEqual((status, answer["going_back"]), (200, 2000))
            self.assertTrue(read_control(folder).stop)  # it saves as it ends
            time.sleep(0.3)
            self.assertEqual(self.recorder.launched, [])  # never two at once: nothing starts before it has ended
            (folder / "time.txt").write_text("1.0\n")  # it ended
            for _ in range(100):
                if self.recorder.launched:
                    break
                time.sleep(0.05)
        command, launched_in = self.recorder.launched[0]
        self.assertNotEqual(launched_in, folder)  # a new world; the old one stays in History
        self.assertEqual(Path(command[command.index("--resume") + 1]), (folder / "saves" / "step-0002000.npz").resolve())
        self.assertIn("--forever", command)
        self.assertNotIn("--continue", command)
        self.assertEqual(read_control(launched_in).mode, "manual")  # the old world's mode
        self.assertTrue((folder / "time.txt").exists())  # the old world stays ended

    def test_back_to_a_missing_save_is_refused(self):
        name = self.saved_world()
        status, answer = self.server.post("/control", {"run": name, "back_to": 3000})
        self.assertEqual((status, answer["error"]), (404, f"{name} has no save at step 3000"))
        self.assertFalse(read_control(self.root / name).stop)  # nothing stopped
        self.assertEqual(self.recorder.launched, [])

    def test_the_page_goes_back_from_a_chart(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ("back_to:", "Go back to step", "/saves?run="):
            self.assertIn(words, page)

    MADE = {"size": 1.2, "grass_gut": 0.3, "meat_gut": 0.9, "gills": 0.0, "lungs": 1.0, "vision_width": 90.0,
            "vision_reach": 5.0, "hearing": 2.0, "fear_of_death": 3.0, "love_of_babies": 20.0,
            "learns_from_others": 0.4, "learning_speed": 0.001, "mutations": 2, "mutation_size": 0.05, "calls": 1,
            "kin": 0.5, "stealth": 0.2, "armour": 0.6, "layers": 2, "neurons": [12, 7]}

    def test_place_is_written_for_the_world(self):
        """The owner, 2026-11-27: "spawn it if I wish in map where ever I click"."""
        name = self.running_world()
        place = {"spot": [0.0, 0.6, 0.8], "genes": self.MADE, "brain": {"kind": "smartest"}}
        self.assertEqual(self.server.post("/control", {"run": name, "place": place})[0], 200)
        first = read_control(self.root / name).place
        self.assertEqual({k: first[k] for k in ("spot", "genes", "brain")}, place)
        self.assertEqual(self.server.post("/control", {"run": name, "place": place})[0], 200)
        self.assertNotEqual(read_control(self.root / name).place["seq"], first["seq"])  # each one placed once

    def test_a_bad_place_is_refused(self):
        name = self.running_world()
        good = {"spot": [0.0, 0.6, 0.8], "genes": self.MADE, "brain": {"kind": "new"}}
        for bad in ({**good, "genes": {**self.MADE, "size": 99.0}}, {**good, "spot": [0.0, 1.0]},
                    {**good, "spot": [0.0, 0.0, 0.0]}, {**good, "brain": {"kind": "borrowed"}},
                    {**good, "brain": {"kind": "creature"}}):
            status, answer = self.server.post("/control", {"run": name, "place": bad})
            self.assertEqual(status, 400, bad)
            self.assertTrue(answer["error"])
        self.assertIsNone(read_control(self.root / name).place)

    def test_the_page_makes_and_places(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        with urllib.request.urlopen(self.server.base + "/static/beasts.js") as response:
            beasts = response.read().decode("utf-8")
        with urllib.request.urlopen(self.server.base + "/static/planet.js") as response:
            planet = response.read().decode("utf-8")
        for words in ('id="tab-make"', "Planet.genesPicture(", "Planet.groundAt(", "place:", "Place it"):
            self.assertIn(words, page)
        self.assertIn("genesPicture(", beasts.split("export function speciesPicture", 1)[1].split("\n}\n", 1)[0])
        self.assertIn("export function groundAt(", planet)
        self.assertRegex(planet, r'export \{[^}]*genesPicture[^}]*\} from "/static/beasts.js"')

    def test_resume_from_a_missing_save_is_refused(self):
        name = self.saved_world()
        status, answer = self.server.post("/start", {"from": {"run": name, "step": 3000}, "mode": "auto"})
        self.assertEqual((status, answer["error"]), (404, f"{name} has no save at step 3000"))
        self.assertEqual(self.recorder.launched, [])

    def test_pause_is_written_to_the_run(self):
        name = self.running_world()
        self.assertEqual(self.server.post("/control", {"run": name, "paused": True})[0], 200)
        self.assertTrue(read_control(self.root / name).paused)
        self.assertEqual([run["paused"] for run in runs(self.root)], [True])
        self.server.post("/control", {"run": name, "paused": False})
        self.assertFalse(read_control(self.root / name).paused)

    def test_runs_say_how_many_saves(self):
        name = self.saved_world()
        self.assertEqual([run["saves"] for run in runs(self.root)], [2])

    def test_page_has_the_history_rows(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ("Smartest first", "Hardest survived", "Intelligence", "× random", "Steers to food"):
            self.assertIn(words, page)

    def test_page_has_history_filters(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ("Animals", "Mode", "At least", "Only smarter than random", "Only with saves", "Clear filters"):
            self.assertIn(words, page)

    def test_now_of_a_world(self):
        name = self.running_world()
        now = {"step": 1234, "alive": 70, "energy": 0.5, "food": 120, "births": 9, "deaths": 3, "time": 1.0}
        (self.root / name / "now.json").write_text(json.dumps(now))
        self.assertEqual(self.server.get(f"/now?run={name}"), now)

    def test_no_now_for_an_unknown_world(self):
        name = self.running_world()
        self.assertIsNone(self.server.get(f"/now?run={name}"))  # none written yet
        self.assertEqual(self.server.status("/now?run=..%2F..%2Felsewhere"), 404)

    def test_the_rules_page_works_out_size_by_its_powers(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for power in ("r.tank_power", "r.speed_power", "r.living_power"):
            self.assertIn(power, page)  # "holds … energy" and the rest follow the size curves

    def test_page_shows_care_for_kin(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ('geneTile("kin"', '["kin", "Care for kin'):
            self.assertIn(words, page)  # a tile and a box to change it

    def test_the_page_shows_the_four_genes(self):  # two guts, gills and lungs (2026-11-27)
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for gene, words in (("grass_gut", "Grass gut"), ("meat_gut", "Meat gut"), ("gills", "Gills"), ("lungs", "Lungs")):
            self.assertIn(f'geneTile("{gene}"', page)  # a tile
            self.assertIn(f'["{gene}", "{words}', page)  # a box to change it
        self.assertIn("meatColour", page)  # a meat eater drawn redder
        for old in ('geneTile("diet"', 'geneTile("water"', '["diet",', '["water",', "g.diet", "g.water", "c.diet", "c.water"):
            self.assertNotIn(old, page)
        with urllib.request.urlopen(self.server.base + "/static/gallery.html") as response:
            gallery = response.read().decode("utf-8")
        self.assertNotIn("diet:", gallery)
        self.assertNotIn("water:", gallery)

    def test_page_shows_stealth(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ('geneTile("stealth"', '["stealth", "Stealth'):
            self.assertIn(words, page)  # a tile and a box to change it

    def test_page_shows_armour(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ('geneTile("armour"', '["armour", "Armour'):
            self.assertIn(words, page)  # a tile and a box to change it

    def test_the_page_gets_the_water(self):
        name = self.running_world()
        png = b"\x89PNG\r\n\x1a\n" + b"rest"
        (self.root / name / "water.png").write_bytes(png)
        with urllib.request.urlopen(self.server.base + f"/water?run={name}") as response:
            self.assertEqual((response.read(), response.headers["Content-Type"]), (png, "image/png"))
        dry = self.running_world("20261123-130000-6-learning")
        self.assertEqual(self.server.status(f"/water?run={dry}"), 404)  # a world without water (or from before it)
        self.assertEqual(self.server.status("/water?run=..%2F..%2Felsewhere"), 404)
        with urllib.request.urlopen(self.server.base + "/static/planet.js") as response:
            self.assertIn("/water?run=", response.read().decode("utf-8"))

    def test_creatures_are_drawn_at_their_true_size(self):  # a body and its ring by its own radius, as rocks and trees
        with urllib.request.urlopen(self.server.base + "/static/planet.js") as response:
            planet = response.read().decode("utf-8")
        with urllib.request.urlopen(self.server.base + "/static/beasts.js") as response:
            beasts = response.read().decode("utf-8")
        self.assertIn("at.scale.setScalar(x.c.radius)", planet)
        self.assertIn("body.scale.setScalar(c.radius)", beasts)
        self.assertNotIn("x.c.size", planet)
        self.assertNotIn("setScalar(Math.max(c.size", beasts)

    def test_page_draws_a_planet(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        with urllib.request.urlopen(self.server.base + "/static/planet.js") as response:
            planet = response.read().decode("utf-8")
        self.assertIn('"three": "/static/three.module.min.js"', page)  # the library from this server
        self.assertIn('from "/static/planet.js"', page)
        self.assertIn('from "three"', planet)
        self.assertIn('/static/OrbitControls.js', planet)
        for name in ("makePlanet", "updatePlanet", "followCreature", "view", "bench"):
            self.assertIn(f"export function {name}", planet)
        self.assertIn("gl.readPixels(", planet)  # the bench waits for each frame to be drawn (gl.finish may not)
        self.assertIn("new THREE.SphereGeometry(R, 96, 64)", planet)  # the ground light enough to draw without a graphics card
        for text in (page, planet):
            self.assertNotIn("http://", text.replace("http://www.w3.org/2000/svg", ""))  # an SVG name, not a download
            self.assertNotIn("https://", text)  # nothing from the internet

    def test_page_has_the_creature_look(self):
        with urllib.request.urlopen(self.server.base + "/static/beasts.js") as response:
            beasts = response.read().decode("utf-8")
        with urllib.request.urlopen(self.server.base + "/static/planet.js") as response:
            planet = response.read().decode("utf-8")
        for gene in ("radius", "love_of_babies", "energy", "meat_gut", "gills", "fear_of_death", "layers", "neurons",
                     "learns_from_others", "learning_speed", "vision_width", "vision_reach", "hearing", "armour",
                     "stealth", "calls", "kin"):
            self.assertIn(f"c.{gene}", beasts, gene)  # each gene shows in the body (size: by its radius, size x body_radius)
        for words in ("export function beastModel(", "export const BEASTS", "farGeometry(", "ringColour(", "export const DOT"):
            self.assertIn(words, beasts)  # a full body near, a simple one far, a simplest one farther, a hunger ring
        self.assertIn('from "/static/beasts.js"', planet)
        # drawn in batches (this box draws without a graphics card): far bodies by body plan, all hunger rings in one,
        # the rocks in one and the ponds in one
        for words in ("look.farKey(", "look.ringGeometry", "mergeGeometries(rocks)", "mergeGeometries(ponds)"):
            self.assertIn(words, planet)
        with urllib.request.urlopen(self.server.base + "/static/gallery.html") as response:
            self.assertIn("beastModel", response.read().decode("utf-8"))  # each gene low, middle and high

    def test_the_species_page_draws_each_species_body(self):
        """The owner, 2026-11-27: "i should see the shape of that specias ! based on the average of genees!"."""
        with urllib.request.urlopen(self.server.base + "/static/beasts.js") as response:
            beasts = response.read().decode("utf-8")
        with urllib.request.urlopen(self.server.base + "/static/planet.js") as response:
            planet = response.read().decode("utf-8")
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        made = beasts.split("export function speciesPicture", 1)
        self.assertEqual(len(made), 2)
        body = made[1].split("\n}\n", 1)[0]
        for words in ("genesPicture(", "fear_of_death: t.fear", "love_of_babies: t.love", "learns_from_others: t.learns"):
            self.assertIn(words, body)  # the species' gene names given as a creature's
        maker = beasts.split("export function genesPicture", 1)[1].split("\n}\n", 1)[0]
        for words in ("genesOf(", "bodyPicture("):
            self.assertIn(words, maker)  # the planet's own body maker
        self.assertRegex(planet, r'export \{[^}]*speciesPicture[^}]*\} from "/static/beasts.js"')
        for function in ("function speciesCard(", "async function drawSpecies("):
            self.assertIn("Planet.speciesPicture(", page.split(function, 1)[1].split("\n}\n", 1)[0], function)

    def test_static_files_are_served(self):
        with urllib.request.urlopen(self.server.base + "/static/three.module.min.js") as response:
            self.assertEqual(response.status, 200)
            self.assertTrue(response.headers["Content-Type"].startswith("text/javascript"))
            self.assertIn(b"Copyright", response.read(400))  # three.js's licence comment
        for wrong in ("/static/../viewer.py", "/static/nothing.js", "/static/%2e%2e/viewer.py"):
            with self.assertRaises(urllib.error.HTTPError) as refused:
                urllib.request.urlopen(self.server.base + wrong)
            self.assertEqual(refused.exception.code, 404, wrong)

    def test_page_shows_the_season(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        self.assertIn('["season", "Season"]', page)  # a tile: how much food grows now against the year's middle

    def test_page_shows_the_calls(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        with urllib.request.urlopen(self.server.base + "/static/planet.js") as response:
            planet = response.read().decode("utf-8")
        for words in ('geneTile("calls"', '["calls", "Calls'):
            self.assertIn(words, page)  # a tile, a box to change it
        self.assertIn("if (x.c.call)", planet)  # a ring as wide as a call carries

    def test_the_alive_tile_reads_the_room_from_the_rules(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        self.assertNotIn("of 80 room", page)  # no fixed room: each world has its own
        self.assertIn("rules.room", page)

    def test_page_has_live_tiles(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ("steps a second", "Average energy", "Born", "Died"):
            self.assertIn(words, page)

    def test_update_relaunches_the_world_in_its_own_run(self):
        name = self.running_world()
        folder = self.root / name
        self.assertEqual(self.server.post("/control", {"run": name, "update": True})[0], 200)
        self.assertTrue(read_control(folder).stop)  # the world is asked to save and end
        (folder / "saves").mkdir()
        (folder / "saves" / "step-0004321.npz").write_bytes(b"")
        (folder / "time.txt").write_text("1.0\n")  # it ended
        for _ in range(100):
            if self.recorder.launched:
                break
            time.sleep(0.05)
        command, launched_in = self.recorder.launched[0]
        self.assertEqual(launched_in, folder)
        self.assertEqual(Path(command[command.index("--resume") + 1]), (folder / "saves" / "step-0004321.npz").resolve())
        self.assertIn("--continue", command)
        self.assertEqual(Path(command[command.index("--out") + 1]), folder.resolve())
        self.assertFalse((folder / "time.txt").exists())
        self.assertFalse(read_control(folder).stop)

    def flat_save(self, folder, step):
        """A save from before the sphere (format 26)."""
        import numpy as np
        (folder / "saves").mkdir(exist_ok=True)
        path = folder / "saves" / f"step-{step:07d}.npz"
        with open(path, "wb") as f:
            np.savez(f, about=np.array(json.dumps({"format": 26, "step": step, "extra": {"seed": 1, "kind": "learning"}})))
        return path

    def test_a_flat_world_is_not_updated(self):
        from src.saves import FLAT_WORLD
        name = self.running_world()
        folder = self.root / name
        self.flat_save(folder, 100)
        status, answer = self.server.post("/control", {"run": name, "update": True})
        self.assertEqual(status, 409)
        self.assertIn(FLAT_WORLD, answer["error"])
        self.assertFalse(read_control(folder).stop)  # it keeps going

    def test_a_planet_world_of_before_is_updated(self):  # its last save is format 27: a planet, not a flat world
        import numpy as np
        name = self.running_world()
        folder = self.root / name
        (folder / "saves").mkdir(exist_ok=True)
        with open(folder / "saves" / "step-0000100.npz", "wb") as f:
            np.savez(f, about=np.array(json.dumps({"format": 27, "step": 100, "extra": {"seed": 1, "kind": "learning"}})))
        status, _ = self.server.post("/control", {"run": name, "update": True})
        self.assertEqual(status, 200)

    def test_a_flat_save_is_not_resumed(self):
        from src.saves import FLAT_WORLD
        name = self.saved_world()
        self.flat_save(self.root / name, 4000)
        status, answer = self.server.post("/start", {"from": {"run": name, "step": 4000}, "mode": "manual"})
        self.assertEqual(status, 409)
        self.assertIn(FLAT_WORLD, answer["error"])
        self.assertEqual(self.recorder.launched, [])

    def test_update_keeps_the_owners_rule_changes(self):
        name = self.running_world()
        folder = self.root / name
        self.assertEqual(self.server.post("/control", {"run": name, "rules": {"sprint_speed": 0.8}})[0], 200)
        self.assertEqual(self.server.post("/control", {"run": name, "update": True})[0], 200)
        (folder / "saves").mkdir()
        (folder / "saves" / "step-0004321.npz").write_bytes(b"")
        (folder / "time.txt").write_text("1.0\n")  # it ended
        for _ in range(100):
            if self.recorder.launched:
                break
            time.sleep(0.05)
        self.assertTrue(self.recorder.launched)
        self.assertEqual(read_control(folder).rules, {"sprint_speed": 0.8})  # still waiting for the world after its restart

    def test_hunters_in_auto_is_written(self):
        name = self.running_world()
        self.assertEqual(self.server.post("/control", {"run": name, "settings": {"hunters": True}})[0], 200)
        control = read_control(self.root / name)
        self.assertEqual((control.mode, control.hunters, control.settings), ("auto", True, None))
        status, answer = self.server.post("/control", {"run": name, "settings": {"lifespan": 7000}})
        self.assertEqual((status, answer["error"]), (409, "switch to Manual to turn the knobs"))

    def test_page_has_update_code_and_the_hunting_chart(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ("Update code", "Fleeing and chasing"):
            self.assertIn(words, page)

    def test_follow_from_above_or_its_eyes(self):  # the owner, 2026-11-27: "2 mods from above and from his eyes"
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        with urllib.request.urlopen(self.server.base + "/static/planet.js") as response:
            planet = response.read().decode("utf-8")
        for words in ('id="follow-view"', 'data-view="above"', 'data-view="eyes"', ">Above<", ">Eyes<"):
            self.assertIn(words, page)
        self.assertNotIn('id="follow-pov"', page)
        for words in ("export function view(", "camera.fov", "vision_width", "body.visible = false"):
            self.assertIn(words, planet)  # its eyes: as wide as its vision, its own body not in the way
        for words in ("export function sensedBy(", "hears.set(", "c.moved"):  # only what it senses: seen, or a sound's blur
            self.assertIn(words, planet)

    def test_a_slider_stretches_to_its_worlds_ends(self):  # a knob's ends are its world's rules (the owner, 2026-11-27)
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ("rules[`knob.${knob.name}.easy`]", "rules[`knob.${knob.name}.hard`]", "slider.min = low",
                      "slider.max = high"):
            self.assertIn(words, page)

    def test_the_planet_shows_what_each_one_does(self):  # the owner, 2026-11-27: "still no animations for most actions"
        with urllib.request.urlopen(self.server.base + "/static/planet.js") as response:
            planet = response.read().decode("utf-8")
        self.assertIn("function drawActions(", planet)
        for words in ('"EAT"', '"EAT_MISS"', '"ATTACK"', '"HIT_BY_ATTACK"', '"CHASE"', '"RUN_AWAY"', "c.baby"):
            self.assertIn(words, planet)

    def test_the_planet_has_room(self):  # the owner, 2026-11-27: a smaller legend, drag the edges, full screen
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ('id="legend-toggle"', ">Legend<", "its hunger", "a call, as far as it carries", 'id="watch-handle"',
                      "localStorage", "requestFullscreen", ">Full screen<"):
            self.assertIn(words, page)
        with urllib.request.urlopen(self.server.base + "/static/planet.js") as response:
            self.assertIn("export function markCanvas(", response.read().decode("utf-8"))  # the legend shows the same marks

    def test_the_card_turns_the_costs_as_rules(self):  # the owner's, in Auto and Manual alike (2026-11-27)
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ("const COST_SLIDERS", "rules: { [name]: "):
            self.assertIn(words, page)
        name = self.running_world()  # in Auto
        status, _ = self.server.post("/control", {"run": name, "rules": {"move_cost": 0.3}})
        self.assertEqual(status, 200)
        self.assertEqual(read_control(self.root / name).rules, {"move_cost": 0.3})
        self.assertEqual(self.server.post("/control", {"run": name, "rules": {"move_cost": 0.5}})[0], 400)  # above nature

    def test_a_folded_legend_is_not_drawn(self):  # the owner, 2026-11-27: Jump in failed ("radius ... is negative")
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        body = page[page.index("function drawLegend() {"):]
        body = body[:body.index("\n}\n")]
        self.assertIn("if (canvas.hidden || !canvas.clientWidth) return;", body)  # hidden: no width to draw in

    def test_the_legend_shows_the_3d_bodies(self):  # the owner, 2026-11-27: "match it to 3d"
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        with urllib.request.urlopen(self.server.base + "/static/beasts.js") as response:
            self.assertIn("export function bodyPicture(", response.read().decode("utf-8"))
        legend = page[page.index("function drawLegend() {"):page.index("async function jumpIn(")]
        self.assertIn("Planet.bodyPicture(c.body)", legend)
        self.assertNotIn("drawCreature(", page)  # the 2D bodies of before are gone
        for words in ("plant eater", "meat eater", "swimmer", "rounder:, loves babies", "antennae:, learns from others",
                      "coat bands:, learns fast", "eyes to the sides:, wide vision", "big eyes:, sees far",
                      "bone plates:, armour", "duller coat:, stealth", "throat sac:, calls", "pale blaze:, cares for kin"):
            self.assertIn(words, page)

    def test_the_marks_sit_above_the_head(self):  # "the green apple for eating ... hides all the creature"
        with urllib.request.urlopen(self.server.base + "/static/planet.js") as response:
            planet = response.read().decode("utf-8")
        for words in ("Math.max(x.c.radius * 0.8, n.clone().multiplyScalar(R).distanceTo(eye) * 0.015)",
                      "// a green open bite"):
            self.assertIn(words, planet)

    def test_the_page_has_a_pixels_view(self):  # the owner: "one is exactly how he sees it with the pixels"
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ('data-view="pixels"', ">Pixels<", "function drawPixels(", "follow: camera.follow"):
            self.assertIn(words, page)
        name = self.running_world()
        self.assertEqual(self.server.post("/control", {"run": name, "follow": 7})[0], 200)
        self.assertEqual(read_control(self.root / name).follow, 7)  # the world sends that one's pixels
        self.assertEqual(self.server.post("/control", {"run": name, "follow": None})[0], 200)
        self.assertIsNone(read_control(self.root / name).follow)
        self.assertEqual(self.server.post("/control", {"run": name, "follow": "x"})[0], 400)

    def test_the_species_book_is_served(self):  # the tree of species (the owner, 2026-11-27)
        name = self.running_world()
        book = {"next_id": 3, "species": [{"id": 1, "parent": None, "began": 0, "ended": None, "counts": [[250, 30]]},
                                          {"id": 2, "parent": 1, "began": 250, "ended": None, "counts": [[250, 6]]}]}
        (self.root / name / "species.json").write_text(json.dumps(book))
        self.assertEqual(self.server.get(f"/species?run={name}"), book)
        self.assertEqual(self.server.status(f"/species?run={self.running_world('20261123-130000-6-learning')}"), 404)
        self.assertEqual(self.server.status("/species?run=..%2F..%2Felsewhere"), 404)

    def test_the_page_draws_the_tree(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ('id="tab-species"', 'id="species-page"', "function drawSpecies(", "/species?run=", "<svg"):
            self.assertIn(words, page)

    def page_files(self):
        files = {}
        for name, path in (("page", "/"), ("planet", "/static/planet.js")):
            with urllib.request.urlopen(self.server.base + path) as response:
                files[name] = response.read().decode("utf-8")
        return files["page"], files["planet"]

    @staticmethod
    def function_of(text, head):
        return text.split(head, 1)[1].split("\n}\n", 1)[0]

    def test_the_tree_of_life_is_circles_with_bodies(self):
        """The owner, 2026-11-28, with a poster of the tree of life: "the tree of life should look like this"."""
        page, _ = self.page_files()
        tree = self.function_of(page, "async function drawSpecies(")
        for words in ("<circle", "Planet.speciesPicture(", "lives("):
            self.assertIn(words, tree)
        self.assertNotIn('class="band"', page)  # the time bands are gone; their facts stay on the card

    def test_land_or_water_and_time_in_water(self):
        """The owner, 2026-11-28: "Land or water is genes based", then "Actually also activity"."""
        page, _ = self.page_files()
        words = self.function_of(page, "function lives(")
        for said in ("water animal", "land animal", "both"):
            self.assertIn(said, words)
        self.assertIn("f.in_water", page.split("const inWater = (f) =>", 1)[1].split("\n", 1)[0])  # one place says it
        self.assertIn("inWater(f)", self.function_of(page, "function speciesCard("))
        self.assertIn("inWater(s.facts)", self.function_of(page, "async function drawSpecies("))  # Alive now says it too

    def test_an_extinct_species_can_be_brought_back(self):
        """The owner, 2026-11-28: "i want to see instinct cretures and able to bring a instikt creature back"."""
        page, _ = self.page_files()
        self.assertIn("Bring it back", self.function_of(page, "function speciesCard("))
        self.assertIn("(extinct)", self.function_of(page, "async function drawMake("))

    def test_the_rings_are_above_the_water(self):
        """The owner, 2026-11-28: "the circle in water invosible we need to bring it a bit higher"."""
        _, planet = self.page_files()
        self.assertIn("const WATER_LIFT", planet)
        self.assertIn("SphereGeometry(R + WATER_LIFT", planet)
        self.assertGreaterEqual(planet.count("WATER_LIFT +"), 3)  # the hunger rings, the call rings, the follow ring

    def test_species_are_named_in_latin(self):
        """The owner, 2026-11-28: "Species 16 is a boring name ... Just not species number"; "Latin"."""
        page, _ = self.page_files()
        naming = self.function_of(page, "function speciesNames(")
        for words in ("Phyto", "Sarco", "Panto", "mys", "therium", "saurus", "ichthys", "cetus", "batrachus", "eats(", "lives("):
            self.assertIn(words, naming)
        table = page.split("const NAME_GENES = [", 1)[1].split("];\n", 1)[0]
        self.assertEqual(table.count("\n  ["), 16)  # one row per gene of the spec's table
        self.assertNotIn("`species ${", page)
        self.assertNotIn('"first life"', page)
        self.assertIn("number ${", self.function_of(page, "function speciesCard("))

    def test_the_names_of_a_small_tree(self):
        """The names of a hand-made tree, worked out by the page's own functions (run in node)."""
        page, _ = self.page_files()
        node = shutil.which("node")
        self.assertIsNotNone(node, "node is needed to run the page's naming")  # on this box: node v24
        code = "\n".join([
            "function eats(" + self.function_of(page, "function eats(") + "\n}",
            "function lives(" + self.function_of(page, "function lives(") + "\n}",
            "const NAME_GENES = [" + page.split("const NAME_GENES = [", 1)[1].split("];\n", 1)[0] + "];",
            "function speciesNames(" + self.function_of(page, "function speciesNames(") + "\n}",
        ])
        middle = {"size": 1.25, "stealth": 0.5, "armour": 0.5, "vision_width": 75, "vision_reach": 15.5, "hearing": 2,
                  "calls": 5, "kin": 0.5, "fear": 50.05, "love": 50.05, "learns": 0.505,
                  "learning_speed": 10 ** -3.5, "mutations": 7, "mutation_size": 0.105, "neurons": 32.5, "layers": 2}
        first = {**middle, "grass_gut": 1, "meat_gut": 0, "gills": 0, "lungs": 1, "size": 1.0, "hearing": 3.6,
                 "fear": 95, "neurons": 60, "learning_speed": 1e-5}
        second = {**first, "meat_gut": 1, "grass_gut": 0, "size": 1.5, "armour": 1, "stealth": 0.9, "neurons": 20}
        fourth = {**second, "gills": 1, "lungs": 0, "size": 1.0, "kin": 0, "mutations": 12, "love": 10}
        book = {"species": [{"id": 1, "parent": None, "start": first}, {"id": 2, "parent": 1, "start": second},
                            {"id": 3, "parent": 1, "start": second}, {"id": 4, "parent": 2, "start": fourth}]}
        code += f"\nconsole.log(JSON.stringify([...speciesNames({json.dumps(book)}).entries()]));"
        out = subprocess.run([node, "-e", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(dict(json.loads(out.stdout)), {
            1: "Phytotherium lentospinosus cerebrauritus",
            2: "Sarcosaurus simplicoloricatus occultogiganteus",
            3: "Sarcosaurus simplicoloricatus occultogiganteus II",
            4: "Sarcoichthys solitomutabilis incuriopumilus"})

    def beasts_file(self):
        with urllib.request.urlopen(self.server.base + "/static/beasts.js") as response:
            return response.read().decode("utf-8")

    def test_every_gene_shows(self):
        """The owner, 2026-11-28: "lungs has 0 affect on look", "grass gus 0 affect", "gills 1 should have 0 legs",
        "neurons in layer 2 ... same changes", "care for kin 0 affect"; "more drastic changes to look"."""
        beasts = self.beasts_file()
        genes = self.function_of(beasts, "function genesOf(")
        for words in ("c.lungs", "c.grass_gut", "c.brain", "c.vision_reach", "/ 29", "* 10) / 10"):
            self.assertIn(words, genes)
        model = self.function_of(beasts, "export function beastModel(")
        for words in ("g.lungs", "g.grass", "g.brain[", "g.kin", "legLen > 0.05"):
            self.assertIn(words, model)

    def test_the_coat_comes_from_the_genes(self):
        """The owner, 2026-11-28: "all animal have same color thats borring"; colour "From its genes"."""
        beasts = self.beasts_file()
        coat = self.function_of(beasts, "function coatOf(")
        for words in ("g.meat", "g.swim", "g.fear", "g.stealth", "g.armour"):
            self.assertIn(words, coat)
        mark = self.function_of(beasts, "function markOf(")
        for words in ("g.love", "g.kin", "g.learns"):
            self.assertIn(words, mark)

    def test_the_picture_frames_the_body(self):
        """A big creature's head is not cut at the picture's edge."""
        self.assertIn("new THREE.Box3()", self.function_of(self.beasts_file(), "export function bodyPicture("))

    def test_make_and_species_pass_each_layer(self):
        """Each brain layer shows its own neurons: Make passes its layers; a species shares its neurons over its layers."""
        page, _ = self.page_files()
        self.assertIn("brain:", self.function_of(page, "function drawMadeBody("))
        genes = self.function_of(self.beasts_file(), "function genesOf(")
        self.assertIn("c.neurons /", genes)
        self.assertIn("c.layers", genes)

    def test_make_shows_the_body_big(self):
        """The owner, 2026-11-28: "the Make screen the view of the animal is too smal screen!", then, after the mockup,
        "yes fix the editor"."""
        page, _ = self.page_files()
        self.assertIn("Planet.genesPicture(", self.function_of(page, "function drawMadeBody("))
        self.assertIn("MADE_PX = 560", page)
        make = page[page.index('<section id="make-page"'):]
        first_panel = make[:make.index("</div>", make.index('<div class="panel'))]
        self.assertIn('id="make-body"', first_panel)  # the body in the wide left column, the sliders to its right

    def test_a_body_picture_is_kept_by_its_size(self):
        """A big picture is never the small one kept before for the same genes."""
        with urllib.request.urlopen(self.server.base + "/static/beasts.js") as response:
            beasts = response.read().decode("utf-8")
        picture = self.function_of(beasts, "export function bodyPicture(")
        self.assertIn("const key = JSON.stringify([genes, size]);", picture)

    def test_a_species_card_tells_its_story(self):  # "how smart a species is, how big of a brain ... Size"
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        card = page[page.index("function speciesCard("):]
        card = card[:card.index("\n}\n")]
        self.assertIn("eats(g)", card)
        self.assertIn("lives(g)", card)
        words_of = page[page.index("function eats("):page.index("function speciesWords(")]  # what it eats, where it lives
        for words in ("hunter", "grazer", "eats both", "water animal", "land animal"):
            self.assertIn(words, words_of)
        for words in ("grass gut", "meat gut", "gills", "lungs", "eyesight", "hearing", "how smart", "brain", "size", "armour",
                      "stealth", "calls", "kin", "fear of death", "love of babies", "learns from others", "babies born",
                      "died of hunger", "hunted", "old age", "oldest", "generations", "changed most"):
            self.assertIn(words, card)

    def test_every_rule_can_be_dragged(self):  # the owner, 2026-11-27: "dragble nobs and not just munbers"
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        box = page[page.index("function ruleInput("):page.index("function ruleRow(")]
        self.assertIn('type="range" data-slider=', box)  # a slider beside every number box (genes' too: geneRows uses it)
        for words in ("function sliderValue(", "function sliderPosition(", "data-log", "data-whole"):
            self.assertIn(words, page)  # even or log scale; whole steps for counts
        rules = page[page.index("async function drawRules("):page.index("function render(")]
        self.assertIn('input[data-slider]', rules)  # a drag moves its box, and is sent when it ends
        self.assertIn("change(slider.dataset.slider", rules)

    def test_a_slowly_watched_world_is_still_running(self):  # the owner, 2026-11-27: Pixels "still doesnt work" at
        # 1 step a second: a row every 250 s, so the page server took the world for stopped and refused whom it follows
        import os
        name = self.running_world()
        folder = self.root / name
        long_ago = time.time() - 600
        os.utime(folder / "scoreboard.csv", (long_ago, long_ago))  # its last row: 10 minutes ago
        (folder / "view.json").write_text("{}")  # but it draws its picture now: it is being watched
        self.assertEqual(self.server.post("/control", {"run": name, "follow": 5})[0], 200)
        self.assertEqual(read_control(folder).follow, 5)

    def test_world_card_has_every_cost(self):  # the owner's cost sliders, never Auto's (2026-11-27)
        answer = self.server.get("/knobs")
        self.assertNotIn("move_cost", {k["name"] for k in answer["knobs"]})  # not Auto's knobs
        costs = {c["name"]: c for c in answer["costs"]}
        self.assertEqual(set(costs), {"move_cost", "step_cost", "sprint_power", "eat_miss", "attack_cost", "attack_damage",
                                      "bump_cost", "vision_cost", "hearing_cost", "neuron_cost", "stealth_cost",
                                      "armour_cost", "pixel_cost", "gut_cost", "breath_cost"})
        self.assertEqual((costs["vision_cost"]["label"][:14], costs["step_cost"]["low"], costs["step_cost"]["high"]),
                         ("Cost of seeing", 0.0, 0.1))  # from 0 to nature

    def test_page_lets_the_owner_change_a_creatures_genes(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ('id="follow-genes"', "function drawGenes(", "edit: {"):
            self.assertIn(words, page)

    def test_page_edits_the_next_worlds_rules_with_nothing_running(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ('send("/next-rules"', "the rules the next new world starts on"):
            self.assertIn(words, page)

    def test_page_can_hide_the_vision_cones(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ('id="show-cones"', "camera.cones"):
            self.assertIn(words, page)

    def test_page_explains_the_creature_look(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ('id="watch-legend"', "ring under it:, its hunger: full", "hungry", "ears:, hears far",
                      "big head:, many neurons", "horns or crest:, brain layers", "quills:, fears death",
                      "its target:, a red line, from the attacker", "chasing", "fleeing"):
            self.assertIn(words, page)
        self.assertNotIn("tan diamond: food", page)  # the owner: the words-only key "is not helping"

    def test_page_draws_cones_and_hearing(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ("function drawSenses", "c.vision_width", "c.hearing", "cone:, what it sees",
                      '"vision_width"', '"hearing"'):
            self.assertIn(words, page)

    def test_page_shows_the_values(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ('"fear_of_death"', '"love_of_babies"', '"learns_from_others"', "fears death", "loves babies",
                      "learns from others"):
            self.assertIn(words, page)

    def test_page_has_the_drawn_legend_and_hunting_tiles(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ("Flees bigger", "Chases smaller", 'id="watch-legend"'):
            self.assertIn(words, page)

    def test_page_explains_hunger_colours(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        self.assertIn("hungerColour", page)

    def test_page_has_pause_and_resume(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ("Pause", "Continue"):
            self.assertIn(words, page)

    def test_page_names_the_new_world_button(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        self.assertIn("Start a new world from this save", page)
        self.assertNotIn("Resume from here", page)

    def test_page_has_jump_in(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ("Jump in", "Jump out"):
            self.assertIn(words, page)


if __name__ == "__main__":
    unittest.main()
