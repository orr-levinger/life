import json
import os
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
        eat = next(r for r in answer["rules"] if r["name"] == "eat_miss")
        self.assertEqual((eat["group"], eat["value"], eat["start"], eat["low"], eat["high"]), ("costs", 0.2, 0.2, 0.0, 5.0))
        (self.root / name / "now.json").write_text(json.dumps({"step": 9, "rules": {**{r.name: r.start for r in RULES},
                                                                                     "eat_miss": 0.4}}))
        answer = self.server.get(f"/rules?run={name}")
        self.assertEqual(next(r for r in answer["rules"] if r["name"] == "eat_miss")["value"], 0.4)  # the world's own

    def test_a_rule_change_out_of_range_is_refused(self):
        name = self.running_world()
        status, answer = self.server.post("/control", {"run": name, "rules": {"eat_miss": 9}})
        self.assertEqual((status, answer["error"]), (400, "Eating at nothing can be at most 5"))
        status, answer = self.server.post("/control", {"run": name, "rules": {"gene.size.low": 3}})
        self.assertEqual(status, 400)
        self.assertIsNone(read_control(self.root / name).rules)
        status, answer = self.server.post("/control", {"run": name, "rules": {"eat_miss": 0.5}})
        self.assertEqual(status, 200)
        status, answer = self.server.post("/control", {"run": name, "rules": {"tank": 150}})
        self.assertEqual(read_control(self.root / name).rules, {"eat_miss": 0.5, "tank": 150.0})  # changes add up

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
                         ["costs", "world", "body", "senses", "birth", "genes", "learning", "auto", "exam", "runner"])
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
        self.assertEqual([k["name"] for k in answer["knobs"]], [k.name for k in KNOBS])
        energy = next(k for k in answer["knobs"] if k["name"] == "food_energy")
        self.assertEqual((energy["label"], energy["easiest"], energy["hardest"], energy["start"]),
                         ("Energy per food", 60.0, 3.0, 30.0))
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
        self.assertEqual((status, answer["error"]), (400, "food_energy must be between 3 and 60, not 0.5"))
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

    def test_page_shows_water(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ('geneTile("water"', '["water", "Water'):
            self.assertIn(words, page)  # a tile and a box to change it

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

    def test_page_draws_a_planet(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        with urllib.request.urlopen(self.server.base + "/static/planet.js") as response:
            planet = response.read().decode("utf-8")
        self.assertIn('"three": "/static/three.module.min.js"', page)  # the library from this server
        self.assertIn('from "/static/planet.js"', page)
        self.assertIn('from "three"', planet)
        self.assertIn('/static/OrbitControls.js', planet)
        for name in ("makePlanet", "updatePlanet", "followCreature", "pointOfView", "bench"):
            self.assertIn(f"export function {name}", planet)
        for text in (page, planet):
            self.assertNotIn("http://", text.replace("http://www.w3.org/2000/svg", ""))  # an SVG name, not a download
            self.assertNotIn("https://", text)  # nothing from the internet

    def test_page_has_the_creature_look(self):
        with urllib.request.urlopen(self.server.base + "/static/beasts.js") as response:
            beasts = response.read().decode("utf-8")
        with urllib.request.urlopen(self.server.base + "/static/planet.js") as response:
            planet = response.read().decode("utf-8")
        for gene in ("size", "love_of_babies", "energy", "diet", "water", "fear_of_death", "layers", "neurons",
                     "learns_from_others", "learning_speed", "vision_width", "vision_reach", "hearing", "armour",
                     "stealth", "calls", "kin"):
            self.assertIn(f"c.{gene}", beasts, gene)  # each gene shows in the body
        for words in ("export function beastModel(", "export const BEASTS", "THREE.LOD"):
            self.assertIn(words, beasts)  # a full body near the camera, a simple one far away
        self.assertIn('from "/static/beasts.js"', planet)
        with urllib.request.urlopen(self.server.base + "/static/gallery.html") as response:
            self.assertIn("beastModel", response.read().decode("utf-8"))  # each gene low, middle and high

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

    def test_page_shows_the_diet(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ('geneTile("diet"', '["diet", "Diet', "meatColour"):
            self.assertIn(words, page)  # a tile, a box to change it, a meat eater drawn redder

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

    def test_page_has_a_point_of_view_button(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        with urllib.request.urlopen(self.server.base + "/static/planet.js") as response:
            planet = response.read().decode("utf-8")
        self.assertIn('id="follow-pov"', page)
        for words in ("export function sensedBy(", "hears.set(", "c.moved"):  # only what it senses: seen, or a sound's blur
            self.assertIn(words, planet)

    def test_world_card_has_the_senses_costs(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ("Cost of seeing", "Cost of hearing", '"vision_cost"', '"hearing_cost"', "sense-cost",
                      '["Costs", ["move_cost", "step_cost", "vision_cost", "hearing_cost"]]'):
            self.assertIn(words, page)

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
        for words in ('id="watch-legend"', "dotted: past, its lifespan", "hungry: red", "function creatureShape",
                      "big ears:, hears far", "big head:, many neurons", "folds:, brain layers", "quills:, fears death",
                      "attacking: a strike, line to its target", "chasing", "fleeing", "function drawStrike"):
            self.assertIn(words, page)
        self.assertNotIn("tan diamond: food", page)  # the owner: the words-only key "is not helping"

    def test_page_draws_cones_and_hearing(self):
        with urllib.request.urlopen(self.server.base + "/") as response:
            page = response.read().decode("utf-8")
        for words in ("function drawSenses", "c.vision_width", "c.hearing", "cone:, what it sees", "dotted circle:, what it hears",
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
