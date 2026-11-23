# The Living World: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One continuous world where creatures with brains learn during life and evolve by having babies from energy,
through five stages from "nobody dies" to "hunters", watched live.

**Architecture:** `src/stages.py` replaces `src/levels.py` (five stages and the world's constants). `World` owns life
and death: eating on contact, births from energy, deaths by stage, old age, the primordial soup, and making babies and
newcomers of its species. `src/rounds.py` and the teams are deleted. The learner hands out and takes back slots as
animals are born and die. `src/evolve.py` runs one species in one world; `src/viewer.py` serves the live page.

**Tech Stack:** Python 3.12, NumPy 2.1.3, Matplotlib 3.10.3, TensorFlow 2.19.0 (learning animals only), unittest.

**Spec:** `docs/superpowers/specs/2026-11-23-living-world-design.md` (approved: "Let's build it, let's see how it
works."). The goal: `README.md`. Rules of work: `CLAUDE.md`.

## Global Constraints

- Commands run from `C:\Users\Administrator\life` with `PY=/c/Users/Administrator/.venvs/life/Scripts/python`.
- No program-made selection anywhere: no rounds, ranking, elites or restarts (`CLAUDE.md` rule 1).
- Spec values: birth at 70 % of maximum energy, the baby gets 40 % of the parent's energy; mutation rate 0.1, scale 0.2;
  size × (1 ± 0.2) within 0.5–2.0; learning speed × e^N(0, 0.2); old age 5,000 steps; safety limit 80 creatures; soup
  below 10 alive in mortal stages; exam every 2,000 steps, 500 steps, up to 20 brains, pass at ≥ 1.5 twice in a row with
  ≥ 10 alive; test world 80 food of energy 10–25, 3 candidates per step; stage starting numbers as in the spec.
- Food never increases from one stage to the next; no stage is tuned to make an exam score reachable.
- The instinct species never imports TensorFlow. Same seed, same machine, same run. Zero delay. At most 2 runs at once.
- Before any run longer than a minute, the live viewer is open in the app's browser (general-coding-standards 18).
- Every task ends with a commit on `evolve-from-scratch`, judged by the commit judge.

## Review Focus

1. The Garden must never kill: energy at zero stays at zero, and no death of any cause is counted there. Test: Task 1
   `test_garden_never_kills`.
2. A birth must not look like a loss to a learner (or it learns to avoid having babies). Test: Task 2
   `test_birth_is_not_a_negative_reward`.
3. A dead learner's slot is reused by a baby without the dead animal's experience or optimizer state. Test: Task 2
   `test_slot_reuse_starts_clean`.
4. Births stop at the safety limit and resume below it; nothing else ever removes a creature to make room. Test: Task 1
   `test_births_pause_at_limit`.
5. Two creatures reaching for the same last bite: one eats, the other gets nothing, nothing crashes. Test: Task 1
   `test_last_bite_goes_to_one`.

---

### Task 1: The living world (instinct animals, no TensorFlow)

**Files:**
- Create: `src/stages.py`, `tests/test_stages.py`
- Delete: `src/levels.py`, `tests/test_levels.py`, `src/rounds.py`, `tests/test_rounds.py`
- Modify: `src/brain.py` (3 outputs), `src/creature.py`, `src/policies.py`, `src/world.py`, `src/exam.py`,
  `src/scoreboard.py`, `src/evolve.py`, `src/check_proof.py`, `src/compare.py`, `src/visualization.py`, `src/main.py`
  and their tests (`tests/helpers.py`, `test_brain`, `test_policies`, `test_rules_policy`, `test_world_rules`,
  `test_exam`, `test_scoreboard`, `test_evolve`, `test_check_proof`, `test_compare`, `test_visualization`)

**Interfaces:**
- `Stage(number, name, mortal: bool, food_energy, max_food, food_per_step, move_cost, step_cost, attacks: bool)`;
  `STAGES` (0 Garden … 4 Harsh, the spec's starting numbers); `stage(number)`; `TEST_WORLD: Stage` (the exam's fixed
  world); constants `BIRTH_AT = 0.7`, `BABY_SHARE = 0.4`, `MAX_AGE = 5000`, `MAX_CREATURES = 80`, `SOUP_MIN = 10`,
  `HUNTER_SHARE = 0.8`, `MUTATION_RATE = 0.1`, `MUTATION_SCALE = 0.2`, `SIZE_MUTATION = 0.2`, `LR_MUTATION = 0.2`.
- `Brain`: `N_OUT = 3` (move x, move y, attack). Actions are `("MOVE", (dx, dy))` and `("ATTACK", Creature)` only.
- `World(width, height, stage: Stage, species: str, rng, np_rng, food_rng, evolving=True, learner=None,
  inherit="learned")` with `species` one of `"brain" | "learner" | "random" | "rules"`; `add_newcomer() -> Creature`
  (a random brain of the species at a random spot); `step(observer=None)` in this order: food, sense, decide, attacks,
  moves, eating on contact (one bite from the nearest food in reach), deaths (mortal stages: hunger, old age; Garden:
  energy floored at 0), births (energy ≥ 70 %, below the limit), soup (mortal stages, fewer than 10 alive: one
  newcomer), aging; counters `births`, `newcomers`, `deaths` (`hunger`, `attack`, `age`), `total_bites`,
  `creature_steps`; `evolving=False` (the exam): no deaths, births, soup or aging.
- `python -m src.evolve --seed N --steps S [--species instinct] [--policy random|rules] [--stage N] [--max-stage N]
  [--window] [--out DIR]` (instinct and baselines start at stage 1); CSV columns: `step, stage, population, births,
  newcomers, generation, bites_per_creature_step, toward_food, flee, direction_concentration, exam_ratio, deaths_hunger,
  deaths_attack, deaths_age` (births, newcomers and deaths per 1,000-step window; `generation` the deepest alive).
- `check_proof` judges the spec's criteria 2–6 (criterion 1 only when the run started in the Garden).

- [ ] **Step 1: Write the failing tests** (new or changed behavior; existing tests are updated to the new interfaces):
  `test_stages`: `test_five_stages_garden_first`, `test_food_never_increases`, `test_only_garden_is_immortal`,
  `test_hunting_from_stage_three`. `test_world_rules`: `test_eating_on_contact`, `test_last_bite_goes_to_one`,
  `test_garden_never_kills`, `test_hunger_kills_in_mortal_stage`, `test_old_age_kills_in_mortal_stage`,
  `test_birth_at_seventy_percent` (baby beside the parent with 40 % of its energy, a mutated brain, generation + 1,
  parent alive), `test_births_pause_at_limit`, `test_soup_refills_a_dying_world`, `test_no_soup_in_garden`,
  `test_exam_world_has_no_births_or_deaths`, `test_same_seed_same_world`. `test_policies`: brain and random walker
  never return `EAT`; the rules yardstick stays on reachable food. `test_exam`: `test_exam_uses_the_fixed_test_world`.
  `test_evolve`: `test_instinct_run_writes_rows` (new columns), `test_same_seed_same_csv`,
  `test_stage_goes_up_after_two_passed_exams`, `test_baseline_never_changes_stage`. `test_check_proof`: one passing
  and one failing case per criterion.
- [ ] **Step 2: Run them to see them fail** · `$PY -m unittest discover tests` · Expected: `ModuleNotFoundError: No
  module named 'src.stages'` and failures naming the new behavior.
- [ ] **Step 3: Implement** the files above; delete the four files.
- [ ] **Step 4: Run them to see them pass** · `$PY -m unittest discover tests` · Expected: `OK`. Then
  `grep -rn "rounds\|next_generation\|EAT_MISS\|\"EAT\"\|src.levels\|--team" src tests` · Expected: no output; and
  `grep -ln "tensorflow" src/*.py` · Expected: only `src/learner.py`.
- [ ] **Step 5: Commit** as `feat: the living world (births from energy, stages from Garden to Harsh)`;
  `git show --stat HEAD` lists exactly this task's files.

### Task 2: Learning animals in the living world

**Files:** Modify: `src/learner.py` (2 choices: move, attack; slots handed out and taken back), `src/world.py` (births,
deaths and newcomers of learner creatures; rewards), `src/exam.py`, `src/evolve.py` (`--species learning`,
`--inherit learned|inborn`, default stage 0), tests (`test_learner`, `test_world_rules`, `test_exam`, `test_evolve`).

**Interfaces:**
- `Learner(slots, seed)`: `allocate(weights: NetWeights, lr: float) -> int` (a free slot with a clean optimizer and no
  experience; `RuntimeError` when none is free), `release(slot)`, `lr(slot) -> float`; `act` returns choice 0 (move) or
  1 (attack, masked when attacks are off); `install` is removed (allocate replaces it).
- A learner creature's reward for a step = its energy change from eating, moving, attacking and being attacked, measured
  before births; a baby's 40 % is never part of it.
- A learner baby's network = `current(parent.slot)` (inherit `"learned"`) or `inborn(parent.slot)` (`"inborn"`), mutated,
  with the parent's learning speed × e^N(0, 0.2).

- [ ] **Step 1: Write the failing tests:** `test_learner`: `test_allocate_and_release`, `test_slot_reuse_starts_clean`,
  `test_attack_masked` (2 choices), `test_learns_toy_task` (unchanged bar: ≥ 0.8). `test_world_rules`:
  `test_learner_baby_inherits_learned_brain`, `test_learner_baby_inherits_born_brain_when_inborn`,
  `test_birth_is_not_a_negative_reward`, `test_dead_learner_frees_its_slot`, `test_garden_learner_keeps_learning_at_zero_energy`.
  `test_evolve`: `test_learning_run_starts_in_the_garden`, `test_same_seed_same_csv_learning`.
- [ ] **Step 2: Run them to see them fail** · `$PY -m unittest discover tests` · Expected: failures naming `allocate`
  and the new behavior.
- [ ] **Step 3: Implement** the files above.
- [ ] **Step 4: Run them to see them pass** · `$PY -m unittest discover tests` · Expected: `OK`.
- [ ] **Step 5: Commit** as `feat: learning animals are born, learn, and pass on their brains`; `git show --stat HEAD`
  lists exactly this task's files.

### Task 3: The live viewer

**Files:** Create: `src/viewer.py`, `src/viewer.html`, `tests/test_viewer.py`.

**Interfaces:** `runs(folder: Path) -> list[dict]` (every `*/scoreboard.csv` under the runs folder: name, rows, whether
finished and its seconds); `python -m src.viewer [--runs runs] [--port 8765]` serves `/` (the page) and `/data` (that
list as JSON). The page redraws every 2 seconds: stage, population, births, generation, exam score, bites, steering to
food, per-run progress.

- [ ] **Step 1: Write the failing tests:** `test_runs_lists_scoreboards`, `test_unfinished_run_has_no_seconds`,
  `test_server_answers_page_and_data` (a server on a free port, two requests).
- [ ] **Step 2: Run them to see them fail** · `$PY -m unittest tests.test_viewer -v` · Expected: `ModuleNotFoundError`.
- [ ] **Step 3: Implement** the viewer (from the scratchpad viewer of 2026-11-23, with the new columns).
- [ ] **Step 4: Run them, then the suite** · `$PY -m unittest discover tests` · Expected: `OK`.
- [ ] **Step 5: Open it** in the app's browser on `http://localhost:8765`; check the page shows a finished run and has
  no console errors.
- [ ] **Step 6: Commit** as `feat: live viewer`; `git show --stat HEAD` lists exactly this task's files.

### Task 4: The check that could prove the design wrong (the Garden)

- [ ] **Step 1:** With the viewer open: `$PY -m src.evolve --seed 1 --steps 30000 --species learning --max-stage 0`
  (background).
- [ ] **Step 2: Check:** the mean exam score of the last 3 exams is above the first 3, and the steering score of the
  last 5 rows is above the first 5.
- [ ] **Step 3:** Show the owner the numbers. **If the check fails, stop:** report it with what was measured and a
  proposal; nothing after this task is run until the owner decides.

### Task 5: The brain's size evolves (added 2026-11-23; the owner: "Ok")

Built only after Task 4's check has passed, so a learning failure cannot be confused with this machinery.

**Files:** Modify: `src/learner.py` (room for 3 layers of 64 neurons, size genes, silent growth), `src/world.py` (a
learner baby's size mutation; the energy cost of a brain), `src/stages.py` (the rates and the cost), `src/scoreboard.py`
(columns `neurons`, `layers`), `src/viewer.html`, tests (`test_learner`, `test_world_rules`, `test_scoreboard`,
`test_viewer`).

**Interfaces:**
- `NetWeights` gains `layers: int` (1–3) and `neurons: tuple[int, int, int]` (1–64 active per layer); weights have room
  for `MAX_LAYERS = 3` × `MAX_NEURONS = 64`. Layer 1 reads the senses; layers 2 and 3 add to what they receive
  (`h + s · tanh(h·W + b)`), with `s` zero when a layer is switched on.
- `NetWeights.mutate_size(rng) -> NetWeights`: with `NEURON_CHANGE = 0.05` each, one neuron on or off; with
  `LAYER_CHANGE = 0.01` each, one layer on or off; a new neuron's outgoing weights and a new layer's `s` are zero.
- `Learner.active_neurons(slot) -> int`; a learner creature pays `NEURON_COST = 0.0005` × its active neurons per step.
- The first animals: 1 layer of 32 neurons.

- [ ] **Step 1: Write the failing tests:** `test_new_neuron_is_silent_at_birth` and `test_new_layer_is_silent_at_birth`
  (the baby's outputs equal the parent's for 100 random senses), `test_removed_neuron_changes_behavior`,
  `test_size_genes_stay_in_bounds` (1,000 mutations), `test_bigger_brain_costs_more_energy`,
  `test_learns_toy_task` (unchanged bar: ≥ 0.8, on a 1 × 32 brain), `test_scoreboard_reports_brain_size`.
- [ ] **Step 2: Run them to see them fail** · `$PY -m unittest discover tests` · Expected: failures naming
  `mutate_size`, `active_neurons` and the new columns.
- [ ] **Step 3: Implement** the files above.
- [ ] **Step 4: Run them to see them pass** · `$PY -m unittest discover tests` · Expected: `OK`.
- [ ] **Step 5: The Garden check again** · Task 4 Step 1's run, with the viewer open · Expected: the exam score of the
  last 3 exams is not below Task 4's. If it is, stop and report both sets of numbers.
- [ ] **Step 6: Commit** as `feat: the brain's size evolves`; `git show --stat HEAD` lists exactly this task's files.
  Show the owner the Garden check's numbers beside Task 4's.

### Task 6: Tune the stages by measurement

**Files:** Modify: `src/stages.py` (values and their measurements in comments), `tests/test_stages.py` (the tuned values).

- [ ] **Step 1: Measure** instinct animals at each mortal stage: `$PY -m src.evolve --seed 1 --steps 20000 --species
  instinct --stage N --max-stage N` for N = 1…4 · Expected after tuning: the population settles between 30 and 60, births
  happen, deaths by old age appear, and the soup stops being needed at stage 1.
- [ ] **Step 2:** Adjust food per step, maximum food and the old-age limit until Step 1 holds, keeping the two rules
  (food never increases; nothing is tuned to an exam score). Every value tried and its result go in the report.
- [ ] **Step 3:** `$PY -m unittest discover tests` · Expected: `OK`.
- [ ] **Step 4: Commit** the 2 files as `tune: stage values, measured`. Show the owner the settled populations.

### Task 7: Let it run, and judge it

- [ ] **Step 1:** Measure the time of `--species learning` for 10,000 steps; with the viewer open, run seeds 1–5 ×
  100,000 steps for `learning` (inherit learned), and seeds 1–5 for `instinct` (background, two at a time). If the
  learning runs would take over 12 hours in all, propose fewer seeds or steps to the owner first.
- [ ] **Step 2:** Run `learning --inherit inborn` on seeds 1–2 for the comparison.
- [ ] **Step 3:** `$PY -m src.check_proof` per species and `$PY -m src.compare` across them.
- [ ] **Step 4:** Show the owner the result of every criterion and the comparison table.

---

## Must be completed to mark this as done

1. **Every test passes:** `$PY -m unittest discover tests` → `OK`, with the count.
2. **No program-made selection is left:** `grep -rn "rounds\|next_generation\|--team" src tests` → no output.
3. **The Garden check ran first** and its numbers were shown to the owner (Task 4), before any long run.
4. **Food never increases stage to stage:** `test_food_never_increases` passes on the tuned values.
5. **Every long run was watched live:** the viewer was open in the app's browser before the Task 4, Task 5 and Task 7 runs.
6. **Brain size evolves:** `test_new_neuron_is_silent_at_birth` and `test_new_layer_is_silent_at_birth` pass, and Task 5 Step 5 showed the Garden exam score not below Task 4's.
7. **The proof was judged:** Task 7 Step 3's output is shown to the owner; every failed criterion is reported as failed,
   first, with its numbers.
8. **Committed:** `git log --oneline` on `evolve-from-scratch` shows one commit per Task 1, 2, 3, 5 and 6.
