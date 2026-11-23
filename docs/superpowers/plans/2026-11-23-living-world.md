# The Living World: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One continuous world where creatures with brains learn during life and evolve by having babies from energy,
in a world that starts easy and gets a little harder whenever nobody dies (the difficulty dial), watched live.

**Architecture:** `src/settings.py` holds the world's settings and constants; `src/dial.py` is the difficulty dial
(Task 4d; Tasks 1–4c were built on five stages in `src/stages.py`, which the dial replaced). `World` owns life
and death: births from energy, deaths only with low energy, the primordial soup, and making babies and
newcomers of its species. `src/rounds.py` and the teams are deleted. The learner hands out and takes back slots as
animals are born and die. `src/evolve.py` runs one species in one world; `src/viewer.py` serves the live page.

**Tech Stack:** Python 3.12, NumPy 2.1.3, Matplotlib 3.10.3, TensorFlow 2.19.0 (learning animals only), unittest.

**Spec:** `docs/superpowers/specs/2026-11-23-living-world-design.md` (approved: "Let's build it, let's see how it
works."). The goal: `README.md`. Rules of work: `CLAUDE.md`.

## Global Constraints

- Commands run from `C:\Users\Administrator\life` with `PY=/c/Users/Administrator/.venvs/life/Scripts/python`.
- No program-made selection anywhere: no rounds, ranking, elites or restarts (`CLAUDE.md` rule 1).
- Spec values: birth at 70 % of maximum energy, the baby gets 40 % of the parent's energy; mutation rate 0.1, scale 0.2;
  size × (1 ± 0.2) within 0.5–2.0; learning speed × e^N(0, 0.2); a lifespan drawn from 4,000–6,000 steps at the start;
  safety limit 80 creatures; soup below 10 alive; exam every 2,000 steps, 500 steps, up to 20 brains, a measurement
  only; test world 80 food of energy 10–25, 3 candidates per step; the dial's numbers as in the spec.
- The dial only nudges a setting harder than the start and takes nudges back in reverse order; no setting is tuned to
  make an exam score reachable.
- The instinct species never imports TensorFlow. Same seed, same machine, same run. Zero delay. At most 2 runs at once.
- Before any run longer than a minute, the live viewer is open in the app's browser (general-coding-standards 18).
- Every task ends with a commit on `evolve-from-scratch`, judged by the commit judge.

## Review Focus

1. Only a creature with low energy dies, and a well-fed one never does. Tests: Task 4d
   `test_well_fed_elder_lives_on`, `test_hungry_elder_dies_of_old_age`, `test_energy_out_is_death_from_the_start`.
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
  world); constants `BIRTH_AT = 0.7`, `BABY_SHARE = 0.4`, `LIFESPAN = (4000, 6000)`, `MAX_CREATURES = 80`, `SOUP_MIN = 10`,
  `HUNTER_SHARE = 0.8`, `MUTATION_RATE = 0.1`, `MUTATION_SCALE = 0.2`, `SIZE_MUTATION = 0.2`, `LR_MUTATION = 0.2`.
- `Brain`: `N_OUT = 3` (move x, move y, attack). Actions are `("MOVE", (dx, dy))` and `("ATTACK", Creature)` only.
- `World(width, height, stage: Stage, species: str, rng, np_rng, food_rng, evolving=True, learner=None,
  inherit="learned")` with `species` one of `"brain" | "learner" | "random" | "rules"`; `add_newcomer() -> Creature`
  (a random brain of the species at a random spot); `step(observer=None)` in this order: food, sense, decide, attacks,
  eating, moves, deaths (mortal stages: hunger, old age; Garden:
  energy floored at 0), births (energy ≥ 70 %, below the limit), soup (mortal stages, fewer than 10 alive: one
  newcomer), aging; counters `births`, `newcomers`, `deaths` (`hunger`, `attack`, `age`), `total_bites`,
  `creature_steps`; `evolving=False` (the exam): no deaths, births, soup or aging.
- `python -m src.evolve --seed N --steps S [--species instinct] [--policy random|rules] [--stage N] [--max-stage N]
  [--window] [--out DIR]` (instinct and baselines start at stage 1); CSV columns: `step, stage, population, births,
  newcomers, generation, bites_per_creature_step, toward_food, flee, direction_concentration, exam_ratio, deaths_hunger,
  deaths_attack, deaths_age` (births, newcomers and deaths per 1,000-step window; `generation` the deepest alive).
- `check_proof` judges the spec's criteria 2–6 (criterion 1 only when the run started in the Garden).

- [x] **Step 1: Write the failing tests** (new or changed behavior; existing tests are updated to the new interfaces):
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
- [x] **Step 2: Run them to see them fail** · `$PY -m unittest discover tests` · Expected: `ModuleNotFoundError: No
  module named 'src.stages'` and failures naming the new behavior.
- [x] **Step 3: Implement** the files above; delete the four files.
- [x] **Step 4: Run them to see them pass** · `$PY -m unittest discover tests` · Expected: `OK`. Then
  `grep -rn "rounds\|next_generation\|EAT_MISS\|\"EAT\"\|src.levels\|--team" src tests` · Expected: no output; and
  `grep -ln "tensorflow" src/*.py` · Expected: only `src/learner.py`.
- [x] **Step 5: Commit** as `feat: the living world (births from energy, stages from Garden to Harsh)`;
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

- [x] **Step 1: Write the failing tests:** `test_learner`: `test_allocate_and_release`, `test_slot_reuse_starts_clean`,
  `test_attack_masked` (2 choices), `test_learns_toy_task` (unchanged bar: ≥ 0.8). `test_world_rules`:
  `test_learner_baby_inherits_learned_brain`, `test_learner_baby_inherits_born_brain_when_inborn`,
  `test_birth_is_not_a_negative_reward`, `test_dead_learner_frees_its_slot`, `test_garden_learner_keeps_learning_at_zero_energy`.
  `test_evolve`: `test_learning_run_starts_in_the_garden`, `test_same_seed_same_csv_learning`.
- [x] **Step 2: Run them to see them fail** · `$PY -m unittest discover tests` · Expected: failures naming `allocate`
  and the new behavior.
- [x] **Step 3: Implement** the files above.
- [x] **Step 4: Run them to see them pass** · `$PY -m unittest discover tests` · Expected: `OK`.
- [x] **Step 5: Commit** as `feat: learning animals are born, learn, and pass on their brains`; `git show --stat HEAD`
  lists exactly this task's files.

### Task 3: The live viewer

**Files:** Create: `src/viewer.py`, `src/viewer.html`, `tests/test_viewer.py`.

**Interfaces:** `runs(folder: Path) -> list[dict]` (every `*/scoreboard.csv` under the runs folder: name, rows, whether
finished and its seconds); `python -m src.viewer [--runs runs] [--port 8765]` serves `/` (the page) and `/data` (that
list as JSON). The page redraws every 2 seconds: stage, population, births, generation, exam score, bites, steering to
food, per-run progress.

- [x] **Step 1: Write the failing tests:** `test_runs_lists_scoreboards`, `test_unfinished_run_has_no_seconds`,
  `test_server_answers_page_and_data` (a server on a free port, two requests).
- [x] **Step 2: Run them to see them fail** · `$PY -m unittest tests.test_viewer -v` · Expected: `ModuleNotFoundError`.
- [x] **Step 3: Implement** the viewer (from the scratchpad viewer of 2026-11-23, with the new columns).
- [x] **Step 4: Run them, then the suite** · `$PY -m unittest discover tests` · Expected: `OK`.
- [x] **Step 5: Open it** in the app's browser on `http://localhost:8765`; check the page shows a finished run and has
  no console errors.
- [x] **Step 6: Commit** as `feat: live viewer`; `git show --stat HEAD` lists exactly this task's files.

### Task 3b: Eating is the creature's action again (added 2026-11-23; the owner: "Eat must be action creature does")

Tasks 1 and 2 were built with eating "on contact" (Claude's change, reverted here). The brain chooses to eat.

**Files:** Modify: `src/stages.py` (`EAT_MISS_COST = 0.2`), `src/brain.py` (4 outputs: move x, move y, eat, attack),
`src/learner.py` (3 choices: move, eat, attack), `src/policies.py`, `src/creature.py` (the eat action and its miss),
`src/world.py` (no eating on contact; actions in the order attack, eat, move), tests (`test_brain`, `test_policies`,
`test_rules_policy`, `test_world_rules`, `test_learner`, `tests/helpers.py`).

**Interfaces:** actions are `("MOVE", (dx, dy))`, `("EAT", Food | None)` and `("ATTACK", Creature)`. A brain eats when
its eat output is above 0 (and attack does not win), a learner when it picks choice 1; the target is the food it stands
on, else the nearest food within reach, else `None` (a miss: `EAT_MISS_COST`, no bite). The random wanderer and the rules
yardstick eat whenever food is within reach.

- [x] **Step 1: Write the failing tests:** `test_brain_eat_in_reach`, `test_brain_eat_miss_when_nothing_in_reach`,
  `test_random_walker_eats_in_reach`, `test_eats_food_in_reach` (rules), `test_eat_action_takes_a_bite`,
  `test_eat_miss_costs_energy`, `test_second_eater_misses_last_bite`, `test_no_bite_without_choosing_to_eat`,
  `test_learner_eat_choice`; the brain has 4 outputs and the learner 3 choices.
- [x] **Step 2: Run them to see them fail** · `$PY -m unittest discover tests` · Expected: failures naming `EAT`.
- [x] **Step 3: Implement** the files above.
- [x] **Step 4: Run them to see them pass** · `$PY -m unittest discover tests` · Expected: `OK`. Then
  `grep -rn "on contact" src` · Expected: no output.
- [x] **Step 5: Commit** as `fix: eating is the creature's own action again`; `git show --stat HEAD` lists exactly this
  task's files.

### Task 4: The check that could prove the design wrong (the Garden)

- [x] **Step 1:** With the viewer open: `$PY -m src.evolve --seed 1 --steps 30000 --species learning --max-stage 0`
  (background).
- [x] **Step 2: Check:** the mean exam score of the last 3 exams is above the first 3, and the steering score of the
  last 5 rows is above the first 5.
- [x] **Step 3:** Show the owner the numbers. **If the check fails, stop:** report it with what was measured and a
  proposal; nothing after this task is run until the owner decides.

### Task 4b: Old age in the Garden (added 2026-11-23; the owner: "when none dies there is 0 evolution only learning" · "Ok")

**Files:** Modify: `src/stages.py` (`Stage.mortal` becomes `Stage.hunger`: whether an empty creature dies), `src/world.py`
(old age and the soup at every stage; hunger only where `stage.hunger`), `src/viewer.html` (the Garden's note), tests
(`test_stages`, `test_world_rules`, `tests/helpers.py`).

- [x] **Step 1: Write the failing tests:** `test_only_the_garden_cannot_starve`, `test_garden_never_starves`,
  `test_old_age_kills_in_the_garden`, `test_a_death_of_old_age_makes_room_for_a_baby`, `test_soup_also_in_the_garden`.
- [x] **Step 2: Run them to see them fail** · `$PY -m unittest tests.test_stages tests.test_world_rules` · Expected:
  failures naming `hunger` and old age in the Garden.
- [x] **Step 3: Implement** the files above.
- [x] **Step 4: Run the suite** · `$PY -m unittest discover tests` · Expected: `OK`.
- [x] **Step 5: Commit** as `feat: old age in the Garden, so evolution runs from the first step`.
- [x] **Step 6: The Garden run again**, with the viewer open: `$PY -m src.evolve --seed 1 --steps 30000 --species
  learning --max-stage 0 --out runs/2-learning` · Expected: births and deaths of old age after step 5,000, the deepest
  generation rising, and an exam score not below Task 4's. Show the owner both runs side by side.

### Task 4c: A lifespan is a range (added 2026-11-23; the owner: "Life span should be in a range")

**Files:** Modify: `src/stages.py` (`LIFESPAN = (4000, 6000)` replaces `MAX_AGE`), `src/creature.py` (`lifespan`),
`src/world.py` (a newcomer's and a baby's lifespan is drawn from the range; a creature dies at its own),
`tests/test_world_rules.py`.

- [x] **Step 1: Write the failing tests:** `test_every_creature_has_its_own_lifespan`,
  `test_a_creature_dies_at_its_own_lifespan`; the old-age tests set a creature's own lifespan.
- [x] **Step 2: Run them to see them fail** · Expected: `cannot import name 'LIFESPAN'`.
- [x] **Step 3: Implement** the files above.
- [x] **Step 4: Run the suite** · `$PY -m unittest discover tests` · Expected: `OK`. `grep -rn MAX_AGE src tests` · no output.
- [x] **Step 5: Commit** as `feat: every creature has its own lifespan`.

### Task 4d: The difficulty dial (added 2026-11-23; the owner: "nudge the parameters of difficulty very delicately … Okay")

Replaces the five stages and the exam gate (spec, "The difficulty dial").

**Files:**
- Create: `src/dial.py`, `tests/test_dial.py`
- Modify: `src/stages.py` (becomes the world's settings: `Settings`, `START`, `TEST_WORLD`, the dial's numbers; the
  `STAGES` table and `stage()` go), `src/world.py` (deaths: energy out, or past lifespan with low energy; lifespans from
  the settings; fractional food per step), `src/evolve.py` (the dial reviews each 1,000-step window; `--level`,
  `--max-level`; the exam only measures; `should_unlock` goes), `src/scoreboard.py` (columns `level` and the
  settings), `src/check_proof.py`, `src/compare.py`, `src/viewer.py`, `src/viewer.html`, `src/visualization.py`, their
  tests and `tests/helpers.py`.

**Interfaces:**
- `Settings(max_food: float, food_energy: tuple[float, float], food_per_step: float, move_cost: float, step_cost: float,
  lifespan: tuple[float, float], attacks: bool)` (frozen); `START` (the easy start); `TEST_WORLD` (the exam's);
  `LOW_ENERGY = 0.3`.
- `Dial(start: Settings = START)`: `level -> int` (nudges in force), `settings -> Settings`,
  `review(deaths: int, alive: int, newcomers: int) -> str` returning `"harder"`, `"easier"` or `"hold"` and changing
  `settings` accordingly; `nudged(settings, n) -> Settings` (the n-th nudge: one setting, about 5 %, within its limits;
  the 25th switches hunters on); `turn_to(level)` starts the dial at that many nudges (a yardstick world is held there).
- `World(width, height, settings, species, …)`; `world.settings` replaces `world.stage`.
- Scoreboard columns: `step, level, population, births, newcomers, generation, bites_per_creature_step, toward_food,
  flee, direction_concentration, exam_ratio, deaths_hunger, deaths_attack, deaths_age, max_food, food_energy,
  food_per_step, move_cost, step_cost, lifespan, hunters`.

- [x] **Step 1: Write the failing tests:** `test_dial`: `test_no_deaths_nudges_exactly_one_setting`,
  `test_nudges_take_turns`, `test_too_many_deaths_take_the_last_nudge_back`, `test_newcomers_take_the_last_nudge_back`,
  `test_some_deaths_hold`, `test_never_easier_than_the_start`, `test_settings_stay_within_limits`,
  `test_hunters_come_as_their_own_nudge`. `test_world_rules`: `test_well_fed_elder_lives_on`,
  `test_hungry_elder_dies_of_old_age`, `test_energy_out_is_death_from_the_start`. `test_evolve`:
  `test_dial_turns_when_nobody_dies`, `test_exam_does_not_change_the_world`.
- [x] **Step 2: Run them to see them fail** · `$PY -m unittest discover tests` · Expected: `No module named 'src.dial'`.
- [x] **Step 3: Implement** the files above (`src/stages.py` is renamed `src/settings.py`).
- [x] **Step 4: Run the suite** · `$PY -m unittest discover tests` · Expected: `OK`.
  `grep -rn "STAGES|should_unlock|max-stage" -E src tests` · Expected: no output.
- [x] **Step 5: Commit** (with Task 4e, built in the same change) as `feat: the difficulty dial (one delicate nudge at
  a time, taken back when too many die)`.
- [ ] **Step 6: Watch it**, viewer open: `$PY -m src.evolve --seed 1 --steps 30000 --species learning` · Show the owner
  the level, the settings, the deaths, intelligence and the average energy over time.

### Task 4e: A clear intelligence gauge, Live and History, average energy (added 2026-11-23, the owner's requests)

The owner: "inteligatesce should be rewards / waste or something similar.. its not super clear how to mesure it" · "at
the end who ever survices is smart in nature" · "if not it should show nothing running" · "only show me charts when its
running" · "and lets save history runs" · "add chart of average energy per creature?"

**Files:** Modify: `src/creature.py`, `src/world.py` (count energy earned and energy spent), `src/exam.py`
(`ExamResult(intelligence, wanderers)`), `src/scoreboard.py`, `src/evolve.py` (a dated folder and a `run.json` per
run), `src/check_proof.py`, `src/compare.py`, `src/viewer.py`, `src/viewer.html`, their tests.

**Interfaces:**
- `World.energy_earned` (eating and hunting) and `World.energy_spent` (moving, living, eating at nothing, attacking).
- `ExamResult(intelligence: Optional[float], wanderers: Optional[float])`: energy earned ÷ energy spent in the test
  world, the examined creatures' and the random wanderers' in the identical copy.
- Scoreboard columns: `step, level, population, energy, births, newcomers, generation, bites_per_creature_step,
  toward_food, flee, direction_concentration, efficiency, intelligence, wanderers, deaths_hunger, deaths_attack,
  deaths_age, max_food, food_energy, food_per_step, move_cost, step_cost, lifespan, hunters`. `energy` is the average,
  over the window, of each creature's energy as a share of its maximum; `efficiency` is the living world's energy
  earned ÷ energy spent in the window.
- A run's folder is `runs/<YYYYMMDD-HHMMSS>-<seed>-<kind>` with `run.json` `{seed, kind, steps, started}`.
- `viewer.runs(folder)` gives each run a `status`: `running` (no end yet, scoreboard written in the last 10 minutes),
  `finished`, or `stopped`. The page has two views: **Live** (charts of the runs in progress, or the words "Nothing is
  running") and **History** (every past run; a click shows its charts).

- [x] **Step 1: Write the failing tests:** `test_exam`: `test_intelligence_is_energy_earned_over_spent`;
  `test_world_rules`: `test_world_counts_energy_earned_and_spent`; `test_scoreboard`:
  `test_row_reports_efficiency_and_average_energy`; `test_viewer`: `test_a_run_is_running_only_while_it_keeps_writing`,
  `test_server_answers_page_and_data` (the page says "Nothing is running"); `test_evolve`: `test_run_writes_its_record`.
- [x] **Step 2: Run them to see them fail** · `$PY -m unittest tests.test_scoreboard tests.test_viewer` · Expected:
  failures naming `energy` and "Nothing is running".
- [x] **Step 3: Implement** the files above.
- [x] **Step 4: Run the suite** · `$PY -m unittest discover tests` · Expected: `OK`.
  `grep -rn "exam_ratio" src tests` · Expected: no output.
- [x] **Step 5: Open it** in the app's browser on `http://localhost:8765`: with no run, Live says "Nothing is
  running" and History lists the past runs; with a run, Live shows its charts.

### Task 4f: Auto turns the knobs by number and fedness; difficulty is one number (added 2026-11-23; the owner's request)

The owner: "an auto mode button that changes and reduces and increases the params based on the number of creatures and
their fedness." Spec: "The control room". Measured reason for the change of signal: the spec's "Why not nobody died".

**Files:** Modify: `src/dial.py`, `tests/test_dial.py`, `src/evolve.py` (the review gets the number alive, the average
energy and the soup's newcomers of the last 1,000 steps), `tests/test_evolve.py`, `src/check_proof.py`,
`tests/test_check_proof.py` (criterion 1 reads the difficulty), `README.md` (rule 4's last paragraph).

**Interfaces:**
- `Knob(name: str, harder: int, easiest: float, hardest: float)`; `KNOBS: tuple[Knob, ...]` for `max_food`,
  `food_energy`, `move_cost`, `step_cost`, `lifespan`, `food_per_step` (a range is named by its middle).
- `value(settings, name) -> float`; `with_value(settings, name, value) -> Settings` (a range keeps its shape around the
  new middle; the value is held within the knob's ends).
- `difficulty(settings) -> float`: nudges harder than `START` (below zero: easier); hunters count as one.
- `Dial(settings: Settings = START, max_level: Optional[int] = None)`: `settings`, `level -> float` (the difficulty),
  `review(alive: int, energy: Optional[float], newcomers: int) -> str` (`"harder"`, `"easier"`, `"hold"`),
  `set(settings)` (the owner's settings: the nudges to take back are forgotten), `turn_to(level)`.
- `MANY = 60`, `WELL_FED = 0.5`, `FEW = 30`, `HUNGRY = 0.35`; `TOO_MANY` goes.

- [x] **Step 1: Write the failing tests** (`test_dial`): `test_many_and_well_fed_nudges_one_knob_harder`,
  `test_nudges_take_turns`, `test_few_takes_the_last_nudge_back`, `test_hungry_takes_the_last_nudge_back`,
  `test_newcomers_take_the_last_nudge_back`, `test_in_between_holds`, `test_auto_can_go_easier_than_the_start`,
  `test_knobs_stay_within_their_ends`, `test_hunters_come_as_their_own_nudge`, `test_difficulty_counts_the_nudges`,
  `test_difficulty_of_the_owner_s_settings`, `test_set_forgets_the_nudges_to_take_back`,
  `test_a_range_keeps_its_shape`, `test_max_level_stops_the_dial`; (`test_evolve`):
  `test_auto_reviews_number_and_fedness`; (`test_check_proof`): `test_a_world_that_stayed_easy_fails` on the difficulty.
- [x] **Step 2: Run them to see them fail** · `$PY -m unittest tests.test_dial` · Expected: failures naming `Knob`,
  `difficulty` and `set`.
- [x] **Step 3: Implement** the files above.
- [x] **Step 4: Run the suite** · `$PY -m unittest discover tests` · Expected: `OK`.
  `grep -rn "TOO_MANY" src tests` · Expected: no output.
- [x] **Step 5: Commit** as `feat: Auto turns the knobs by number and fedness, both ways`; `git show --stat HEAD` lists
  exactly this task's files.

### Task 4g: A world that runs until stopped and takes its knobs while it runs (added 2026-11-23; the owner's request)

**Files:** Create: `src/control.py`, `tests/test_control.py`. Modify: `src/evolve.py` (`--forever`, `--mode`; the
control file read every 25 steps; a scoreboard row every 250 steps; Auto's review every 1,000; Stop), `src/scoreboard.py`
(column `manual`), `src/check_proof.py`, `src/compare.py` (their windows are counted in steps, not rows),
`src/viewer.py` (a run counts as running for 3 minutes after its last row), their tests.

**Interfaces:**
- `Control(mode: str, settings: Settings, stop: bool)` (frozen); `MODES = ("auto", "manual")`.
- `settings_from(values: Mapping[str, object], base: Settings) -> Settings`: the knobs named in `values` set on `base`;
  raises `ValueError` in plain words for an unknown knob, a value that is not a number, or one outside the knob's ends.
- `read_control(folder: Path) -> Optional[Control]`; `write_control(folder: Path, control: Control) -> None` (written
  to a temporary file and renamed, so the run never reads half a file).
- `new_run_folder(runs: Path, seed: int, kind: str) -> Path` (`<YYYYMMDD-HHMMSS>-<seed>-<kind>`; the one place a run's
  folder is named: `evolve` and the viewer both use it).
- `python -m src.evolve … [--steps N | --forever] [--mode auto|manual]`; `run.json` gets `steps: null` for a run
  without a limit. Constants: `ROW_EVERY = 250`, `REVIEW_EVERY = 1000`, `CONTROL_EVERY = 25`.
- Scoreboard column `manual` (1 while the owner holds the knobs, else 0), after `hunters`.

- [x] **Step 1: Write the failing tests:** `test_control`: `test_control_file_round_trip`,
  `test_unknown_knob_is_refused`, `test_value_outside_the_ends_is_refused`, `test_no_file_is_no_control`,
  `test_run_folders_are_dated`; `test_evolve`: `test_a_row_every_250_steps`, `test_manual_knobs_reach_the_world`
  (a control file written while the run goes: the next row shows the value and `manual` 1),
  `test_auto_leaves_manual_knobs_alone`, `test_back_to_auto_goes_on_from_the_owner_s_settings`,
  `test_stop_ends_a_run_without_a_limit` (it writes `time.txt` and its best brains),
  `test_steps_or_forever_not_both`; `test_check_proof`, `test_compare`: their fixtures get rows every 250 steps.
- [x] **Step 2: Run them to see them fail** · `$PY -m unittest tests.test_control tests.test_evolve` · Expected:
  `No module named 'src.control'`.
- [x] **Step 3: Implement** the files above.
- [x] **Step 4: Run the suite** · `$PY -m unittest discover tests` · Expected: `OK`.
- [x] **Step 5: Measure** the cost of the control check: 2,000 steps of `--species instinct` with and without a control
  file present · Expected: the same time within 2 %. The two times go in the ledger.
- [x] **Step 6: Commit** as `feat: a world that runs until stopped and takes its knobs while it runs`;
  `git show --stat HEAD` lists exactly this task's files.

### Task 4h: The control room on the live page (added 2026-11-23; the owner's request)

**Files:** Modify: `src/viewer.py` (`POST /start`, `POST /control`, `GET /knobs`; each run's `mode`), `src/viewer.html`
(Start, Stop, the Auto / Manual switch, the seven knobs, the change lines on the charts, the Difficulty chart),
`tests/test_viewer.py`, `README.md` (how to play).

**Interfaces:**
- `GET /knobs` → `[{name, label, easiest, hardest, start, step}]` and the hunters switch: the page builds its sliders
  from the server's list, so the ends are written in one place (`src/dial.py`).
- `POST /start` `{species, seed, mode, settings?}` → `200 {name}`; `400` in plain words for a bad value; `409` "2
  worlds are running already" at the limit.
- `POST /control` `{run, mode?, settings?, stop?}` → `200`; `400` for a bad value; `404` for a run that is not running.
- Both accept only `Content-Type: application/json`; the server stays on `127.0.0.1`.
- `viewer.runs()` gives each run `mode` (from its control file; `auto` when it has none).

- [x] **Step 1: Write the failing tests:** `test_knobs_list_the_ends_of_every_slider`, `test_start_launches_a_world`
  (the process launch is replaced by a recorder: the command, the folder, the control file),
  `test_start_refuses_a_third_world`, `test_start_refuses_a_bad_species`, `test_control_writes_the_run_s_file`,
  `test_control_refuses_a_value_outside_the_ends`, `test_control_refuses_a_run_that_is_not_running`,
  `test_posts_must_be_json`, `test_stop_is_written_to_the_run`, `test_page_has_the_control_room` (the words Start,
  Stop, Auto, Manual).
- [x] **Step 2: Run them to see them fail** · `$PY -m unittest tests.test_viewer` · Expected: 404s and missing words.
- [x] **Step 3: Implement** the files above.
- [x] **Step 4: Run the suite** · `$PY -m unittest discover tests` · Expected: `OK`.
- [x] **Step 5: Play with it** in the app's browser on `http://localhost:8765`: start a Manual world of learning
  animals; lower energy per food; the next point shows the value, the line marks the change, average energy falls;
  switch to Auto and see a knob move by itself within 1,000 steps; press Stop and find the world in History. No errors
  in the browser's console. Each of these is written in the ledger with what was seen.
- [x] **Step 6: Commit** as `feat: the control room (Auto, Manual, live knobs, Start and Stop)`; `git show --stat HEAD`
  lists exactly this task's files. Tell the owner it is ready to play with.

### Task 4i: Jump into the world (added 2026-11-23; the owner's request)

Spec: "Jump into the world".

**Files:** Modify: `src/control.py` (`Control.watch_until: Optional[float]`, `Control.watch_speed: float`;
`WATCH_SPEEDS = (1, 30)`, `WATCH_SPEED = 10`, `WATCH_LEASE = 10` seconds), `src/evolve.py` (while watched: the control
file every step, `view.json` after each step, a wait that holds the speed), `src/viewer.py` (`GET /view?run=`;
`POST /control` takes `watch: bool` and `watch_speed`), `src/viewer.html` (Jump in / Jump out, the picture, the speed
slider, the renewal every 2 seconds), their tests.

**Interfaces:**
- `world_view(world) -> dict`: `{step, creatures: [{x, y, size, energy, heading: [dx, dy], did, generation}],
  foods: [{x, y, energy}]}` (`energy` of a creature as a share of its maximum); in `src/evolve.py`.
- `POST /control {run, watch: true, watch_speed?}` sets `watch_until` to now + 10 s; `{run, watch: false}` clears it;
  a speed outside 1-30 is refused in plain words.
- `GET /view?run=<name>` → the run's `view.json`; `404` when the world is not being watched.

- [x] **Step 1: Write the failing tests:** `test_control`: `test_watch_round_trip`, `test_watch_speed_outside_its_ends_is_refused`;
  `test_evolve`: `test_watched_world_writes_its_view` (creatures and food where the world has them),
  `test_watching_never_changes_the_world` (the same scoreboard watched at 1,000 steps a second and not watched),
  `test_watch_runs_out_by_itself` (a lease in the past: no view written, no wait), `test_watched_world_waits`
  (100 steps at 50 a second take at least 1.8 s); `test_viewer`: `test_jump_in_sets_the_lease`,
  `test_jump_out_clears_it`, `test_view_of_a_watched_world`, `test_no_view_when_not_watched`,
  `test_page_has_jump_in` (the words Jump in, Jump out).
- [x] **Step 2: Run them to see them fail** · `$PY -m unittest tests.test_control tests.test_evolve tests.test_viewer` ·
  Expected: failures naming `watch_until`, `world_view` and `/view`.
- [x] **Step 3: Implement** the files above.
- [x] **Step 4: Run the suite** · `$PY -m unittest discover tests` · Expected: `OK`.
- [x] **Step 5: Measure** full speed is untouched: 2,000 instinct steps not watched, before and after this task ·
  Expected: within 2 %. Both times in the ledger.
- [x] **Step 6: Try it** in the app's browser: Jump in on a running world; the picture moves at about 10 steps a second
  and the creatures' circles, headings and food are visible; the speed slider changes the pace; Jump out and the step
  counter races again; close the tab while watching and see the world back at full speed within 15 seconds. No errors in
  the browser's console. Each in the ledger with what was seen.
- [x] **Step 7: Commit** as `feat: jump into a running world and watch it slowly`; `git show --stat HEAD` lists exactly
  this task's files. Tell the owner it is ready.

### Task 4j: Saves every 2,000 steps, resuming from any save, and Pause (added 2026-11-23; the owner's request)

Spec: "Saves, resuming from any save, and Pause".

**Files:** Create: `src/saves.py`, `tests/test_saves.py`. Modify: `src/learner.py` (`state()` / `load_state(state)`),
`src/dial.py` (`Dial.state()` / `Dial.from_state(state)`), `src/evolve.py` (a save every 2,000 steps and at Stop;
`--resume <save>`; Pause), `src/control.py` (`Control.paused: bool`), `src/viewer.py` (`GET /saves?run=`, `POST /start`
with `from: {run, step}`, `POST /control` with `paused`), `src/viewer.html` (Pause / Continue; the saves of a world and
Resume from here), their tests, `README.md`.

**Interfaces:**
- `SAVE_EVERY = 2000`; `SAVE_FORMAT = 1`; a save is `runs/<run>/saves/step-<step:07d>.npz` (arrays) with its
  description inside it (JSON: format, step, the run's seed and kind, creatures, food, settings, Auto's state, counters,
  generator states).
- `save_world(path: Path, world: World, dial: Dial, extra: dict) -> None` (written to a temporary file and renamed);
  `load_world(path: Path, learner_seed: int) -> Tuple[World, Dial, dict]`; `saves_of(run: Path) -> List[int]` (steps);
  `SaveError` for a save of another format or a broken file.
- `Learner.state() -> Dict[str, np.ndarray]` and `Learner.load_state(state)`: every slot's weights, inborn weights,
  optimizer moments, step counts, learning speeds and which slots are free.
- `python -m src.evolve --resume runs/<run>/saves/step-0016000.npz [--mode auto|manual] [--out DIR]`: the seed and
  species come from the save; the new run's `run.json` adds `from: {run, step}`; its scoreboard starts with the old
  run's rows up to that step.
- `GET /saves?run=<name>` → `[2000, 4000, …]`; `POST /start {from: {run, step}, mode}`; `POST /control {run, paused}`.

- [x] **Step 1: Write the failing tests:** `test_saves`: `test_save_and_load_give_the_same_world` (creatures, food,
  knobs, counters equal), `test_a_resumed_world_goes_on_as_the_old_one_did` (instinct: 2,000 steps from a save at 2,000
  equal the old world's steps 2,001-4,000), `test_learners_keep_what_they_learned` (weights, inborn weights, optimizer
  moments and learning speeds equal after loading), `test_auto_keeps_its_nudges` (a take-back after resuming undoes the
  same nudge it would have), `test_a_save_of_another_format_is_refused`, `test_saves_of_lists_the_steps`;
  `test_evolve`: `test_a_save_every_2000_steps`, `test_stop_saves`, `test_resume_starts_a_new_world_from_the_save`
  (new folder, `from` in run.json, the old rows up to the step, the next row at step + 250, the old world untouched),
  `test_pause_takes_no_steps` (paused for 1 s: no step taken, then it goes on); `test_control`: `test_paused_round_trip`;
  `test_viewer`: `test_saves_are_listed`, `test_resume_from_a_save_launches_a_world`, `test_resume_from_a_missing_save_is_refused`,
  `test_pause_is_written_to_the_run`, `test_page_has_pause_and_resume` (the words Pause, Continue, Resume from here).
- [x] **Step 2: Run them to see them fail** · `$PY -m unittest tests.test_saves tests.test_evolve tests.test_control
  tests.test_viewer` · Expected: `No module named 'src.saves'` and failures naming `paused`, `/saves`.
- [x] **Step 3: Implement** the files above.
- [x] **Step 4: Run the suite** · `$PY -m unittest discover tests` · Expected: `OK`.
- [x] **Step 5: Measure** a learning world's save: its size on disk and how long it takes; and that saving every 2,000
  steps leaves the speed within 2 % (2,000 instinct steps with and without the save). Numbers in the ledger.
- [x] **Step 6: Try it** in the app's browser: start a learning world; after 4,000 steps see its saves 2k and 4k; Pause
  and see the step counter hold, Continue and see it go on; stop it; in History pick it, pick 2k, Resume from here; the
  new world's charts start with the old history up to 2k and go on from there; Jump in shows its creatures. No errors in
  the browser's console. Each in the ledger with what was seen.
- [x] **Step 7: Commit** as `feat: a save every 2,000 steps, resume from any save, and Pause`; `git show --stat HEAD`
  lists exactly this task's files. Tell the owner it is ready.

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
- [ ] **Step 5: The dial run again** · Task 4d Step 6's run, with the viewer open · Expected: intelligence over the
  last 3 exams and the level reached are not below Task 4d's. If either is, stop and report both sets of numbers.
- [ ] **Step 6: Commit** as `feat: the brain's size evolves`; `git show --stat HEAD` lists exactly this task's files.
  Show the owner the run's numbers beside Task 4d's.

### Task 6: Tune the dial by measurement

**Files:** Modify: `src/settings.py`, `src/dial.py` (values and their measurements in comments), `tests/test_settings.py`,
`tests/test_dial.py` (the tuned values).

- [ ] **Step 1: Measure** the dial on learning and instinct animals: `$PY -m src.evolve --seed 1 --steps 30000 --species
  learning` and `… --species instinct`, viewer open · Read from the scoreboard: how often the dial turned harder, how
  often it took a nudge back, and how many died in the window after each nudge.
- [ ] **Step 2:** Adjust the nudge size, the window, the "too many died" share and the starting settings only where
  Step 1 shows a problem (a nudge that kills more than a quarter at once, or a dial that never turns). Nothing is tuned
  to an exam score. Every value tried and its result go in the report.
- [ ] **Step 3:** `$PY -m unittest discover tests` · Expected: `OK`.
- [ ] **Step 4: Commit** as `tune: the dial's numbers, measured`. Show the owner what was measured.

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
4. **The dial is delicate:** `tests/test_dial.py` passes (one setting per nudge, the last nudge taken back when too many
   die, never easier than the start), and `grep -rn "STAGES|should_unlock|max-stage|exam_ratio" -E src tests` → no output.
5. **Every long run was watched live:** the viewer was open in the app's browser before the Task 4, Task 4d, Task 5 and
   Task 7 runs; with no run the page says "Nothing is running", and past runs are in History.
5a. **The control room works:** Task 4h Step 5 was done in the app's browser and each part seen (a knob moved by hand
   reaches the running world and shows on the charts; Auto moves a knob by itself; Stop ends the world and it is in
   History), with no error in the browser's console; `tests/test_control.py` and `tests/test_viewer.py` pass.
6. **Brain size evolves:** `test_new_neuron_is_silent_at_birth` and `test_new_layer_is_silent_at_birth` pass, and Task 5
   Step 5 showed intelligence and the level reached not below Task 4d's.
7. **The proof was judged:** Task 7 Step 3's output is shown to the owner; every failed criterion is reported as failed,
   first, with its numbers.
8. **Committed:** `git log --oneline` on `evolve-from-scratch` shows one commit per Task 1, 2, 3, 4d (with 4e), 4f, 4g, 4h, 4i, 4j, 5 and 6.
