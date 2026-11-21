# Brains That Learn From Scratch by Evolution: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the failing TensorFlow brain with small NumPy brains that start random and improve only by evolution,
in a world that gets harsher level by level, and prove across 5 seeds that they learn to eat and to flee.

**Architecture:** The world stays `World` + `Creature` + `Food`. A creature's action comes from one of three policies in
`src/policies.py` (brain, random walker, rules); only baseline runs use the rules. `src/senses.py` turns what a creature
sees into the brain's 12 inputs. `src/evolve.py` runs a seeded simulation headless with zero delay, writes a scoreboard
(`src/scoreboard.py`), holds exams against random walkers (`src/exam.py`), unlocks levels (`src/levels.py`), keeps a hall
of fame (`src/hall_of_fame.py`) and restarts after extinction. `src/check_proof.py` checks the spec's proof criteria.

**Tech Stack:** Python 3.12, NumPy 2.1.3, Matplotlib 3.10.3, unittest.

**Spec:** `docs/superpowers/specs/2026-11-21-evolve-brains-from-scratch-design.md` (approved by the owner: "Build it").

## Global Constraints

- Every command runs from `C:\Users\Administrator\life` with `PY=/c/Users/Administrator/.venvs/life/Scripts/python`
  (Git Bash). Task 1 recreates that venv from `requirements.txt`.
- Dependencies: only `numpy==2.1.3` and `matplotlib==3.10.3` plus the standard library. No TensorFlow, no ML library.
- No hand-written behavior steers a brain creature. The rules live only in `src/policies.py rules_action`, called only
  for creatures whose `policy == "rules"`.
- Every random draw comes from the run's seeded generators (`random.Random(seed)`, `np.random.default_rng(seed)`), never
  the global `random` or `np.random` state: same seed, same machine, same run.
- Zero delay: nothing in `src/` sleeps or calls `plt.pause`; the window refreshes with `canvas.draw_idle()` +
  `canvas.flush_events()` only.
- At most 2 simulation runs at once on the box (4 shared cores).
- Spec values: brain 12 → 16 (tanh) → 4 (tanh); mutation rate 0.1 and 0.2, scale 0.2; size × (1 ± 0.5) within 0.5–2.0;
  start 20 creatures, size 0.5–2.0, energy 50–100; cap 40; hall of fame 10; exam every 2,000 steps, 500 steps, up to 20
  brains against walkers matched one to one; unlock at exam ratio ≥ 1.5 in two exams in a row with ≥ 10 alive; hunters
  gain 80 %; old age 3,000 steps (tuned in Task 9); energy capped at the maximum; split energy ÷ 3; one scoreboard row
  every 1,000 steps; output under `runs/`.
- Each task ends with a commit on branch `evolve-from-scratch`; the commit judge runs on every commit. Nothing is pushed
  or merged by this plan.

## Review Focus

1. A score with nothing to measure (no food seen, no bigger creature in view, walkers with 0 bites): reported as empty,
   never a crash or a fake 0 that unlocks a level. Tests: Task 5 `test_no_samples_is_none`, Task 6
   `test_ratio_none_when_walkers_never_bite`.
2. A creature pressed into a wall or corner: it stays inside the world and its senses stay right. Test: Task 4
   `test_move_clamped_in_corner`.
3. Two creatures going for the same last bite or the same victim in one step: the second gets a miss, nothing crashes.
   Test: Task 4 `test_second_eater_misses_last_bite`, `test_dead_target_is_a_miss`.
4. Extinction before anyone has eaten: the restart still has brains to start from (the dead are offered to the hall
   of fame whatever they ate). Test: Task 7 `test_extinction_restarts_from_hall_of_fame`.
5. Sizes at the limits (0.5 against 2.0): relative size is clipped, not out of range. Test: Task 3
   `test_relative_size_clipped`.

---

### Task 1: The NumPy brain and a Windows-installable requirements file

**Files:**
- Create: `src/brain.py`, `tests/test_brain.py`
- Modify: `requirements.txt` (only `numpy==2.1.3`, `matplotlib==3.10.3`), `.gitignore` (add `runs/`, `saved_brains/`,
  `__pycache__/`)

**Interfaces:**
- Produces: `class Brain` with `N_IN = 12`, `N_HIDDEN = 16`, `N_OUT = 4`;
  `Brain(w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray)`;
  `Brain.random(rng: np.random.Generator) -> Brain` (weights normal with scale `1/sqrt(fan_in)`, biases 0);
  `forward(x: np.ndarray) -> np.ndarray` (shape (12,) → (4,), tanh on both layers; another shape raises `ValueError`);
  `copy() -> Brain`; `mutate(rng, rate: float, scale: float) -> Brain` (new brain; each weight and bias, with
  probability `rate`, gets `+ N(0, scale)`); `save(path) -> None` and `Brain.load(path) -> Brain` (`.npz`).

- [x] **Step 1: Recreate the venv from the new requirements**

  Write `requirements.txt` and `.gitignore` as above. Run:
  `rm -rf /c/Users/Administrator/.venvs/life && "/c/Program Files/Python312/python" -m venv /c/Users/Administrator/.venvs/life && $PY -m pip install -q -r requirements.txt && $PY -c "import numpy, matplotlib; print(numpy.__version__, matplotlib.__version__)"`
  Expected: `2.1.3 3.10.3`, exit 0.

- [x] **Step 2: Write the failing tests** in `tests/test_brain.py`:
  - `test_forward_shape_and_range`: `Brain.random(np.random.default_rng(0)).forward(np.ones(12))` has shape `(4,)`, every
    value in `[-1, 1]`.
  - `test_random_is_seeded`: two `Brain.random(default_rng(7))` give equal `forward` on the same input.
  - `test_copy_is_equal_and_independent`: the copy's `forward` equals the original's; after `copy.w1[0, 0] += 1` the
    original's `forward` is unchanged.
  - `test_mutate_changes_about_rate`: `mutate(default_rng(1), 0.2, 0.2)` changes between 15 % and 25 % of the 276
    parameters; the parent's parameters are unchanged.
  - `test_save_load_round_trip`: saved to a temp folder and loaded, `forward` is equal.
  - `test_wrong_input_size_raises`: `forward(np.ones(11))` raises `ValueError`.

- [x] **Step 3: Run them to see them fail**
  Run: `$PY -m unittest tests.test_brain -v` · Expected: `ModuleNotFoundError: No module named 'src.brain'`.

- [x] **Step 4: Implement `src/brain.py`** with the interface above.

- [x] **Step 5: Run them to see them pass**
  Run: `$PY -m unittest tests.test_brain -v` · Expected: `Ran 6 tests … OK`.

- [x] **Step 6: Measure the brain's speed**
  Run: `$PY -m timeit -s "import numpy as np; from src.brain import Brain; b=Brain.random(np.random.default_rng(0)); x=np.ones(12)" "b.forward(x)"`
  Expected: a time per call, kept for the report (TensorFlow today: about 12 ms per creature-step, spec §"What we found").

- [x] **Step 7: Commit** `requirements.txt .gitignore src/brain.py tests/test_brain.py` as
  `feat: NumPy brain, Windows-installable requirements`. Check: `git show --stat HEAD` lists exactly those 4 files.

### Task 2: Levels

**Files:**
- Create: `src/levels.py`, `tests/test_levels.py`

**Interfaces:**
- Produces: frozen dataclass `Level(number: int, food_energy: tuple[float, float], max_food: int, food_per_step: int,
  move_cost: float, step_cost: float, attacks: bool)`; `LEVELS: tuple[Level, ...]` with the spec's 5 rows and
  `food_per_step = 3` (today's `int(50 * 30 * 0.002)`); `level(number: int) -> Level` (raises `ValueError` outside 1–5);
  constants `HUNTER_SHARE = 0.8`, `MAX_AGE = 3000`, `EAT_MISS_COST = 0.2` (today's cost of eating where there is no food;
  Claude's pick of the two costs, spec "Left to the builder").

- [x] **Step 1: Write the failing tests** in `tests/test_levels.py`:
  - `test_five_levels_in_order`: `[l.number for l in LEVELS] == [1, 2, 3, 4, 5]`.
  - `test_level_one_matches_spec`: `level(1)` has `food_energy == (20, 40)`, `max_food == 200`, `move_cost == 0.5`,
    `step_cost == 0.05`, `attacks is False`.
  - `test_attacks_from_level_four`: `[l.attacks for l in LEVELS] == [False, False, False, True, True]`.
  - `test_unknown_level_raises`: `level(0)` and `level(6)` raise `ValueError`.

- [x] **Step 2: Run them to see them fail** · `$PY -m unittest tests.test_levels -v` · Expected: `ModuleNotFoundError`.

- [x] **Step 3: Implement `src/levels.py`** with the interface and the spec's table.

- [x] **Step 4: Run them to see them pass** · Expected: `Ran 4 tests … OK`.

- [x] **Step 5: Commit** `src/levels.py tests/test_levels.py` as `feat: levels from forgiving to harsh`.
  Check: `git show --stat HEAD` lists exactly those 2 files.

### Task 3: Senses

**Files:**
- Create: `src/senses.py`, `tests/test_senses.py`
- Modify: `src/sensors.py` (delete the unused grid `VisionSensor`; `ProximitySensor` stays the one place that lists
  what is in range), `src/creature.py` (add `is_family(a, b) -> bool` and `Creature.can_reach(food) -> bool`; nothing
  else changes in this task)

**Interfaces:**
- Consumes: `Creature` fields `x, y, size, energy, max_energy, radius, id, parent_id`; `ProximitySensor.get_reading`.
- Produces:
  - `is_family(a: Creature, b: Creature) -> bool`: parent, child, or the same non-zero `parent_id`.
  - `Creature.can_reach(food: Food) -> bool`: distance ≤ `(radius + food.radius) × EAT_RANGE_FACTOR`, the one eat-range
    rule.
  - frozen dataclass `Senses(vector: np.ndarray, food: Food | None, other: Creature | None, on_food: Food | None)`.
  - `sense(creature: Creature, world: World) -> Senses`: the 12 inputs exactly as spec §1; `other` is the nearest
    non-family creature in range; `on_food` the food the creature overlaps.
  - `display_intent(creature: Creature, action: Action, senses: Senses) -> str`: `"ATTACK"`; `"GO_TO_FOOD"` when moving
    within 45° of `senses.food`; `"RUN_AWAY"` when moving away from an `other` at least 1.5 × bigger; `"REST"` for a
    zero move or a bite; else `"WANDER"`.

- [x] **Step 1: Write the failing tests** in `tests/test_senses.py` (the world is a
  `types.SimpleNamespace(creatures=[…], foods=[…])`, all that `sense` reads; foods are built with an explicit energy):
  - `test_nothing_seen`: a lone creature of size 1.25 and energy half its maximum: `vector[0] == 0.5`, `vector[1] == 0`,
    `vector[2:] ` all 0.
  - `test_food_seen_east`: food 1.5 to the east: `vector[3:7] == [1, 0.5, 1, 0]`, `senses.food` is that food.
  - `test_food_out_of_range`: food 3.5 away: `vector[3] == 0`, `senses.food is None`.
  - `test_family_is_invisible`: a child 1 away and a stranger 2 away: `senses.other` is the stranger, `vector[8] == 2/3`.
  - `test_relative_size_clipped`: other 2.0 against own 0.5 → `vector[11] == 1.0`; 0.5 against 2.0 → `-1.0`; equal → 0.
  - `test_on_food_flag`: standing on food → `vector[2] == 1` and `senses.on_food` is it.
  - `test_display_intent_toward_food` and `test_display_intent_run_away`: the labels above.
  - `test_is_family`: parent–child and siblings true; strangers and two creatures with `parent_id == 0` false.

- [x] **Step 2: Run them to see them fail** · `$PY -m unittest tests.test_senses -v` · Expected:
  `ModuleNotFoundError: No module named 'src.senses'`.

- [x] **Step 3: Implement** `src/senses.py`, `is_family` and `can_reach`; delete `VisionSensor`.

- [x] **Step 4: Run them to see them pass** · Expected: `Ran 9 tests … OK`. Then
  `grep -rn VisionSensor src tests` · Expected: no output.

- [x] **Step 5: Commit** `src/senses.py src/sensors.py src/creature.py tests/test_senses.py` as
  `feat: senses for the brain`. Check: `git show --stat HEAD` lists exactly those 4 files.

### Task 4: Policies, world rules, and the TensorFlow brain removed

**Files:**
- Create: `src/policies.py`, `tests/test_policies.py`, `tests/test_rules_policy.py`, `tests/test_world_rules.py`
- Modify: `src/creature.py`, `src/world.py`, `src/food.py` (energy becomes a required argument),
  `src/visualization.py` (number inside = `food_eaten`; corner = `Step`, `Creatures`, `MaxScore` = most food eaten by a
  living creature, `Level`), `tests/test_visualization.py` (rewritten for the new creature), `tests/test_senses.py`
  (constructor calls only)
- Not runnable from this task until Task 7: `src/main.py` (Task 7 rewires it to the runner)
- Delete: `src/neural_network.py`, `tests/test_neural_network_learning.py`, `tests/test_attacking_and_predation.py`
  (a TensorFlow training test), `tests/test_world_and_creature.py`, `tests/test_food_and_eating.py`,
  `tests/test_continuous_movement.py`, `tests/test_parent_child_relationship.py` (their cases move to the three new test
  files, against the new interfaces)

**Interfaces:**
- Consumes: `Brain` (Task 1), `Level`, `HUNTER_SHARE`, `MAX_AGE`, `EAT_MISS_COST` (Task 2), `Senses`, `sense`,
  `display_intent`, `is_family`, `can_reach` (Task 3).
- Produces:
  - `Action = tuple[str, object]`: `("MOVE", (dx, dy))`, `("EAT", Food | None)`, `("ATTACK", Creature)`.
  - `brain_action(creature, senses, attacks_on: bool) -> Action`: outputs `o = brain.forward(senses.vector)`; attack
    when `attacks_on and o[3] > 0 and o[3] >= o[2] and senses.other`; else eat when `o[2] > 0` (target `senses.on_food`,
    else `senses.food` if `can_reach`, else `None`); else move `(o[0], o[1]) × velocity`, capped at `velocity`.
  - `random_action(creature, senses, rng: random.Random) -> Action`: eat when food is in reach; else move at a random
    angle and a speed uniform in `[0, velocity]`.
  - `rules_action(creature, world, rng: random.Random) -> Action`: today's `Creature._decide_without_brain` behavior
    (flee from ≥ 1.5 × bigger, attack ≤ 1/1.5 in range, never family, chase and eat food, else wander or rest), moved
    here.
  - `decide(creature, world, senses, rng) -> Action`: picks by `creature.policy`.
  - `Creature(id: int, x, y, size, energy, policy: str, brain: Brain | None = None, parent_id: int = 0,
    generation: int = 0)` (`policy` is `"brain"`, `"random"` or `"rules"`; `"brain"` without a brain raises
    `ValueError`); new fields `age`, `food_eaten`; `apply_action(action, world)`; `should_split()`;
    `split(world) -> list[Creature]`; `get_color()` (purple for brain, green otherwise). Removed: `decide`,
    `_decide_without_brain`, `_set_intent_and_vector`, the teacher, `get_nn_score`, `_save_brain`, `MIN_STEPS`.
  - `World(width: int, height: int, level: Level, rng: random.Random, np_rng: np.random.Generator,
    food_rng: random.Random, evolving: bool = True)`; `next_id() -> int`; `set_level(level)`; `spawn_food()` (draws
    `level.food_per_step` candidates every step whatever the food count, keeps those that fit, up to `max_food`, energy
    uniform in `level.food_energy`; with `evolving=False` it skips the overlap check with creatures, so two exam copies
    get the same food); `step(observer: Callable[[Creature, Senses, Action], None] | None = None)`; counters
    `deaths: collections.Counter` (`"hunger"`, `"attack"`, `"age"`), `total_bites`, `creature_steps`;
    `dead_this_step: list[Creature]`. With `evolving=False`: no splits, no deaths, no aging.

- [x] **Step 1: Write the failing tests.**
  - `tests/test_policies.py` (a `Brain` built with zero weights and chosen output biases gives fixed outputs):
    `test_brain_move_vector` (outputs 0.6, 0.8, −1, −1 → `("MOVE", (0.6v, 0.8v))`), `test_brain_move_capped` (1, 1 →
    length `v`), `test_brain_eat_in_reach`, `test_brain_eat_miss_when_nothing_in_reach` (→ `("EAT", None)`),
    `test_brain_attack_needs_attacks_on`, `test_brain_never_targets_family` (only a child in view → not `ATTACK`),
    `test_random_walker_eats_in_reach`, `test_random_walker_speed_within_velocity`.
  - `tests/test_rules_policy.py`, moved from today's tests: `test_flee_from_bigger`, `test_attack_smaller_in_range`,
    `test_parent_does_not_attack_child`, `test_child_does_not_attack_parent`, `test_goes_toward_visible_food`,
    `test_eats_food_in_reach`.
  - `tests/test_world_rules.py`: `test_step_cost` (a zero move loses `step_cost`), `test_move_cost_and_speed_cap`,
    `test_move_clamped_in_corner`, `test_bite_moves_one_energy`, `test_second_eater_misses_last_bite` (miss costs
    `EAT_MISS_COST`), `test_hunter_gains_eighty_percent` (attacker energy `+ 0.8 × taken − attack_cost`),
    `test_kill_leaves_corpse_and_counts_attack_death`, `test_dead_target_is_a_miss`, `test_family_cannot_be_hurt`,
    `test_energy_capped_at_max`, `test_old_age_death` (age `MAX_AGE` → removed, `deaths["age"] == 1`),
    `test_split_three_children` (parent gone; 3 children with `max_energy / 3` each, unique ids, generation + 1, one brain
    equal to the parent's, two different, sizes within 0.5–2.0), `test_no_split_at_cap_of_40`,
    `test_food_respects_level`, `test_same_seed_same_world` (two worlds, same seeds, 200 steps → same positions),
    `test_exam_world_never_kills_or_splits`, `test_brain_creatures_never_call_rules` (`rules_action` patched to raise; 500
    steps of 20 brain creatures run without raising).
  - `tests/test_visualization.py` (Agg backend): `test_render_empty_world`, `test_render_with_creatures_and_food`,
    `test_corner_shows_level`.

- [x] **Step 2: Run them to see them fail** ·
  `$PY -m unittest tests.test_policies tests.test_rules_policy tests.test_world_rules tests.test_visualization -v` ·
  Expected: `ModuleNotFoundError: No module named 'src.policies'` and failures in the visualization tests.

- [x] **Step 3: Implement** `src/policies.py`, the new `Creature` and `World`, `Food` with required energy, the
  visualizer change; delete the files listed above.

- [x] **Step 4: Run them, then the whole suite** · `$PY -m unittest discover tests` · Expected: `OK`, no failures or
  errors. Then `grep -rn "tensorflow\|neural_network\|nn_score\|MIN_STEPS" src tests` · Expected: no output.

- [x] **Step 5: Commit** every file of this task as `feat: brain-only creatures, world rules, TensorFlow removed`.
  Check: `git show --stat HEAD` lists exactly the files above.

### Task 5: Scoreboard

**Files:**
- Create: `src/scoreboard.py`, `tests/test_scoreboard.py`

**Interfaces:**
- Consumes: `Creature`, `Senses`, `Action`, `World` counters (Task 4).
- Produces: `class Scoreboard` with `observe(creature, senses, action) -> None` (the world's observer) and
  `row(world: World, exam_ratio: float | None, restarts: int) -> dict` with keys, in order: `step, level, population,
  max_generation, bites_per_creature_step, toward_food, flee, direction_concentration, exam_ratio, deaths_hunger,
  deaths_attack, deaths_age, restarts`. Scores cover the window since the last row and reset after it; one with no
  samples is `None`. Definitions: toward-food = mean cosine between the move and the direction of `senses.food`, over
  moves of length > 0 with food seen; flee = share of steps with an `other` ≥ 1.5 × bigger where the creature moved away
  (dot product < 0); direction concentration = length of the mean unit move vector.

- [x] **Step 1: Write the failing tests** in `tests/test_scoreboard.py`: `test_straight_at_food_scores_one`,
  `test_sideways_scores_zero`, `test_flee_counts_moving_away`, `test_eating_beside_a_hunter_is_not_fleeing`,
  `test_concentration_one_direction_is_one`, `test_concentration_four_directions_is_zero`, `test_no_samples_is_none`,
  `test_row_resets_window`, `test_row_has_exact_keys`.

- [x] **Step 2: Run them to see them fail** · `$PY -m unittest tests.test_scoreboard -v` · Expected:
  `ModuleNotFoundError`.

- [x] **Step 3: Implement `src/scoreboard.py`.**

- [x] **Step 4: Run them to see them pass** · Expected: `Ran 9 tests … OK`.

- [x] **Step 5: Commit** `src/scoreboard.py tests/test_scoreboard.py` as `feat: scoreboard`. Check: `git show --stat
  HEAD` lists exactly those 2 files.

### Task 6: Exam and hall of fame

**Files:**
- Create: `src/exam.py`, `src/hall_of_fame.py`, `tests/test_exam.py`, `tests/test_hall_of_fame.py`

**Interfaces:**
- Consumes: `World(evolving=False)`, `Brain`, `Level`, `random_action` (Task 4).
- Produces:
  - frozen dataclass `ExamResult(brain_bites_per_step: float, walker_bites_per_step: float, ratio: float | None)`.
  - `exam_worlds(brains: list[Brain], sizes: list[float], level: Level, seed: int) -> tuple[World, World]`: two copies,
    one with the brains, one with random walkers of the same sizes at the same spots, same food.
  - `run_exam(brains, sizes, level, seed, steps: int = 500) -> ExamResult` (`ratio` is `None` when walkers bite 0).
  - `walker_score(level: Level, seed: int, steps: int = 500, n: int = 20) -> float` (sizes 0.5–2.0 from the seed), used
    by Task 9's tuning.
  - `class HallOfFame(size: int = 10)`: `offer(brain, food_eaten: float)`, `brains() -> list[Brain]` (best first),
    `save(folder: Path)` (`best_00.npz` …).

- [x] **Step 1: Write the failing tests.** `tests/test_exam.py`: `test_copies_start_identical` (same food positions and
  creature spots), `test_same_seed_same_result`, `test_ratio_none_when_walkers_never_bite` (a level with
  `max_food = 0`), `test_exam_leaves_brains_unchanged`. `tests/test_hall_of_fame.py`: `test_keeps_best_ten_of_fifteen`,
  `test_empty_is_empty_list`, `test_save_writes_files`.

- [x] **Step 2: Run them to see them fail** · `$PY -m unittest tests.test_exam tests.test_hall_of_fame -v` · Expected:
  `ModuleNotFoundError`.

- [x] **Step 3: Implement** `src/exam.py` and `src/hall_of_fame.py`.

- [x] **Step 4: Run them to see them pass** · Expected: `Ran 7 tests … OK`.

- [x] **Step 5: Commit** the 4 files as `feat: exam against random walkers, hall of fame`. Check: `git show --stat HEAD`
  lists exactly those 4 files.

### Task 7: Runner, window and visualizer

**Files:**
- Create: `src/evolve.py`, `tests/test_evolve.py`
- Modify: `src/main.py` (runs `evolve.main(["--seed", "0", "--window"])` when given no arguments, else passes its
  arguments plus `--window`)

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `parse_args(argv: list[str] | None) -> argparse.Namespace`: `--seed` (int, required), `--steps` (int, 100000),
    `--policy` (`brain|random|rules`, `brain`), `--level` (int, 1: the start level, held fixed for baselines),
    `--max-level` (int, 5: the highest level a brain run may unlock; used by Task 10's level-1 check), `--window`,
    `--out` (default `runs/<seed>-brain`, baselines `runs/<seed>-<policy>-L<level>`).
  - `should_unlock(ratios: list[float | None], population: int, level: int, max_level: int) -> bool`: the last two
    ratios both ≥ 1.5, population ≥ 10, `level < max_level`.
  - `run(args: argparse.Namespace, levels: tuple[Level, ...] = LEVELS) -> Path`: the seeded loop; returns the CSV path
    (`scoreboard.csv` in `--out`, a row flushed every 1,000 steps and printed); exams every 2,000 steps for brain runs
    (up to 20 living brains sampled with the run's rng); dead brains offered to the hall of fame each step; at
    extinction 20 creatures restart (brain runs: mutated hall-of-fame brains, rate 0.1, scale 0.2; baselines: fresh
    creatures of their policy); hall of fame saved at the end.
  - `main(argv: list[str] | None = None) -> None`.

- [x] **Step 1: Write the failing tests.** `tests/test_evolve.py`: `test_short_run_writes_two_rows` (seed 1, 2,000 steps
  → 2 rows with the Task 5 keys), `test_same_seed_same_csv` (seed 3, 1,000 steps, twice → identical files),
  `test_unlock_rule` (`[1.6, 1.7]`, 10 alive → True; `[1.6, 1.4]` → False; `[None, 1.6]` → False; 9 alive → False;
  `level == max_level` → False), `test_extinction_restarts_from_hall_of_fame` (a test level with no food and high costs:
  `restarts ≥ 1` and 20 alive after the restart), `test_baseline_never_unlocks` (random policy, 4,000 steps, level stays
  1).

- [x] **Step 2: Run them to see them fail** · `$PY -m unittest tests.test_evolve -v` · Expected:
  `ModuleNotFoundError: No module named 'src.evolve'`.

- [x] **Step 3: Implement** `src/evolve.py` and `src/main.py`.

- [x] **Step 4: Run them, then the whole suite** · `$PY -m unittest discover tests` · Expected: `OK`, no failures or
  errors. Then `grep -rn "pause\|sleep" src` · Expected: no output.

- [x] **Step 5: Measure speed** · `time $PY -m src.evolve --seed 1 --steps 10000` · Expected: 10 rows printed and the
  elapsed time kept for the report. **If 10,000 steps take over 3 minutes** (so 100,000 over 30): stop and propose to the
  owner a task that vectorizes the senses with NumPy (spec "Left to the builder"); it is added to this plan only once the
  owner agrees.

- [x] **Step 6: Commit** the task's files as `feat: zero-delay runner and scoreboard CSV`. Check:
  `git show --stat HEAD` lists exactly those files.

- [x] **Step 7: Show the owner** the 10 scoreboard rows of Step 5 (random brains at level 1: the starting line).

### Task 8: Proof checker

**Files:**
- Create: `src/check_proof.py`, `tests/test_check_proof.py`

**Interfaces:**
- Consumes: the CSV of Task 7.
- Produces: `check(brain_csvs: list[Path], rules_csvs: list[Path]) -> list[tuple[str, bool | None, str]]` (criterion,
  pass, the numbers behind it; criterion 5 is `None`: reported only) and a CLI
  `python -m src.check_proof --brain <csv>... --rules <csv>...` printing one line per criterion. Criteria exactly as spec
  "How we prove it works": (1) per seed, mean exam ratio of the last 5 exams ≥ 2 and above the first 5; (2) per seed,
  mean toward-food over the last 10 rows ≥ 0.7; (3) at least 4 of 5 seeds reach level 4, and in each of those the mean
  flee over its rows at level ≥ 4 ≥ 0.70; (4) per seed, population > 0 in the last row and `restarts` unchanged over the
  last 20 rows; (5) the rules' bites per creature-step per level.

- [x] **Step 1: Write the failing tests** with small synthetic CSVs: `test_all_pass`, one failing test per criterion
  (`test_low_exam_ratio_fails`, `test_flat_exam_ratio_fails`, `test_low_toward_food_fails`,
  `test_three_seeds_at_level_four_fails`, `test_low_flee_fails`, `test_late_restart_fails`, `test_extinct_at_end_fails`),
  `test_rules_reported_not_judged`.

- [x] **Step 2: Run them to see them fail** · `$PY -m unittest tests.test_check_proof -v` · Expected:
  `ModuleNotFoundError`.

- [x] **Step 3: Implement `src/check_proof.py`.**

- [x] **Step 4: Run them to see them pass** · Expected: `Ran 9 tests … OK`.

- [x] **Step 5: Commit** the 2 files as `feat: proof checker`. Check: `git show --stat HEAD` lists exactly those 2 files.

### Task 9: Tune the levels and old age by measurement

**Files:**
- Modify: `src/levels.py` (level values, `MAX_AGE`; each value's measurement in a comment), `tests/test_levels.py`
  (`test_level_one_matches_spec` updated to the tuned values, with the measurement named)

- [x] **Step 1: Measure random walkers at every level**
  Run: `$PY -c "from src.exam import walker_score; from src.levels import LEVELS; print([round(walker_score(l, 1), 3) for l in LEVELS])"`
  Expected after tuning: every value ≤ 0.3 (spec tuning target). While any is above, lower that level's `max_food` or
  `food_energy` and run again; each value tried and its result go in the report.

- [ ] **Step 2: Measure turnover at level 1**
  Run: `$PY -m src.evolve --seed 1 --steps 20000 --max-level 1 --out runs/tune-age`
  Expected: once the population reaches 40, `deaths_age` > 0 in later rows and `max_generation` still rising over the
  last 10,000 steps. If `max_generation` stalls, lower `MAX_AGE` and run again.

- [ ] **Step 3: Run the suite** · `$PY -m unittest discover tests` · Expected: `OK`.

- [ ] **Step 4: Commit** the 2 files as `tune: level values and old age, measured`. Check: `git show --stat HEAD` lists
  exactly those 2 files.

- [ ] **Step 5: Show the owner** the walker scores per level and the turnover rows.

### Task 10: First evidence at level 1 (the owner's small step)

- [ ] **Step 1: Run** `$PY -m src.evolve --seed 1 --steps 30000 --max-level 1 --out runs/1-brain-L1`
  (background).

- [ ] **Step 2: Check** in `runs/1-brain-L1/scoreboard.csv`: the mean exam ratio of the last 3 exams is above the first
  3, and the toward-food score of the last 5 rows is above the first 5.

- [ ] **Step 3: Show the owner** the rows. **If either check fails, stop**: report the numbers and propose a change
  (for example a larger hidden layer or different mutation), which joins the plan only once the owner agrees.

### Task 11: The full proof

- [ ] **Step 1: Brain runs** · seeds 1–5: `$PY -m src.evolve --seed <n> --steps 100000` (background, two at a time).

- [ ] **Step 2: Rules yardstick** · levels 1–5: `$PY -m src.evolve --seed 1 --steps 20000 --policy rules --level <n>`
  (background, two at a time).

- [ ] **Step 3: Check** · `$PY -m src.check_proof --brain runs/{1,2,3,4,5}-brain/scoreboard.csv --rules runs/1-rules-L{1,2,3,4,5}/scoreboard.csv`
  Expected: PASS on criteria 1–4 and the rules' numbers printed. A FAIL is reported as not done, with its numbers; if
  the runs look too short to judge, a longer run is proposed, never claimed as a pass.

- [ ] **Step 4: Show the owner** the checker's output and each seed's level progression.

---

## Must be completed to mark this as done

1. **Every test passes.** `$PY -m unittest discover tests` → `OK`, with the count of tests run and no failures or errors.
2. **TensorFlow is gone.** `grep -rn "tensorflow\|neural_network" src tests requirements.txt` → no output.
3. **No rules steer a brain.** `test_brain_creatures_never_call_rules` passes, and `grep -n "rules_action" src/*.py`
   shows only its definition and the `"rules"` branch of `decide` in `src/policies.py`.
4. **It installs on Windows.** Task 1 Step 1's fresh venv and install exit 0, and item 1 passes in that venv.
5. **Zero delay.** `grep -rn "pause\|sleep" src` → no output; the 10,000-step time of Task 7 Step 5 is reported.
6. **The brain is proven.** Task 11 Step 3's checker prints PASS for criteria 1–4 and the rules' numbers. Any FAIL means
   this item is not done and is reported first, with the numbers.
7. **The owner saw each small step.** Scoreboards shown after Task 7, Task 9, Task 10 and Task 11, in the conversation.
8. **Every task committed** on `evolve-from-scratch`: `git log --oneline master..evolve-from-scratch` shows the design
   commit, this plan's commit and one commit per Task 1–9.
