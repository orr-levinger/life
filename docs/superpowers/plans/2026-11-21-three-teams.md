# Three Teams (Evolution, Learning, Both): Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run three teams on the same world (evolution only in NumPy, PPO learning only in TensorFlow, and both) and
compare which makes creatures smartest.

**Architecture:** The world, senses, levels, exam, scoreboard and runner built by Tasks 1–8 of
`2026-11-21-evolve-brains-from-scratch.md` stay. Splitting, old age and the extinction restart go; `src/rounds.py`
breeds each round's next generation. `src/learner.py` holds every learning creature's actor-critic network together in
TensorFlow and runs PPO. `src/evolve.py` runs a team in rounds (evolution, both) or as a continuous world with newborns
replacing the dead (learning, baselines). `src/compare.py` tabulates the teams. This plan replaces Tasks 9–11 of the first
plan.

**Tech Stack:** Python 3.12, NumPy 2.1.3, Matplotlib 3.10.3, TensorFlow 2.19.0, unittest.

**Spec:** `docs/superpowers/specs/2026-11-21-three-teams-design.md` (approved in conversation: "Okay, let's build it").

## Global Constraints

- Commands run from `C:\Users\Administrator\life` with `PY=/c/Users/Administrator/.venvs/life/Scripts/python`.
- Dependencies: `numpy==2.1.3`, `matplotlib==3.10.3`, `tensorflow==2.19.0`. The evolution team never imports TensorFlow.
- Spec values: rounds of 1,000 steps with 30 creatures; 8 fittest kept unchanged, 22 mutated children (rate 0.1, scale
  0.2, size × (1 ± 0.5) within 0.5–2.0); fitness = food eaten in the round; learner 12 → 32 (tanh) → choice of
  move/eat/attack + normal move with learned spread + critic; reward = energy change per step; PPO every 64 steps,
  γ 0.99, λ 0.95, clip 0.2, 4 passes, value weight 0.5, entropy weight 0.01, learning rate 3e-4 (Both: inherited,
  × e^N(0, 0.2)); exam every 2,000 steps with learning paused.
- Same seed, same machine, same run: TensorFlow seeded and with op determinism on.
- Zero delay; at most 2 runs at once; every task ends with a commit on `evolve-from-scratch`, judged by the commit judge.

## Review Focus

1. A creature that dies in the middle of a 64-step rollout: its last step is marked done and it never bootstraps from
   the next creature in its slot. Test: Task 4 `test_death_marks_done_and_slot_restarts`.
2. Attacks off: the learner never picks attack (masked), and its odds stay valid. Test: Task 3 `test_attack_masked`.
3. A round where everyone starves early: the next generation is still bred from whoever ate most (ties by rank).
   Test: Task 2 `test_breeds_when_everyone_ate_nothing`.
4. A child must carry the parent's inborn weights, never the learned ones. Test: Task 4 `test_child_inherits_inborn`.
5. An exam must not train the brains it tests. Test: Task 4 `test_exam_leaves_learners_unchanged`.

---

### Task 1: Commit the measured level values

**Files:** Modify: `src/levels.py` (level 1 max food 40, level 2 100; already edited, with its measurements in a
comment), `tests/test_levels.py` (`test_level_one_matches_spec` expects `max_food == 40`).

- [ ] **Step 1:** Run `$PY -m unittest tests.test_levels` · Expected: FAIL in `test_level_one_matches_spec` (200 ≠ 40).
- [ ] **Step 2:** Update the test's expected value to 40 · Run the same · Expected: `OK`.
- [ ] **Step 3:** Commit the 2 files as `tune: level food counts, measured against random walkers`; `git show --stat
  HEAD` lists exactly those 2 files.

### Task 2: Rounds and the Evolution team (no TensorFlow)

**Files:**
- Create: `src/rounds.py`, `tests/test_rounds.py`
- Modify: `src/world.py` (no splitting, no old age: `_deaths_and_births` becomes deaths only), `src/creature.py` (remove
  `split`, `should_split`, `MUTATION_*`, `SIZE_MUTATION`), `src/levels.py` (remove `MAX_AGE`), `src/scoreboard.py`
  (`max_generation` → `round`; `deaths_age` removed), `src/evolve.py` (`--team evolution`; rounds; continuous mode with
  newborns replacing the dead for baselines; `restart_population` removed), `src/check_proof.py` (new column names),
  their tests (`tests/test_world_rules.py`, `tests/test_scoreboard.py`, `tests/test_evolve.py`,
  `tests/test_check_proof.py`)

**Interfaces:**
- `Member(genome, size: float, lr: float | None)` (frozen dataclass; `genome` is a `Brain` or, from Task 3, a
  `NetWeights`; both have `mutate(np_rng, rate, scale)`).
- `next_generation(scored: list[tuple[Member, float]], rng: random.Random, np_rng: np.random.Generator, n: int = 30,
  keep: int = 8) -> list[Member]`: the `keep` fittest unchanged (stable on ties), then `n - keep` mutated children of
  them in rank order, round-robin; a child's `lr` is the parent's × e^N(0, 0.2) when not `None`.
- `parse_args`: `--team evolution|learning|both` (default `evolution`) beside `--policy random|rules` for baselines;
  default out `runs/<seed>-<team>` or `runs/<seed>-<policy>-L<level>`.
- `run(args, levels=LEVELS) -> Path` as before; CSV column `round` (round number for round teams, empty otherwise).

- [ ] **Step 1: Write the failing tests.** `tests/test_rounds.py`: `test_keeps_eight_fittest_unchanged`,
  `test_children_are_mutated_copies_in_rank_order`, `test_sizes_stay_in_range`, `test_lr_inherited_and_mutated`,
  `test_breeds_when_everyone_ate_nothing`. In `tests/test_world_rules.py`: replace the split, cap and old-age tests with
  `test_full_energy_does_not_split` and `test_no_old_age`. In `tests/test_evolve.py`: `test_evolution_team_writes_rounds`
  (seed 1, 2,000 steps → rows with `round` 1 and 2), `test_same_seed_same_csv` (team evolution), `test_baseline_replaces_dead`
  (random policy, a starving level: population stays 30). Update `tests/test_scoreboard.py` and
  `tests/test_check_proof.py` to the new keys.
- [ ] **Step 2: Run them to see them fail** · `$PY -m unittest discover tests` · Expected: `ModuleNotFoundError: No
  module named 'src.rounds'` and failures naming `round` / `split`.
- [ ] **Step 3: Implement** the files above.
- [ ] **Step 4: Run them to see them pass** · `$PY -m unittest discover tests` · Expected: `OK`. Then
  `grep -rn "should_split\|MAX_AGE\|deaths_age\|restart_population" src tests` · Expected: no output; and
  `grep -n "tensorflow" src/brain.py src/rounds.py src/world.py src/evolve.py` · Expected: no output.
- [ ] **Step 5: Commit** as `feat: evolution in rounds`; `git show --stat HEAD` lists exactly this task's files.
- [ ] **Step 6: First evidence for the Evolution team** · `$PY -m src.evolve --seed 1 --steps 30000 --team evolution
  --max-level 1 --out runs/1-evolution-L1` (background) · Show the owner the rows. Check: the mean exam ratio of the last
  3 exams is above the first 3. If not, stop and report the numbers with a proposal (the plan changes only with the
  owner's yes).

### Task 3: The PPO learner (TensorFlow)

**Files:** Create: `src/learner.py`, `tests/test_learner.py`. Modify: `requirements.txt` (add `tensorflow==2.19.0`).

**Interfaces:**
- `NetWeights` (frozen dataclass of NumPy arrays: hidden layer, choice head, move-mean head, move log-spread, critic
  head) with `NetWeights.random(np_rng)`, `mutate(np_rng, rate, scale) -> NetWeights`, `save(path)`.
- `class Learner(slots: int, seed: int)`:
  - `install(slot: int, weights: NetWeights, lr: float) -> None`: puts a network in a slot, keeps `weights` as the slot's
    inborn weights, clears the slot's optimizer state and unfinished experience.
  - `inborn(slot) -> NetWeights`; `current(slot) -> NetWeights` (the learned weights).
  - `act(slots: list[int], senses: np.ndarray, attacks_on: bool, learning: bool = True) -> list[tuple[int, tuple[float,
    float]]]`: one TensorFlow call for all; per slot the choice (0 move, 1 eat, 2 attack; attack masked when
    `attacks_on` is false) and a move vector in −1..1 per axis (the caller scales by speed). With `learning` it keeps the
    step's odds and critic value for training.
  - `record(slot: int, reward: float, done: bool) -> None`; `update() -> int`: one PPO update for every slot that has 64
    recorded steps (all in one batched call), returning how many slots it updated.
  - `gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95) -> (advantages, returns)` as a module function.

- [ ] **Step 1: Write the failing tests** in `tests/test_learner.py`: `test_act_shapes_and_ranges`,
  `test_attack_masked` (attacks off: 1,000 choices, none is 2), `test_gae_matches_hand_example` (3 steps, γ 0.5,
  λ 1: advantages equal the hand-computed values), `test_update_changes_only_full_slots`, `test_install_resets_slot`,
  `test_same_seed_same_actions`, `test_learns_toy_task` (a stand-in world: random food direction in the senses, reward =
  cosine between the move and the food direction; after 300 updates of 8 slots the mean reward over 640 steps is ≥ 0.8;
  before learning it is about 0).
- [ ] **Step 2: Run them to see them fail** · `$PY -m unittest tests.test_learner -v` · Expected:
  `ModuleNotFoundError: No module named 'src.learner'`.
- [ ] **Step 3: Implement `src/learner.py`** (PPO as the spec says; weights of all slots stacked so a step is one call).
- [ ] **Step 4: Run them to see them pass** · Expected: `Ran 7 tests … OK`.
- [ ] **Step 5: Measure** · `$PY -m timeit` of `act` for 30 slots and of `update` for 30 full slots · Expected: times
  kept for the report.
- [ ] **Step 6: Commit** as `feat: PPO learner`; `git show --stat HEAD` lists exactly this task's files.

### Task 4: Learning and Both teams in the world

**Files:** Modify: `src/creature.py` (policy `"learner"`, a `slot`), `src/policies.py` (`decide_all(world, creatures,
senses) -> list[Action]`: learner creatures in one `act` call, others as before), `src/world.py` (optional `learner`;
`step` senses everyone, then `decide_all`, then after actions and deaths records each learner creature's energy change
and death, then `learner.update()`), `src/exam.py` (exam of learner networks with learning paused), `src/evolve.py`
(`--team learning`: continuous, the dead replaced by fresh random networks; `--team both`: rounds whose members carry
`NetWeights` and `lr`, children from the inborn weights), tests (`tests/test_policies.py`, `tests/test_world_rules.py`,
`tests/test_exam.py`, `tests/test_evolve.py`).

- [ ] **Step 1: Write the failing tests:** `test_learner_creatures_act_and_record` (one step: each learner slot holds one
  recorded reward equal to its energy change), `test_death_marks_done_and_slot_restarts`, `test_exam_leaves_learners_unchanged`,
  `test_child_inherits_inborn` (after learning has changed `current(slot)`, a child's genome is a mutation of
  `inborn(slot)`, not of `current(slot)`), `test_learning_team_replaces_dead`, `test_both_team_writes_rounds`,
  `test_same_seed_same_csv_learning`.
- [ ] **Step 2: Run them to see them fail** · `$PY -m unittest discover tests` · Expected: failures naming `learner` /
  `decide_all`.
- [ ] **Step 3: Implement** the files above.
- [ ] **Step 4: Run them to see them pass** · `$PY -m unittest discover tests` · Expected: `OK`.
- [ ] **Step 5: Measure** · `time $PY -m src.evolve --seed 1 --steps 10000 --team both --out runs/t4-speed` · Expected:
  the time kept. If 10,000 steps take over 6 minutes (100,000 over an hour), stop and report it with a proposal.
- [ ] **Step 6: Commit** as `feat: learning and both teams`; `git show --stat HEAD` lists exactly this task's files.

### Task 5: The comparison table

**Files:** Create: `src/compare.py`, `tests/test_compare.py`.

**Interfaces:** `compare(runs: dict[str, list[Path]]) -> list[dict]` (one row per team: mean over its seeds of the exam
ratio of the last 5 exams, toward-food over the last 10 rows, flee over rows at level ≥ 4, highest level, minutes per
run from `runs/<…>/time.txt` written by the runner) and a CLI `python -m src.compare --team NAME CSV... [--team …]`
printing the table.

- [ ] **Step 1: Write the failing tests:** `test_one_row_per_team`, `test_means_over_seeds`, `test_missing_values_show_dash`.
- [ ] **Step 2: Run them to see them fail** · `$PY -m unittest tests.test_compare -v` · Expected: `ModuleNotFoundError`.
- [ ] **Step 3: Implement** `src/compare.py`, and the runner writes `time.txt` (seconds) beside its CSV.
- [ ] **Step 4: Run them, then the suite** · `$PY -m unittest discover tests` · Expected: `OK`.
- [ ] **Step 5: Commit** as `feat: comparison table`; `git show --stat HEAD` lists exactly this task's files.

### Task 6: First look at level 1

- [ ] **Step 1:** `$PY -m src.evolve --seed 1 --steps 30000 --team <t> --max-level 1 --out runs/1-<t>-L1` for `learning`
  and `both` (background, two at a time; the evolution team's ran in Task 2).
- [ ] **Step 2:** `$PY -m src.compare --team evolution runs/1-evolution-L1/scoreboard.csv --team learning
  runs/1-learning-L1/scoreboard.csv --team both runs/1-both-L1/scoreboard.csv` · Show the owner the table.

### Task 7: The full comparison

- [ ] **Step 1:** seeds 1–5 × teams evolution, learning, both: `$PY -m src.evolve --seed <n> --steps 100000 --team <t>`
  (background, two at a time). If Task 4's measured time puts the whole set over 12 hours, propose fewer seeds to the
  owner first.
- [ ] **Step 2:** the rules yardstick, levels 1–5: `$PY -m src.evolve --seed 1 --steps 20000 --policy rules --level <n>`.
- [ ] **Step 3:** `$PY -m src.compare --team evolution runs/{1..5}-evolution/scoreboard.csv --team learning
  runs/{1..5}-learning/scoreboard.csv --team both runs/{1..5}-both/scoreboard.csv` and, per team,
  `$PY -m src.check_proof --brain runs/{1..5}-<t>/scoreboard.csv --rules runs/1-rules-L{1..5}/scoreboard.csv`.
- [ ] **Step 4:** Show the owner the table and each team's proof lines.

---

## Must be completed to mark this as done

1. **Every test passes:** `$PY -m unittest discover tests` → `OK`, with the count and no failures or errors.
2. **The evolution team has no TensorFlow:** `grep -n "tensorflow" src/brain.py src/rounds.py src/world.py
   src/evolve.py src/policies.py` → no output (TensorFlow only in `src/learner.py`, imported only by learner runs).
3. **PPO learns:** `test_learns_toy_task` passes (mean reward ≥ 0.8 after training, about 0 before).
4. **The full comparison ran:** the Task 7 table and the three teams' `check_proof` lines are shown to the owner, with the
   seeds and steps actually run. A team that fails its proof criteria is reported as failing, with its numbers.
5. **The owner saw each step's numbers:** Task 2 Step 6, Task 6 Step 2, Task 7 Step 4, in the conversation.
6. **Committed:** `git log --oneline` on `evolve-from-scratch` shows this plan's commit and one commit per Task 1–5.
