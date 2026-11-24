# The Rules Page: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a Rules page where the owner sees and changes every cost, value and gene of the running world, applied at once;
the rules that were missing (a 120° cone, hearing without size, size costing energy, the brain's size in the genes with a
cost per neuron); one world at a time on Live.

**Architecture:** one new module, `src/rules.py`, holds every rule once (name, table, words, start, range); the world
holds its own values (`World.rules`) and every module reads its numbers from there instead of constants. The page reads
the rules from a new `GET /rules` and changes them through the control file, as the knobs do; saves keep them.

**Tech Stack:** Python 3.12, NumPy, TensorFlow 2.19 (learners), stdlib HTTP server, one HTML page, unittest.

**Spec:** `docs/superpowers/specs/2026-11-24-rules-page-design.md` (the owner, 2026-11-24, on the mockups: "fix it, add
the page").

## Global Constraints

- Commands run from `C:\Users\Administrator\life` with `PY=/c/Users/Administrator/.venvs/life/Scripts/python`.
- One place for each rule: after Task R1 no number of the world is a constant outside `src/rules.py` (`grep` proves it).
- The owner's world is never stopped by Claude; only the owner presses Update code (the owner: "let me just run it
  myself").
- A creature's genes never change in its life; new gene limits apply to babies born afterwards.
- One commit per task, after its suite run passes.

## Review Focus

- Every value the code uses comes from `World.rules`; nothing reads a removed constant.
- A rule change reaches the running world within 25 steps and is kept in its saves.
- The owner's world converts without losing a creature (format 5 → 6).

### Task R0: Commit what the owner's world runs now

The owner's world has run Tasks 4u, 4v and 4w since step 158,000 (the owner stopped the fair check: "stop it let me just
run it myself").

- [x] **Step 1:** `$PY -m unittest discover tests` → `OK`, with the count.
- [x] **Step 2:** commit as `feat: learning from others, senses and rewards in the genes`, with both plans' steps marked
  (the fair check marked stopped by the owner); `git show --stat HEAD` lists the files of Tasks 4u, 4v and 4w only.

### Task R1: Every rule in one place

**Files:** Create `src/rules.py`, `tests/test_rules.py`. Modify `src/settings.py` (its constants move into rules),
`src/creature.py`, `src/world.py`, `src/senses.py`, `src/policies.py`, `src/dial.py`, `src/saves.py` (format 6),
their tests.

**Interfaces:** `Rule(name, table: "costs"|"values"|"genes", words, start, low, high)`; `RULES: tuple[Rule, ...]`;
`starting_rules() -> dict[str, float]`; `check(rules, name, value) -> str | None` (a plain message when out of range, or a
gene's lowest above its highest). Auto's nudge size stays a constant of `src/dial.py`: changing it live would rescale
the difficulty number (put to the owner).
`World.rules: dict[str, float]`. Genes are rules named `gene.<gene>.low|high|new_low|new_high|start|change`.

- [x] **Step 1: Write the failing tests:** `test_every_rule_starts_inside_its_range`,
  `test_a_value_outside_its_range_is_refused`, `test_the_world_reads_its_rules` (changing `eat_miss` in `World.rules`
  changes what eating at nothing costs), `test_rules_are_kept_in_a_save`, `test_a_format_5_save_gets_the_starting_rules`.
- [x] **Step 2:** `$PY -m unittest tests.test_rules tests.test_saves` → the five fail (no `src.rules`).
- [x] **Step 3: Implement.**
- [x] **Step 4:** `$PY -m unittest discover tests` → `OK`; `grep -rn "EAT_MISS_COST\|BIRTH_AT\|BABY_SHARE\|SENSE_COST\|HUNTER_SHARE" src` → only `src/rules.py`.
- [x] **Step 5: Commit** as `feat: every rule in one place`.

### Task R2: The Rules page

**Files:** Modify `src/control.py` (`Control.rules`), `src/evolve.py` (applies them every 25 steps), `src/viewer.py`
(`GET /rules`, `POST /control {rules}` checked), `src/viewer.html` (the Rules tab: Costs, Values, Genes; number boxes,
range, start, Reset; the "Auto" mark on being alive and moving), `tests/test_control.py`, `tests/test_viewer.py`,
`tests/test_evolve.py`.

- [x] **Step 1: Write the failing tests:** `test_rules_endpoint_lists_every_rule`, `test_a_rule_change_out_of_range_is_refused`,
  `test_a_rule_change_reaches_the_running_world`, `test_page_has_costs_values_and_genes`,
  `test_page_has_no_switches_and_no_read_only_rows`.
- [x] **Step 2:** run them → they fail.
- [x] **Step 3: Implement.**
- [x] **Step 4:** `$PY -m unittest discover tests` → `OK`.
- [x] **Step 5: Commit** as `feat: the Rules page`.

### Task R3: The rules that were missing

**Files:** Modify `src/rules.py` (cone 30–120°, reach 1–10, separate vision and hearing costs, the sprint's power, the
cost of size), `src/creature.py` (being alive and moving × size; the sprint's power), `src/senses.py` (hearing: 4 inputs,
no size; 20 in all), `src/brain.py` and `src/learner.py` (20 inputs), `src/saves.py` (the owner's world: cones 120° reach
5, the heard-size input removed), their tests.

- [x] **Step 1: Write the failing tests:** `test_a_cone_is_at_most_120`, `test_hearing_gives_no_size`,
  `test_a_bigger_body_pays_more_to_live_and_move`, `test_vision_and_hearing_are_priced_apart`,
  `test_the_sprint_power_is_a_rule`, `test_the_owners_world_converts` (cones 120°/5, 20 inputs, the other weights kept).
- [x] **Step 2:** run them → they fail.
- [x] **Step 3: Implement.**
- [x] **Step 4:** `$PY -m unittest discover tests` → `OK`.
- [x] **Step 5: Commit** as `feat: the rules that were missing`.

### Task R3b: A baby's genes change by at most 5 % (added 2026-11-24; the owner: "just change the max mutation to 5%")

Spec: section 2b. **Files:** Modify `src/rules.py` (every gene's `change` starts at 0.05), `tests/test_rules.py`.

- [x] **Step 1: Write the failing tests:** `test_a_baby_gene_changes_by_at_most_5_percent` (every `gene.*.change` starts
  at 0.05), `test_a_baby_changes_one_gene` (each baby differs from its parent in at most one gene; over many babies
  every gene is picked sometime). **Files** also: `src/world.py` (`_birth` picks the one gene), `tests/test_world_rules.py`.
- [x] **Step 2:** run them → they fail (0.2; every gene changes).
- [x] **Step 3: Implement.**
- [x] **Step 4:** `$PY -m unittest discover tests` → `OK`.
- [x] **Step 5:** the owner's world: `POST /control` with every `gene.*.change` at 0.05; its `now.json` shows them.
- [x] **Step 6: Commit** as `feat: a baby's genes change by at most 5 %`.

### Task R4: The brain's size evolves, each neuron costs

As the approved Task 5 of `2026-11-23-living-world.md` (its interfaces and tests), with its numbers as rules
(`neuron_cost` 0.0005, `neuron_change` 0.05, `layer_change` 0.01, `gene.neurons`, `gene.layers`).

- [ ] **Step 1: Write the failing tests:** Task 5's (`test_new_neuron_is_silent_at_birth`, `test_new_layer_is_silent_at_birth`,
  `test_removed_neuron_changes_behavior`, `test_size_genes_stay_in_bounds`, `test_bigger_brain_costs_more_energy`,
  `test_learns_toy_task`, `test_scoreboard_reports_brain_size`).
- [ ] **Step 2:** run them → they fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** `$PY -m unittest discover tests` → `OK`.
- [ ] **Step 5: Commit** as `feat: the brain's size evolves`.

### Task R5: One world at a time

**Files:** Modify `src/viewer.py` (`MAX_RUNNING = 1`), `src/viewer.html` (Live: the one world; charts of that world;
lines only for the owner's changes; the cone's edge in Jump in), `tests/test_viewer.py`.

- [ ] **Step 1: Write the failing tests:** `test_a_second_world_is_refused`, `test_only_the_owners_changes_draw_lines`,
  `test_page_draws_the_cone_edge`.
- [ ] **Step 2:** run them → they fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** `$PY -m unittest discover tests` → `OK`.
- [ ] **Step 5: Commit** as `feat: one world at a time`.

### Task R6: The owner's world, and a look

- [ ] **Step 1:** restart the page server (`src.viewer --port 8765`) on the new code; the owner's world untouched
  (`/data` still lists it).
- [ ] **Step 2:** the owner presses Update code. Then, in the app's browser: the Rules page with Costs, Values and Genes;
  a change (eating at nothing 0.2 → 0.3) shown in the world within 25 steps; Live with the one world; the cone's edge;
  `read_console_messages` → no errors; screenshots to the owner.
- [ ] **Step 3:** the world over the next 5,000 steps, live: alive and the gene averages to the owner. Below 40 alive at
  the end is reported first, as a fail.

## Must be completed to mark this as done

1. **Every test passes:** `$PY -m unittest discover tests` → `OK`, with the count.
2. **One place for each rule:** `grep -rn "EAT_MISS_COST\|BIRTH_AT\|BABY_SHARE\|SENSE_COST\|HUNTER_SHARE\|SPEED =\|SPRINT =" src`
   → only `src/rules.py`.
3. **The page:** Task R6 Step 2's screenshots shown to the owner, no console error.
4. **The owner's world:** converted with every creature kept (Task R3's test) and Task R6 Step 3's numbers shown.
5. **Committed:** `git log --oneline` shows the six commits of Tasks R0–R5.
