# Every Cost a Knob Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every cost is an Auto-turned, draggable knob with 0 as its easy end; a new world starts each cost at a tenth
of today's value.

**Architecture:** The ten costs move from `src/rules.py` into `Settings` (`src/settings.py`) and `KNOBS`
(`src/dial.py`); the creature reads them from `world.settings`. The dial gets a first value for knobs at 0. Saves,
scoreboards and the exam fill the new fields with today's values.

**Tech Stack:** Python 3.12.

**Spec:** `docs/superpowers/specs/2026-11-26-every-cost-a-knob-design.md`

## Global Constraints

| Knob | Today (old worlds, exam) | New world starts | Easy end | Hard end |
|------|------|------|------|------|
| move_cost | 0.5 | 0.05 | 0 | 3.0 |
| step_cost | 0.05 | 0.005 | 0 | 0.5 |
| sprint_power | 2.0 | 0.2 | 0 | 4.0 |
| eat_miss | 0.2 | 0.02 | 0 | 2.0 |
| attack_cost | 1.0 | 0.1 | 0 | 5.0 |
| attack_damage | 5.0 | 0.5 | 0 | 25.0 |
| bump_cost | 0.5 | 0.05 | 0 | 3.0 |
| vision_cost | 0.05 / (π·9) | a tenth | 0 | 10 × today |
| hearing_cost | 0.05 / (π·9) | a tenth | 0 | 10 × today |
| neuron_cost | 0.0005 | 0.00005 | 0 | 0.005 |
| stealth_cost | 0.3 | 0.03 | 0 | 3.0 |
| armour_cost | 0.3 | 0.03 | 0 | 3.0 |

Every Auto nudge, for every knob (food and lifespan too), moves it by max(5 % of its value, 1 % of its range), held
within its ends. The difficulty counts each knob's nudges from START by that rule (finite at 0). Tests stay under 300 s;
never a full life simulation as a test.

## Review Focus

- A knob at 0: the difficulty number stays finite (no log of 0), and Auto can lift it.
- An old save (no new fields) and an old scoreboard (fewer columns) resume with today's costs.
- The owner drags a cost to 0 in Manual, then switches to Auto: Auto goes on from there.
- The exam: its creatures pay today's costs, whatever the living world's knobs say.
- A rule named in the Rules page's next-world changes that is now a knob (an old next_world.json): ignored, not an error.

---

### Task 1: The costs are settings and knobs

**Files:** `src/settings.py` (ten new fields; START at a tenth; `TODAY_COSTS` with today's values), `src/dial.py`
(KNOBS with the ends above; a nudge moves max(5 % of the value, 1 % of the range); `difficulty` stays finite at 0), `src/rules.py` (the ten rules go), `src/creature.py` (reads `world.settings`), `src/exam.py`
(today's costs), `src/viewer.py` (`KNOB_LABELS` for the ten), `src/viewer.html` (a Costs group of sliders);
tests `tests/test_dial.py`, `tests/test_creature.py`, `tests/test_exam.py`, and every test that named a moved rule.

- [ ] Step 1: tests first: `test_every_cost_is_a_knob` (the twelve names, easy end 0, hard ends as above, START a tenth
  of today), `test_auto_lifts_a_cost_from_zero` (a knob at 0, one harder nudge → 1 % of its range; difficulty finite),
  `test_a_nudge_is_at_least_a_hundredth_of_the_range` (a small value moves by 1 % of the range, a large one by 5 %),
  `test_a_sprint_costs_what_the_knob_says` (a creature's sprint uses `world.settings.sprint_power`),
  `test_the_exam_pays_todays_costs`. Run → FAIL.
- [ ] Step 2: the changes. Run `python -m unittest tests.test_dial tests.test_creature tests.test_exam tests.test_rules
  tests.test_viewer tests.test_control` → OK (tests that named a moved rule read it from settings).
- [ ] Step 3: commit "Every cost is a knob Auto turns, from 0; a new world starts at a tenth of today's costs".

### Task 2: Old worlds resume as they were

**Files:** `src/dial.py` (`settings_from_state` fills missing fields from `TODAY_COSTS`), `src/scoreboard.py` (the old
header without the new columns is read; rows written after it have them), `src/viewer.py` (a next-world change naming a
moved rule is dropped); tests `tests/test_saves.py`, `tests/test_scoreboard.py`, `tests/test_viewer.py`.

- [ ] Step 1: tests first: `test_an_old_save_keeps_its_costs`, `test_an_old_scoreboard_goes_on`,
  `test_an_old_next_world_rule_that_is_now_a_knob_is_dropped`. Run → FAIL.
- [ ] Step 2: the changes. Run those modules → OK.
- [ ] Step 3: `python scripts/run_tests.py` OK under 300 s; commit "Old worlds resume with today's costs".

### Task 3: Watch it

- [ ] Step 1: restart the 8766 room the safe way (the owner's world saved and resumed after, as Update code does);
  start a new planet; open the live graphs for the owner.
- [ ] Step 2: report hunger deaths in its first 1,000 steps against today's (~320 of 600) and the first
  `toward_food` numbers; push to `evolve-from-scratch`.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | Every cost a knob, 0 easy end, a tenth at start | `python -m unittest tests.test_dial` | `OK` |
| 2 | Creatures and the exam pay the right costs | `python -m unittest tests.test_creature tests.test_exam` | `OK` |
| 3 | Old saves and scoreboards resume | `python -m unittest tests.test_saves tests.test_scoreboard` | `OK` |
| 4 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 5 | New planet survives its start | the run's scoreboard | hunger deaths in 1,000 steps reported, far below ~320 |
| 6 | Pushed | `git push origin HEAD:evolve-from-scratch` | pushed |

Anything not done or not proven is reported as such, first.
