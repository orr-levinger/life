# Costs Fixed by Nature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Costs are fixed rules a new world grows into; Auto turns only food and lifespan.

**Spec:** `docs/superpowers/specs/2026-11-27-costs-fixed-by-nature-design.md`

## Global Constraints

- The twelve cost rules and values as in the spec's table, each from 0 to 3 × its value (sprint_power to 4);
  `grow_up_generations` 10 (0-100). The owner turns them on the Rules page and on the card's sliders, in Auto and
  Manual; Auto never does.
- Growing-up share 0.1 + 0.9 × min(generation, G) ÷ G by the deepest living generation, never lower than before.
- Old saves keep their costs (share 1); the exam keeps today's costs.
- Checks on this box use a half-size world (radius 18.5, 100 creatures), the browser closed after. Tests under 300 s.

### Task 1: Costs are rules; Auto turns food and lifespan

**Files:** Modify `src/rules.py` (the twelve cost rules in Costs, `grow_up_generations`; the cost knob ends go),
`src/dial.py` (`KNOBS` the five), `src/settings.py` (the cost fields go), `src/creature.py` and `src/world.py` (a
world's cost: its rule × its grown-up share), `src/viewer.py` and `src/viewer.html` (no cost sliders),
`src/scoreboard.py` (the cost columns keep the costs paid now), `src/viewer.html` (the card's Costs sliders post rule
changes in either mode, ends from the rule's range); Test `tests/test_costs.py` (new), `tests/test_dial.py`,
`tests/test_rules.py`, `tests/test_viewer.py`, and the tests that set a cost through settings (they set the rule).

- [ ] Step 1: tests first: `test_the_costs_are_rules_with_these_values`, `test_auto_turns_only_food_and_lifespan`,
  `test_the_card_turns_the_costs_as_rules` (its Costs sliders post `rules`, in Auto too),
  `test_a_cost_change_reaches_the_running_world`. Run → FAIL.
- [ ] Step 2: the changes. Run `python -m unittest tests.test_costs tests.test_dial tests.test_rules tests.test_viewer
  tests.test_creature tests.test_water` → OK.
- [ ] Step 3: commit "Costs are fixed rules; Auto turns only food and lifespan".

### Task 2: Growing up, old saves, the exam

**Files:** Modify `src/world.py` (the grown-up share, kept), `src/saves.py` (keeps the share; a save from before: cost
rules from its saved settings, share 1), `src/exam.py` (today's costs); Test `tests/test_costs.py`,
`tests/test_saves.py`, `tests/test_exam.py`.

- [ ] Step 1: tests first: `test_a_new_world_grows_into_its_costs` (0.1, 0.55, 1; never lower),
  `test_an_old_save_keeps_its_costs`, `test_the_exam_pays_todays_costs`. Run → FAIL.
- [ ] Step 2: the changes. Run `python -m unittest tests.test_costs tests.test_saves tests.test_exam tests.test_evolve`
  → OK.
- [ ] Step 3: `python scripts/run_tests.py` → OK under 300 s; commit "A new world grows into its costs; old saves keep
  theirs".

### Task 3: Checked in the world

- [ ] Step 1: a half-size world run a few thousand steps (a scratch script reading each step's spending by kind:
  living, brain, senses, moving, the rest); report the shares and steer-to-food against random lifetimes.
- [ ] Step 2: a share far off its target: say which ratio would change and why, and wait for the owner; else push to
  `evolve-from-scratch`.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | Costs are fixed rules; Auto turns five knobs | `python -m unittest tests.test_costs tests.test_dial` | `OK` |
| 2 | Growing up; old saves; the exam | `python -m unittest tests.test_costs tests.test_saves tests.test_exam` | `OK` |
| 3 | Spending as in nature | the half-size world's shares | about half living and brain, about a third moving (numbers reported) |
| 4 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 5 | Pushed | `git push origin HEAD:evolve-from-scratch` | pushed |

Anything not done or not proven is reported as such, first.
