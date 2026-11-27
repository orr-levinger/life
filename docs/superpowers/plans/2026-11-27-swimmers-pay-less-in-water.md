# Swimmers Pay Less in Water Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A swimmer's moving and living cost less in water and more on land, by its water gene.

**Spec:** `docs/superpowers/specs/2026-11-27-swimmers-pay-less-in-water-design.md`

## Global Constraints

- New rules: `swim_saving` 0.5 (0-1), `land_cost` 0.5 (0-2), group "body".
- In water × (1 − water × swim_saving); on land × (1 + water × land_cost). A move priced where it starts; a step alive
  where it is after its action.
- Tests under 300 s; never a full life simulation as a test.

### Task 1: The rule

**Files:** Modify `src/rules.py` (the two rules, their texts; the water gene's text says it), `src/creature.py` (a
`_home_factor(world)`; `_move` and the living cost use it); Test `tests/test_water.py`, `tests/test_rules.py`.

- [x] Step 1: tests first: `test_a_swimmer_pays_less_in_water_and_more_on_land` (pond world, costs of everything else
  0 except move_cost and step_cost: water 1 in the pond: a move and a step alive × 0.5; on land × 1.5; water 0.5:
  × 0.75 and × 1.25; water 0 on land × 1); `test_the_swimming_cost_rules` (start, range, text). Run
  `python -m unittest tests.test_water tests.test_rules` → FAIL.
- [x] Step 2: the changes. Run `python -m unittest tests.test_water tests.test_rules tests.test_creature tests.test_exam
  tests.test_viewer` → OK.
- [x] Step 3: `python scripts/run_tests.py` → OK under 300 s; commit "Swimmers pay less to move and live in water,
  more on land"; push to `evolve-from-scratch`.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | The rule, in water and on land, by the gene | `python -m unittest tests.test_water` | `OK` |
| 2 | The rules on the Rules page with their texts | `python -m unittest tests.test_rules` | `OK` |
| 3 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 4 | Pushed | `git push origin HEAD:evolve-from-scratch` | pushed |

Anything not done or not proven is reported as such, first.
