# Fruit Has Its Own Room Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trees drop fruit up to their own limit, whatever the plants do.

**Spec:** `docs/superpowers/specs/2026-11-27-fruit-has-its-own-room-design.md`

## Global Constraints

- Rule `max_fruit` 300 (0-3000), group "world"; `max_food` counts plants and meat only.
- Tests under 300 s; never a full life simulation as a test.

### Task 1: Fruit's own limit

**Files:** Modify `src/rules.py` (the rule and its text), `src/world.py` (`spawn_food`: plants against `max_food`
without fruit, fruit against `max_fruit`); Test `tests/test_world_rules.py` or `tests/test_terrain.py` (the food
tests), `tests/test_rules.py`.

- [x] Step 1: tests first: `test_trees_drop_fruit_when_plants_fill_their_room`, `test_fruit_stops_at_its_own_limit`,
  `test_the_fruit_limit_rule`. Run → FAIL.
- [x] Step 2: the change. Run `python -m unittest tests.test_terrain tests.test_rules tests.test_world_rules
  tests.test_water` → OK.
- [x] Step 3: `python scripts/run_tests.py` → OK under 300 s; commit "Fruit has its own room"; push to
  `evolve-from-scratch`.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | Fruit grows whatever the plants do, up to its own limit | the food tests | `OK` |
| 2 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 3 | Pushed | `git push origin HEAD:evolve-from-scratch` | pushed |

Anything not done or not proven is reported as such, first.
