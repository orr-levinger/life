# Every Death Leaves a Corpse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A creature that dies of hunger or old age leaves a corpse, the same as one killed by a hit.

**Architecture:** The corpse moves from `Creature._attack` into `World.kill`, the one place every death goes through.

**Tech Stack:** Python 3.12, unittest.

**Spec:** `docs/superpowers/specs/2026-11-27-corpses-from-every-death-design.md`

## Global Constraints

- The corpse stays as it is: meat, `corpse_energy` × size, `corpse_steps` long, at the spot. No new rule.
- Who eats it gets bite × its diet (meat), as every corpse today; approved 2026-11-27: "Yes, but whoever eats it gets
  energy best if it's carnivore or not".
- Tests: small hand-made worlds, a step or two; the whole suite under 300 s.

## Review Focus

- One place makes a corpse (`World.kill`); a hit makes exactly one, not two.

### Task 1: The corpse in `World.kill`

**Files:** Modify `src/world.py` (`kill` makes the corpse), `src/creature.py` (`_attack` no longer makes it);
Test `tests/test_world_rules.py` (two new tests in `TestHunting`).

**Interfaces:** `World.kill(creature, cause)` unchanged in its arguments.

- [x] Step 1: tests first: `test_a_hungry_death_leaves_a_corpse` (a creature of size 1 at 0.01 energy, one step of a
  world with no food → one meat food of energy 8, 5 steps, at its spot) and `test_an_old_death_leaves_a_corpse` (past
  its lifespan with low energy → the same). Run `python -m unittest tests.test_world_rules` → the two FAIL (no food).
- [x] Step 2: the change. Run `python -m unittest tests.test_world_rules tests.test_diet tests.test_on_sphere` → OK
  (the hit test still finds exactly one corpse).
- [x] Step 3: `python scripts/run_tests.py` → OK under 300 s; commit "Every death leaves a corpse" (`git show --stat` →
  the three files, this plan and the spec); push to `evolve-from-scratch`.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | Hunger and old age leave a corpse | `python -m unittest tests.test_world_rules` | `OK` (the two new tests among them) |
| 2 | A hit still leaves exactly one | the same run: `test_kill_leaves_corpse_and_counts_attack_death` | passes |
| 3 | One place makes it | `grep -n "corpse_energy" src/*.py` | only `src/world.py` and `src/rules.py` |
| 4 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 5 | Pushed | `git rev-parse --short origin/evolve-from-scratch` | the commit's hash |

Anything not done or not proven is reported as such, first.
