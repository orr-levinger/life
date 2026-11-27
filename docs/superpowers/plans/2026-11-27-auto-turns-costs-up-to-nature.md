# Auto Turns the Costs up to Nature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto turns every cost between 0 and its natural value; a new world starts at a tenth; the owner can set any
cost and Auto goes on from there.

**Spec:** `docs/superpowers/specs/2026-11-27-auto-turns-costs-up-to-nature-design.md`

## Global Constraints

- Costs stay the world's rules (Rules page, card sliders); range 0 to nature, start a tenth of nature.
- Auto's knobs: food, lifespan and the twelve costs; a cost knob's value is its rule (`world.rules`), nudged by the Dial
  in place; Auto's take-backs restore the rules too.
- Growing up (`grown`, `grow_up_generations`) goes; a save's `grown` is ignored.
- Tests under 300 s; never a full life simulation as a test.

### Task 1: Costs are Auto's knobs again, up to nature

**Files:** Modify `src/rules.py` (cost rules: start nature ÷ 10, high nature; `grow_up_generations` goes),
`src/dial.py` (cost knobs read and write the rules; `KNOBS` the seventeen; difficulty counts them; take-backs keep the
rules), `src/world.py` (`cost` is the rule; `grown` and `grow_up` go), `src/evolve.py` (no `grown`), `src/saves.py`
(no `grown`), `src/viewer.html` (the card's cost sliders: Manual or Auto, as today); Test `tests/test_costs.py`,
`tests/test_dial.py`, `tests/test_saves.py`.

- [x] Step 1: tests first: `test_a_cost_goes_from_0_to_nature_starting_at_a_tenth`,
  `test_auto_raises_a_cost_up_to_nature_and_no_further`, `test_auto_lowers_a_cost_to_zero`,
  `test_auto_goes_on_from_the_owners_cost`, `test_difficulty_counts_cost_nudges`,
  `test_an_old_saves_cost_above_nature_comes_down` (the growing-up tests go). Run → FAIL.
- [x] Step 2: the changes. Run `python -m unittest tests.test_costs tests.test_dial tests.test_saves tests.test_exam
  tests.test_viewer tests.test_evolve` → OK.
- [x] Step 3: `python scripts/run_tests.py` → OK under 300 s; commit "Auto turns the costs again, never above nature";
  push to `evolve-from-scratch`.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | Costs 0 to nature, Auto turns them, owner's value kept | `python -m unittest tests.test_costs tests.test_dial` | `OK` |
| 2 | Old saves and the exam | `python -m unittest tests.test_saves tests.test_exam` | `OK` |
| 3 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 4 | Pushed | `git push origin HEAD:evolve-from-scratch` | pushed |

Anything not done or not proven is reported as such, first.
