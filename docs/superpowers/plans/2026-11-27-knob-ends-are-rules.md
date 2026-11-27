# Knob Ends Are Rules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every knob's easy and hard ends are rules the owner changes on the Rules page, live in the running world.

**Spec:** `docs/superpowers/specs/2026-11-27-knob-ends-are-rules-design.md`

## Global Constraints

- Rules `knob.<name>.easy` / `knob.<name>.hard`, group "knobs" (title "Knob ends"), start today's ends, allowed 0 to
  10 × the larger end today; the easy end stays on its easy side of the hard end.
- Auto, Manual and the slider use the world's ends; the difficulty number uses today's fixed ends (`KNOBS`).
- Tests under 300 s; never a full life simulation as a test.

### Task 1: The ends as rules, used by Auto and Manual

**Files:** Modify `src/rules.py` (the rules from `KNOBS`' ends, the group, their texts, `check` refuses a wrong-way end),
`src/dial.py` (`knobs_of(rules)`: `KNOBS` with the world's ends; `with_value` and the Dial's nudges use them),
`src/control.py` (`settings_from(values, base, rules)` checks against the world's ends), `src/viewer.py` (passes the
world's rules to `settings_from`); Test `tests/test_dial.py`, `tests/test_control.py`, `tests/test_rules.py`.

- [x] Step 1: tests first: `test_the_knob_ends_are_rules`, `test_a_wrong_way_knob_end_is_refused`,
  `test_auto_eases_up_to_the_worlds_end` (fruit easy 5: easier nudges pass 1 and stop at 5),
  `test_the_least_nudge_uses_the_worlds_range`, `test_difficulty_does_not_move_with_the_ends`,
  `test_manual_takes_values_within_the_worlds_ends` (3 taken, 6 refused). Run `python -m unittest tests.test_dial
  tests.test_control tests.test_rules` → FAIL.
- [x] Step 2: the changes. Run `python -m unittest tests.test_dial tests.test_control tests.test_rules tests.test_evolve
  tests.test_saves` → OK.
- [x] Step 3: commit "A knob's ends are rules: Auto and Manual use the world's".

### Task 2: The page's sliders stretch to the world's ends

**Files:** Modify `src/viewer.html` (a card's slider ends and end labels from its world's `knob.<name>.easy/hard` in
`/now`'s rules, renewed as they change; the Rules page shows the "Knob ends" group); Test `tests/test_viewer.py`.

- [x] Step 1: test first: `test_a_slider_stretches_to_its_worlds_ends` (the page reads `knob.${...}.easy` and `.hard`
  from the world's rules and sets the slider's min and max). Run `python -m unittest tests.test_viewer` → FAIL.
- [x] Step 2: the change. Run `python -m unittest tests.test_viewer` → OK.
- [x] Step 3: `python scripts/run_tests.py` → OK under 300 s; commit "The page's sliders stretch to the world's knob
  ends"; push to `evolve-from-scratch`. (No browser check here: the owner checks on the Mac.)

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | Ends are rules; Auto and Manual keep within them | `python -m unittest tests.test_dial tests.test_control tests.test_rules` | `OK` |
| 2 | The slider follows the world's ends | `python -m unittest tests.test_viewer` | `OK` |
| 3 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 4 | Pushed | `git push origin HEAD:evolve-from-scratch` | pushed |

Anything not done or not proven is reported as such, first.
