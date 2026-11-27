# Rules Page Sliders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every number on the Rules page can be dragged as well as typed.

**Spec:** `docs/superpowers/specs/2026-11-27-rules-page-sliders-design.md`

## Global Constraints

- The page only (`src/viewer.html`). Log scale when low > 0 and high ÷ low ≥ 100; whole steps for whole-number rules.
- A drag sends on release, through the same `change` a typed number uses. Tests under 300 s.

### Task 1: Sliders beside the boxes

**Files:** Modify `src/viewer.html` (`ruleInput` and the gene rows: a slider beside each box, tied both ways;
`sliderOf(rule)`: even or log, whole or not); Test `tests/test_viewer.py`.

- [x] Step 1: test first: `test_every_rule_can_be_dragged`. Run `python -m unittest tests.test_viewer` → FAIL.
- [x] Step 2: the change. Run `python -m unittest tests.test_viewer` → OK.
- [x] Step 3: browser pane, a test server on another port, half-size next world: a screenshot of the Rules page with a
  slider dragged; closed after.
- [x] Step 4: `python scripts/run_tests.py` → OK under 300 s; commit "Sliders on the Rules page"; push.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | Every rule has a slider tied to its box | `python -m unittest tests.test_viewer` | `OK` |
| 2 | It shows and works | screenshot sent | sliders, one dragged, its box moved |
| 3 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 4 | Pushed | `git push origin HEAD:evolve-from-scratch` | pushed |

Anything not done or not proven is reported as such, first.
