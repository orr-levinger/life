# Back in Time Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Click a point on any Live chart, press "Go back to step N": the world running now saves and stops, and a world
goes on from its save at N with today's code.

**Architecture:** The page turns a chart click into a step and the save at or before it (`/saves`); `/control` gets
`back_to`; the page server stops the world and, once it has ended, starts a world from that save through the same code
as `/start` with `from` (one way to start a world).

**Tech Stack:** Python 3.12 (page server, unittest), JavaScript (the page).

**Spec:** `docs/superpowers/specs/2026-11-27-back-in-time-design.md` (approved by the owner, 2026-11-27: "Yes").

## Global Constraints

- No change to the world's rules, Auto or the save loader. One way to start a world from a save.
- Tests: no full life simulation (small hand-made worlds, few steps); the whole suite under 300 s.
- The check world is half size (planet radius 18.5, 100 first creatures); the browser pane is closed after.

## Review Focus

- The old world is never left running beside the new one, and never stopped without a save to go back to.
- A save from older code goes on with today's rules (costs held at nature after Auto's first look).

### Task 1: Back in time

**Files:** Modify `src/viewer.py` (`Controls.start`: its launch from a save moved into `_start_from(save, mode)`, used
by `/start` and by `back_to`; `Controls.control`: `back_to` checked with `_save`, then the stop, then
`_go_back(folder, save, mode)` in a thread, like `_hand_over`); `src/viewer.html` (each Live chart: a click marks a
step, the save at or before it from `/saves`, the button "Go back to step N" and "Cancel"; Esc cancels); Test
`tests/test_viewer.py`, `tests/test_saves.py`.

**Interfaces:** `/control {run, back_to: <step>}` → `{..., "going_back": <step>}`; 404 when the run has no save at
that step (nothing stopped).

- [x] Step 1: tests first:
  - `test_back_to_a_save_stops_the_world_then_starts_from_it` (test_viewer): a running world with saves 2,000 and
    4,000; `/control {back_to: 2000}` → 200; the world is told to stop; once it has ended (`time.txt`), a world is
    launched with `--resume <its step-0002000.npz>`, `--forever`, a new folder, the old world's mode.
  - `test_back_to_a_missing_save_is_refused` (test_viewer): `back_to: 3000` → 404, the world not stopped, nothing
    launched.
  - `test_the_page_goes_back_from_a_chart` (test_viewer): the page sends `back_to` and has "Go back to step".
  - `test_a_save_from_older_code_goes_on` (test_saves): `tests/data/before_guts/save.npz` loaded, a cost set above
    nature (`eat_miss` 0.3), Auto's look (`dial.review`) holds it at nature (0.1); 3 steps with no error; every
    creature's four new genes between 0 and 1.
  Run `python -m unittest tests.test_viewer tests.test_saves` → FAIL (the three page and server tests: no `back_to`;
  `test_a_save_from_older_code_goes_on` passes against today's code: a value the task must keep).
- [x] Step 2: the server change. Run `python -m unittest tests.test_viewer` → the two server tests OK; the
  existing `test_resume_from_a_save_launches_a_world` still OK.
- [x] Step 3: the page change. Run `python -m unittest tests.test_viewer tests.test_saves` → OK.
- [x] Step 4: on a half-size check world (`python -m src.viewer --runs <scratch> --port 8771`, a world started with
  planet radius 18.5, 100 first creatures and `save_every` 500 so it is quick on this box, run past its saves at 500
  and 1,000 steps): click the chart near step 800, press "Go back to step 500"; the old world shows in History, the
  new one runs "from the HH:MM world at 500" with its history before it; screenshot; no console error. Stop the world and the viewer; close
  the browser pane.
- [x] Step 5: `python scripts/run_tests.py` → OK under 300 s (box busy: cut at 300 s with `test_evolve_runs` still running; run alone, OK, 19 tests in 228 s; every other file OK); commit "Back in time from any chart" (`git show
  --stat` → the files above, this plan and the spec); push to `evolve-from-scratch`.
- [ ] Step 6: after the owner's first jump on the Mac (`./update`, Ctrl+C, `./start`, then the button), Claude reads
  the new world's `report/events.jsonl` and `errors.jsonl` from the bucket: a `resume` event at the chosen step, no
  error.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | Stop, then start from the save; a missing save refused | `python -m unittest tests.test_viewer` | `OK` |
| 2 | An old save goes on with today's rules | `python -m unittest tests.test_saves` | `OK` |
| 3 | The owner can do it from a chart | the screenshot of Step 4 | the old world in History, the new one from step 500 |
| 4 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 5 | Pushed | `git ls-remote origin evolve-from-scratch` | the commit's hash |
| 6 | Works on the owner's Mac | the bucket's events and errors of the new world | a `resume` at the chosen step; no error |

Anything not done or not proven is reported as such, first.
