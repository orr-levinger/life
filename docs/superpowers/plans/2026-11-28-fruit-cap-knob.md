# The Fruit Cap Knob Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The most fruit on the planet at once is a knob ("Fruit pieces at most") beside the grass's ("Grass pieces at
most"), shown on Live, set in Manual and turned by Auto; Live shows the grass and fruit lying now apart.

**Architecture:** `max_fruit` moves from the rules (`src/rules.py`) to the settings (`Settings`, `START`) and joins the
knobs (`KNOB_ENDS`), so the dial (`src/dial.py`), the control file (`src/control.py`), the page's knobs
(`src/viewer.py` `KNOB_LABELS`) and the scoreboard (`src/scoreboard.py`) take it as they take every knob. The world reads
`self.settings.max_fruit`. A save from before is converted when loaded (`src/saves.py`, as the water steps were).

**Tech Stack:** Python 3.12 (unittest), JavaScript (the page).

**Spec:** `docs/superpowers/specs/2026-11-28-fruit-cap-knob-design.md`.

## Global Constraints

- The one world rule change is the owner's: the fruit cap a knob. The fruit rate, the grass rate and the costs stay.
- One place for each knob's ends: `KNOB_ENDS`. No `max_fruit` rule left anywhere.
- Tests: small hand-made worlds, few steps; the whole suite under 300 s. The check world is half size.

## Review Focus

- A save from before goes on with exactly its fruit cap (its rule's value), so its world does not change when loaded.
- The scoreboard's new column at the end, and the layout before it read with that column empty.

### Task 1: The fruit cap knob

**Files:** Modify `src/settings.py` (`Settings.max_fruit`; `START` `max_fruit=300`), `src/rules.py` (`KNOB_ENDS`:
`("max_food", 800.0, 20.0, "Grass pieces at most")`, `("max_fruit", 800.0, 20.0, "Fruit pieces at most")`; the
`max_fruit` rule and its words removed), `src/world.py` (`spawn_food`: `self.settings.max_fruit`), `src/exam.py` (the
exam's `Settings`: `max_fruit=0`, it has no fruit), `src/saves.py` (`_fruit_room(about)`: a save's settings take its
rule `max_fruit`, 300 when it has none, and the rule is dropped), `src/scoreboard.py` (`max_fruit` the last column;
`BEFORE_FRUIT_ROOM` the layout before it), `src/evolve.py` (`world_now`: `"grass"` and `"fruit"`, the pieces lying now;
the scoreboard row's `max_fruit`), `src/viewer.py` (`KNOB_LABELS`: `"max_food": ("Grass pieces at most", 1.0)`,
`"max_fruit": ("Fruit pieces at most", 1.0)`), `src/viewer.html` (the food tile: "grass N · fruit M"). Tests:
`tests/test_terrain.py`, `tests/test_seasons.py`, `tests/test_rules.py` (their `max_fruit` rule uses become the
setting), `tests/test_dial.py`, `tests/test_saves.py`, `tests/test_scoreboard.py`, `tests/test_evolve_runs.py` (the now
file's keys), `tests/test_viewer.py`.

- [ ] Step 1: tests first:
  - `test_fruit_has_its_own_knob` (test_terrain): the fruit world with `settings.max_fruit` 3 holds 3 fruit at most
    after 20 steps (today's `fruit_world` test, given the setting instead of the rule).
  - `test_auto_turns_the_fruit_cap` (test_dial): `knob_named("max_fruit")` exists with ends 800 and 20; one harder
    nudge on it lowers it.
  - `test_a_save_from_before_keeps_its_fruit_cap` (test_saves): `tests/data/before_guts/save.npz` loads with
    `settings.max_fruit` 300 and no `max_fruit` in its rules.
  - `test_the_scoreboard_keeps_the_fruit_cap` (test_scoreboard): `ROW_KEYS[-1] == "max_fruit"`; a scoreboard in the
    layout before reads with it empty.
  - `test_now_has_grass_and_fruit_apart` (test_evolve_runs): the now file's keys include `"grass"` and `"fruit"`.
  - `test_live_shows_both_caps` (test_viewer): the knobs answer has "Grass pieces at most" and "Fruit pieces at most";
    the page's food tile shows `grass` and `fruit`.
  Run `python -m unittest tests.test_terrain tests.test_dial tests.test_saves tests.test_scoreboard tests.test_viewer`
  and `python -m unittest tests.test_evolve_runs` → FAIL (no fruit knob yet).
- [ ] Step 2: the changes. Run the same commands → OK; `grep -rn '"max_fruit"\]' src tests` → no rule use left
  (`rules["max_fruit"]`).
- [ ] Step 3: on a half-size check world: Live shows "Grass pieces at most" and "Fruit pieces at most" and the tile
  "grass N · fruit M"; in Manual the fruit cap set to 50: the fruit lying stays at 50 or under after 500 steps; back to
  Auto. Screenshot. Stop the world and its viewer; close the browser pane.
- [ ] Step 4: `python scripts/run_tests.py` → OK under 300 s; commit "The fruit cap is a knob" (`git show --stat` →
  the files above and this plan); push to `evolve-from-scratch`.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | The fruit cap a knob, saves, scoreboard, now | Step 2's commands | `OK` |
| 2 | No rule left | `grep -rn 'rules\["max_fruit"\]' src tests` | nothing |
| 3 | The owner sees it | Step 3's screenshot | both caps and the grass · fruit tile |
| 4 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 5 | Pushed | `git ls-remote origin evolve-from-scratch` | the last commit's hash |

Anything not done or not proven is reported as such, first.
