# Creature Maker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A "Make" tab: a slider for every gene and the body they make; "Place it" puts that creature where the owner
clicks on the live planet, with a new brain, the smartest's or the followed creature's.

**Architecture:** The page sends `/control {run, place: {spot, genes, brain}}`; the server checks it (`check_genes`)
and writes it in the run's control file once (`seq`, as an edit); the runner hands it to `World.place`, which makes the
creature as `add_newcomer` does but at that spot with those genes and that brain, and writes the outcome in `now.json`
(which the page already reads) and an event in the report.

**Tech Stack:** Python 3.12 (world, runner, page server, unittest), JavaScript (the page, three.js).

**Spec:** `docs/superpowers/specs/2026-11-27-creature-maker-design.md` (approved by the owner, 2026-11-28: "Approve").

## Global Constraints

- No change to the world's rules, to births, to Auto, or to how the soup's newcomers are made. One body maker
  (`beasts.js`). One way a newcomer's body is set up: `add_newcomer`'s, shared by `place`.
- A placed creature is not counted in `world.newcomers` (Auto reads that count as "nearly died out").
- Tests: small hand-made worlds, few steps; the whole suite under 300 s. The check world is half size; the browser
  pane is closed after.

## Review Focus

- `place` and `add_newcomer` share one body set-up; no copied gene list.
- A copied brain is an exact copy (`learner.baby` with no gene changed); its size sets the creature's layers and
  neurons, never the sliders.

### Task 1: The world places a creature

**Files:** Modify `src/world.py` (`add_newcomer` split: its body set-up into `_set_up(creature, genes)`, used by both;
new `place(spot, genes, brain) -> Creature` raising `PlaceRefused` with a plain reason; `smartest()`; `world_now`
gains `placed`: the last outcome `{seq, id | why}`), `src/control.py` (`Control.place: Optional[Dict]`, read and
written), `src/evolve.py` (`take`: a new `place` seq → `world.place`, the outcome kept for `now.json`, an event
`placed` or `place_refused` in the report); Test `tests/test_place.py` (new).

**Interfaces:** `World.place(spot: Sequence[float], genes: Dict[str, Any], brain: Dict) -> Creature`; `brain` is
`{"kind": "new"}`, `{"kind": "smartest"}` or `{"kind": "creature", "id": int}`. `PlaceRefused(why: str)`.

- [x] Step 1: tests first (`tests/test_place.py`):
  - `test_placed_where_and_as_asked`: a placed creature's point is the given spot (normalised), every given gene
    exactly as given; generation 0; `world.newcomers` unchanged; its species "first life".
  - `test_a_new_brain_is_a_newcomers`: `{"kind": "new"}` in a learning world: a learner slot with the given layers,
    neurons and learning speed.
  - `test_a_copied_brain_is_an_exact_copy`: `{"kind": "creature", "id": n}`: the new slot's weights equal creature
    n's (`learner.current`), its layers and neurons are n's (not the given ones), its species is n's.
  - `test_the_smartest_brain`: three creatures with set earned and spent; `{"kind": "smartest"}` copies the one with
    the highest earned ÷ spent (of those that spent at least `MIN_SPENT`).
  - `test_a_place_it_cannot_live_is_refused`: a spot inside a rock, a pond for gills 0, a full room, a creature gone:
    each `PlaceRefused` with its reason; nothing added.
  - `test_the_runner_places_once`: a control with a `place` seq: placed once though read twice; `now.json`'s
    `placed` has the new id; the report has a `placed` event.
  Run `python -m unittest tests.test_place` → FAIL (no `place`).
- [x] Step 2: the changes. Run `python -m unittest tests.test_place tests.test_world_rules tests.test_guts
  tests.test_saves` → OK (the soup's newcomers made as before: `test_world_rules` and the saves' exact replays).
- [x] Step 3: commit "The world places a creature" (`git show --stat` → the files above and this plan).

### Task 2: The server and the page

**Files:** Modify `src/viewer.py` (`control`: `place` checked: a spot of three numbers, genes by `check_genes`, a brain
of the three kinds; written with a new `seq`), `src/static/beasts.js` (`genesPicture(genes, size)`: a creature's genes
by their gene names, through `genesOf` and `bodyPicture`; `speciesPicture` uses it), `src/static/planet.js`
(`groundAt(event)`: the point on the ground under the pointer, as a unit vector; `onGround` called on a click while
placing), `src/viewer.html` (the "Make" tab: sliders from `/rules`' gene limits, the body, "Start from", "Brain",
"Place it"; in Live, placing mode: each click on the ground sends `place`; the outcome from `/now`'s `placed`; Esc or
"Done" ends it); Test `tests/test_viewer.py`.

- [x] Step 1: tests first (`tests/test_viewer.py`):
  - `test_place_is_written_for_the_world`: `/control {run, place: {spot, genes, brain}}` → 200; the control file's
    `place` has those and a `seq`; a second one gets a new `seq`.
  - `test_a_bad_place_is_refused`: a gene outside its limits, a spot not of three numbers, an unknown brain kind:
    each 400 with its reason; the control file unchanged.
  - `test_the_page_makes_and_places`: the page has the "Make" tab, `Planet.genesPicture(`, `Planet.groundAt(` and
    sends `place:`; `beasts.js`'s `speciesPicture` calls `genesPicture(`.
  Run `python -m unittest tests.test_viewer` → FAIL (no `place`).
- [x] Step 2: the server change. Run `python -m unittest tests.test_viewer` → the two server tests OK.
- [x] Step 3: the page. Run `python -m unittest tests.test_viewer` → OK.
- [x] Step 4: on a half-size check world (planet radius 18.5, 100 first creatures): in "Make", move size and the guts:
  the body changes; "Place it", click the ground: the creature is there in the view where clicked (its id in the
  outcome, its genes on its card when followed); a click on a rock is refused with its reason; screenshots; no console
  error. Stop the world and the viewer; close the browser pane. Done: the body changed with the guts and size (a rust hunter with quills); a click on
  the grass placed creature 203 (its `placed` event at the clicked point with the given genes); a click on the water: "Not
  placed: that is water, and it has no gills"; the dropdowns styled as the page's others. The console's 404s are the
  view asked for before the world drew its first picture after jumping in (as before).
- [x] Step 5: `python scripts/run_tests.py` → OK under 300 s (OK in 286.0 s; three older tests updated for the change: `_Knobs` takes the reporter in `test_costs` and `test_evolve`, `now.json`'s new `placed` in `test_evolve_runs`); commit "Make a creature and place it" (`git show
  --stat` → the files above and this plan); push to `evolve-from-scratch`.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | Placed where and as asked; the three brains; refusals | `python -m unittest tests.test_place` | `OK` |
| 2 | The soup and saves unchanged | `python -m unittest tests.test_world_rules tests.test_guts tests.test_saves` | `OK` |
| 3 | The server checks and writes it; the page has it | `python -m unittest tests.test_viewer` | `OK` |
| 4 | The owner can make and place one | the screenshots of Task 2 Step 4 | the body changing; the creature where clicked; a refusal |
| 5 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 6 | Pushed | `git ls-remote origin evolve-from-scratch` | the last commit's hash |

Anything not done or not proven is reported as such, first.
