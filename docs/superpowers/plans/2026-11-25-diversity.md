# Diversity — plan

**Goal:** every body and way of life can win, so the world keeps many kinds alive (the owner, 2026-11-25: "I want more
variety… how better can we simulate nature?").
**Spec:** `docs/superpowers/specs/2026-11-25-diversity-design.md` (its decisions 1–10 are what this plan builds).
**Architecture:** the world (`src/world.py`, `src/creature.py`, `src/senses.py`, `src/terrain.py`) gives costs,
benefits and senses; genes live on the creature and are listed in `src/rules.py` (`_gene`, `EDITABLE_GENES`,
`GENE_MEANS`); every number is a `Rule`; saves convert forward one format a task (`src/saves.py`); the page
(`src/viewer.html`) shows each gene as a tile and a Genes-panel box.
**Tech Stack:** Python 3.12, NumPy, TensorFlow (`src/learner.py`), stdlib HTTP server, one HTML page.
**Approvals:** the ideas — the owner, 2026-11-25, "I LIKE ALL THE IDEAS", "care for kin amazing idea", "write the plan";
Task 0 — "the world need to be much bigger with much more max creatures", "add this to the plan". The plan —
"Approved, start Task 0" (2026-11-26); Task 3 changed — "lets just build it and fine tune it later" (2026-11-26).
This plan is approved only by the owner's explicit yes.

## Global Constraints

- Nothing decides who attacks or helps whom: only costs, benefits and senses (spec, "What is decided").
- Every new number is a `Rule` with a meaning in `MEANS`; every new trait is a gene: `_gene(...)` with its range,
  newcomer range and smallest change, an `EDITABLE_GENES` entry, a `GENE_MEANS` line, a tile (`GENE_TILES`, `TILES`)
  and a Genes-panel box (`GENE_BOXES`).
- A new sense is appended at the end of the inputs (`N_SENSES` in `src/brain.py`, today 25 — measured:
  `grep -n '^N_SENSES' src/brain.py`); old brains get zero weights for it (`_widen_inputs`), so they act as before.
- A new action starts almost never chosen: its bias −5 in old and new brains.
- Saves go one format up per task (today 15 — measured: `grep -n '^SAVE_FORMAT' src/saves.py`).
- Tests first; one commit per task, through the commit judge; the whole suite before every commit
  (`python -m unittest discover -s tests`, today 357 tests `OK` — measured).
- The page server is restarted only when `src/viewer.py` or what it imports changed; if a world ends, it is restored
  from its last save (memory: life-viewer-owns-worlds).

## Review Focus

- Saves: every conversion keeps old creatures and brains acting as before.
- Speed: a step stays near today's 13 ms with 80 creatures (measured on a copy of the owner's world, 2026-11-25);
  every task re-measures it with the timing script.
- Plain meanings on the Rules page for each new rule.

## Tasks

Every task has the same six steps; each step's proof is named in it.

### Task 0 — A bigger world with more creatures (spec decision 0)
**Files:** `src/rules.py` (`room` start and highest 300, `auto_target` start 240 and highest 300, `first_creatures`
100 / highest 300, `soup` 40 / highest 300, `world_width` 120, `world_height` 72, `obstacles` 35 / highest 150, `trees`
46 / highest 200), `src/saves.py` (learner slots 80 → 300), `src/viewer.html` (the Alive tile's "of N room" from the
world's own `room` rule, not a fixed 80), `tests/test_rules.py`, `tests/test_saves.py`, `tests/test_viewer.py`.
**Interfaces:** `ROOM_SLOTS` (= the `room` rule's highest, 300) sizes the learner; a format-15 save's learner arrays
are padded from 80 slots to 300 (empty slots, free).
1. Tests: the new starts and limits; a format-15 save with 80 slots loads with 300 and its 80 brains unchanged; the
   page reads the room from the rules. Run `python -m unittest tests.test_rules tests.test_saves tests.test_viewer`
   → fails on the new starts (the right reason).
2. Code. Step-1 command → `OK`.
3. Format 16: rules keep the world's own values (its size and room stay as they are); learner slots padded.
   `tests.test_saves` → `OK`.
4. Whole suite → `OK`. Speed: a copy of a new world run to 300 alive; the ms per step reported to the owner before
   any world is switched (today 13 ms with 80).
4b. Speed (the owner, 2026-11-26: "Yes, fix the speed"; measured: 618 ms a step with 300 alive, 64.5 of 93.7 s in
   `Terrain.sight_blocked`): `sight_blocked(..., only=mask)` checks only the pairs already within reach and cone,
   in one pass over those pairs; `senses` passes its masks. Test first (`tests/test_terrain.py`): for random terrain,
   points and masks, `sight_blocked(..., only=mask)` equals the full check `& mask` → fails (`TypeError: only`),
   then `OK`. Timing script with 80 and 300 alive → the ms reported.
4c. Fast tests (the owner, 2026-11-26: "hard limits for tests! 5 minutes! 1 second above it should fail the tests!",
   "we should never run full life simulations as tests!"): `scripts/run_tests.py` runs every test file, 4 at a time,
   and fails when the whole run passes `--limit` seconds (300); its test (`tests/test_run_tests.py`) first fails (no
   script), then `OK`. Test worlds are small whatever the starting rules (`tests/helpers.SMALL_WORLD`: 50 × 30, room
   80, 30 first, soup 10, 6 obstacles, 8 trees, target 64) for `make_world`, Auto's tests and the runs the tests
   start. `python scripts/run_tests.py` → `OK` under 300 s. The tests that still run whole worlds are listed to the
   owner with a proposal to replace them.
5. Commit "A bigger world…" (judge passes; stat shows only this task's files). The owner's world updated (it keeps
   50 × 30 and 80 room until the owner changes them or starts a new world).
6. Report the measured step time and, if the owner starts a new world, its alive count after 20,000 steps.

### Task 1 — Size curves (spec decision 1)
**Files:** `src/rules.py`, `src/creature.py`, `src/world.py`, `src/senses.py`, `src/saves.py`, `src/viewer.html`,
`tests/test_size_curves.py` (new), `tests/test_saves.py`.
**Interfaces:** rules `tank_power` 1.3, `living_power` 0.75, `speed_power` 0.5, `lifespan_power` 0.5, `heard_power`
1.0, `seen_power` 0.5 (group `body`); `Creature.max_energy`, `velocity`, living cost, the lifespan drawn at birth, and
`senses._in_sight` / hearing use them.
1. Write `tests/test_size_curves.py`: for sizes 0.5, 1, 2 the tank, living cost, speed and lifespan follow the powers;
   a size-0.5 mover is heard at most at hearing × 0.5 and seen at most at reach × √0.5. Run
   `python -m unittest tests.test_size_curves` → fails with `KeyError: 'tank_power'` (the right reason).
2. Code the rules and formulas; the Rules page's "what it means" lines (`WORKS_OUT`) use the powers. Run the step-1
   command → `OK`.
3. `src/saves.py` format 17 adds the six rules at their starts; `tests/test_saves.py` gains "a format 16 save gets the
   size powers". Run `python -m unittest tests.test_saves` → `OK`.
4. Whole suite → `OK`. Timing script on a copy of the owner's world → a step's ms reported.
5. Commit "Size by curves…" (the judge passes; `git show --stat` lists exactly this task's files). Update the owner's
   world (Update code); its `now.json` shows `tank_power`.
6. After 20,000 steps read the Size tile (lowest–average–highest) and report it beside today's 0.500–0.532.

### Task 2 — Diet gene (spec decision 2)
**Files:** `src/rules.py`, `src/creature.py`, `src/food.py` (`meat` flag; a corpse is meat), `src/world.py`,
`src/senses.py`, `src/brain.py` (`N_SENSES` 27), `src/saves.py`, `src/evolve.py`, `src/viewer.html`,
`tests/test_diet.py` (new), `tests/test_saves.py`.
**Interfaces:** gene `diet` (0–1, newcomers 0–1); `Creature._eat` gain × (1 − diet) for plants and fruit, × diet for
meat; `_attack` gain × diet, damage × (1 + diet × `diet_bite`); inputs 25 (nearest seen food is meat), 26 (own diet).
1. Write `tests/test_diet.py` for those gains, the bite, the two inputs. Run → fails (`AttributeError: diet`).
2. Code it; tile, Genes-panel box, the creature drawn redder the higher its diet. Step-1 command → `OK`.
3. Format 18: creatures get diet 0.5, corpses meat, brains widened by 2; a save test. `tests.test_saves` → `OK`.
4. Whole suite → `OK`; timing reported.
5. Commit "Diet gene…" (judge passes; stat shows only this task's files). World updated; `now.json` genes has `diet`.
6. After 20,000 steps report the Diet tile.

### Task 3 — Calls of several kinds, each with a volume (spec decision 4, changed by the owner 2026-11-26)
**Files:** `src/rules.py`, `src/learner.py` (choices: the 3 of today + 10 calls; the move head gains a volume),
`src/brain.py` (`N_SENSES` + 17), `src/creature.py`, `src/world.py`, `src/senses.py`, `src/policies.py`,
`src/saves.py`, `src/evolve.py`, `src/viewer.html` (a call drawn as a ring as wide as it carries; a Calls tile and
box), `tests/test_call.py` (new), `tests/test_saves.py`.
**Interfaces:** gene `calls` 0–10 (newcomers 0–0, smallest change 1); rule `call_reach` (how far a call at full volume
carries); actions `CALL_1`…`CALL_10`, those above the creature's `calls` closed (masked like unused neurons), each
starting at bias −5; the loudest call heard this step gives 17 inputs: which call (10), how loud it arrived
(1 − distance ÷ its reach), direction (cos, sin), the voice (diet, size, water, stealth scaled 0–1; 0 until those
genes exist). No energy cost.
1. Write `tests/test_call.py`: a creature with calls 0 cannot call; with calls 3 it can make calls 1–3 only; a call
   at volume 0.5 is heard within 0.5 × `call_reach` and not beyond; the hearer gets the call's kind, loudness,
   direction and voice; similar genes → voices within 0.05; no call → zeros; calling costs no energy; a newcomer has
   calls 0; a converted brain calls in under 2 % of 1,000 steps. Run → fails.
2. Code. Step-1 command → `OK`.
3. Format 19: creatures get calls 0; the choice head widened by 10 with bias −5; the move head gains the volume;
   inputs widened by 17. `tests.test_saves` → `OK`.
4. Whole suite → `OK`; timing reported.
5. Commit "Calls…" (judge passes). World updated.
6. After 20,000 steps report the Calls tile and the share of steps spent calling, near hunters and away from them.

### Task 4 — Care for kin (spec decision 5)
**Files:** `src/rules.py`, `src/world.py` (`_teach`), `src/creature.py` (family rule removed), `src/senses.py`
(family seen; input 35 similarity), `src/brain.py` (`N_SENSES` 36), `src/saves.py`, `src/viewer.html`,
`tests/test_kin.py` (new), `tests/test_saves.py`.
**Interfaces:** gene `kin` (0–1); `similarity(a, b)` = 1 − mean |gene difference ÷ gene range| over size, diet,
water, stealth, armour, vision width, vision reach, hearing; reward += kin × Σ similarity × their energy change over
those it perceives.
1. Write `tests/test_kin.py`: kin 1 gains a share of a similar perceived creature's change, none of an unlike one;
   kin 0 none; family can be attacked and is seen; the similarity input. Run → fails.
2. Code. Step-1 → `OK`. 3. Format 20 (kin 0.5; widened by 1). `tests.test_saves` → `OK`.
4. Suite `OK`; timing. 5. Commit "Care for kin…" (judge). World updated.
6. After 20,000 steps report the Kin tile and attacks between similar and unlike creatures.

### Task 5 — Water (spec decision 6)
**Files:** `src/terrain.py`, `src/creature.py`, `src/world.py`, `src/senses.py` (inputs 36–37), `src/brain.py`
(`N_SENSES` 38), `src/saves.py`, `src/viewer.html`, `tests/test_water.py` (new), `tests/test_saves.py`.
**Interfaces:** gene `water`; rules `land_penalty` 0.7, `pond_food_share` 0.2; `Terrain.in_pond`; ponds no longer block
sight; inputs: in water, nearest obstacle in view is water.
1. Write `tests/test_water.py`: water 0 cannot enter a pond (bumps); water 1 swims at full speed, walks at 0.3;
   food grows in ponds; a pond does not hide food, a rock does; the inputs. Run → fails.
2. Code. → `OK`. 3. Format 21 (water 0; widened by 2). → `OK`. 4. Suite; timing.
5. Commit "Water…" (judge). World updated. 6. After 20,000 steps report the Water tile.

### Task 6 — Stealth (spec decision 7)
**Files:** `src/rules.py`, `src/creature.py`, `src/senses.py`, `src/saves.py`, `src/viewer.html`,
`tests/test_stealth.py` (new), `tests/test_saves.py`.
**Interfaces:** gene `stealth`; rules `stealth_strength` 0.7, `stealth_cost` 0.3.
1. Test: seen and heard distances × (1 − stealth × 0.7); living cost × (1 + stealth × 0.3). Run → fails.
2. Code → `OK`. 3. Format 22 (stealth 0) → `OK`. 4. Suite; timing. 5. Commit "Stealth…" (judge); world updated.
6. Report the Stealth tile after 20,000 steps.

### Task 7 — Armour (spec decision 8)
**Files:** `src/rules.py`, `src/creature.py`, `src/saves.py`, `src/viewer.html`, `tests/test_armour.py` (new),
`tests/test_saves.py`.
**Interfaces:** gene `armour`; rules `armour_strength` 0.7, `armour_slow` 0.3, `armour_cost` 0.3.
1. Test: a hit takes × (1 − armour × 0.7); speed and living cost change. Run → fails.
2. Code → `OK`. 3. Format 23 (armour 0) → `OK`. 4. Suite; timing. 5. Commit "Armour…" (judge); world updated.
6. Report the Armour tile after 20,000 steps.

### Task 8 — Seasons (spec decision 9)
**Files:** `src/rules.py`, `src/world.py`, `src/saves.py`, `src/viewer.html` (season marker), `tests/test_seasons.py`
(new), `tests/test_saves.py`.
**Interfaces:** rules `year_steps` 20,000, `season_depth` 0.5; food and fruit per step × (1 + depth × sin(2π step ÷
year)).
1. Test: per-step food follows that curve at four points of the year. Run → fails.
2. Code → `OK`. 3. Format 24 (the rules) → `OK`. 4. Suite; timing. 5. Commit "Seasons…" (judge); world updated.
6. Report births and alive over two years (40,000 steps).

### Task 9 — Regions (spec decision 10)
**Files:** `src/terrain.py`, `src/world.py`, `src/rules.py`, `tests/test_terrain.py`.
**Interfaces:** rules `forests` 2, `forest_spread` 6, `pond_edge_food` 2.0; `make_terrain` clusters trees in forests;
food candidates near a pond's edge count × `pond_edge_food`.
1. Test: trees lie within `forest_spread` of a forest centre; pond edges get more food than open ground over 1,000
   steps. Run → fails.
2. Code → `OK`. 3. No save change (a world keeps its ground; a new world gets regions). 4. Suite; timing.
5. Commit "Regions…" (judge). A new world is started only if the owner asks.
6. A screenshot of a new world's ground, shown to the owner.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | Each task's tests were written first and failed for the right reason | each task's step-1 run, output kept in the session | a failure naming the missing piece |
| 2 | Each task's code passes its own tests and the whole suite | `python -m unittest discover -s tests` | `OK` after every task |
| 3 | Every new number is a rule that says what it means | `python -m unittest tests.test_rules` (every rule and gene says what it means) | `OK` |
| 4 | Every new trait has a tile and a Genes-panel box | `python -m unittest tests.test_viewer` naming each gene; a screenshot of the tiles | `OK`; the tile visible |
| 5 | Old saves load into every new format with brains acting as before | `python -m unittest tests.test_saves` | `OK` (a conversion test per format 16–24) |
| 6 | One commit per task, each passed by the judge | `git log --oneline` | 10 commits, one per task (0–9) |
| 7 | The owner's world runs each task's code | its `now.json` after each update | the task's new rule or gene present |
| 8 | Each task's effect is measured | the tile read after 20,000 steps | the numbers reported to the owner |
| 10 | A world holds 300 creatures | a new world's `now.json` after it fills | `alive` reaches above 80, and room 300 |
| 9 | A step stays fast | the timing script on a copy of the owner's world after each task | the ms per step reported; a rise above 20 ms is said |

Anything not done or not proven is reported as such, first.
