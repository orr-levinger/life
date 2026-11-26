# Sphere World Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (the owner's chosen way: read and followed
> from the plugin's file, as for the diversity plan). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the world becomes the surface of a sphere (radius 37, room for 600) that the owner spins and zooms in a
three.js page, with creatures drawn in 3D from their genes.

**Architecture:** a geometry module (`src/sphere.py`) gives positions as unit vectors, arc distances, ground directions
and moves; creatures, food, terrain, senses and the world use it instead of x/y; senses and moves become the creature's
own frame (ahead, left). The page (`src/viewer.html` + `src/static/`) draws the planet with three.js from the world's
view (3D positions); the Python page server serves the library from the repository.

**Tech Stack:** Python 3.12, NumPy, TensorFlow (unchanged); three.js 0.170.0 (`three.module.min.js`,
`OrbitControls.js`, `BufferGeometryUtils.js`), vendored.

**Spec:** `docs/superpowers/specs/2026-11-26-sphere-world-design.md` (approved by the owner: "yes", 2026-11-26).

## Global Constraints

- Every number a rule with a plain meaning (`MEANS`); a gene's look follows its value.
- Tests first; `python scripts/run_tests.py` under 300 s; no full life simulation as a test (small test worlds, few
  steps).
- Directions a creature senses and moves by are (ahead, left) in its own frame; nothing uses east/north.
- Old flat saves (format ≤ 26) are refused in plain words before anything stops; their files stay.
- three.js is downloaded only after the owner's yes (file, source, size) and served by `src/viewer.py`; the page loads
  nothing from the internet.
- Speed: a step with 600 alive timed and told; above 200 ms a neighbour grid is added (Task 11). A frame drawn back to
  back under 33 ms on this box with 600 alive.

## Review Focus

1. Antipodal and near-identical points (distance π R and 0): arc distance, bearing and move stay finite — Task 1 tests.
2. A creature crossing any spot (old "edges", the former poles): moves, senses and facing continuous — Task 2 test
   "walks straight around the planet".
3. A cap (rock) straddling the point opposite the base of the test map — Task 3 test with caps at random spots.
4. Resuming a sphere save gives the same next steps as the world that wrote it — Task 7 round-trip test.
5. The page with 0 creatures, 1 creature, and 600; a creature whose genes are all at their limits — Task 10 gallery
   check renders without error.

---

### Task 0: three.js in the repository (the owner's yes first)

**Files:** Create `src/static/three.module.min.js`, `src/static/OrbitControls.js`, `src/static/BufferGeometryUtils.js`
(from `https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.min.js` and `…/examples/jsm/controls/`,
`…/examples/jsm/utils/`); `src/static/LICENSE-three.txt`. Modify `src/viewer.py` (serve `/static/<name>` from
`src/static/`, content types `.js` → `text/javascript`). Test: `tests/test_viewer.py`.

- [ ] **Step 1:** Ask the owner: files, source, sizes (the three about 700 KB together). Wait for yes.
- [ ] **Step 2:** Test `test_static_files_are_served`: GET `/static/three.module.min.js` → 200, `text/javascript`,
  body starts with the three.js licence comment; GET `/static/../viewer.py` → 404.
- [ ] **Step 3:** Run `python -m unittest tests.test_viewer` → FAIL (404).
- [ ] **Step 4:** Download the files; add the static route (only names in `src/static/`, no `..`).
- [ ] **Step 5:** Step-3 command → OK. Commit "three.js in the repository, served by the page server".

### Task 1: Sphere geometry

**Files:** Create `src/sphere.py`, `tests/test_sphere.py`.

**Interfaces — Produces** (all arrays NumPy, points unit vectors shape (…, 3), R the radius):
- `random_point(rng) -> np.ndarray` (uniform on the sphere).
- `arc(p, q, R) -> np.ndarray` = `R * atan2(|p × q|, p · q)` (broadcasts).
- `tangent_toward(p, q) -> np.ndarray` unit ground direction at `p` toward `q` (zeros when q = ±p).
- `bearing(p, f, q) -> (ahead, left)` = components of `tangent_toward(p, q)` along `f` and `p × f`.
- `walk(p, f, ahead, left, d, R) -> (p2, f2)`: move `d` along the ground direction `ahead·f + left·(p×f)` (normalised);
  `f2` = that direction carried to `p2` (the new facing, as today a creature faces the way it moved).
- `local_frame(p) -> (e1, e2)` an arbitrary orthonormal ground frame at `p` (for tests and newcomers).
- `exp_map(base, e1, e2, x, y, R) -> np.ndarray` the point `R`-arc `(x, y)` away from `base` (the test helpers' map).

- [ ] **Step 1: Tests** in `tests/test_sphere.py`:
  `test_a_quarter_turn_is_a_quarter_of_the_way` (arc(e_x, e_y, 37) == π·37/2);
  `test_opposite_and_same_points` (arc of ±p, finite; tangent_toward(p, ±p) == 0);
  `test_walking_straight_comes_back` (2000 walks of 2π·37/2000 ahead return within 0.001 of start, facing within
  0.001, from 5 random starts);
  `test_bearing_is_the_creatures_own_view` (q ahead → (1, 0); q to the left → (0, 1); the same at 50 random points and
  facings); `test_exp_map_keeps_distance_from_its_base` (arc(base, exp_map(…, x, y)) == hypot(x, y)).
- [ ] **Step 2:** `python -m unittest tests.test_sphere` → FAIL (no module).
- [ ] **Step 3:** Implement the functions above.
- [ ] **Step 4:** Step-2 command → OK. Commit "Sphere geometry".

### Task 2: Bodies and food on the sphere

**Files:** Modify `src/creature.py`, `src/food.py`, `src/world.py` (`width`/`height` → `radius`), `tests/helpers.py`.
Test: `tests/test_on_sphere.py` (new).

**Interfaces:**
- Consumes: Task 1.
- Produces: `Creature(id, p, size, energy, policy, …)` with `c.p` (unit vector), `c.facing` (ground unit vector),
  `c.radius` unchanged; `Food(p, remaining_duration, energy, fruit=False, meat=False)` with `f.p`;
  `World(radius, settings, species, rng, np_rng, food_rng, …)` with `world.radius` (the rule `planet_radius`);
  a move action `("MOVE", (ahead, left))` in the creature's frame; `tests/helpers.py`:
  `BASE` point and frame, `at(x, y) -> p` (= `exp_map` on the test planet), `add(w, x, y, …)` and `face(c, fx, fy)`
  keep the old test words (x east, y north on the local map).
- `SMALL_WORLD` gains `planet_radius: 1000.0` (a test planet nearly flat over the few units tests use) and loses
  `world_width`/`world_height`.

- [ ] **Step 1: Tests** in `tests/test_on_sphere.py`: `test_a_creature_walks_straight_around_the_planet` (on radius 37,
  moves of its top speed ahead return it to its start within 0.01 after 2πR); `test_moving_costs_the_arc_moved`;
  `test_reach_is_measured_along_the_ground` (eating at arc 0.9 × reach works, 1.1 × fails);
  `test_a_corpse_lies_where_the_prey_died`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Change `Creature`, `Food`, `_move` (`walk`), `can_reach` and `_attack` (arc), corpses (`p`), world
  size (`radius`); clamp-to-edges code goes (no edges).
- [ ] **Step 4:** Migrate the tests' creature and food making to the helpers (`add(w, x, y)` → on the map); any test
  whose expected number depended on flat geometry gets the sphere's number, each one listed in the ledger.
- [ ] **Step 5:** `python scripts/run_tests.py` → only Tasks 3–7's modules may still fail; record which. Commit
  "Bodies and food on the sphere".

### Task 3: Rocks, ponds, trees and forests on the sphere

**Files:** Modify `src/terrain.py`. Test: `tests/test_terrain.py`, `tests/test_water.py`.

**Interfaces:** Produces `Obstacle(kind, caps)` with caps `((point, arc_radius), …)`; `Tree(p, canopy)`;
`Terrain.blocked(p, radius, swims=False)`, `in_pond(p)`, `near_pond(p, width)`, `pond_spot(rng) -> p`,
`under_tree(points) -> bool array`, `sight_blocked(a, b, only=None)` (a rock cap crossed by the great-circle arc a→b),
`nearest_edges(points) -> (edge arc, ground directions)`; `make_terrain(rules, rng, R, keep_clear)`.

- [ ] **Step 1: Tests:** the existing terrain and water tests moved to caps (via `at(x, y)`), plus
  `test_a_rock_anywhere_hides_what_is_behind_it` (20 random rock spots on radius 37, including opposite the map's base)
  and `test_a_new_planet_has_its_forests` (trees within `forest_spread` arc of a centre; 4 centres).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → OK. Commit "Terrain on the sphere".

### Task 4: Senses in the creature's own view

**Files:** Modify `src/senses.py`, `src/policies.py`, `src/sensors.py`. Test: `tests/test_senses.py`,
`tests/test_policies.py`, `tests/test_call.py`, `tests/test_kin.py`, `tests/test_stealth.py`.

**Interfaces:** Consumes Tasks 1–3. Produces: every direction pair in the 47 inputs is `bearing(...)` (ahead, left);
distances are arcs; the learner's move output (ahead, left) → `("MOVE", (ahead·top, left·top))`; the yardsticks
(random, rules) move by (ahead, left).

- [ ] **Step 1: Tests:** existing senses tests on the map; `test_food_ahead_is_ahead_anywhere` (food 2 units ahead of
  a creature at 30 random points and facings → inputs (1, 0)); `test_a_call_from_the_left_is_heard_from_the_left`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → OK. Commit "Senses in the creature's own view".

### Task 5: The world on the sphere

**Files:** Modify `src/world.py` (newcomer spots, babies near parents, food and pond-edge spots, fruit, kin and calls
by arc), `src/rules.py`. Test: `tests/test_world_rules.py`, `tests/test_seasons.py`, `tests/test_terrain.py`.

**Interfaces:** Produces rules `planet_radius` (start 37, low 5, high 2000; group world; for new worlds),
`room` and `auto_target` highest 600 (room start 600, auto_target start 480), `first_creatures` 200 (highest 600),
`soup` 80 (highest 600), `obstacles` 70 (highest 300), `trees` 92 (highest 400), `forests` 4, `exam_radius` 10.9
(replaces `exam_width`/`exam_height`); `world_width`/`world_height` go. `ROOM_SLOTS` = 600. `START` food per step and
food at most doubled.

- [ ] **Step 1: Tests:** `test_a_new_planet_has_its_numbers` (the values above); `test_a_baby_is_born_near_its_parent`
  (within 1 arc unit); `test_food_spreads_evenly_over_the_planet` (2,000 candidates: each octant within 25 % of an
  eighth); the existing world tests on the map.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → OK. Commit "The world on the sphere".

### Task 6: The exam on its own small planet

**Files:** Modify `src/exam.py`. Test: `tests/test_exam.py`.

- [ ] **Step 1:** Test `test_the_exam_planet_stays_as_it_was` (radius `exam_radius` 10.9, straight-line bodies kept).
- [ ] **Step 2:** FAIL → **Step 3:** implement → **Step 4:** OK. Commit "The exam on its own small planet".

### Task 7: Saves, the runner and the page's data

**Files:** Modify `src/saves.py` (format 27: points, facings, caps, trees as 3D; `planet_radius`), `src/evolve.py`
(`world_view` gives `p`, `facing` as 3-lists and terrain caps; `--window` and `src/visualization.py` go), `src/viewer.py`
(`_hand_over`, start-from-save and resume refuse a save of format ≤ 26 with "a flat world from before the sphere: it
cannot be resumed"), `src/scoreboard.py` (toward food, flees, chases by bearings; direction concentration = length of
the mean of all facings in 3D), `src/check_proof.py` (thresholds kept). Delete `src/visualization.py`,
`tests/test_visualization.py`. Test: `tests/test_saves.py`, `tests/test_evolve.py`, `tests/test_viewer.py`,
`tests/test_scoreboard.py`.

- [ ] **Step 1: Tests:** `test_a_sphere_save_resumes_the_same_world` (save, load, 50 steps each: same picture);
  `test_a_flat_save_is_refused_in_plain_words` (format 26 → SaveError naming the sphere; the viewer's update and resume
  answer 409 with that sentence and the run keeps going); `test_the_view_gives_points_on_the_planet`; the old format
  conversion tests (1–26) go, as flat saves can no longer be read.
- [ ] **Step 2:** FAIL → **Step 3:** implement → **Step 4:** `python scripts/run_tests.py` → OK (every module).
  Commit "Saves, runner and page data on the sphere".

### Task 7b: Short test runs (the owner, 2026-11-26: "ok" to shortening test_evolve's whole-world runs)

On the sphere a step costs 2-3× the flat one, so test_evolve passed 300 s. The test world's runner timings are 5×
quicker (`SMALL_WORLD`: save_every 400, row_every 50, auto_every 100, auto_measure 200, exam_every 400, exam_steps
100, control_every 5) and every step number in tests/test_evolve.py and the resume test in tests/test_saves.py is
divided by 5. Proof: `python scripts/run_tests.py` → OK under 300 s; the same assertions, scaled.

### Task 8: The planet on the page

**Files:** Modify `src/viewer.html` (the live view's canvas becomes a three.js scene; tiles, charts, Rules page
unchanged). Create `src/static/planet.js` (scene, ground, terrain, camera, controls). Test: `tests/test_viewer.py`.

**Interfaces:** Consumes Task 7's view (`p`, `facing`, caps). Produces in `planet.js`: `makePlanet(container, view)`,
`updatePlanet(view)`, `followCreature(id | null)`, `pointOfView(id | null)`, `bench(n) -> {msPerFrame}` (frames drawn
back to back, for Task 10).

- [ ] **Step 1:** Test `test_page_draws_a_planet` (the page imports `/static/planet.js`, which imports
  `/static/three.module.min.js`; no `http` URL in either).
- [ ] **Step 2:** FAIL → **Step 3:** implement: textured grass (light and dark patches, fine blades); rocks as rough
  boulders from their caps rising out of the ground; ponds flat and a little sunk; trees with trunks and crowns; drag
  to spin, wheel/pinch to zoom from the whole planet to a close-up; follow keeps the creature centred; point of view
  shows only what it senses (today's rule).
- [ ] **Step 4:** OK; screenshots of the whole planet and a close-up sent to the owner. Commit "The planet on the page".

### Task 9: Creatures in 3D from their genes

**Files:** Create `src/static/beasts.js` (from the approved gallery: body plans grazer / hunter / swimmer blending by
diet and water; every gene's feature as in the spec; hunger ring; cones, hearing and call rings on the ground).
Modify `src/static/planet.js`. Test: `tests/test_viewer.py`; a gallery page `src/static/gallery.html` (each gene low,
middle, high, as shown to the owner).

**Interfaces:** Produces `beastModel(genes, detail: "near" | "far") -> THREE.Group`, `geneKey(genes)` (models cached by
rounded genes), used by `planet.js` with near models within `near_distance` (a page setting, 15 units) of the camera.

- [ ] **Step 1:** Test `test_page_has_the_creature_look` (beasts.js names every gene the spec maps: size, love, energy
  ring, diet, water, fear, layers, neurons, learns, learning, width, reach, hearing, armour, stealth, calls, kin).
- [ ] **Step 2:** FAIL → **Step 3:** implement → **Step 4:** OK; the gallery's pictures sent to the owner. Commit
  "Creatures in 3D from their genes".

### Task 10: Speed of the page

**Files:** `src/static/planet.js` (instancing per model where models repeat, far models merged per frame only when the
view changes). Test: a timing in the browser pane: `planet.bench(30)` (frames drawn back to back).

- [ ] **Step 1:** With a world of 600 alive watched: `bench(30)` → median ms per frame; must be under 33 ms. If above,
  lower the far model and repeat; report each number.
- [ ] **Step 2:** Commit "Page speed: N ms a frame with 600" (the measured N).

### Task 11: Speed of a step

**Files:** the timing script (scratchpad) on a new planet with 600 alive.

- [ ] **Step 1:** Time 150 steps (as for the diversity plan) → report the ms per step. Above 200 ms: add a neighbour
  grid in `src/sphere.py` (`neighbours(points, R, reach) -> pairs` from a grid of cells on the cube-sphere) used by
  senses and kin; its test `test_the_grid_finds_the_same_pairs` (against all pairs, 300 random points); time again.
- [ ] **Step 2:** Commit (if code changed) with the numbers in the message.

### Task 11b: Neighbours by k-d tree (the owner, 2026-11-26: "use the most advanced optimiztions algorithems", "do it")

Measured first (Task 11 Step 1): 1099 ms a step with ~600 alive; senses ~320 ms, kin shares ~215 ms, learning from
others ~170 ms (every pair), the learner ~260 ms. Nothing about what a creature senses changes: only how the near ones
are found.

**Files:** `requirements.txt` (scipy, downloaded with the owner's yes), `src/sphere.py`, `src/senses.py`,
`src/world.py`, `src/terrain.py`; tests in `tests/test_sphere.py`, `tests/test_senses.py`, `tests/test_terrain.py`.

- [ ] **Step 1:** `pip install scipy`; pin its version in `requirements.txt`.
- [ ] **Step 2:** Test first `test_tree_finds_the_same_pairs` (300 random points, each its own reach; against every
  pair by arc) and `test_tree_pairs_on_a_tiny_and_a_huge_planet`; RED; then `sphere.neighbours(points, reaches, R)`:
  a `scipy.spatial.cKDTree` on the 3D points, each one's reach as the chord 2R·sin(reach / 2R), the pairs exact by
  arc; GREEN.
- [ ] **Step 3:** Test first `test_senses_same_with_and_without_the_tree` (a hand-made world of 40 creatures and
  rocks: every input identical to the every-pair result); RED; then senses, kin shares and learning from others
  use the neighbour pairs, their bearings and distances computed in one NumPy pass; GREEN.
- [ ] **Step 4:** Test first `test_sight_blocked_same_with_the_rock_tree` (random lines and rocks, against every
  rock); RED; then a sight line tests only the rocks a k-d tree of rock caps finds near it; GREEN.
- [ ] **Step 5:** Time 150 steps again (the Task 11 script) and profile; report each part. Still above 200 ms:
  report what is left and ask the owner (Numba would be a separate download).
- [ ] **Step 6:** `python scripts/run_tests.py` OK under 300 s; commit "Neighbours by k-d tree: N ms a step with 600".

### Task 11c: Brains copied in one go, water by lookup (the owner, 2026-11-26: "most important is to make it fast first")

After Task 11b: 501 ms a step with ~600 alive; ~1,100 small TensorFlow calls a step copy brains slot by slot (~118 ms),
"in a pond?" asked one creature at a time against every cap (~48 ms). What a creature does and senses stays the same.

- [ ] **Step 1:** Find every slot-by-slot TensorFlow read and write (profile callers); test first that a batch of
  slot writes and reads gives the same weights as one at a time (`tests/test_learner.py`); RED; then the learner
  gathers them and does one read / one write a step; GREEN.
- [ ] **Step 2:** Test first `test_in_pond_same_with_the_tree` (random points and ponds, against every cap); RED;
  then `in_pond` / `blocked` try only the caps a k-d tree finds within reach; GREEN.
- [ ] **Step 2b:** (the owner, 2026-11-26: "ok") Measured: deciding for the same number of creatures 19 ms, for a new
  number 350-580 ms (the compiled brain maths is compiled again for each new batch size). Test first
  `test_the_brains_decide_in_one_batch_size` (every call to the compiled graph has the learner's full batch, whatever
  the number acting); RED; then the batch is padded to every slot, the dice drawn for the real ones only (each still
  decides as before); GREEN; the timing script's alive counts stay 600 / 389 / 60.
- [ ] **Step 3:** Time 150 steps again and profile; report each part.
- [ ] **Step 4:** `python scripts/run_tests.py` OK under 300 s; commit "Brains in one go, water by lookup: N ms a step".

### Task 11d: Food sliders reach further (the owner, 2026-11-26: "yes" to slider tops 800 / 20 / 100)

The easiest ends of three knobs (`src/dial.py`): food pieces at most 400 -> 800, new food per step 6 -> 20, energy per
food 60 -> 100. Auto may go that easy too. New worlds still start at START (400, 6, 20-40).

- [ ] **Step 1:** Test first `test_food_sliders_reach_further` (`tests/test_dial.py`: the three knobs' easiest ends,
  and START unchanged); RED; then the three numbers; GREEN; `python -m unittest tests.test_dial tests.test_viewer` OK.
- [ ] **Step 2:** Commit "Food sliders reach 800 pieces, 20 a step, 100 energy".

### Task 11e: Fruit slider reaches 1 (the owner, 2026-11-26: "Fruit a tree drops per step set max to 1")

- [ ] **Step 1:** Test first `test_fruit_slider_reaches_one` (`tests/test_dial.py`: the fruit_rate knob's easiest end
  is 1.0; START unchanged); RED; then the number; GREEN; `python -m unittest tests.test_dial tests.test_control
  tests.test_viewer` OK.
- [ ] **Step 2:** Commit "Fruit slider reaches 1 a step".

### Task 12: Final

- [ ] **Step 1:** `python scripts/run_tests.py` → OK under 300 s.
- [ ] **Step 2:** Start a new planet on the page (8765); screenshots: whole planet, a close-up of creatures, follow,
  point of view; sent to the owner.
- [ ] **Step 3:** Fresh whole-branch review (executing-plans), fixes test-first.
- [ ] **Step 4:** Merge to `evolve-from-scratch` and push (the owner pulls it on the Mac).

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | Walking straight comes back; distances and bearings right everywhere | `python -m unittest tests.test_sphere tests.test_on_sphere` | `OK` |
| 2 | Every trait works on the planet | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 3 | A sphere save resumes the same world; flat saves refused in plain words | `python -m unittest tests.test_saves` | `OK` |
| 4 | A new planet: radius 37, room 600, 200 first, 70 obstacles, 92 trees in 4 forests | the new run's `now.json` rules | those values |
| 5 | The page shows the planet, spins and zooms; creatures in 3D from genes | screenshots (whole, close-up, follow, POV) sent | the owner sees them |
| 6 | Page speed | `bench(30)` with 600 alive | under 33 ms a frame (number reported) |
| 7 | Step speed | timing script, 600 alive | ms per step reported; grid added if above 200 ms |
| 7b | Neighbours by k-d tree give the same senses | `python -m unittest tests.test_sphere tests.test_senses tests.test_terrain` | `OK`; step time reported before and after |
| 8 | three.js served locally | `test_static_files_are_served`; no `http` import in the page | `OK` |
| 9 | One commit per task | `git log --oneline` | Tasks 0–12 |
| 10 | On GitHub for the Mac | `git push origin evolve-from-scratch` | pushed |

Anything not done or not proven is reported as such, first.
