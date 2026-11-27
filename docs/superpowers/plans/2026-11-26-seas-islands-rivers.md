# Seas, Islands and Rivers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Half the planet is water (ocean, seas, lakes), with islands walkers cannot reach and rivers with a few
crossings, so groups of land creatures live apart.

**Architecture:** A new `src/water.py` holds the water of a planet: a smooth height (a dozen waves) below a sea level,
round islands raised in open water, rivers as chains of points with crossings. It is made once per world and turned
into a cube-sphere grid of cells (0.25 across), so "is this water?" is one array lookup. `Terrain` gets the water and
answers every pond question (in water, blocked, near the shore, a spot in water, water ahead) from caps (old saves) or
the grid. The page gets the water as a picture (`/water?run=`, a PNG drawn from the grid) wrapped on the ball.

**Tech Stack:** Python 3.12, NumPy, SciPy (already in), three.js (already in).

**Spec:** `docs/superpowers/specs/2026-11-26-seas-islands-rivers-design.md`

## Global Constraints

- New rules: `water_share` 0.5 (0-0.8), `islands` 8 (0-30), `island_gap` 6.0 (0-20), `rivers` 3 (0-10),
  `river_crossings` 2 (0-6), `river_width` 1.5 (0.5-4). `obstacles` 70 stays and means rocks only.
- No change to senses' meaning, genes, decisions or learning; water is what a pond was.
- The exam planet does not change.
- Tests: `python scripts/run_tests.py` under 300 s; never a full life simulation as a test.
- Step time on a new planet with 600 alive within 5% of before (Task 11c's timing script).
- Page under 33 ms a frame (`bench(30)`).

## Review Focus

- A planet of radius 5 or 2000, or water_share 0 or 0.8: making the water never fails or hangs (bounded tries).
- An island that cannot be placed (no open water far enough): fewer islands, said in the log, never a hang.
- A river that ends in a hollow, not the sea: it ends in a lake there.
- A point exactly on a cell's edge, or at a cube corner: the lookup gives one answer, never an error.
- An old sphere save (format 27, ponds as caps): resumes, its ponds still water.

Each line gets its test in the task that owns it.

---

### Task 1: The new rules

**Files:** Modify `src/rules.py` (rules list, help texts; `obstacles` text says rocks); Test `tests/test_rules.py`.

**Produces:** `rules["water_share"]`, `["islands"]`, `["island_gap"]`, `["rivers"]`, `["river_crossings"]`,
`["river_width"]`.

- [ ] Step 1: test `test_the_water_rules` (each new rule's start and range as in Global Constraints; each has a help
  text). Run `python -m unittest tests.test_rules` → FAIL (no rule).
- [ ] Step 2: add the six rules (group "world") and texts. Run → OK.
- [ ] Step 3: commit "Rules for seas, islands and rivers".

### Task 2: The water of a planet (`src/water.py`)

**Files:** Create `src/water.py`; Test `tests/test_water_map.py`.

**Produces:**
- `class Water`: `Water(radius, waves, sea_level, islands, rivers, crossings, river_width)` where `waves` is an
  array `[k, 5]` (unit direction xyz, frequency, phase), `islands` `[m, 4]` (centre xyz, radius), `rivers` a list of
  arrays `[n_i, 3]` (points along each river), `crossings` `[c, 3]` (points where a river is land).
- `Water.height(points) -> array` (the smooth height), `Water.wet_exact(points) -> bool array` (the rule itself),
  `Water.is_water(points) -> bool array` and `Water.is_water_at(p) -> bool` (grid lookup), `Water.share(n, rng)`,
  `Water.state() -> dict`, `Water.from_state(dict) -> Water`, `Water.picture(width=1024) -> bytes` (a PNG, equirectangular,
  blue where water, clear elsewhere).
- `make_water(rules, rng, radius) -> Water`.

- [ ] Step 1: tests (each on radius 37 unless said):
  `test_about_half_is_water` (share of 100,000 random points within 0.03 of `water_share`; also 0.0 → none and
  0.8 → 0.8 ± 0.03);
  `test_islands_are_cut_off` (every point at island radius + 0.5 .. + `island_gap` - 0.5 from an island's centre is
  water; at least 6 of 8 placed);
  `test_rivers_reach_water_and_have_crossings` (each river's last point is in the sea or a lake; `river_crossings`
  land points on each, inside its width);
  `test_the_grid_is_the_rule` (10,000 random points: `is_water` equals `wet_exact` at the cell's centre; points on
  cube edges and corners answer);
  `test_water_comes_back_from_a_save` (state → from_state → same `is_water` on 10,000 points);
  `test_any_planet_size_makes_water` (radius 5 and 2000, water_share 0 and 0.8: done in under 5 s each, no error);
  `test_the_picture_is_a_png` (starts with the PNG signature; its size 1024 × 512).
  Run `python -m unittest tests.test_water_map` → FAIL (no module).
- [ ] Step 2: write `src/water.py`: waves (12, random directions, frequencies 1-4 per radian, phases); sea level =
  the height quantile that leaves `water_share` + the islands' area share as water (20,000 samples); islands: centres
  tried (at most 200 tries each) where every sampled point within radius + gap is water; rivers: start at land points
  high above sea level, walk 0.5 a stride down the height's slope to water (at most 400 strides; ending in a hollow
  makes a lake of radius 2 there); crossings at random points along each river away from its ends; grid: 6 cube faces
  of N × N cells (N = ceil(π R / 2 / 0.25)), each cell's centre judged by `wet_exact`; lookup by the cube face and
  cell of a point (vectorised). Run → OK.
- [ ] Step 3: commit "The water of a planet: seas, islands, rivers".

### Task 3: Terrain asks the water

Changed with the owner on 2026-11-27 ("Yes, use the map"): the first wording (9 points around a body on every move,
rays sampled every 0.5) measured 56 µs a move and 17 ms a step of sight lines, about 45 % slower a step. Instead a
distance-to-water map is made once with the grid: each cell's distance along the ground to the nearest water (0 in
water), so every question is one lookup.

**Files:** Modify `src/water.py` (the distance map: `distance_to_water(points)`, `distance_at(p)` one point the fast
way), `src/terrain.py` (Terrain takes `water: Optional[Water] = None`; `make_terrain` builds rocks only, then the water,
then the trees off both), `src/senses.py` (`_obstacles_seen` adds water ahead), `src/world.py` (`spawn_food` uses
`terrain.has_water()`; with a planet's water, land food grows on land); Test `tests/test_water_map.py`,
`tests/test_water.py`.

**Produces:** `Water.distance_to_water(points) -> array`, `Water.distance_at(p) -> float` (the same answer);
`Terrain.has_water() -> bool`; `in_pond(p)` true in water (caps or map); `blocked(p, radius, swims)` true for a walker
whose body reaches water (its distance to water under its radius); `near_pond(p, width)` true on land with water within
`width`; `pond_spot(rng)` a random water point; `water_ahead(points, facing, reach, half) -> (distance, ahead, left)`
(inf when none): 5 rays across each looker's cone, only for lookers with water within reach, each ray stepping
max(the map's distance there, 0.5) until water or its reach.

- [x] Step 1: tests: `test_the_distance_map` (distance to a straight shore within a cell of the true one; one point the
  fast way equals many points, cube edges and corners included); `test_a_walker_never_enters_the_sea` (a walker made
  to walk into the sea stays on land; a swimmer goes in); `test_in_water_and_water_seen_on_a_planet` (the two senses
  and the distance on a planet with water; a rock nearer than the water is what is seen); `test_food_grows_in_the_sea`
  (the pond share of new food lands in water, the rest on land); `test_old_ponds_still_water` (a Terrain of caps only
  answers as before). Run `python -m unittest tests.test_water_map tests.test_water` → FAIL.
- [x] Step 2: the changes above. Run `python -m unittest tests.test_water_map tests.test_water tests.test_terrain
  tests.test_senses tests.test_on_sphere` → OK.
- [x] Step 3: commit "Water is water: walkers kept out, swimmers in, senses and food".

### Task 4: New planets and saves

**Files:** Modify `src/evolve.py` (new world: `make_terrain` with water), `src/saves.py` (format 28 keeps the water;
27 still resumes; `FIRST_PLANET = 27`: only a save before it is a flat world), `src/viewer.py` (Update code refuses only
a flat world, `< FIRST_PLANET`: added with the owner on 2026-11-27, "Yes, add the fix", so the owner's running world of
format 27 can still be updated); Test `tests/test_saves.py`, `tests/test_evolve.py` (small world only),
`tests/test_viewer.py`.

- [x] Step 1: tests: `test_a_save_keeps_the_water` (save, resume: the same water); `test_a_planet_save_of_before_resumes`
  (a format-27 save with ponds resumes, its ponds water); `test_a_new_planet_has_seas` (the small test world:
  water share within 0.05 of its rule); `test_a_planet_world_of_before_is_updated` (Update code on a world whose last
  save is format 27 is not refused). Run → FAIL.
- [x] Step 2: the changes. Run `python -m unittest tests.test_saves tests.test_evolve tests.test_viewer` → OK.
- [x] Step 3: commit "New planets have seas, islands and rivers; saves keep them".

### Task 5: The page shows the water

**Files:** Modify `src/viewer.py` (route `/water?run=<name>`: the watched world's water picture, PNG, 404 when none),
`src/evolve.py` (writes `water.png` in the run folder when the world starts or resumes), `src/static/planet.js`
(a sphere just above the ground with the water picture: blue seas, lakes and rivers; land clear); Test
`tests/test_viewer.py`.

- [ ] Step 1: test `test_the_page_gets_the_water` (the route gives a PNG for a world with water; planet.js loads
  `/water?run=`). Run → FAIL.
- [ ] Step 2: the changes. Run `python -m unittest tests.test_viewer` → OK.
- [ ] Step 3: in the browser pane on a new planet: screenshots (whole planet, an island close up, a river) sent to
  the owner; `bench(30)` under 33 ms.
- [ ] Step 4: commit "The page shows seas, islands and rivers".

### Task 6: Final

- [ ] Step 1: Task 11c's timing script on a new planet with 600: ms a step within 5% of before; reported.
- [ ] Step 2: `python scripts/run_tests.py` OK under 300 s.
- [ ] Step 3: a fresh reviewer reads the whole change against the spec; fixes test first.
- [ ] Step 4: the owner's world updated to the new code (Update code), a new planet started for the owner to watch.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | Half water, islands cut off, rivers with crossings | `python -m unittest tests.test_water_map` | `OK` |
| 2 | Walkers kept out of water, swimmers in, senses and food | `python -m unittest tests.test_water` | `OK` |
| 3 | Saves keep the water; old sphere saves resume | `python -m unittest tests.test_saves` | `OK` |
| 4 | The page shows the water | screenshots sent; `bench(30)` | the owner sees seas, islands, rivers; under 33 ms |
| 5 | No slower | timing script, 600 alive | within 5% of before (number reported) |
| 6 | Everything else still works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 7 | One commit per task | `git log --oneline` | Tasks 1-5 |

Anything not done or not proven is reported as such, first.
