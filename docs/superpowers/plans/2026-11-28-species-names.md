# Latin Names for Species Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every species is called by a three-word Latin name made from nine of its genes at its birth, everywhere the
page now says "species 16" or "first life".

**Architecture:** One function on the page, `speciesNames(book)`, turns a tree of species (`species.json`, as the world
already writes it) into a name for each species, using `eats()` and `lives()` for the first word and one table,
`NAME_GENES`, for the other two. The species page and the Make tab each name the book they read.

**Tech Stack:** JavaScript (the page), Python 3.12 unittest; Node (on this box, `node` v24) runs the naming function
in one test.

**Spec:** `docs/superpowers/specs/2026-11-28-species-names-design.md` (approved by the owner, 2026-11-28: "Ok").

## Global Constraints

- Nothing in the world changes: no rule, no save, no file the world writes. Only `src/viewer.html` and its test.
- Starts after the tree-of-life plan (`2026-11-28-species-tree-water-extinct-ring.md`) has made its commits: both
  change `src/viewer.html`, and each task is one commit of its own files.
- Tests: no full life simulation; the whole suite under 300 s. The check world is half size (planet radius 18.5, 100
  first creatures); the browser pane is closed after.

## Review Focus

- A name depends only on the species' and its parent's `start` (written once, at birth) and on fixed ranges, so it
  never changes; species are named in the order of their numbers, so a new species never renames an older one.
- One place: the first word comes from `eats()` and `lives()`, the same words the card shows.

### Task 1: The names

**Files:** Modify `src/viewer.html`:
- `NAME_GENES`: one row per gene of the spec's table, in its order: `[gene, low, high, upRoot, upWord, downRoot,
  downWord]`, the ranges from `starting_rules()` (size 0.5–2, stealth 0–1, armour 0–1, vision width 30–120, vision
  reach 1–30, hearing 0–4, calls 0–10, kin 0–1, fear 0.1–100, love 0.1–100, learns 0.01–1, learning speed 1e-5–1e-2 on a
  log scale, mutations 1–13, mutation size 0.01–0.2, neurons 1–64, layers 1–3; measured:
  `python -c "from src.rules import starting_rules, gene_limits; ..."`, 2026-11-28).
- `speciesNames(book)` → a `Map` of species number → name: species in number order; each one's genes at its birth
  (`start`) against its parent's `start` (first life: each range's middle), each gene as a share of its range; the
  four that moved most (ties: the table's order; a gene missing from an old tree is not ranked); the first word by
  `eats()` (Phyto, Sarco, Panto) and `lives()` with `start.size` (mys under 0.9, therium to 1.4, saurus over;
  ichthys under 1.2, cetus from 1.2; batrachus for both); each other word a root, an "o" (none before a vowel) and a
  word; a name already given gets " II", " III" after it; a species with no `start` yet is "not yet named".
- `speciesName(id)` reads the names of the book on the Species page (`bookNames`, made when that book is read); the
  Make tab names its own book with `speciesNames(book)`.
- Italics: `<i>` on the card's title, "from", "daughters" and "Alive now"; `font-style="italic"` on the tree's
  labels; the Make list's options as plain text. The card adds, small and grey, "number N in this world".

Test `tests/test_viewer.py`.

**Interfaces:** `speciesNames(book: {species: [{id, parent, start}]}) → Map<number, string>`.

- [x] Step 1: tests first (`tests/test_viewer.py`):
  - `test_species_are_named_in_latin`: the page has `function speciesNames(` and `NAME_GENES` with 16 rows; the
    starts Phyto, Sarco, Panto and the six endings are in `speciesNames`; no "`species ${`" and no `"first life"` left
    in the page; the card says `number ${`.
  - `test_the_names_of_a_small_tree`: takes `eats`, `lives`, `NAME_GENES` and `speciesNames` out of the page as
    served, runs them in `node` on a hand-made tree, and checks the four names:
    - first life: a land plant eater of size 1, every gene at its range's middle except hearing 3.6, fear 95,
      neurons 60, learning speed 1e-5 → `Phytotherium lentospinosus cerebrauritus`;
    - species 2 (from 1): its genes, but a meat eater of size 1.5, armour 1, stealth 0.9, neurons 20 →
      `Sarcosaurus simplicoloricatus occultogiganteus`;
    - species 3 (from 1): species 2's genes → `Sarcosaurus simplicoloricatus occultogiganteus II`;
    - species 4 (from 2): species 2's genes, but gills 1, lungs 0, size 1, kin 0, mutations 12, love 10 →
      `Sarcoichthys solitomutabilis incuriopumilus`.
    The test fails loudly when `node` is missing (it is on this box: v24).
  Run `python -m unittest tests.test_viewer` → FAIL (no `speciesNames`).
- [x] Step 2: the changes. Run `python -m unittest tests.test_viewer` → OK (103 tests).
- [x] Step 3: on a half-size check world: the tree, a card, "Alive now" and the Make list show Latin names in italics
  (the list in plain text), the card "number N in this world"; screenshot; no new console error. Stop the world and
  the viewer; close the browser pane.
  Done 2026-11-28 on the 42-species check world resumed from step 4,424: for example *Pantobatrachus cerebrotacitus
  constomyops* (first life) and *Phytoichthys cerebroprofundus circumoconspicuus* "number 3 in this world"; the Make
  list the same names; console: only 404s of the tree asked for before the world's first chart point wrote it.
- [x] Step 4: `python scripts/run_tests.py` → OK under 300 s (OK in 281.6 s); commit "Species are named in Latin" (`git show --stat` →
  `src/viewer.html`, `tests/test_viewer.py`, this plan); push to `evolve-from-scratch`.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | Names made in one place, none left by number | `python -m unittest tests.test_viewer` (`test_species_are_named_in_latin`) | `OK` |
| 2 | The names are as the spec makes them | `python -m unittest tests.test_viewer` (`test_the_names_of_a_small_tree`) | `OK`: the four names |
| 3 | The owner sees them | Step 3's screenshot | Latin names on the tree, card, table and Make |
| 4 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 5 | Pushed | `git ls-remote origin evolve-from-scratch` | the last commit's hash |

Anything not done or not proven is reported as such, first.
