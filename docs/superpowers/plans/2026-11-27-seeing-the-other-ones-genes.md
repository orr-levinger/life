# Seeing the Other One's Genes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A creature senses the genes of the one it sees, and the genes and distance of the one that hit it.

**Spec:** `docs/superpowers/specs/2026-11-27-seeing-the-other-ones-genes-design.md`

## Global Constraints

- `N_SENSES` 47 → 84: 18 seen genes, then the hitter's distance and 18 genes, at the end of the vector.
- Genes scaled 0-1 by their ranges (learning speed on a log scale); zeros when none.
- Every brain and learner state of before loads with zero weights for the new inputs: the same choices as before.
- Tests under 300 s; never a full life simulation as a test.

### Task 1: The senses

**Files:** Modify `src/brain.py` (`N_SENSES` 84), `src/senses.py` (`genes_seen(other, rules)`; the seen one's genes;
the hitter's distance and genes), `src/creature.py` (`hurt` keeps the hitter's genes and distance); Test
`tests/test_senses.py`.

- [x] Step 1: tests first: `test_it_sees_the_other_ones_genes`, `test_it_knows_who_hit_it_and_how_far`. Run
  `python -m unittest tests.test_senses` → FAIL.
- [x] Step 2: the changes. Run `python -m unittest tests.test_senses tests.test_world_rules tests.test_on_sphere` → OK.
- [x] Step 3: commit "A creature sees the other one's genes, and knows who hit it".

### Task 2: Brains of before keep what they learned

**Files:** Modify `src/brain.py` (a 47-input brain grows zero rows), `src/learner.py` (`load_state` and the saved
weights: a 47-input state grows zero rows for `w1` and `wh`), `src/saves.py` if a saved brain is read there; Test
`tests/test_learner.py`, `tests/test_saves.py`.

- [x] Step 1: tests first: `test_a_brain_of_before_acts_the_same`, `test_a_learner_of_before_acts_the_same`,
  `test_a_save_of_before_resumes`. Run → FAIL.
- [x] Step 2: the changes. Run `python -m unittest tests.test_learner tests.test_saves tests.test_brain` → OK.
- [x] Step 3: step time before and after (the same half-size world, 150 steps): within 5 % (the owner: speed is the
  only reason to drop genes).
- [x] Step 4: `python scripts/run_tests.py` → OK under 300 s; commit "Brains of before keep what they learned"; push to
  `evolve-from-scratch`.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | Genes of the seen one and of the hitter sensed | `python -m unittest tests.test_senses` | `OK` |
| 2 | Old brains and saves act the same | `python -m unittest tests.test_learner tests.test_saves` | `OK` |
| 2b | Not slower | `task11_time.py`-style timing, 150 steps, before and after on the same seed | within 5 % (numbers reported) |
| 3 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 4 | Pushed | `git push origin HEAD:evolve-from-scratch` | pushed |

Anything not done or not proven is reported as such, first.
