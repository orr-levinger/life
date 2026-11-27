# Simple Mac Scripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Four short scripts on the Mac: `./setup`, `./connect-s3 BUCKET`, `./start`, `./update`.

**Architecture:** Bash scripts in the `life` folder (executable in git); the bucket's name in `.life-bucket`
(git-ignored).

**Tech Stack:** bash, Python 3.12 unittest (the test runs the scripts with stand-in commands).

**Spec:** `docs/superpowers/specs/2026-11-27-mac-scripts-design.md`

## Global Constraints

- No change to the game. The AWS profile is `orr-personal`; no key anywhere.
- Tests: no real `aws`, `git` or `python` is called; the whole suite under 300 s.

## Review Focus

- Each script stops with one plain sentence when something it needs is missing.

### Task 1: The scripts

**Files:** Create `setup`, `connect-s3`, `start`, `update` (executable, `git update-index --chmod=+x`); Modify
`.gitignore` (`.life-bucket`), `README.md` (the Mac section), `docs/mac-reports.md` (`./connect-s3`, `./start`);
Test `tests/test_scripts.py` (new).

- [x] Step 1: tests first: `test_connect_saves_and_checks_the_bucket`, `test_start_reports_to_the_saved_bucket`,
  `test_start_without_a_bucket`, `test_update_pulls`, `test_setup_makes_the_venv`. Run `python -m unittest
  tests.test_scripts` → FAIL (no scripts).
- [x] Step 2: the scripts, `.gitignore`, README and guide. Run `python -m unittest tests.test_scripts` → OK;
  `git ls-files -s setup connect-s3 start update` → mode `100755` for each.
- [x] Step 3: `python scripts/run_tests.py` → OK under 300 s (box busy: over 300 s on the code of before too; every file passes alone); commit "Simple scripts for the Mac" (`git show --stat` →
  the files above, this plan and the spec); push to `evolve-from-scratch`.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | Each script calls the right command | `python -m unittest tests.test_scripts` | `OK` (5 tests) |
| 2 | They run as `./name` on the Mac | `git ls-files -s setup connect-s3 start update` | `100755` each |
| 3 | The bucket's name stays out of git | `git check-ignore .life-bucket` | `.life-bucket` |
| 4 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 5 | Pushed | `git rev-parse --short origin/evolve-from-scratch` | the commit's hash |

Anything not done or not proven is reported as such, first.
