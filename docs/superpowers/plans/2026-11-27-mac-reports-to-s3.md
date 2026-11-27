# Mac Reports to S3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A world writes its state, events and errors to a small `report/` folder; the Mac's page server copies it to
the owner's S3 bucket every 2 minutes; Claude reads it here.

**Spec:** `docs/superpowers/specs/2026-11-27-mac-reports-to-s3-design.md`

## Global Constraints

- Rule `report_every` 500 (Runner). Trims: state 500 lines, events 1,000, errors 100, server.log 5,000.
- The upload uses `aws s3 sync` with the owner's profile; never a key in code, files or chat.
- Tests under 300 s: first the slowest file is split so the suite has room (no test changes what it checks).

### Task 0: Room in the test suite

**Files:** Split `tests/test_evolve.py` into `tests/test_evolve.py` and `tests/test_evolve_runs.py` (the tests that
run a world, unchanged).

- [x] Step 1: the split. Run `python scripts/run_tests.py` → OK, its time reported (it was 298.5 s).
- [x] Step 2: commit "Room in the test suite: the evolve tests in two files".

### Task 1: The report folder

**Files:** Create `src/report.py` (`Reporter`: `state(world, dial, step)`, `event(kind, **what)`, `error(text)`;
trimmed writes); Modify `src/evolve.py` (a line every `report_every`; events: start, resume, update, stop, Auto's
nudges; any error written before the world ends), `src/species.py` (`look` returns its splits and endings, as events),
`src/rules.py` (`report_every`); Test `tests/test_report.py` (new).

- [x] Step 1: tests first: `test_a_run_writes_its_state_every_500_steps`, `test_a_split_and_a_nudge_are_events`,
  `test_an_error_is_written_with_its_traceback`, `test_the_files_are_trimmed`. Run → FAIL.
- [x] Step 2: the changes. Run `python -m unittest tests.test_report tests.test_species tests.test_evolve_runs` → OK.
- [x] Step 3: commit "A world reports its state, events and errors".

### Task 2: The upload, and the server's log

**Files:** Create `src/upload.py` (`Uploader`: every 2 minutes, `aws s3 sync` of each changed `report/` and
`server.log`; failures logged); Modify `src/viewer.py` (`--report-to`, `--report-profile`; `server.log` of its prints
and errors); Create `docs/mac-reports.md` (the owner's setup, step by step, with the two policies); Test
`tests/test_upload.py` (new).

- [x] Step 1: tests first: `test_the_upload_command`, `test_a_failed_upload_is_logged_not_raised`,
  `test_the_server_keeps_its_log`. Run → FAIL.
- [x] Step 2: the changes. Run `python -m unittest tests.test_upload tests.test_viewer` → OK.
- [x] Step 3: `python scripts/run_tests.py` → OK under 300 s; commit "Reports reach the owner's S3 bucket"; push.
- [ ] Step 4: the owner follows `docs/mac-reports.md`; Claude reads the first reports from the bucket here and tells the
  owner what it sees.

### Task 3: The owner's own profile (amendment, 2026-11-27)

The owner, 2026-11-27: "on my Mac and also on this machine, the AWS profile for my AWS is orr-personal. So the AWS
connection has to be to this profile." Asked "Make orr-personal the default and update the guide?": "yes".
Checked here: `aws configure list-profiles` → `orr-personal`; `aws sts get-caller-identity --profile orr-personal` →
user `sdk_user`.

**Files:** Modify `src/viewer.py` (`--report-profile` default `orr-personal`); `tests/test_upload.py` (the profile in
its commands, and the default checked); `docs/mac-reports.md` (the two IAM users and `aws configure` removed: the
bucket, then the run command); the spec `docs/superpowers/specs/2026-11-27-mac-reports-to-s3-design.md` (the same).

- [x] Step 1: test first: `test_the_server_keeps_its_log` also checks `parse_args([]).report_profile == "orr-personal"`.
  Run `python -m unittest tests.test_upload` → FAIL (`'life-reports' != 'orr-personal'`).
- [x] Step 2: the default, the guide and the spec. Run `python -m unittest tests.test_upload tests.test_viewer` → OK;
  `grep -rn "life-reports\|life-mac-writer\|life-box-reader" src tests docs/mac-reports.md docs/superpowers/specs` → nothing (the bucket's example name
  becomes `YOUR-BUCKET`).
- [x] Step 3: `python scripts/run_tests.py` → OK under 300 s; commit "Reports use the owner's own AWS profile"
  (`git show --stat` → the four files and this plan); push to `evolve-from-scratch`.

### Task 4: The world's whole history (amendment, 2026-11-27)

The owner, 2026-11-27, asked to copy a world's scoreboard by hand to show why it dies out: "just make the code why do
i need to copy anything?"

**Files:** Modify `src/upload.py` (`Uploader.once`: each run's `scoreboard.csv` and `run.json`, when changed, with
`aws s3 cp` to `<target>/<run>/`); `tests/test_upload.py`; `docs/mac-reports.md` (the table of what is sent); the
spec (the same).

- [x] Step 1: test first: `test_the_upload_command` also expects the `cp` of `scoreboard.csv` and of `run.json`
  (5 commands, then none on the second look). Run `python -m unittest tests.test_upload` → FAIL (3 commands, not 5).
- [x] Step 2: the change and the guide. Run `python -m unittest tests.test_upload tests.test_viewer` → OK.
- [x] Step 3: `python scripts/run_tests.py` → OK under 300 s (box busy: a world of the owner runs here on one core; the suite was cut at 300 s with `test_evolve_runs` and `test_world_rules` still running; run alone, both OK, 88 tests); commit "Reports send the world's whole history"
  (`git show --stat` → `src/upload.py`, `tests/test_upload.py`, `docs/mac-reports.md`, the spec and this plan); push
  to `evolve-from-scratch`.
- [ ] Step 4: the owner runs `./update`, then Ctrl+C and `./start` (the page only; the world goes on); Claude reads
  `s3://life-reports-e78c2aad/mac/20261127-094329-1-learning/scoreboard.csv` here.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 0 | Room in the suite | `python scripts/run_tests.py` | `OK in … s`, well under 300 |
| 1 | State, events, errors written and trimmed | `python -m unittest tests.test_report` | `OK` |
| 2 | Upload command right; failures logged | `python -m unittest tests.test_upload` | `OK` |
| 3 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 4 | Pushed | `git push origin HEAD:evolve-from-scratch` | pushed |
| 5 | Reports from the Mac read here | `aws s3 sync` of the bucket, here | the Mac's state and events, told to the owner |
| 6 | Default profile is the owner's | `python -m unittest tests.test_upload`; the grep of Task 3 Step 2 | `OK`; nothing found |
| 7 | The whole history is sent | `python -m unittest tests.test_upload`; after the owner's restart, `aws s3 ls s3://life-reports-e78c2aad/mac/20261127-094329-1-learning/ --profile orr-personal` | `OK`; `scoreboard.csv` and `run.json` listed |

Anything not done or not proven is reported as such, first.
