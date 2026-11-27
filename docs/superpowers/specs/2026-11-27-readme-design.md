# A README for the game: design

Asked by the owner on 2026-11-27: "create a well writem readme for this game with possible trouble shootings". Outline
approved: "Yes, this outline".

## The goal

One page that lets the owner (or anyone) install the simulator on a Mac or on Windows, start a world, use every part of
the page, and fix the usual problems alone; and that still holds the project's goal and rules word for word.

## What the README holds, in this order

1. **What it is**: three plain lines (creatures with brains learn during life and evolve across generations, on a
   planet with seas and islands, in a world that starts easy and gets harder).
2. **Quick start**: Mac and Windows, from `git clone` to the page at http://localhost:8765 (Python 3.12, a virtual
   environment, `pip install -r requirements.txt`, `python -m src.viewer`).
3. **Using the page**: Start, Stop, Pause and Continue; Auto and Manual and what each knob means; the planet view and
   following a creature; History, Resume from here, Update code; the Rules page.
4. **Without the page**: the command line (`python -m src.evolve ...`, `python -m src.viewer --port ... --runs ...`),
   and the tests (`python scripts/run_tests.py`, its 5-minute limit).
5. **Troubleshooting**: each item is what you see, then the fix: the page is frozen or says "Nothing is running"; port
   8765 is taken; `python3.12` is not found or TensorFlow does not install; the page is slow; every creature starves;
   an old save does not resume or Update code is refused; a world still runs old code after `git pull`.
6. **The goal, the rules of this project, the decisions made on the owner's behalf**: kept word for word as today.
7. **Where things stand**: written for today (it still says 2026-11-23).

## Not in this

- No change to any code, test or rule of the world.
- The words of sections 6 (the goal, the rules, the decisions) do not change.

## How it is proven

- Every command in the README is run on this box (Windows) and gives what the README says; the Mac commands are the
  same commands with the Mac's paths (named as not run here).
- Every button, knob and page named in the README exists in `src/viewer.html` (searched by its label).
- `git diff` shows sections 6 unchanged.
