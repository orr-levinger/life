# Life: instructions for Claude

**Read `README.md` first.** Its "The goal" section is the owner's own words and the only source of truth for what this
simulator is. Read it again before every design decision.

## The goal in one line

A living world where creatures with brains and senses get smarter from generation to generation, through reinforcement
learning during life and evolution across generations, in a world that starts easy and gets harder as they get smarter.

## Never do these (each one happened on 2026-11-21/23 and cost the owner's trust)

1. **Never replace nature with a program.** No rounds, no ranking, no "the best N are copied", no world restarts.
   Babies come from energy. If natural reproduction does not get going, make the world easier; do not swap in a
   program that picks parents.
2. **Never bend the world to fit a test.** A measurement must fit the world (give the exam its own fixed test world);
   the world's levels are never tuned to make a score reachable.
3. **Never change a rule of the world inside a long document.** Say the change alone, in one plain line, before
   anything else, and wait for the owner's yes.
4. **Never build a design without the one-minute arithmetic.** How long does a creature live? How many steps does it
   need to learn? Does anything carry over between lives? If the numbers say it cannot work, do not build it.
5. **Never chase a score away from the goal.** A higher exam ratio that needed a non-biological rule is a failure.
6. **Never leave a long run in the background.** Open the live graphs first (owner's rule, 2026-11-23).

## How to work here

- Before building anything, tell the owner three short lines: what they want, what you would change, and the one
  simple check that could prove it wrong.
- Plain words, short answers, one question at a time. Explain like to a smart person who is not a programmer.
- Specs in `docs/superpowers/specs/`, plans in `docs/superpowers/plans/`. No code without an approved spec and plan.
- Tests: `python -m unittest discover tests`. Runs: `python -m src.evolve`. Python environment:
  `C:\Users\Administrator\.venvs\life` on the Windows box.
