# Life: an evolution simulator

## The goal

In the owner's words (2026-11-23):

> "To create a real evolution: brain creatures evolve over time with brain, with reinforcement learning that improve
> from generation to generation, with evolution, with sensors. And we start with easy settings that get harder and
> harder. Maybe at step 0 they even never die. And only after a certain level of intelligence they can start die."

> "I want it to be real evolution! Real life simulator! To really mimic biology!"

Everything in this repo serves that goal. A change that raises a score but moves away from it is wrong.

## What the goal means: the rules of this project

1. **Creatures have a brain and senses.** A brain is a neural network. Nothing hand-written tells a creature what to do.
2. **Brains improve in two ways, together.**
   - **Learning during life:** reinforcement learning. A creature's brain changes from its own experience: eating is
     good, wasting energy is bad, and the whole walk that led to food gets the credit, not only the last step.
   - **Evolution across generations:** a baby inherits its parent's brain with small mutations. Creatures that do well
     have more babies, so good brains spread and get better from generation to generation.
3. **Nature selects, never a program.** A creature that gathers enough energy has a baby. No program ranks creatures,
   restarts the world, or picks parents.
4. **The world starts easy and gets harder only as the creatures get smarter.** At the start nobody dies. Once the
   creatures pass an intelligence test, hunger appears; later old age, scarcer food, and hunters.
5. **Intelligence is measured, not guessed:** a standard exam against creatures that wander randomly, in a fixed test
   world that never changes.
6. **Every long run is watched live** as graphs that update while it runs.

## Decisions made on the owner's behalf

The owner handed the "how" to Claude ("I don't know what's the right way! This is what you were here to solve!").
These are Claude's calls; each can be overruled.

- **What a baby inherits:** the parent's brain as it is when the baby is born, including what the parent has learned,
  with mutations. This is the one deliberate step away from strict genetics (in biology a baby inherits only what the
  parent was born with). It is chosen because it is what makes learning improve from generation to generation, which is
  the goal. Strict-genes inheritance stays available as an experiment to compare.
- **Eating happens on contact.** A creature touching food eats; its brain does not decide to eat. The skill to learn
  and evolve is steering to food and staying on it.
- **The learning method is PPO** (an actor that tries moves by rolling dice on its own odds, and a critic that judges
  how good each moment is).
- **A primordial soup:** while very few creatures live, random newcomers keep appearing. Once life sustains itself,
  none are needed, and that moment is part of the proof.

## Where things stand (2026-11-23)

Branch `evolve-from-scratch`. Built and tested: the 12 senses, a NumPy brain, a PPO learner in TensorFlow, the world's
movement, eating, hunting and family rules, the exam, the scoreboard, a zero-delay runner, a comparison table.

Not yet matching the goal, to be replaced by `docs/superpowers/specs/2026-11-23-living-world-design.md`:
- reproduction runs in program-made "rounds" (the 8 best eaters are copied every 1,000 steps), not by energy;
- the levels are tuned to the exam, so the first level is not the easiest;
- creatures can die from the first step; there is no old age;
- eating is a brain decision;
- a "learning only" team exists that passes nothing on (it scored below random wandering).

## Run and watch

Design and plans live in `docs/superpowers/`. Read the newest design before changing anything.
