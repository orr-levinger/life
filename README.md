# Life: an evolution simulator

## The goal

In the owner's words (2026-11-23):

> "To create a real evolution: brain creatures evolve over time with brain, with reinforcement learning that improve
> from generation to generation, with evolution, with sensors. And we start with easy settings that get harder and
> harder. Maybe at step 0 they even never die. And only after a certain level of intelligence they can start die."

> "I want it to be real evolution! Real life simulator! To really mimic biology!"

> "What is critical to make sure is that the learning algorithm, the reinforcement learning, sees the penalties, sees
> the rewards, sees the steps it took so far, and learns. We must make sure that the learning algorithm is the best we
> can use for this life simulation. … Even the size should be dynamic. Like evolution should decide to add layer, remove
> layer, add a neuron, remove a neuron. This also an evolution step."

Everything in this repo serves that goal. A change that raises a score but moves away from it is wrong.

## What the goal means: the rules of this project

1. **Creatures have a brain and senses.** A brain is a neural network. Nothing hand-written tells a creature what to do.
   **Everything a creature does is its own action:** each step its brain chooses to move, to eat, or to attack. Eating
   is never automatic (the owner, 2026-11-23: "Eat must be action creature does. It was always an action!!").
2. **Brains improve in two ways, together.**
   - **Learning during life:** reinforcement learning. A creature's brain changes from its own experience: eating is
     good, wasting energy is bad, and the whole walk that led to food gets the credit, not only the last step.
   - **Evolution across generations:** a baby inherits its parent's brain with small mutations. Creatures that do well
     have more babies, so good brains spread and get better from generation to generation.
   - **The brain's size evolves too:** a mutation can add or remove a neuron or a layer. A bigger brain costs a little
     more energy every step, as in nature, so evolution finds the size that pays.
3. **Nature selects, never a program.** A creature that gathers enough energy has a baby. No program ranks creatures,
   restarts the world, or picks parents.
4. **The hardness of survival drives intelligence, and it is tuned delicately.** The owner, 2026-11-23:
   > "Only animals with low energy should die. … When no creature dies, that's the signal that we can increase the
   > difficulty level, either by lowering lifespan, lowering food value, making less food … slowly make survival
   > harder. … Maybe one at a time. If we see that when we reduce something, the survival rate goes down too far, we
   > should increase it again. … We should nudge the parameters of difficulty very delicately and be prepared to raise
   > them when they die too much. The hardness of survival is supposed to drive intelligence."

   So: the world starts easy. Only a creature with low energy dies (its energy ran out, or it is past its lifespan
   and its energy is low). In Auto, while the creatures are many and well fed, one setting of the world is nudged a
   little harder, one setting at a time; when they are few or hungry, the last nudge is taken back (the owner: "an auto
   mode button that changes and reduces and increases the params based on the number of creatures and their
   fedness"). In Manual the owner turns the settings by hand, live. There are no fixed stages and no exam gate.
5. **Intelligence is measured, not guessed,** in two ways. The owner: "At the end, whoever survives is smart in
   nature." · "Intelligence should be rewards / waste or something similar."
   - **The difficulty level the creatures survive at.** It only rises while nobody dies.
   - **Energy earned ÷ energy spent** (earned by eating and hunting; spent on moving, living, eating at nothing and
     attacking), in a standard exam in a fixed test world that never changes, beside creatures that wander randomly.
     The exam only measures; it does not change the world.
6. **Every long run is watched live** as graphs that update while it runs. With no run going the page says "Nothing is
   running"; every past run is kept in its History.

## Decisions made on the owner's behalf

The owner handed the "how" to Claude ("I don't know what's the right way! This is what you were here to solve!").
These are Claude's calls; each can be overruled.

- **What a baby inherits:** the parent's brain as it is when the baby is born, including what the parent has learned,
  with mutations. This is the one deliberate step away from strict genetics (in biology a baby inherits only what the
  parent was born with). It is chosen because it is what makes learning improve from generation to generation, which is
  the goal. Strict-genes inheritance stays available as an experiment to compare.
- **The learning method is PPO** (an actor that tries moves by rolling dice on its own odds, and a critic that judges
  how good each moment is).
- **A primordial soup:** while very few creatures live, random newcomers keep appearing. Once life sustains itself,
  none are needed, and that moment is part of the proof.
- **A brain costs energy:** each active neuron costs a tiny amount per step (approved by the owner, 2026-11-23). Without
  a cost, brain size would drift upward for no reason.
- **Not built yet, proposed for later: memory.** Today a creature decides from what it senses this moment; its learning
  uses its past steps, but its acting does not. A brain with memory would let it remember food that left its view.

## Where things stand (2026-11-23)

Branch `evolve-from-scratch`. Built and tested: the 12 senses, a NumPy brain, a PPO learner in TensorFlow, one living
world (births from energy, deaths only with low energy, lifespans in a range, the primordial soup), eating and hunting
as the creature's own actions, the difficulty dial, the intelligence exam, the scoreboard, a zero-delay runner, the live
page with Live and History, a comparison table and the proof checker.

Still to build (the plan: `docs/superpowers/plans/2026-11-23-living-world.md`): the brain's size evolving, tuning the
dial's numbers by measurement, the long runs and their judgement. The design:
`docs/superpowers/specs/2026-11-23-living-world-design.md`.

Play with it: run `python -m src.viewer` and open http://localhost:8765. Press **Start** to begin a world (it runs
until you press **Stop**). In **Auto** the world turns its own knobs; in **Manual** you move the sliders (food, energy
per food, new food per step, the costs of moving and living, lifespan, hunters) and the running world takes each change
within a second. The charts get a new point every 250 steps, and every change of a knob is marked on them. Past worlds
are in **History**.

A world can also be run from the command line: `python -m src.evolve --seed 1 --steps 30000 --species learning`.
