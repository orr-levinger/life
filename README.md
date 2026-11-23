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
   and its energy is low). Whenever nobody is dying, one setting of the world is nudged a little harder, one setting at
   a time. If too many die after a nudge, that nudge is taken back. There are no fixed stages and no exam gate.
5. **Intelligence is measured, not guessed:** a standard exam against creatures that wander randomly, in a fixed test
   world that never changes. The exam only measures; it does not change the world.
6. **Every long run is watched live** as graphs that update while it runs.

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
