# The living world: design

Date: 2026-11-23 · Repo: `orr-levinger/life` · Branch: `evolve-from-scratch`

This is the current design. It replaces `2026-11-21-evolve-brains-from-scratch-design.md` and
`2026-11-21-three-teams-design.md`, which are kept only as history.

## The goal, in the owner's words

> "To create a real evolution: brain creatures evolve over time with brain, with reinforcement learning that improve
> from generation to generation, with evolution, with sensors. And we start with easy settings that get harder and
> harder. Maybe at step 0 they even never die. And only after a certain level of intelligence they can start die."

> "I want it to be real evolution! Real life simulator! To really mimic biology!"

> "They have babies if they split! If they get energy!" · "The whole point of learners is that their babies inherit
> their brain with mutations." · "I don't know what's the right way! This is what you were here to solve!"

## What was wrong before (so it is not repeated)

- A program picked parents in "rounds" and restarted the world every 1,000 steps. That is not evolution by nature.
- A "learning only" team passed nothing on. Arithmetic that was available and ignored: the learner needed about 29,000
  steps of practice for the simplest task (measured 2026-11-23, `tests.test_learner`: 450 updates × 64 steps); a creature
  lived about 400 steps (measured: 76 deaths per 1,000 steps among 30). It scored 0.33 × random wandering.
- The first level had the least food (40 pieces against 100 at level 2), because the levels were tuned to the exam.

## The design in plain words

### One world, no program in charge

One continuous world. Nothing restarts it. Nothing ranks creatures. The only things that happen are what creatures do
and what the world's rules do to them.

### A creature

- **Senses** (as built): its own energy and size, whether it stands on food, the nearest food (seen?, how far, which
  way), the nearest stranger (seen?, how far, which way, how much bigger or smaller). Family is never a target.
- **A brain** that turns senses into one action each step: move (direction and speed), eat, or, once hunting exists,
  attack.
- **Eating is the creature's own action** (the owner, 2026-11-23: "Eat must be action creature does. It was always an
  action!!"). Choosing to eat takes one bite of food within reach; with no food in reach it is a miss that costs 0.2
  energy, as in the original simulation. An earlier version of this design made eating automatic "on contact"; that
  was Claude's change, made without saying it changed the owner's rule, and it is reverted.
- **The exam's random wanderers** move randomly and eat whenever food is within reach, as in the original yardstick.

### How brains improve

1. **Learning during life (reinforcement learning, PPO, as built):** each creature learns from its own life. Its
   reward each step is its change in energy from eating, moving, hunting and being hunted. Energy given to a baby is
   not counted as a loss. The whole walk that led to food gets the credit.
2. **Evolution across generations:** a creature whose energy reaches 70 % of its maximum has one baby beside it and
   gives it 40 % of its energy; the parent lives on. The baby gets the parent's brain **as it is at that moment,
   including what the parent has learned**, with small mutations (rate 0.1, scale 0.2), and the parent's size × (1 ± 0.2)
   and learning speed × e^N(0, 0.2).

Why learned brains are inherited (Claude's call; the one step away from strict genetics): the arithmetic. A mortal
creature lives a few thousand steps; learning from nothing needs about 29,000. If a baby started from what its parent
was born with, every generation would start over and learning could never build up. Inheriting the learned brain is what
makes "reinforcement learning that improves from generation to generation" possible. Strict-genes inheritance (the
parent's born brain only) stays as an option (`--inherit inborn`) to compare.

### The brain's size evolves (added 2026-11-23, approved by the owner: "Ok")

The owner: "Even the size should be dynamic. Like evolution should decide to add layer, remove layer, add a neuron,
remove a neuron. This also an evolution step."

- A learning animal's brain has room for up to 3 layers of 64 neurons. Genes say how many layers and how many neurons
  per layer are switched on; the first animals have 1 layer of 32 (today's size).
- At birth a mutation may switch one neuron on or off (5 % each) or one layer on or off (1 % each). These rates are
  not measured; they are tuned in the plan.
- A new neuron or layer starts silent: the baby behaves exactly like its parent at birth, and learning and later
  mutations put the new part to use. Removing one is a real change.
- **A brain costs energy:** each active neuron costs 0.0005 energy per step (not measured; 32 neurons cost about a
  third of the Garden's cost of living). Without a cost, size would drift upward for no reason.
- The scoreboard and the live page show the average number of neurons and layers.
- The instinct yardstick keeps its fixed brain. The critic (a training aid, not the brain) keeps its fixed size.
- This is built only after the Garden check has shown that learning works with the fixed brain, so that a failure there
  cannot be confused with the size machinery.

### Easy first, harder as they get smarter

Stages change only when the creatures pass the intelligence exam, never by the clock, and never go back.

| Stage | What can kill | Food | Moving and living cost | Hunting |
|---|---|---|---|---|
| 0 Garden | old age only: nobody starves | plenty, rich | cheap | off |
| 1 Mortal | hunger, old age | plenty, rich | cheap | off |
| 2 Scarcity | hunger, old age | less, poorer | double | off |
| 3 Hunters | hunger, old age, hunters | less, poorer | double | on |
| 4 Harsh | hunger, old age, hunters | little, poor | triple | on |

- **Garden:** a creature at zero energy simply stays at zero: nobody starves. Lives are long, so there is time to
  learn.
- **Old age is there from the first step** (changed 2026-11-23, approved by the owner: "Ok"). The first Garden run
  showed why: with no deaths at all the world filled to its limit of 80 by step 1,000 and not one baby was born in the
  next 14,000 steps (measured, `runs/1-learning`), so there was learning but no evolution. Each death of old age frees
  a place for a baby of a creature that has gathered the energy. It starts at 5,000 steps (not measured; tuned in the
  plan).
- **Starting numbers** (not measured; tuned in the plan): food pieces at most 200, 200, 100, 100, 50; food energy 20–40,
  20–40, 10–25, 10–25, 5–15; cost per unit moved 0.5, 0.5, 1.0, 1.0, 1.5; cost per step 0.05, 0.05, 0.10, 0.10, 0.15.
  Two rules hold through any tuning: food never increases from one stage to the next, and each mortal stage settles
  between 30 and 60 creatures by food alone.
- **How many can live:** food decides. A safety limit of 80 pauses births only to keep the simulation fast.
- **Primordial soup:** at every stage, while fewer than 10 creatures live, one random newcomer appears each step.
  Newcomers are counted; when life sustains itself, the count stops.

### The intelligence exam

Every 2,000 steps, up to 20 living creatures' brains (learning paused) and the same number of random wanderers each
spend 500 steps in identical copies of a **fixed test world that never changes** (80 food pieces of energy 10–25; there,
random wanderers score 0.24 bites per step, measured 2026-11-21 with `walker_score`). The score is the brains' bites ÷
the wanderers'. A stage is passed at 1.5 or more, twice in a row, with at least 10 alive.

### Two kinds of animal, for comparison

- **Learning animals** (the goal): PPO during life, learned brain inherited.
- **Instinct animals** (a yardstick): brains never learn; babies inherit the brain with mutations. They start at stage
  1, because without learning and without death nothing can change.

Each runs in its own world: `python -m src.evolve --seed N --steps S --species learning|instinct [--inherit learned|inborn]`.

### Watching it live

`python -m src.viewer` serves a page that redraws every 2 seconds: stage, population, births, generations, exam score,
bites, steering to food, and each run's progress. It is opened in the app's browser before a run starts.

## The one-minute arithmetic (done before building this time)

- Can a creature learn in one life? Garden: yes, nobody dies, so 29,000 steps of practice fit. Mortal stages: a life is
  at most 5,000 steps, but a baby continues from its parent's learned brain, so practice adds up over generations.
- Does anything carry over between lives? Yes: the brain, by inheritance.
- Can evolution get going? Babies need 70 % energy, not 100 %; the Garden cannot kill; the soup refills a dying world.
- Speed (not measured for this design): the learner took 6.75 ms per step for 30 creatures and 136 ms per update
  (measured 2026-11-23); with up to 80 creatures a 100,000-step run is roughly 40 minutes. The plan measures it.

## The simple check that could prove this wrong

Run learning animals in the Garden for 30,000 steps. If their exam score does not rise, creatures cannot learn from
their own lives in this world, and nothing after the Garden can work. That check comes before anything else is built on
top.

## How we know it works

Learning animals, seeds 1–5:
1. **They learn in the Garden:** every seed passes stage 0.
2. **Life sustains itself:** no newcomer in the last 20,000 steps.
3. **Generations follow each other:** the deepest generation alive at the end is at least 20.
4. **Smarter from generation to generation:** the exam score over the last 5 exams is at least 2 and above the first 5.
5. **They steer to food:** steering score over the last 10,000 steps at least 0.7.
6. **They flee hunters:** at least 4 of 5 seeds reach the Hunters stage, fleeing at least 70 % of the time there.
7. **Brain size evolves:** the average number of active neurons at the end differs from the start in every seed, and
   adding the size machinery does not lower the Garden check's exam score.
Then two comparisons: learning animals against instinct animals, and learned against strict-genes inheritance.
The thresholds are Claude's call, not measured. A failed criterion is reported as failed, with its numbers.

## Kept from what is built

Senses, the NumPy brain and its mutation, the PPO learner, movement and its costs, hunting with 80 % gain, family
protection, corpses, the exam's mechanics, the scoreboard, the zero-delay runner, the comparison table, the proof
checker, the random-wanderer and rules yardsticks.

## Removed

Rounds (`src/rounds.py`), the three teams, the `round` column, the eat decision and its miss cost, levels tuned to the
exam.

## Constraints

Python 3.12; `numpy==2.1.3`, `matplotlib==3.10.3`, `tensorflow==2.19.0` (learning animals only). At most 2 runs at once.
Same seed, same machine, same run. Zero delay.
