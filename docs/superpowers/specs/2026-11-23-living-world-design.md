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

### The difficulty dial (replaces the five stages; the owner, 2026-11-23)

The owner: "Only animals with low energy should die." · "When no creature dies, that's the signal that we can increase
the difficulty level, either by lowering lifespan, lowering food value, making less food." · "Maybe one at a time. If we
see that when we reduce something, the survival rate goes down too far, we should increase it again. … nudge the
parameters of difficulty very delicately and be prepared to raise them when they die too much. The hardness of survival
is supposed to drive intelligence. Okay."

- **Who dies:** only a creature with low energy. It dies when its energy runs out ("hunger"), or when it is past its
  own lifespan and its energy is below 30 % of its maximum ("old age"). A well-fed creature never dies. Hunters, once
  they exist, kill by taking energy.
- **A lifespan is a range, not one number** ("Life span should be in a range"): each creature draws its own when it is
  born, between 4,000 and 6,000 steps at the start.
- **The world's settings** start easy: at most 200 food pieces of energy 20–40, 3 new food candidates per step, moving
  costs 0.5 per unit, living costs 0.05 per step, no hunters.
- **The dial:** every 1,000 steps the world looks at who died in that window.
  - **Nobody died:** survival is too easy. One setting is nudged about 5 % harder, one setting at a time, in turn: less
    food at once, less energy per food, costlier moves, costlier living, shorter lifespans, less new food. After 24
    nudges the next one switches hunters on.
  - **Too many died** (more than a quarter of the creatures alive, or random newcomers had to be added): the last nudge
    is taken back. If it happens again, the one before it, and so on down to the easy start.
  - **Some died, not too many:** nothing changes. This is where survival is hard enough to select.
- The dial's level (how many nudges are in force) and every setting are on the scoreboard and the live page.
- All numbers here (5 %, 1,000 steps, a quarter, 30 %, 24, the starting settings and each setting's limits) are
  Claude's starting values, not measured; the plan tunes them by measurement.
- **Primordial soup:** while fewer than 10 creatures live, one random newcomer appears each step.

### The intelligence exam

The owner: "At the end, whoever survives is smart in nature." · "Intelligence should be rewards / waste or something
similar."

Two gauges, both on the live page:
1. **The difficulty level the animals survive at** (the dial, above). This is nature's own measure: the level only
   rises while nobody dies.
2. **Intelligence = energy earned ÷ energy spent.** Earned: by eating and hunting. Spent: on moving, on living, on
   eating at nothing, on attacking. Above 1 a creature gains more than it burns.
   - **In the exam:** every 2,000 steps, up to 20 living creatures' brains (learning paused) and the same number of
     random wanderers each spend 500 steps in identical copies of a **fixed test world that never changes** (80 food
     pieces of energy 10–25). The creatures' number is `intelligence`; the wanderers' is the yardstick beside it.
     Because the test world never changes, exams compare across time.
   - **In the living world:** the same division over each 1,000-step window (`efficiency`). It moves with the dial,
     because a harder world pays less.

The exam only measures; it does not change the world.

### Two kinds of animal, for comparison

- **Learning animals** (the goal): PPO during life, learned brain inherited.
- **Instinct animals** (a yardstick): brains never learn; babies inherit the brain with mutations. Both kinds start at
  the dial's easy start.

Each runs in its own world: `python -m src.evolve --seed N --steps S --species learning|instinct [--inherit learned|inborn]`.

### Watching it live

`python -m src.viewer` serves a page that redraws every 2 seconds. It is opened in the app's browser before a run
starts. The owner: "If not it should show nothing running." · "Only show me charts when it's running." · "And let's
save history runs." · "Add chart of average energy per creature?"

- **Live:** the charts of the runs in progress. With no run in progress it shows no chart, only "Nothing is running".
- **History:** every past run is kept (each run writes its own dated folder); a click on one shows its charts.
- **The charts:** difficulty level, intelligence (with the random wanderers beside it), energy earned ÷ spent in the
  living world, average energy per creature (as a share of its maximum, with the 70 % "has a baby" and 30 % "the old
  die below this" lines), deaths, creatures alive, steering to food, generations, bites.

## The one-minute arithmetic (done before building this time)

- Can a creature learn in one life? Garden: yes, nobody dies, so 29,000 steps of practice fit. Mortal stages: a life is
  at most 5,000 steps, but a baby continues from its parent's learned brain, so practice adds up over generations.
- Does anything carry over between lives? Yes: the brain, by inheritance.
- Can evolution get going? Babies need 70 % energy, not 100 %; the Garden cannot kill; the soup refills a dying world.
- Speed (not measured for this design): the learner took 6.75 ms per step for 30 creatures and 136 ms per update
  (measured 2026-11-23); with up to 80 creatures a 100,000-step run is roughly 40 minutes. The plan measures it.

## The simple check that could prove this wrong

Run learning animals in the easiest world for 30,000 steps. If their exam score does not rise, creatures cannot learn
from their own lives in this world, and nothing after it can work. That check came before anything else was built on
top (run 2026-11-23, when the easiest world was the "Garden" stage: the score rose from 0.55 to 1.22 times the
wanderers').

## How we know it works

Learning animals, seeds 1–5:
1. **Survival got harder:** every seed ends at dial level 10 or more.
2. **Life sustains itself:** no newcomer in the last 20,000 steps.
3. **Generations follow each other:** the deepest generation alive at the end is at least 20.
4. **Smarter from generation to generation:** intelligence over the last 5 exams is at least twice the random
   wanderers' and above the first 5 exams'.
5. **They steer to food:** steering score over the last 10,000 steps at least 0.7.
6. **They flee hunters:** in at least 4 of 5 seeds the dial switches hunters on, and creatures flee at least 70 % of
   the time while hunters exist.
7. **Brain size evolves:** the average number of active neurons at the end differs from the start in every seed, and
   adding the size machinery does not lower intelligence or the level reached.
Then two comparisons: learning animals against instinct animals, and learned against strict-genes inheritance.
The thresholds are Claude's call, not measured. A failed criterion is reported as failed, with its numbers.

## Kept from what is built

Senses, the NumPy brain and its mutation, the PPO learner, movement and its costs, hunting with 80 % gain, family
protection, corpses, the exam's mechanics, the scoreboard, the zero-delay runner, the comparison table, the proof
checker, the random-wanderer and rules yardsticks.

## Removed

Rounds (`src/rounds.py`), the three teams, the `round` column, levels tuned to the exam, the five stages and the exam
as a gate (replaced by the dial), the "bites compared with wanderers" score (replaced by energy earned ÷ spent).

## Constraints

Python 3.12; `numpy==2.1.3`, `matplotlib==3.10.3`, `tensorflow==2.19.0` (learning animals only). At most 2 runs at once.
Same seed, same machine, same run. Zero delay.
