# Brains that learn from scratch by evolution: design

> **History only. Replaced on 2026-11-23 by `2026-11-23-living-world-design.md`; the goal is in `README.md`.**
> What this design got wrong: it dropped learning during life (the owner wants reinforcement learning and evolution
> together), it tuned the levels to the exam so the first level was not the easiest, and creatures could die from the
> first step. What it got right and is kept: the senses, the steering fix, the exam idea, the scoreboard.

Date: 2026-11-21 · Repo: `orr-levinger/life` · Branch: `evolve-from-scratch`

## Intent, in the owner's words

- "Run simulation until we are able to prove that the brain works, because so far, every time the brain took over, the
  creatures became really dumb. Either just start working in straight line, not running from predator, not chasing food."
- "Ideally, there wouldn't even be the first step where they are deterministic. I wish learning to be from scratch."
- "Make the first steps more forgiving and harsher as times move on … then evolution get time to evolve to harsher
  conditions."
- "Simulations should have zero delay between frames so we can get faster test."
- "Do small steps until we see some improvements."

The owner picked **evolution only** and **levels unlocked by performance** ("Yeah, let's do it. Test it."), and two
world rules that evolution needs: **hunters gain what they take** and **creatures die of old age** ("Let's do it").

## What we found (measured 2026-11-21, no repo code changed)

Measured with two read-only scripts that import the repo's own `World`, `Creature` and `NeuralNetwork` (kept in the
session scratchpad, not in the repo). The new scoreboard (§5) has the same measures on the new code; the old numbers are
not re-run, since today's code is replaced:
- `python measure_brain.py <seed> 1500` for seeds 0 and 1: today's simulation run headless, `Creature.decide` wrapped in
  that process only to record each decision.
- `python probe_move_head.py`: 5 fresh brains × 500 random sensed states, recording the move direction and speed each can
  output.

| | Rules driving (steps < 500) | Brain driving (after step 500) |
|---|---|---|
| Bites per creature-step | 0.39 (seed 0), 0.37 (seed 1) | 0.05 (seed 0), 0.00 (seed 1) |
| Direction of moves (1 = always the same) | 0.12, 0.10 | 1.00, 1.00 |
| Outcome | population grows to 5 (seed 1) | all dead at step 554 (seed 0), 1235 (seed 1) |

Causes, each seen in the code and in those numbers:
1. **Movement output is broken.** Direction (`2π·p[6]`) and speed (`p[7]`) come out of the same softmax as the action
   choice (`src/neural_network.py` `_map_output_to_action`), so they are small, nearly constant numbers: fresh brains
   head 40°–77° at 7–28 % of top speed whatever they sense (`probe_move_head.py`).
2. **The brain cannot sense what matters.** Its 12 inputs (`_process_sensory_inputs`) never say whether another creature
   is bigger, smaller or family, so fleeing or attacking cannot be learned.
3. **The learning cannot work.** After the handover only the last action before a bite or a death is rewarded
   (`apply_reward`, terminal steps only), never the walk that led there, and the brain always takes its top choice
   (`argmax`), so it never tries alternatives. Seed 0's brain ate 100 % of the time (misses cost energy, it died); seed
   1's rested 89 % and starved (`measure_brain.py`).
4. **It is slow.** 41 s for 3,460 creature-steps, about 12 ms per creature-step, almost all TensorFlow
   (`measure_brain.py` seed 1). How much faster a NumPy brain is: not measured yet (the plan measures it).
5. **Today's tests mostly fail already.** `python -m unittest tests.test_parent_child_relationship
   tests.test_world_and_creature tests.test_food_and_eating tests.test_continuous_movement tests.test_visualization`:
   23 run, 2 failures, 11 errors. The tests' stand-in decision functions take 2 arguments but `World.step` passes 3, and
   the family tests expect a different split. The two TensorFlow training tests were not run (minutes of training each).

## Goals

- Every brain starts random. No rules, no teacher, no fallback drives a brain creature at any time.
- Brains improve only by evolution: creatures that eat well split, and their children carry mutated copies of the brain.
- The world starts forgiving and gets harsher in levels, each unlocked when the population does well enough.
- A fast headless runner with zero delay, and a scoreboard that shows in numbers whether brains are getting smarter.
- Proof across 5 seeds (see "How we prove it works").

## Non-goals

- Learning during a creature's life (reinforcement learning). Added later only if evolution gets stuck, in its own spec.
- TensorFlow, GPUs, any machine-learning library.
- New world mechanics beyond the levels and the two rules in §2 (hunting pays, old age).

## Anti-goals (a failure even if the code runs)

- Any hand-written behavior steering a brain creature.
- A world so easy that random wandering does as well as a brain: then nothing is selected and nothing proves the brain.
  The scoreboard always compares brains with random walkers at the same level.
- Claiming progress from one lucky seed.

## Constraints

- Python 3.12 on the Windows box (`python --version`: 3.12.10) and the owner's Mac. Only `numpy` and `matplotlib`.
  Today's `requirements.txt` is a Mac freeze that does not install on Windows (`pip install -r requirements.txt` failed:
  no `tensorflow-io-gcs-filesystem==0.37.1` for Windows), so it becomes those two packages at today's pins
  (`numpy==2.1.3`, `matplotlib==3.10.3`).
- The box has 4 cores shared with other work: at most 2 simulation runs at once.
- Every run is reproducible from its seed on the same machine (Claude's call: matching bit for bit across the Windows
  box and a Mac is not a goal).

## Approaches considered

1. **Evolution only (picked).** Brains never change during a life; selection and mutation do the work. The sim already
   splits creatures and mutates brains (`Creature.split`, `NeuralNetwork.mutate`); it is simple, and the proof is clear:
   scores climb over generations.
2. **Learning during life.** Each creature updates its brain from rewards. This is what fails today; fragile and slow to
   compute. Not now.
3. **Both.** Possibly strongest, but if it improves we cannot tell which part did it. Not now.

Levels: a fixed clock was rejected, because a population that has not learned yet would be wiped out when conditions
harden. Levels unlock by performance instead (proposed by Claude, agreed by the owner).

## Design

### 1. The brain (NumPy)

- A small network: 12 inputs → 16 hidden units (tanh) → 4 outputs (tanh). Weights start random (normal, scale
  `1/sqrt(number of inputs to the layer)`). No gradient training: weights change only by mutation.
- **Inputs (senses)**, each roughly in −1..1:
  1. energy ÷ max energy
  2. size, scaled: `(size − 1.25) ÷ 0.75`
  3. standing on food (1 or 0)
  4. –7. nearest food: seen (1/0), distance ÷ sense range, cos and sin of its direction (all 0 when none is seen)
  8. –12. nearest creature that is not family: seen (1/0), distance ÷ sense range, cos and sin of its direction,
     relative size `log2(its size ÷ own size)` clipped to −1..1 (all 0 when none is seen)
- **Outputs → one action per step**:
  - outputs 1–2: a move vector, times the creature's top speed, capped at the top speed;
  - output 3, eat: when > 0, take a bite of the nearest food within eat range (or the food it stands on) instead of
    moving; with no food in reach it is a miss and costs energy, as today;
  - output 4, attack: when > 0, ≥ the eat output, and attacks are on at this level, attack the nearest non-family
    creature (today's damage and range; the attacker gains part of the energy it takes, see §2). Below the attack level
    this output is ignored.
- **Family** (parent, child, same parent) can never be attacked: a world rule, as today's tests require, not a behavior.
- **Display intent** for the visualizer's arrows is derived from what the creature did: moving within 45° of seen food →
  green, moving away from a bigger creature in view → blue, attack → red, otherwise gray. Display only, never input.

### 2. Evolution

- Splitting stays as today: at max energy a creature becomes one exact copy and two mutated children (mutation rate 0.1
  and 0.2, scale 0.2; size × (1 ± 0.5), kept within 0.5–2.0), and the parent is gone. One change (Claude's call): the
  energy is shared by the 3 children (÷ 3); today it is ÷ 4 and a quarter is lost.
- Start: 20 creatures with random brains, sizes 0.5–2.0, energy 50–100. Cap: 40 creatures (as today); a full world
  delays splits.
- **Two world rules evolution needs** (found by the builder check, approved by the owner):
  - **Hunting pays.** Today an attacker gains no energy, only a corpse worth at most about 5–6 energy, so big creatures
    are never a threat and fleeing can never be selected. Now the attacker gains 80 % of the energy it takes from its
    victim (Claude's call on the share, not measured). The corpse stays as today.
  - **Old age.** Today nobody dies of old age, so once 40 good-enough creatures fill the world, births stop and
    selection stops. Now a creature dies when it reaches its maximum age, starting at 3,000 steps (Claude's call, not
    measured; tuned in the plan so that a full world keeps renewing). Energy can no longer rise above the maximum.
- **Hall of fame:** the 10 brains whose creatures ate the most in their lifetime, saved under the run's folder.
- **Extinction:** when the population reaches 0, 20 creatures restart from mutated copies of hall-of-fame brains at the
  same level. Every restart is counted on the scoreboard.

### 3. Levels: forgiving to harsh

Starting values (Claude's call, not measured; tuned in the plan by measurement so that at each level random walkers
survive but food-chasers clearly have more children):

| Level | Food energy | Max food | Move cost (per unit of distance) | Cost per step | Attacks |
|---|---|---|---|---|---|
| 1 | 20–40 | 200 | 0.5 | 0.05 | off |
| 2 | 10–25 | 150 | 1.0 | 0.10 | off |
| 3 | 10–25 | 80 | 1.0 | 0.10 | off |
| 4 | 10–25 | 80 | 1.0 | 0.10 | on |
| 5 | 5–15 | 50 | 1.5 | 0.15 | on |

- **Unlock:** the population moves up when its **exam ratio** (§4) is ≥ 1.5 in two exams in a row, with at least 10
  creatures alive. Levels only go up.
- **Tuning target** (measured in the plan): at every level, random walkers' exam bites per creature-step stay at or
  below 0.3, so that 2 × is reachable under the one-bite-per-step ceiling.
- A new level's values apply from the next step (costs at once, food as it spawns).

### 4. Baselines (yardsticks, never teachers)

- **Random walkers:** move in a random direction at a random speed (0 to top speed) each step and eat whenever food is
  in reach. The bar a brain must clearly beat.
- **The exam** (Claude's call; it keeps crowding out of the comparison: in the evolving world each creature's share of
  food depends on how many there are, not only on how smart it is). Every 2,000 steps, up to 20 brains sampled at random
  from the living population, and 20 random walkers, each spend 500 steps in two identical copies of a fresh exam world
  at the current level: same seed (from the run seed and the exam number), same food, no splitting, no deaths. **Exam
  ratio** = the brains' bites per creature-step ÷ the walkers'. The exam never changes the evolving world.
- The toward-food score (random walkers: 0) and the flee score (random walkers: 50 %) are measured in the evolving
  world: by symmetry, crowding does not move what random walking scores on them.
- **Rules:** today's hand-written behavior (`Creature._decide_without_brain`), moved into its own policy module used only
  by baseline runs, never by brain creatures. Reported beside the brains, not required to beat.
- A baseline runs in its own world with the same seed and level table, held at one level per run.

### 5. Runner and scoreboard (zero delay)

- `python -m src.evolve --seed 1 --steps 100000 [--policy brain|random|rules] [--level N] [--window] [--out DIR]`.
  Headless by default, with no pause at all. `--window` draws with no added delay (only the minimum matplotlib needs to
  refresh the window). `src/main.py` loses its pause too.
- Every 1,000 steps, one printed line and one CSV row: step, level, population, highest generation, bites per
  creature-step, **toward-food score** (mean cosine between the move and the direction of the nearest seen food, over
  steps where food is seen and the creature moves; 1 = straight at it, 0 = random), **flee score** (% of steps with a
  bigger non-family creature in view where the creature moved away), **direction concentration** (1 = every move in the
  same direction, today's straight-line symptom; 0 = spread evenly), the latest exam ratio, deaths by hunger, by attack
  and by age, restarts.
- Runs write to `runs/<seed>-<policy>/` (gitignored).

### 6. Visualizer

Same look. Brain creatures stay purple, baseline creatures green. The number inside a creature (today's "NN score")
becomes the food it has eaten in its life; the corner's "MaxScore" becomes the most food any living creature has eaten,
and the corner adds the level.

### 7. Tests

`python -m unittest discover tests`. Tests of removed behavior (the teacher phase, REINFORCE, the brain score) are
removed; today's tests of rule behavior move to the rules baseline. New tests cover the brain (shapes, mutation, copy),
each sense, the output-to-action mapping, family protection, levels and unlocks, the extinction restart, and the
scoreboard metrics. Removed with them (no dead code): TensorFlow, `train_supervised*`, `train_reinforce`, `apply_reward`,
the 500-step teacher (`MIN_STEPS`), `get_nn_score`, the brain save at score 80, and the unused grid `VisionSensor`.

## How we prove it works

Seeds 1–5, 100,000 steps each, random brains at the start, no rules:
1. The exam ratio over the last 5 exams ≥ 2, and higher than over the first 5 exams (a ratio, so a harder level does not
   hide progress).
2. Toward-food score over the last 10,000 steps ≥ 0.7.
3. At least 4 of the 5 runs reach level 4 (attacks on), and there the flee score is ≥ 70 % (random walkers ≈ 50 %).
4. The population is alive at the end, with no restart in the last 20,000 steps.
5. The rules' numbers are reported beside the brains', not required.

The thresholds are Claude's call. If 100,000 steps turn out too short to judge, that is reported with the numbers and a
longer run is proposed; it is never claimed as a pass. After each plan step that changes behavior, one seed's scoreboard
is shown to the owner (the owner's "small steps until we see some improvements").

## Left to the builder

The hidden-layer size if 16 proves too small; mutation details within this design; the level values, tuned by
measurement; the CSV layout; vectorizing the senses with NumPy if a 100,000-step run takes over 30 minutes; unique
creature ids instead of Python's `id()` (reused, not reproducible); which eat-miss cost applies (today 1.0 and 0.2).
