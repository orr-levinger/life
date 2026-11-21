# Three teams: evolution, learning, and both: design

Date: 2026-11-21 · Repo: `orr-levinger/life` · Branch: `evolve-from-scratch`

This design replaces the evolution part of `2026-11-21-evolve-brains-from-scratch-design.md` and adds learning. What
that design built and this one keeps is listed under "Kept as built".

## Intent, in the owner's words

- "I did want to use machine learning to improve learning as well. How can we enjoy from both worlds? … combine it with
  evolution, like in nature, because brains do learn in nature … They should just not only check the last reward."
- "We should use the best algorithm possible for this type of life simulator."
- "Let's build it. One simulation with no TensorFlow. The one you said will be super fast. One with TensorFlow and one
  both. Let's see what works best."

The owner agreed to PPO for learning during life and evolution in rounds ("Okay, let's build it").

## Why the change (measured 2026-11-21)

Evolution by splitting at full energy never started: over 20,000 steps the deepest family line reached generation 1, with
22 extinctions (`python -m src.evolve --seed 1 --steps 20000 --max-level 1`). Gentler costs, free eat misses, more food,
and splitting at 60 % of full energy (each tried in-process on the same runner, no repo change) reached generation 6 at
best. A population of 20–40 that must fill its energy before having babies gets too few generations.

## Goals

- Three teams on the same world, levels, exam and scoreboard:
  - **Evolution** (no TensorFlow): brains never learn during life; rounds pick parents. The fast one.
  - **Learning** (TensorFlow): each creature learns during its life with PPO; no inheritance.
  - **Both**: rounds pick parents, and each creature also learns during its life with PPO.
- A comparison of the three on equal terms: same seeds, same number of steps, same exam.

## Non-goals

- One shared brain for all creatures (each creature has its own brain, as in nature).
- Passing learned weights to children (children get the parent's inborn brain; see "Both").
- NEAT or evolving the network's shape.

## Anti-goals

- A team that "wins" because it was given an easier world: all teams use the same levels, food and exam.
- Calling a team better from one seed.

## Approaches considered

- **Learning algorithm:** PPO (picked: an actor picks moves by rolling dice on its own odds, a critic judges how good each
  moment is, so every step of a walk gets credit; updates are clipped so learning never jumps; the standard for games and
  robots, and what OpenAI's hide-and-seek agents used). REINFORCE with baseline (the owner's inspiration video, Mutual
  Information, "Policy Gradient Methods | Reinforcement Learning Part 6") is the same family but noisier and less stable.
  Q-learning (DQN) needs a fixed list of moves; our steering is continuous.
- **Evolution:** rounds (picked; a generation every round, about 100 in a 100,000-step run). Splitting at full energy was
  tried and measured (above).

## Design

### Kept as built (Tasks 1–8 of the first plan)

The world and its rules (step and move costs, bites, eat misses, hunting with 80 % gain, family protection, corpses), the
12 senses, the 5 levels and their unlock rule (exam ratio ≥ 1.5 twice in a row, ≥ 10 alive), the exam against random
walkers, the scoreboard, the runner's zero delay and CSV, the hall of fame, the rules and random-walker baselines, the
visualizer. Level 1 keeps 40 food and level 2 keeps 100, measured so random walkers score ≤ 0.3 at every level
(`walker_score` seed 1: 0.243, 0.261, 0.238, 0.238, 0.088).

### Removed

Splitting at full energy, old age (`MAX_AGE`), and the extinction restart: rounds give every generation its births and
its end. The scoreboard loses `deaths_age`; `max_generation` becomes the round number for the round-based teams.

### Rounds (Evolution and Both)

- 30 creatures per round, 1,000 steps per round, in a fresh world at the current level (food prefilled as in the exam).
  Creatures can die of hunger or attack during the round; a dead creature stops scoring.
- **Fitness** = food eaten during the round.
- The next round: the 8 fittest keep their inborn brain and size unchanged; 22 children are mutated copies of the 8
  (taken in rank order, best first, round-robin): weights mutated with rate 0.1, scale 0.2; size × (1 ± 0.5) within
  0.5–2.0.
- Every round's fittest are offered to the hall of fame.

### Evolution team (no TensorFlow)

The NumPy brain built in Task 1 (12 → 16 → 4, decided as today: move, eat or attack). Never learns.

### The learner (Learning and Both, TensorFlow)

- Each creature has its own actor-critic network: 12 senses → 32 hidden (tanh) → actor heads (a choice among move, eat,
  attack; a move direction and speed as a normal distribution with a learned spread) and a critic head (how good this
  moment is).
- **Reward** each step = the creature's change in energy (a bite +1, costs negative). Death ends its experience.
- **PPO**: every 64 steps a creature updates its own network from its own last 64 steps: rewards discounted (γ 0.99) and
  credited with generalized advantage estimation (λ 0.95), clip 0.2, 4 passes, value weight 0.5, entropy weight 0.01.
  Learning rate 3e-4 for the Learning team; in Both each creature's learning rate is inherited and mutated (× e^N(0, 0.2)),
  so evolution can tune how fast creatures learn.
- **Speed:** all creatures' networks are stored together and run in one TensorFlow call per step, and updated together;
  each keeps its own weights and its own optimizer state. Its speed is not measured yet; the plan measures it.

### Learning team

One continuous world, 30 creatures; when one dies, a newborn with a fresh random network takes its place (no
inheritance). Creatures keep learning as long as they live.

### Both team

Rounds as above. Each creature starts the round with its inborn network and learns during the round. Children inherit
the parent's inborn weights (mutated) and learning rate, not what the parent learned, as in nature.

### Exams with learning brains

The exam samples up to 20 living creatures and uses their current weights, acting as they do (rolling their dice), with
learning paused. The evolution team's brains are exam-ed as before.

### Runner

`python -m src.evolve --seed N --steps 100000 --team evolution|learning|both` (baselines keep `--policy random|rules`).
Output `runs/<seed>-<team>/scoreboard.csv` and the run's wall time.

### Comparison

`python -m src.compare runs/...` prints one table: per team, the mean over seeds of the exam ratio over the last 5 exams,
the toward-food score over the last 10,000 steps, the flee score at level ≥ 4, the highest level reached, and minutes per
run. The proof criteria of the first design are reported per team by `src.check_proof`.

## How we know what works best

Seeds 1–5, 100,000 steps each, for all three teams (fewer seeds proposed if the TensorFlow teams prove too slow, with the
measured time). The answer is the comparison table; a team "works" when it passes the first design's proof criteria (exam
ratio ≥ 2 and rising, toward-food ≥ 0.7, flee ≥ 70 % at level 4+, alive at the end). If no team passes, the table and
what it shows are reported, never claimed as a pass.

## Constraints

- `requirements.txt` adds `tensorflow==2.19.0` (installed on this Windows box with Python 3.12 on 2026-11-21, without
  the Mac-only `tensorflow-io-gcs-filesystem`).
- At most 2 runs at once on the box. Every run reproducible from its seed on the same machine.

## Left to the builder

Network sizes if 32 proves too small; how the stored networks and optimizer state are laid out; the exact prefill.
