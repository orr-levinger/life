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
- **The dial as first built** (Task 4d) looked at who died every 1,000 steps: nobody died, one setting about 5 %
  harder, one setting at a time; more than a quarter died (or random newcomers were needed), the last nudge taken back,
  never easier than the start. **Its signal is replaced** by the number of creatures and how well fed they are: see
  "The control room" below, which also says why.
- The difficulty and every setting are on the scoreboard and the live page.
- All numbers here (5 %, 1,000 steps, 30 %, 24, the starting settings and each setting's ends) are Claude's starting
  values, not measured; the plan tunes them by measurement.
- **Primordial soup:** while fewer than 10 creatures live, one random newcomer appears each step.

### The control room: play with the world live (the owner, 2026-11-23)

The owner: "Create a webpage where I can play with each parameter, the ease of each step, like lifespan, energy per
food, all these things, so I can set and play with it myself and have the simulation run as I please. And I can at any
point reduce any and change any param at real time. … I want to be able to play in real time live and see all the
charts, how they are changing based on my change. It should be the same web app that you're presenting to me when it's
automatic. We should have like an auto mode button that changes and reduces and increases the params based on the
number of creatures and their fedness. And there should be manual mode where I can tweak, and then steps just keep
running and running forever until we stop it."

**On the same page, in Live:**
- **Start** a world: learning or instinct animals, a seed, Auto or Manual. It runs until **Stop** is pressed; there is
  no step limit. At most 2 worlds run at once. A stopped world stays in History.
- **Seven knobs**, each a slider that shows its value now: food pieces at most, energy per food, new food per step, the
  cost of moving, the cost of living per step, lifespan, and hunters on or off.
- **Manual:** the owner moves a knob and the running world takes the new value within 25 steps (under a second).
  Nothing but the owner changes a knob.
- **Auto:** the world turns its own knobs (the rule below). The sliders move by themselves and cannot be dragged.
  Switching to Manual keeps the values of that moment; switching back to Auto goes on from wherever the knobs are.
- **The charts get a new point every 250 steps** (before: every 1,000), so a change shows within seconds. Every change
  of a knob is a thin vertical line on every chart; pointing at it says what changed ("energy per food 30 → 24, by you").
- **Difficulty is one number that works in both modes:** how many 5 % nudges harder than the easy start the world is
  (each knob's distance from its starting value, counted in 5 % steps; hunters count as one). Below zero means easier
  than the start. With Auto alone from the start it is exactly the number of nudges in force.

**Auto's rule.** Every 1,000 steps:
- **Many and well fed** (60 or more creatures, and their average energy at least 50 % of full): one knob is nudged 5 %
  harder, one knob at a time, in turn: less food at once, less energy per food, costlier moves, costlier living,
  shorter lifespans, less new food. At difficulty 24 the next nudge switches hunters on.
- **Few or hungry** (fewer than 30 creatures, or average energy below 35 %, or random newcomers had to be added): the
  last harder nudge is taken back. With none to take back, one knob is nudged 5 % easier, in turn. Auto may go easier
  than the start.
- **Anything else:** nothing changes.

Each knob has an easiest and a hardest value it never passes; they are the ends of its slider.

**Auto aims at 64 creatures, stepping by the distance and the rate** (the owner, 2026-11-24: "the ease and hardening
should be somewhat relative to the rate of energy and creature amount so we get to the sweet spot faster" · "it's NOT
about total amounts, it's about rate of change!" · "it should be based on derivatives" · "like alpha in gradient
descent" · "we should always be below the max amount of creatures... maybe 80%" · "80 creatures means too easy, 40 too
hard, we are trying to get to 80% so life is hard but not too hard"). This replaces the fixed thresholds above.
- **The target:** 64 creatures alive, 80 % of the room for 80. At 64 life is hard but not too hard.
- **Every 500 steps Auto takes one step,** in nudges (+ harder, − easier):
  step = (alive − 64) ÷ 8 + (rate of the creatures) ÷ 10 % + (rate of the energy, only when it falls) ÷ 5 points.
  The rates are per 1,000 steps, each the slope of a straight line through the last 1,000 steps' points (4 points), so one
  odd point does not swing it. Like gradient descent, the step is a fixed factor (α) times what is measured; like
  clipping, it is rounded and never more than 4 nudges either way.
- **What it does:** 80 alive and steady → 2 harder. 40 alive and steady → 3 easier. 64 and steady → nothing (the sweet
  spot). 50 alive but rising 20 % a 1,000 steps → nothing (it is recovering by itself). 80 → 45 alive in 1,000 steps with
  energy 82 % → 36 % (the owner's crash of 2026-11-23) → 4 easier at once, where the old rule held.
- Easier steps take back Auto's own harder nudges first, the newest first. Random newcomers needed → 4 easier.
- The numbers (64, 8, 10 %, 5 points, 500 and 1,000 steps, 4) are Claude's, not measured; Task 6 measures them.

**Why not "nobody died" any more.** Measured 2026-11-23 on the first dial run (learning animals, seed 1, steps
1,000–3,000): the world was full (80 creatures) and gaining energy (earned ÷ spent 1.11–1.20), yet 41–62 creatures
died of hunger in every 1,000 steps while 62–82 were born. A full world always has deaths, so "nobody died" never
came and the dial stayed at 0. How many live and how well fed they are say directly whether life is easy.

The numbers (60, 30, 50 %, 35 %, 5 %, 1,000 steps, 25 steps, 250 steps, each knob's ends) are Claude's starting values,
not measured; the plan tunes them by measurement.

**Underneath.** The page sends Start, the mode, a knob's value and Stop to the viewer's server. The server listens on
this machine only and accepts only JSON requests, which another website cannot send to it. Start launches one
`python -m src.evolve … --forever` process. The mode, the knobs and Stop are written to that run's own `control.json`,
which the run looks at every 25 steps. Every value from the page is checked against the knob's ends before it is
written; a value outside them is refused in plain words.

### Live tiles beside the knobs (the owner, 2026-11-23)

The owner: "it's weird that there is an entire empty section because of the knobs taking half the screen" · "also add
some live metrics that change over time, the screen looks too static, something like [tiles] for the interesting
metrics that update live".

- In Live, each running world is one wide panel: its knobs on the left, **nine live tiles** on the right (on a phone,
  the tiles go under the knobs): step (and steps a second), alive, average energy, intelligence × random (and where it
  started), difficulty, steers to food, born and died in the last 1,000 steps, deepest generation. Each tile explains
  itself on hover like History's numbers. Under the tiles, a small line of the average energy over the last minutes.
- **They change every second:** the world writes a small `now.json` (step, alive, average energy, food on the ground,
  births and deaths so far) every 25 steps, as it already looks at its control file; the page asks for it every second
  (`GET /now?run=`). Tiles that only change with the scoreboard (intelligence, steering, difficulty) change with it.
  A tile shows which way its number moved since the last 1,000 steps (green better, red worse).

### Jump into the world (the owner, 2026-11-23)

The owner: "add a jump into world button that show me the world in slower motion so can actually see what the creatures
are doing then jump out back to fast mode".

- Each running world's card gets **Jump in**. It opens a picture of that same world (not a copy) above the charts and
  slows the world to **10 steps a second**, with a speed slider from 1 to 30. **Jump out** closes the picture and the
  world goes back to full speed at once.
- **The picture:** the 50 × 30 world; each creature a circle as big as its size, filled as full as its energy, with a
  short line showing where it is heading; its outline says what it did this step (eat, eat at nothing, attack, move);
  food as green dots as big as their energy. The step number, how many live, and a legend.
- **Slowing never changes what happens:** the world only waits between steps, so the same seed gives the same world
  watched or not.
- **Never stuck slow:** the page renews the slow mode every 2 seconds while it is open; when it stops renewing (the
  tab is closed, the laptop sleeps), the world goes back to full speed by itself 10 seconds later.
- The charts keep updating, slowly (a point every 250 steps, about every 25 seconds at 10 steps a second).

**Underneath.** The control file gets `watch_until` (a time) and `watch_speed` (steps a second). While a world is
watched it looks at the control file every step and, after each step, writes `view.json` (where every creature and
food is) the same safe way as the control file; the page asks the server for it 10 times a second (`GET /view?run=`).

### Saves, resuming from any save, and Pause (the owner, 2026-11-23)

The owner: "we should be able to save the state of the world, update the code, update the UI and resume with the level
of intelligence we arrived at" · "let's auto save the game every 2000 steps so I can jump into how the world looked
like at this stage of evolution in the past" · "basically just the every 2000 steps save is enough actually" · "load
world simulation A at step 16k for example, I should be able to" · "we need also a pause button".

- **A save every 2,000 steps**, by itself, in every world, kept for good (no Save button: the owner said the automatic
  save is enough). A world also saves when it is stopped.
- **What a save holds:** everything that makes the world what it is at that step: every creature (where it is, its
  size, energy, age, own lifespan, generation, parent, how much it has eaten), every brain as it is now (a learning
  animal's learned network, the network it was born with, its learning speed and its learning state), every piece of
  food, the knobs and Auto's state (the nudges it can take back, whose turn it is), the counters, and the random number
  generators' state. Only a learning animal's last few unfinished steps of experience (fewer than 65) are not kept: it
  starts its next batch of learning afresh.
- **Resume from a save:** in History or on a running world's card, pick the world and one of its saves (2k, 4k …
  16k …) and press **Resume from here**, in Auto or Manual. A new world starts from that save, with today's code and
  page, and runs until Stop. Its charts begin with the old world's history up to that step, so the story is continuous;
  its name says where it came from ("from seed 1 at 16k"). The world it came from is never changed.
- **Pause:** a running world's card gets **Pause** / **Continue**. Paused, the world takes no steps (nothing changes,
  nothing is learned) until Continue; Jump in still shows it, frozen.
- **Code changes:** each save says which save format it is. A change of code that changes what a save holds (the
  brain's size evolving, Task 5, will) also brings the conversion of older saves, written and tested with it. A save
  the code cannot read is refused in plain words; it is never half-loaded or guessed.
- **Size,** measured 2026-11-23: a learning animal's network has 1,032 numbers; a learning world's save is about
  1.3 MB before compression (80 animals × 4 copies × 4 bytes), so 100,000 steps of saves is about 65 MB.

### History: one row per world, with its scores (the owner, 2026-11-23)

The owner: "what are the numbers, what was the score, this view gives very little value and info" · on the mockup:
"yes, row per run, hovering over a number should explain it, I have no idea what 0.9x random means".

- **One row per past world**, best first ("Smartest first"; also "Hardest world survived" and "Newest first"); the
  smartest has a "Smartest" mark. A click opens its charts and its saves below, as today.
- **Each row:** the world (animals, seed, when it started, where it was resumed from), how many steps, **hardest
  difficulty survived** (and where it ended), **intelligence × random** (and where it started), **steers to food**,
  the deepest generation, how many were alive and how well fed at the end, Auto or Manual, and how many saves.
- **Intelligence × random** = the creatures' energy earned ÷ spent in the test, divided by the random wanderers' in
  the same test (the last exam; "was" = the first exam). Above 1× they get more out of each bit of energy than
  wandering would; below 1× less. Steering is averaged over the last 1,000 steps.
- **Every number explains itself on hover,** in plain words with that world's own numbers, for example: "In the test,
  these creatures got 0.35 energy back for each 1 they spent; creatures wandering at random got 0.40. That is 0.9×:
  10 % worse than wandering at random. Above 1× means smarter than random." Good numbers are green, bad ones red.
- The same explanations on the numbers of a running world's card in Live.
- **Sorting and filters** (the owner: "add sorting and filters to history"): a click on any column's title sorts by it
  (a second click reverses it; an arrow shows which); the Sort list stays as a shortcut. Filters above the rows:
  animals (all, learning, instinct), mode (all, Auto, Manual), "at least" a number of steps (any, 1k, 5k, 10k, 50k),
  "only smarter than random", "only with saves". A line says how many of the worlds are shown, with "Clear filters".
  The page remembers the sort and filters in this browser.

### A running world takes new code, and what Jump in shows (the owner, 2026-11-24)

The owner: "we should have this decoupled so a running simulation can handle new updated code" · "if it's too hard,
pause the running one, take a snapshot, fix the code and resume my run" · "it's in 33K steps, I don't want to lose it" ·
"motion of creatures should be smooth" · "now that Hunters is on we need a new chart showing running away from bigger
creatures and chasing smaller creatures" · "creature size is fixed! they don't shrink when they lose energy … an elephant
loses weight but doesn't shrink" · "why isn't the Hunters toggle editable?"

- **Update code:** a running world's card gets **Update code**. The world saves itself at the step it is at, ends, and
  goes on at once from that save **in the same run** with the code as it is now: the same card, the same History row,
  its charts unbroken (one mark "code updated" on them). Nothing of the world is lost; a learning animal's last few
  unfinished steps of experience (fewer than 65) start afresh, as with any save. A world the page server did not start
  (from the command line) can be updated the same way. The page server itself can be restarted without stopping any
  world: worlds do not belong to it.
- **Smooth motion in Jump in:** the picture glides each creature from where it was to where it is between two frames,
  instead of jumping. (How far a creature may move in one step is unchanged.)
- **Size never changes:** a creature's circle is always as big as its size (its genes). Its energy shows as how full the
  circle is coloured (pale: hungry, solid: full), never as a smaller circle.
- **The hunting chart:** "Fleeing and chasing": when a stranger at least 1.5 times bigger is in sight, the share of
  moves away from it (as today's flee score); when a stranger at most two-thirds its size is in sight, the share of
  moves toward it (new: `chase`), and attacks per 1,000 steps. Shown while hunters are on.
- **The hunters switch works in Auto too:** the owner may switch hunters on or off in either mode; in Auto it stays as
  set until Auto's own turn to switch it at difficulty 24.

### How a creature looks in Jump in (the owner, 2026-11-24)

The owner: "we should give creatures a distinct look based on their age, intelligence, size, energy" · "create a mockup"
· on the mockup: "Yes, build it".
- **Size:** the circle, from its genes; it never changes.
- **Hunger:** the body colour, red empty → yellow half → green full enough for a baby.
- **Age:** an arc around it, filling as it lives through its own lifespan; a dotted outer ring once it is past it.
- **Intelligence:** two eyes looking where it heads; small and grey below random wanderers, larger and white the
  smarter it is (its own energy earned ÷ spent over its life, divided by the random wanderers' in the latest exam).
  Each creature now counts its own energy earned and spent.
- **Heading and action:** the line shows where it heads; a ring flashes green when it eats, amber when it eats at
  nothing, red when it attacks.
- **Food** is a small tan diamond, so it is never mistaken for a full creature.
- **Pointing at a creature** shows its numbers: size, energy, age of lifespan, intelligence, generation, what it did.
- Saves become format 3 (each creature's energy earned and spent); older saves are converted, those counts starting
  at zero from the conversion on.

### Quarter speed (the owner, 2026-11-24: "Yes, quarter speed")

A creature's top speed was 1 ÷ its size per step, so the smallest crossed 2 units a step while it can only eat within
about 1, and kept overshooting food. Now every creature moves at most a quarter of that: `SPEED` 0.25 ÷ size (the
smallest 0.5 a step, the biggest 0.125). A step is a smaller slice of time; moving costs the same per distance, so energy
is unaffected; crossing the world takes four times as many steps. Lifespans and every other number in steps stay as
they are.

### Sprinting (the owner, 2026-11-24)

The owner: "bigger are slower than smaller right now right? bigger will never catch a smaller that runs away. We need
to better mimic nature. Speed can be increased for short bursts of time at the cost of more energy, both for running
away and chasing. A creature can decide how fast to run to flee or chase, up to some max speed based on size, and energy
should reduce relative to the speed chosen."

- **Cruising speed** stays 0.25 ÷ size a step (small 0.5, big 0.125); moving at or below it costs what it costs today
  (the cost of moving × distance).
- **Top (sprint) speed** grows with size, as in nature, where big hunters out-sprint small prey: 0.4 × √size a step, and
  never below cruising speed. Size 0.5: 0.5 (it is already fast); size 1: 0.4; size 2: 0.57.
- **A creature chooses its speed** each step, anywhere from standing still to its top speed: its brain's move is now
  scaled to its top speed instead of its cruising speed.
- **Faster than cruising costs more, steeply:** the cost of moving × distance × (speed ÷ cruising speed)². So a big
  creature sprinting at 0.57 pays about 21 times the cruising cost per unit: about 6 energy a step at today's costs, of
  its 200; a short burst, not a way of life. At or below cruising speed nothing changes.
- Random wanderers (the exam's yardstick) keep moving at cruising speed or less.
- The numbers (0.4, √size, the square) are Claude's, not measured; Task 6 measures them.

### Learning from what others live (the owner, 2026-11-24)

The owner: "fleeing and chasing seems to not be learned at all" · "how does death cost help, the creature is dead? Let's do
it so everyone who sees the battle gains from it. If a creature sees another small creature die, it somehow learns from
it. In general let's add the ability to learn from others, for every penalty and reward" · watchers half as strongly as
from their own life: "Yes, rewrite it".

**What was measured (save 152,000 of the owner's world, 1,000 steps run in memory, outside `runs/`):** with a creature
1.5 times bigger in sight, 19 % of steps moved away and 15 % toward (49 % ate, 17 % attacked); with one two-thirds its
size or smaller, 17 % moved toward and 19 % away. Both are what random moves give: nothing is learned.

**Why:** a learner's reward each step is only its change in energy (`World._teach`), and it learns only from its own
life. Dying costs nothing, and could not teach the one who died anyway: its brain is gone, and its last unfinished
stretch of steps (fewer than a rollout's 64) is thrown away. A baby is neutral. A hit takes 5-10 energy. So fleeing,
which costs moving energy now, never pays.

**The change:**
- **Dying costs a full belly:** a learner that dies (of hunger, an attack or age) gets, on its last step, its change in
  energy minus its maximum energy (100 x size).
- **A baby pays a full belly:** a learner that has a baby gets its maximum energy added to its next step's reward. The
  energy it gives the baby stays no penalty, as today.
- **Every learner learns from what it sees others live:** each step of a learner (what it sensed, what it did, what it got,
  death and baby included) is also a lesson for every other learner within its sense range (3 units) at that step, family
  included. A watcher learns from it half as strongly as from a step of its own life (`WATCH_WEIGHT = 0.5`). So a creature
  that sees a small one killed beside a big one learns that staying there is bad, without dying itself.
- How a watcher learns from another's step: it judges the step with its own brain (its own guess of how good the place
  was and is), and weighs it by how likely it would have done the same itself, never more than its own step would count.
  Its lessons are kept until its own next learning round (every 64 of its steps) and used in that round.
- Only the learning changes. The world's rules, energy, the scoreboard and the exam are as they are. Saves keep no
  unfinished lessons, as today they keep no unfinished experience.
- The numbers (a full belly both ways, half as strongly) are Claude's, not measured; Task 6 measures them.
- **Cost, measured** (`scratchpad/diag/density.py` on save 156,000): a creature has 1.5 others in sight on average (6 at
  most), so each learning round has about 2.5 times today's lessons; the world's speed is measured in the check below.

**How it is checked:** the same save, run 20,000 steps twice in memory (outside `runs/`, same seed): today's learning and
the new one. For each: with a bigger creature in sight, the share of moves away from it; with a smaller one, the share of
moves toward it, over the last 5,000 steps (50 % is random); and seconds per 1,000 steps. Both runs' numbers are shown to
the owner as they are, a fail as a fail. If chasing stays random, the next idea (not in this change) is a bigger payoff
for a kill.

### Rewards in the genes (the owner, 2026-11-24)

The owner: "learning from others should also be a gene · the how much I learn from others" · "rewards themselves should
be genetic... whoever's rewards set him for death just dies... the perfect reward and penalty should be mutable... we give
them basic sensible amounts but in time mutation can sway them".

**What was measured.** Task 4u's check (save 156,000, 20,000 steps in memory, the difficulty held, no Auto, one run
each): today's learning kept 80 alive; "Learning from what others live" with a full belly for a death and a baby crashed
to 10. Each part alone, the same way: the baby pay crashed to 12 by step 4,000; the lessons crashed to 10 by step 9,000;
the death cost changed nothing at all (the very same numbers as today's learning: a creature's last unfinished steps are
never learned from, so a death teaches only its watchers). Then the fair check: the real runner, with Auto, from the same
save to step 164,000, three seeds each: today's learning ended with 78, 80 and 73 alive, the lessons with 61, 59 and 74.
**No crash in either:** the earlier crashes came from the unfair method (a world held at difficulty 76 tips over once its
experienced families die, and one run says nothing). A reward of 100 against steps of about ±1 is still a risk, so the
genes start at 10.

**The owner, on what is a gene and what is not:** "the creatures have all the input... isn't it redundant? All these
feelings are the parameters they get in and how they are interpreted by the neurons" · fear "if it's hard-coded, then I
don't like it" · "a pain sensor should help... I should know I'm being attacked" · "the body doesn't feel pain, the brain
decides how much a pain means". So: anything that moves energy is already in the reward (the change in energy), and the
neurons decide what each input means; only what energy cannot say is a gene (a death, a baby, learning from others);
nothing that triggers a feeling is chosen by hand (no fear of danger); pain is an input, not a gene.

**The change (it replaces the fixed numbers of "Learning from what others live"):** each creature carries three values in
its genes, inherited like its size and senses:
- **How bad dying feels** (`fear_of_death`): taken from its reward on the step it dies. Starts at **10** (ten bites).
- **How good a baby feels** (`love_of_babies`): added to its reward on the step after it has a baby. Starts at **10**.
- **How much it learns from others** (`learns_from_others`): the weight of a lesson against a step of its own life.
  Starts at **0.5**.
- **Starting amounts:** the first creatures and every newcomer get exactly these; a baby gets its parent's × (1 ± up to
  0.2), kept within 0.1 to 100 for the first two and 0.01 to 1 for the third. Mutation sways them; selection keeps what
  helps a family live and have babies.
- **A watcher values a lesson by its own genes:** the lesson is the actor's change in energy, minus the watcher's own fear
  of death if the actor died, plus the watcher's own love of babies if the actor had one; weighted by the watcher's own
  learning from others (and, as before, by how likely the watcher would have done the same, at most fully).
- **The owner's world:** its creatures start with the starting amounts and feel no pain at first; saves become format 5
  (format 4 converted: the three amounts, no pain, the brains' 4 new inputs at zero).
- **A pain sensor (4 new inputs, 21 in all):** for one step after a creature is hit it senses that it was hit, how hard (the
  share of its maximum energy taken) and the direction of the one who hit it (cos, sin); otherwise all four are 0. What it
  means is the neurons' (no pain gene: the energy lost is already in the reward). The owner's world's brains get the 4
  inputs with zero weights, so nothing they do changes until they learn or mutate.
- **Charts and tiles:** the average of each of the three, so their evolution can be watched; pointing at a creature lists
  its own.
- **History keeps every world:** a scoreboard written before a column existed is read with that column empty (today such
  worlds vanish from History whenever columns are added).
- The starting amounts and limits are Claude's (sized so a death or a baby is worth ten bites, not a hundred); Task 6
  measures them.

**How it is checked:** the fair way: the real runner, with Auto, from a copy of the owner's save 156,000 (never the
owner's own run folder) to step 164,000, three seeds with today's learning and three with this one. A run ending below 40
alive is a fail. For each: alive every 1,000 steps, fleeing and chasing over the last 4,000 steps, and the three averages
at the end, shown to the owner as they come out, and live while they run: a second control room on port 8766 shows the
six runs' charts updating (the owner's rule: a long run is shown as live graphs).

### Live: a table of running worlds, each on its own screen (the owner, 2026-11-24)

The owner: "the UI is terrible! each run should get its own screen, create a table of live runs.. not all runs should
share a screen" · on the proposal: "Yes, write it".

**Today:** Live shows every running world's controls one under another, and one set of charts with all of them drawn
together.

**The change (the page only):**
- **Live opens on a table of the running worlds,** one row each: its name (seed and kind), step, alive, average energy,
  difficulty, latest intelligence × random, steps a second, Auto or Manual, running or paused. The rows update every
  second (each world's live numbers, as the tiles use today). "Nothing is running" when the table is empty. The Start bar
  stays above it.
- **Clicking a row opens that world's own screen:** exactly what a world has today (its knobs, live tiles, Update code,
  Pause, Stop, Jump in) and charts of that world alone. A "← All live worlds" button goes back to the table.
- **Each screen has its own address** (`#live/<world>`), so a refresh, the browser's Back and a shared link land on the
  same screen; a world that ends sends its screen back to the table.
- **Charts are always of one world:** the one on screen; the table shows no charts, and no chart ever draws two worlds
  (the owner: "charts are always to the single world we are now testing").
- **Charts mark only the owner's own changes:** a vertical line where the owner moved a knob or the hunters switch by
  hand; Auto's nudges (one every 500 steps since it steers by rates: about 300 lines a world, a solid wall when several
  worlds share a chart) are no longer lines, only words in the hover text at their step. The owner, on the wall of
  lines: "why charts look like this?!" · "Yes, add it".
- History, the world screen's contents and every control are as they are.

**How it is checked:** a page test for the table, its rows and the addresses, and that Auto's changes draw no line; then, in the app's browser with the six
check worlds running, the table updating, a click opening one world's own screen with only its charts, Back and a
refresh, phone width, no console errors, a screenshot to the owner.

### Senses in the genes: a vision cone and hearing (the owner, 2026-11-24)

The owner: "add visualization of the vision in the live view... it should be subtle not to overwhelm the screen... some
transparent angle part of a circle" · "also for hearing add circle visualization" · vision as a forward cone and hearing
as a full circle: "Yes, spec it" · "better vision or better hearing must come with costs" · the cost is energy each
step by the area sensed: "Yes, by area".

**Today:** a creature sees 3 units all around (360°) and hears nothing; its brain gets the nearest food and the nearest
stranger in that circle.

**The change:**
- **Vision is a cone** pointing where the creature faces (the direction of its last move; resting keeps it). Each creature
  has two vision genes: the cone's **width** (30° to 360°) and its **reach** (1 to 8 units). It sees food and creatures
  only inside its cone.
- **Hearing is a circle** around it, all directions, of one gene: its **radius** (0 to 4 units). It hears creatures, not
  food: the nearest stranger it hears but does not see gives the brain where it is, how far, and how big against itself
  (5 new inputs, after today's 12: 17 in all).
- **Senses cost energy each step, by the area they cover:** `SENSE_COST` × (cone area + hearing area), where a cone's area
  is width ÷ 360° × π × reach² and hearing's π × radius². `SENSE_COST` is set so today's senses (a full circle of 3 units,
  area 28.3) would cost 0.05 a step: 0.00177 a square unit. Seeing wide or far, or hearing far, burns more.
- **Inherited like size:** a baby gets each sense gene of its parent × (1 ± up to 0.2), kept in its limits. Newcomers get
  random ones in the limits.
- **Learning from others** (the section above) counts a watcher who sees or hears the actor.
- **The owner's world keeps going:** on Update code its creatures start with today's vision (360°, 3 units) and a hearing
  radius of 1.5; their brains get the 5 new inputs with zero weights, so nothing they do changes until they learn or
  mutate. Saves become format 4 (format 3 saves are converted the same way).
- **Jump in shows them, subtly:** each creature's cone as a faint wedge (about 6 % opacity of its body colour) and its
  hearing as a faint dotted circle; pointing at a creature draws its own a little stronger and lists its senses. The
  legend gets a cone and a hearing circle.
- **The scoreboard** gets the average cone width, reach and hearing radius, each a chart in History and the live tiles,
  so their evolution can be watched.
- The limits, the 0.05 and the starting hearing radius are Claude's, not measured; Task 6 measures them.

**How it is checked:** tests for the cone (food behind it unseen), hearing (heard behind, size given, food not heard), the
cost by area, inheritance in the limits, the format 3 → 4 save conversion, and the 17-input brains acting as before with
zero new weights. Then Update code on the owner's world, and Jump in looked at in the app's browser: cones and hearing
circles faint, the hovered one stronger, no console errors.

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
1. **Survival got harder:** in Auto, every seed ends at difficulty 10 or more.
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
