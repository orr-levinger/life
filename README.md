# Life: an evolution simulator

Creatures with a brain and senses live on a small planet with seas, islands and rivers. Each one learns during its own
life (reinforcement learning), and its babies inherit its brain and genes with small changes (evolution). The world
starts easy and gets harder as they get better at living in it.

## Quick start

You need Git and Python 3.12 (TensorFlow, the learning library, does not install on Python 3.13 or newer yet).

### On a Mac

In Terminal:

```
git clone https://github.com/orr-levinger/life.git
cd life
git checkout evolve-from-scratch
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.viewer
```

Then open http://localhost:8765 in your browser and press **Start**.

Next time, only:

```
cd life
source .venv/bin/activate
python -m src.viewer
```

To get the newest code: `git pull origin evolve-from-scratch`, then start the page again.

### On Windows

In PowerShell:

```
git clone https://github.com/orr-levinger/life.git
cd life
git checkout evolve-from-scratch
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.viewer
```

Then open http://localhost:8765 and press **Start**.

## Using the page

The top bar has three tabs: **Live**, **History** and **Rules**.

### Live: running worlds

- **Start a world:** choose learning animals (they learn during life) or instinct animals (they only evolve), a seed (the
  same seed makes the same world), and **Auto** or **Manual**, then press **Start**. A world runs until you press
  **Stop**.
- **Pause** holds a world still; **Continue** lets it go on.
- **Auto** and **Manual:** in Auto the world turns its own knobs: while the creatures are many and well fed, one knob is
  made a little harder; when they are few or hungry, the last change is taken back. In Manual you move the sliders and
  the running world takes each change within a second.
- **The knobs:** food (Food pieces at most, New food per step, Energy per food, Fruit a tree drops per step), the
  costs (moving, living, sprinting, eating at nothing, an attack, a hit's damage, bumping into a rock, seeing, hearing,
  each neuron, stealth, armour) and Lifespan. Every cost can go down to 0.
- **The charts** get a new point every 250 steps, and each change of a knob is marked on them.
- **Jump in** shows the world as a planet: drag to spin it, use the wheel (or pinch) to zoom, and click a creature to
  follow it. While you follow one: **POV** shows only what it sees, **Genes** shows its genes (change them and press
  **Apply to this creature**), **Stop following** lets it go. **Vision cones** turns the cones on or off; **steps a
  second** sets how fast the world runs while you watch. **Jump out** goes back to full speed.
- **Update code:** after you pull new code, this saves the world at the step it is at and goes on at once with the new
  code, in the same run. Nothing is lost.

### History: past worlds

Every world, running or finished, with its charts. Every world saves itself every 2,000 steps. Pick a save, choose
Auto or Manual, and press **Start a new world from this save** to go on from it as a new world.

### Rules

Every rule of the world, with what it means, its value, what it may be and where it started; **Reset** puts one back.
With nothing running, the page says **Nothing is running** and the rules are the ones the next new world starts on: a
change is kept and used when you press **Start**. Some rules (the planet's size, its seas, islands and rivers) are
only used when a new world is made.

## Without the page

Run one world from the command line:

```
python -m src.evolve --seed 1 --steps 30000 --species learning
```

`--forever` runs it until **Stop** is pressed on the page; `--mode manual` leaves the knobs to you; `--resume <save>`
goes on from a save. `python -m src.evolve --help` lists every option.

Start the page on another port, or with another folder of runs: `python -m src.viewer --port 8770 --runs runs`.

Run the tests (every test file, 4 at a time; the whole run fails if it takes more than 5 minutes):

```
python scripts/run_tests.py
```

## Troubleshooting

**The page says "connecting…" and nothing moves.** The page's server is not running. Start it again with
`python -m src.viewer` (in the `life` folder, with `.venv` active), then reload the page.

**The page says "Nothing is running".** No world is running. Press **Start**, or go to **History** and go on from a
save.

**A world looks frozen.** It may be paused: its card shows **Continue**; press it. While you watch a planet, the world
runs at the **steps a second** you set; **Jump out** puts it back to full speed.

**"Address already in use" when the page starts (Mac).** Another program (often an earlier page server) uses port 8765.
Close it (on a Mac: `lsof -i :8765` shows which one), or start on another port: `python -m src.viewer --port 8770` and
open http://localhost:8770.

**`python3.12: command not found` (Mac).** Install Python 3.12: `brew install python@3.12` (with Homebrew), or the
installer from python.org. Then make the `.venv` again.

**TensorFlow does not install.** Check `python --version` inside `.venv`: it must be 3.12. If it is 3.13 or newer,
delete the `.venv` folder and make it again with Python 3.12. Also try `pip install --upgrade pip` first.

**`ModuleNotFoundError` (numpy, scipy or tensorflow).** The `.venv` is not active. Run `source .venv/bin/activate`
(Mac) or `.venv\Scripts\Activate.ps1` (Windows), then start again.

**The planet view is slow.** Turn **Vision cones** off, zoom out (far creatures are drawn simpler), and close other
heavy tabs. Use a browser with hardware acceleration on (Chrome: Settings → System).

**Every creature starves.** In **Auto** the world makes itself easier on its own when they are few or hungry; give it a
few thousand steps. In **Manual**, lower the costs (moving and sprinting most of all) or raise the food. While fewer
than 80 live, random newcomers keep appearing, so a world never dies out.

**A save will not go on, or Update code is refused: "a flat world from before the sphere".** That save is from before
the planet and cannot be resumed with this code. Start a new world.

**After `git pull`, a world still runs the old code.** A running world keeps the code it started with. Press its
**Update code**. For the page itself to be new, stop the page server (Ctrl+C in its window) and start it again, then
reload the page. Stopping the page server may stop the worlds it started too; their saves stay, and **History** →
**Start a new world from this save** goes on from the last one.

**The tests fail with "limit".** The whole test run took more than 5 minutes; something is too slow (or the computer
is busy with a running world). Run `python scripts/run_tests.py` again with nothing else running.

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

## Where things stand (2026-11-27)

Branch `evolve-from-scratch`. Built and tested: the planet (a sphere, distances along its ground); seas, islands and
rivers that only swimmers enter; rocks and forests; creatures with genes (size, diet, water, stealth, armour, senses,
calls, brain size and more), each deciding every step with its own brain; a PPO learner in TensorFlow; births from
energy, deaths only with low energy; Auto that turns every knob, every cost among them, starting gentle; saves and
resume; the live page with the planet in 3D, following a creature, its point of view and its genes.

Next: the page draws the seas, islands and rivers (`docs/superpowers/plans/2026-11-26-seas-islands-rivers.md`, Task
5), then that plan's last checks. Every design is in `docs/superpowers/specs/`, every plan in `docs/superpowers/plans/`.
