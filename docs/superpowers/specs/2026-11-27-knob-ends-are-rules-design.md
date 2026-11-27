# A knob's ends are rules: design

Asked by the owner on 2026-11-27: "i want the rable of the min and max of nobs to be configurable in the rules.. for
example the Fruit a tree drops per step i want to raise the max value of it". Where a change reaches: the running world
too ("do it", to that recommendation).

## What changes

1. **Each knob's two ends are rules** on the Rules page, in a new group "Knob ends": `knob.<name>.easy` (its easy
   end) and `knob.<name>.hard` (its hard end), starting at today's ends (fruit: easy 1, hard 0.002). Each may be set
   from 0 to 10 times the knob's larger end today. An easy end must stay on its easy side of the hard end (more food is
   easier, a lower cost is easier); a change that crosses them is refused in plain words.
2. **A change reaches the running world within a second**, like every other rule change: its card's slider stretches
   to the new ends, Manual accepts any value between them, Auto turns the knob up to them. A knob already outside new,
   narrower ends is brought inside at the world's next look (Auto) or the owner's next drag (Manual).
3. **Auto's smallest nudge** (1 % of a knob's range) uses the world's ends.
4. **The difficulty number** keeps being counted on today's fixed ends, so moving an end never makes the difficulty
   line jump.
5. A world saved before gets the new rules at today's ends (as every newer rule does).

## Not in this

- No change to what a knob does, its start value or Auto's choice of which knob to turn.

## How it is proven

- Tests: each knob has its two rules at today's ends; a wrong-way end is refused; with fruit's easy end at 5, Manual
  takes 3 and refuses 6, Auto eases fruit past 1 up to 5 and not beyond; the smallest nudge is 1 % of the new range;
  the difficulty of the same settings is the same whatever the ends; the page stretches a card's slider to its world's
  ends; an old save resumes with today's ends.
- The whole suite under 300 s.
