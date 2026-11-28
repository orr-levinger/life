# The fruit cap is a knob: design

Asked by the owner on 2026-11-28, after a picture of their world with empty plains ("huge areas are empty because
most food is only under trees") and the numbers from their Mac's reports (92 trees dropping 0.34 fruit each a step: 89 %
of new food under trees): "there should be sepeate totals for fruits and grass"; told that fruit already has its own
cap (300, a rule that Auto does not turn and Live does not show): "total fruts should also be nobed".

**The world rule change (the owner's yes: "total fruts should also be nobed"):** the most fruit that can lie on the
planet at once becomes a knob, like the grass's: shown on Live, set by the owner in Manual, turned by Auto.

## What changes

- **Two caps, side by side on Live:** "Grass pieces at most" (today's "Food pieces at most": it was only ever the
  grass's) and "Fruit pieces at most" (today the rule `max_fruit`, 300). Both from 20 (hardest) to 800 (easiest);
  a new world starts with grass 400 and fruit 300, as today.
- **Auto turns the fruit cap** as it turns every knob: one knob at a time, in turn, harder (fewer) and easier (more).
- **On Live, what lies on the ground now:** the food tile shows grass and fruit apart (for example "grass 240 · fruit
  290"), so the split under trees can be seen.
- **The scoreboard** keeps a column for the fruit cap, so the charts can show it over time.
- **A save from before:** its world goes on with the fruit cap it had (its rule's value, 300 unless changed), now a
  knob; the rule is gone. A scoreboard from before is read with that column empty.

## The one-minute arithmetic

- In the owner's world now: 92 trees × 0.27 fruit a step = 25 fruit a step, up to the cap. With the cap at 300, fruit
  fills up in about 12 steps after being eaten. Auto turning the cap down (each nudge at least 1 % of 20–800: 7.8
  pieces) brings fewer fruit under the trees as the world gets harder; turning it up gives more when creatures starve.
- Nothing else changes: the fruit rate, the grass rate and their caps' meaning stay; what a creature does or pays stays.

## Not in this

- How big Auto's nudges on the fruit rate are (the owner's "1 % of the total range" floor makes them jump): asked
  separately.
- Auto turning grass and fruit together: not chosen.

## How it is proven

- Tests (small hand-made worlds, few steps): a world whose fruit cap is 3 holds 3 fruit at most; Auto nudging the fruit
  cap harder lowers it; a save from before goes on with its fruit cap 300 and no `max_fruit` rule; the scoreboard has
  the column; the now file has grass and fruit apart.
- On a half-size check world: Live shows both caps and the grass · fruit tile; the cap turned in Manual holds.
- The whole suite under 300 s.
