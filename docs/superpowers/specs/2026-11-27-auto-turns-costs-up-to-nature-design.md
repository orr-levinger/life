# Auto turns the costs, never above nature: design

Asked by the owner on 2026-11-27, after costs became fixed rules (commits b7e5b52, beb71a7): "if we set the max cost
for each action, the ideal cost, the natural cost, it's okay to be automatic in the auto because it will never rise
above what the default should be, what the normal value should be. And in that case, I'm okay with it being managed by
the auto mode as well."

## The world-rule change, in one line

Auto turns every cost again, only between 0 and its natural value (never above); a new world starts each cost at a
tenth and Auto raises it as the creatures thrive; the owner can still set any cost (Rules page or its slider) and Auto
goes on from there.

## What changes

1. **Each cost's range is 0 to its natural value** (the table of 2026-11-27: living 0.1, moving 0.4, sprint power 2,
   brain 0.0006 a neuron, eyes and ears half of before, eating at nothing 0.1, bump 0.2, attack 1, hit 10, stealth and
   armour 0.3). The "3 × nature" max goes.
2. **Auto turns the costs with food and lifespan**, one knob at a time, by the same nudges (5 %, at least 1 % of the
   range): harder raises a cost toward nature, easier lowers it toward 0. In Manual, Auto turns nothing.
3. **A new world starts each cost at a tenth of nature** (each cost rule's start value).
4. **The owner sets a cost** on the Rules page (Costs) or its slider on the card, in Auto or Manual; the world takes it
   within a second, and Auto goes on from that value.
5. **Growing up by generations goes**: Auto does that job now.
6. **The difficulty number** counts the cost nudges again.
7. **An old save** keeps its costs, except a cost above nature comes down to nature at Auto's first look (today's
   moving 0.5 becomes 0.4, bumping 0.5 becomes 0.2, eating at nothing 0.2 becomes 0.1).
8. **The exam** keeps today's costs.

## Not in this

- Sprint power's natural value (2, or 1 after the half-size measurement): a separate question to the owner.

## How it is proven

- Tests: each cost's range is 0 to nature and its start a tenth; Auto's harder nudges raise a cost and stop at nature;
  easier nudges lower it to 0; the owner's value is where Auto goes on from; difficulty counts cost nudges; Manual turns
  nothing; an old save's cost above nature comes down at the first look; the exam pays today's costs.
- The whole suite under 300 s.
