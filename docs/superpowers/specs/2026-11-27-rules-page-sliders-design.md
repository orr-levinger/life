# Sliders on the Rules page: design

Asked by the owner on 2026-11-27: "Change the rules page to also be dragble nobs and not just munbers to set".

## What changes (the page only)

1. **Every number on the Rules page gets a slider beside its box** (costs, world, body, eating and hunting, birth and
   death, genes' lowest, highest and newcomer ranges, learning, Auto, exam, runner, knob ends). The box stays, for an
   exact number; dragging the slider moves the box, and the box moves the slider.
2. **A slider runs over the rule's allowed range** (its lowest to its highest). A range that spans a hundredfold or
   more, from above 0 (learning speed, the costs of senses and neurons, ...), slides on a log scale, so its small
   values can be reached; the others slide evenly. A whole-number rule (counts) moves in whole steps.
3. **A change is sent when the drag ends** (as the world card's sliders do), checked as a typed number is: a value
   the world refuses (a lowest above its highest, an easy end past its hard end) springs back and says why.
4. With nothing running, the sliders set the next new world's rules, as the boxes do today.

## Not in this

- No change to any rule, its range or what it does.

## How it is proven

- A test reads the page: every rule row has a slider (`type="range"`) tied to its box; the log-scale and whole-number
  rules are marked; a drag sends the same request a typed number does.
- In the browser pane on a half-size world: a screenshot of the Rules page with sliders, one dragged; the browser
  closed after.
- The whole suite under 300 s.
