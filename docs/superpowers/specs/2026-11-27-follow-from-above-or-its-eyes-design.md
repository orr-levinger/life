# Following a creature: from above, or from its eyes: design

Asked by the owner on 2026-11-27: "the follow creature and the the wrold poc should have 2 mods from above and from his
eyes"; then, of today's POV (a camera behind its back): "pov still doesnt look from creature eyes!!".

## What changes (the page only)

While a creature is followed, the POV button becomes a two-way switch: **Above | Eyes**.

- **Above** (the start): as following is today: the camera above, the creature kept in the middle, spin and zoom free,
  everything drawn.
- **Eyes**: the camera at its eyes (its head's height above the ground, at the front of its body), looking where it
  faces, a little down; the picture's width is its vision width (30-120 degrees); only what it senses is drawn (as POV
  is today: what it sees, and what it hears as a blur); its own body is not drawn (the camera is inside its head). Spin
  and zoom are off; Jump out, Stop following or Above leave it.

## Not in this

- No change to the world, its senses or its rules; the page's other views stay.

## How it is proven

- A test reads the page: the switch has Above and Eyes; planet.js has the eyes camera (its field of view from the vision
  width, the body hidden).
- In the browser pane: a screenshot from a creature's eyes and one from above, sent to the owner.
- `bench(30)` in Eyes under 33 ms.
