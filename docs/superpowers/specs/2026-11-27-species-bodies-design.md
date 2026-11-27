# Each species' body on the Species page: design

Asked by the owner on 2026-11-27, looking at the Species page: "i should see the shape of that specias ! based on the
average of genees!" Asked "Shall I write a short spec for drawing each species' body on the Species page?": "Yes,
write it".

## What the owner sees

- **The species card** (on the right, when a species is clicked): at its top, a picture of the species' typical member,
  about 160 pixels, drawn exactly as the planet draws a creature with those genes (the same body plan, coat, skin and
  every gene's sign, the look the owner approved in the gallery on 2026-11-26).
- **"Alive now"**: each row starts with a small picture (about 40 pixels) of that species' typical member, so two
  species that look different are told apart at a glance.
- The typical member is the one the page already lists: each gene the mean of the species' members at the last look
  (`species.json`, `now`). Its size makes the picture bigger or smaller, as on the planet (up to the same cap).
- A species that has ended shows its last typical member on its card.

## How it is built

- `src/static/beasts.js` already turns a creature's genes into a body (`genesOf`, then `beastModel`) and draws a small
  picture of one body (`bodyPicture`, used by the legend). A new export there, `speciesPicture(typical, size)`, turns
  the typical member's genes (the names `species.json` uses: `fear`, `love`, `learns`, beside the others) into the
  same body through `genesOf`, and draws it with `bodyPicture`. One place makes bodies; the species page only asks.
- `src/static/planet.js` passes it on (the page loads only that module), as it does `bodyPicture`.
- `src/viewer.html`: the card and each row of "Alive now" put the picture in. While the 3D module is still loading,
  the picture is left out and comes at the next refresh of the page's data.

## Not in this

- No change to the world, its rules, its species, or what `species.json` holds; no change to the tree's bands.
- No new picture style: the bodies are the planet's own.

## How it is proven

- Tests (`tests/test_viewer.py`, the page's files as served): `beasts.js` exports `speciesPicture` and builds it with
  `genesOf` and `bodyPicture` (no second body maker); `planet.js` passes it on; the page's species card and "Alive
  now" rows call it.
- On a small check world on this box (half size: planet radius 18.5, 100 first creatures), the Species page shows the
  picture on the card and in each row; checked by eye with a screenshot, then the browser pane closed.
- The whole suite under 300 s.
