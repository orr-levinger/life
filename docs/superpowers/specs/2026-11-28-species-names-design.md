# Latin names for species: design

Asked by the owner on 2026-11-28: "Species 16 is a boring name. The system should generate cool names based on the
genes and use some common way to name animals… Just not species number." Then, choosing between Latin and English
names: "Latin". Then: "Names should be generated only when a new species is created." Then, on two species that
could share a name: "if we use few genes for the name, we need to add more genes for the name. We need to make it
more complex."

Nothing in the world changes: no rule, no save, no file the world writes. Only the page changes.

## When a species gets its name

- A species is named from its genes at its birth: its typical member at its first look (`start` in `species.json`,
  written once when the species is born and never changed). Its name therefore never changes, however its genes
  drift later.
- The name is worked out by the page from the tree of species the world already writes, so every world has names at
  once: the running one, the ones in History, and the Mac's world after a page reload. Nothing has to be copied or
  converted.
- A name is never given twice in one world. Species are named in the order they were born (by their number); a later
  species never changes an earlier one's name.

## How the name is made: three Latin words

As zoologists name a subspecies (*Panthera tigris altaica*): the first word (the genus) says its body plan; the
second and third say the four genes that set it apart most, two genes in each word. Nine genes make every name
(both guts, gills, lungs and size in the first word; four more in the other two).

**The first word: what it eats + where it lives and how big it is.**

| What it eats (as the card says it: grass gut against meat gut) | Start |
|---|---|
| plants | Phyto |
| meat | Sarco |
| both | Panto |

| Where it lives (as the card says it: gills against lungs) | Ending |
|---|---|
| land, size under 0.9 | mys (a mouse) |
| land, size 0.9 to 1.4 | therium (a beast) |
| land, size over 1.4 | saurus (a lizard) |
| water, size under 1.2 | ichthys (a fish) |
| water, size 1.2 or more | cetus (a whale) |
| land and water | batrachus (a frog) |

For example: *Phytotherium*, *Sarcoichthys*, *Pantobatrachus*.

**The second and third words: the four genes it differs most in from its parent species at its parent's birth.**
Each gene is measured as a share of its whole range (learning speed on a log scale, as the Make slider). "First
life" has no parent: its genes are measured from the middle of each range. The genes are ranked by how far they
moved; a gene that went up gives its up word, one that went down its down word.

Each word joins two genes: the first's root, an "o", and the second's whole word (the "o" is left out before a
vowel). The second word is genes 1 and 2; the third word genes 3 and 4. For example fear of death up (spin-) and
hearing up (auritus) make *spinauritus*; neurons up (cerebr-) and learning speed down (lentus) make *cerebrolentus*.

| Gene | Up: root, word | Down: root, word |
|---|---|---|
| size | gigant-, giganteus | pumil-, pumilus |
| stealth | occult-, occultus | conspic-, conspicuus |
| armour | loric-, loricatus | nud-, nudus |
| vision width | circum-, circumspectus | angust-, angustus |
| vision reach | ocul-, oculatus | myop-, myops |
| hearing | aur-, auritus | surd-, surdus |
| calls | voc-, vocalis | tacit-, tacitus |
| care for kin | collar-, collaris | solit-, solitarius |
| fear of death | spin-, spinosus | aud-, audax |
| love of babies | pi-, pius | incuri-, incuriosus |
| learns from others | docil-, docilis | indocil-, indocilis |
| learning speed | veloc-, velox | lent-, lentus |
| mutations | mutabil-, mutabilis | const-, constans |
| mutation size | vari-, varius | fidel-, fidelis |
| neurons | cerebr-, cerebrosus | simplic-, simplex |
| layers | profund-, profundus | plan-, planus |

For example: *Phytotherium spinauritus cerebrolentus*: a mid-size plant eater on land whose fear of death rose
most, then its hearing, then its neurons, while its learning slowed. *Sarcocetus loricoccultus vocosimplex*: a big
meat eater in water, armoured and stealthy, calling more, with fewer neurons.

**Never twice.** Two species share a name only when they eat the same, live in the same place, are the same size
class and the same four genes moved most, in the same order and the same way. Then the later one gets a number
after its name: *Phytotherium spinauritus cerebrolentus II*.

The words are Latin roots joined the way animal names are; Latin grammar between the words (matching endings) is
not followed.

## Where the name shows

Everywhere the page now says "species 16" or "first life": the tree of life (under each circle), the species card's
title and its "from" and "daughters", the "Alive now" table and its "from" column, and the Make tab's "Start from"
list. The name is written in italics, as Latin names are. The card says once, in small grey words, "number 16 in this
world", so a species can still be matched to the world's events.

## One place

The names are made in one function on the page (`latinName`), next to `lives()` and `eats()`, which already say land
or water and what it eats, so the name and the card always agree. One table on the page holds, for each of the 16
genes, its two roots, its two words and the range it is measured by. The ranges are fixed: the world's starting ranges of today
(for example size 0.5 to 2, hearing 0 to 4, fear of death 0.1 to 100, from `starting_rules()`). A world's own ranges
move as it gets harder, so measuring by them would change old names; fixed ranges keep every name as it was given.

## Not in this

- No rule of the world, no save and no file the world writes changes.
- English names (the owner chose Latin).

## How it is proven

- Tests (the page's files as served): `latinName` exists, every species name on the page comes from it (no
  "species ${" and no "first life" words left), each of the 16 genes has both roots and both words, the six endings and
  three starts are there.
- A test that runs the naming on a small hand-made tree of species (a first life, two daughters with set genes, and a
  third daughter whose nine genes match the second's) and checks the four names: three words each, the joined
  words as the table makes them, and the number II on the third daughter.
- On a half-size check world on this box: the tree, the card, "Alive now" and Make show Latin names; a screenshot;
  the browser pane closed.
- The whole suite under 300 s.
