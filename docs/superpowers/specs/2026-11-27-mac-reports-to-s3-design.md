# Reports from the owner's Mac, through S3: design

Asked by the owner on 2026-11-27: "The game that runs on my Mac should be able ... report data and logs and events. So
I don't have to tell you everything about what's going on in my Mac. Like something like every 500 steps, report the
state of the game, all the information ... any errors, any important events, just report back somehow." Not GitHub
("I don't want it to be stored forever as source code"); "a shared S3 bucket on my personal AWS account": "Yes".

## What a world writes (on any machine)

Each run's folder gets a `report/` folder, rewritten in place and trimmed, so it never grows:
- `state.jsonl`: one line every `report_every` steps (500, a new rule in Runner): the step, how many alive, the
  species alive and the tree's size, the deepest generation, how smart (earned ÷ spent; steering to food), the deaths
  since the last line by cause, Auto's level, every knob and cost, the brain sizes; the last 500 lines kept.
- `events.jsonl`: what happened, as it happens: a species split off (from whom) or ended; Auto's nudges (which knob,
  from what to what); the world started, resumed, was updated, stopped; the last 1,000 kept.
- `errors.jsonl`: every error in the world with its traceback, before the world ends; the last 100 kept.
- `species.json`: the tree of species (already written).

The page server writes its own `runs/server.log` (what it prints, and any error with its traceback; trimmed to its
last 5,000 lines).

## How it reaches this box

- **On the Mac**: `python -m src.viewer --report-to s3://<bucket>/mac ` makes the page
  server copy every run's `report/` folder that changed, and `server.log`, to the bucket every 2 minutes, with the
  AWS command-line tool (`aws s3 sync`, the keys of the owner's own profile `orr-personal`, the
  default of `--report-profile`). A failure (no network, no tool) is written to
  `server.log` and tried again next time; it never stops the page or a world.
- **Here**: Claude reads the bucket with `aws s3 sync s3://<bucket>/mac <scratch> --profile orr-personal` (the tool is
  installed here) and reads the files.
- **Nothing piles up**: files are overwritten; a bucket rule deletes anything older than 7 days.

## What the owner sets up once (`docs/mac-reports.md`, step by step)

1. A private bucket in their own account (block all public access), with a lifecycle rule: expire after 7 days.
2. Nothing else: the owner's own profile `orr-personal` is already set up on the Mac and on this box (the owner,
   2026-11-27: "the AWS connection has to be to this profile"); Claude never handles a key.

## Not in this

- No change to the world's behaviour; no new Python package (the `aws` tool is used).

## How it is proven

- Tests: a short small-world run writes `state.jsonl` every `report_every` steps with the named fields; a species split
  and an Auto nudge become events; an error in a run is written to `errors.jsonl` with its traceback; the files are
  trimmed to their limits; the uploader builds the right `aws s3 sync` command, runs every 2 minutes, and a failing
  command is logged, never raised; `server.log` gets the server's errors.
- On the owner's Mac (after the owner's setup): Claude reads the first reports from the bucket and tells the owner what
  it sees.
- The whole suite under 300 s (it is at 298.5 s: this plan first moves the slowest test file into two, so there is
  room).
