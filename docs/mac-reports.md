# Reports from your Mac to Claude, through your S3 bucket

Your Mac's page server copies each world's `report/` folder (its state every 500 steps, its events, its errors), its
tree of species, its whole history (the scoreboard) and the server's own log to a private bucket in your AWS account every 2 minutes. Claude reads the
bucket from its box. Both use your own AWS profile, `orr-personal`, already set up on the Mac and on Claude's box; Claude
never sees a key. Files are overwritten; the bucket deletes anything older than 7 days.

Below, `YOUR-BUCKET` is the bucket's name: pick your own (bucket names are global), and use it everywhere.

## 1. The bucket (AWS console, once)

1. S3 → **Create bucket**: name `YOUR-BUCKET`, any region; keep **Block all public access** on; create.
2. The bucket → **Management** → **Create lifecycle rule**: name `expire-after-7-days`, apply to all objects, action
   **Expire current versions of objects** after **7** days (and, if versioning is on, **Permanently delete noncurrent
   versions** after 1 day); create.

## 2. Connect, then start

On your Mac, in the `life` folder, once:

```
./connect-s3 YOUR-BUCKET
```

It says `connected` when your `orr-personal` profile reaches the bucket. From then on, `./start` sends the reports: it
says `reports copied to s3://YOUR-BUCKET/mac every 120 s (profile orr-personal)`. A failed copy is written to `runs/server.log`
("upload failed: ...") and tried again 2 minutes later; the game never stops for it.

## 3. Tell Claude the bucket's name

Claude then reads with `aws s3 sync s3://YOUR-BUCKET/mac <a folder> --profile orr-personal` and tells you what it sees:
each world's state, its events (species splitting off and ending, Auto's nudges, starts, updates, stops), any error
with its traceback, the tree of species, and the server's log.

## What gets sent

| File | What | Kept |
|------|------|------|
| `<run>/report/state.jsonl` | every 500 steps: alive, species, generations, how smart, deaths by cause, Auto's level, every knob and cost, brain sizes | last 500 lines |
| `<run>/report/events.jsonl` | starts, resumes, updates, stops; species splits and endings; Auto's nudges | last 1,000 |
| `<run>/report/errors.jsonl` | every error in the world, with its traceback | last 100 |
| `<run>/species.json` | the tree of species | the latest |
| `<run>/scoreboard.csv` | the world's whole history, a row every 250 steps (what the charts draw) | the latest |
| `<run>/run.json` | the world's record: its seed, when it started, its updates | the latest |
| `server.log` | the page server's own log and errors | last 5,000 lines |
