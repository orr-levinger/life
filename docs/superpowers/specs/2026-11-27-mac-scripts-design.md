# Simple scripts for the Mac

**The owner, 2026-11-27:** "I don't want to have to write these complex commands. Give me simple scripts to run.
/connect s3 /start game /whatever. ... Push the scripts to GitHub and I will run scripts with maybe basic variable
names."

## The scripts (in the `life` folder, run from Terminal)

| Script | What it does |
|--------|--------------|
| `./setup` | Once: makes `.venv` with Python 3.12 and installs `requirements.txt`. |
| `./connect-s3 BUCKET` | Saves the bucket's name in `.life-bucket` (not in git) and checks the `orr-personal` profile can reach it (`aws s3 ls`); says "connected" or what went wrong. |
| `./start` | Starts the page with `.venv`, with reports to the saved bucket if there is one (`--report-to s3://BUCKET/mac`), and opens http://localhost:8765. |
| `./update` | Gets the newest code (`git pull origin evolve-from-scratch`) and says to press **Update code** on each running world. |

Each script stops with a plain sentence when something is missing (no `.venv`: "run ./setup first"; no `aws`: "brew
install awscli").

## Not in this

- No change to the game. No script for Windows (the README's Windows commands stay).
- The page server keeps running in its Terminal window; Ctrl+C stops the page, never the worlds (as today).

## How it is proven

- A test runs each script with `bash` in a temporary folder, with stand-in `python`, `aws`, `git` and `open` commands
  that record how they were called: `connect-s3 my-bucket` writes `.life-bucket` and calls `aws s3 ls s3://my-bucket
  --profile orr-personal`; `start` calls `python -m src.viewer --report-to s3://my-bucket/mac`; without
  `.life-bucket`, `start` calls `python -m src.viewer` alone; `update` calls `git pull origin evolve-from-scratch`;
  `setup` calls `python3.12 -m venv .venv`.
- The README's Mac section shows the four scripts instead of the long commands; `docs/mac-reports.md` uses them.
- `python scripts/run_tests.py` → OK under 300 s.
