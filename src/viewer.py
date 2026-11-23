"""The live viewer: a page that redraws every 2 seconds from the runs' scoreboards. Read-only.

    python -m src.viewer [--runs runs] [--port 8765]        then open http://localhost:8765

Open it before a long run starts, so the run is watched as graphs while it goes.
"""
from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .scoreboard import ROW_KEYS, read_rows
from .stages import STAGES

PAGE = Path(__file__).with_name("viewer.html")
RUN_NAME = re.compile(r"^(\d+)-(.+)$")  # <seed>-<kind>, as src.evolve names its output folders


def runs(folder: Path) -> List[Dict]:
    """Every run under `folder` whose scoreboard has today's columns: its name, seed, kind, rows as numbers (an empty
    cell is None), and the seconds it took once it has finished."""
    found = []
    for run in sorted(folder.iterdir()):
        named = RUN_NAME.match(run.name)
        scoreboard = run / "scoreboard.csv"
        if not named or not scoreboard.exists():
            continue
        rows = read_rows(scoreboard)
        if rows and tuple(rows[0]) != ROW_KEYS:
            continue  # a scoreboard of an older design
        timing = run / "time.txt"
        found.append({
            "name": run.name, "seed": int(named.group(1)), "kind": named.group(2),
            "rows": [{key: _number(row[key]) for key in ROW_KEYS} for row in rows],
            "seconds": float(timing.read_text()) if timing.exists() else None,
        })
    return found


def _number(cell: str) -> Optional[float]:
    if cell == "":
        return None
    value = float(cell)
    return int(value) if value.is_integer() else value


def make_server(folder: Path, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.startswith("/data"):
                body, kind = json.dumps(runs(folder)).encode(), "application/json"
            elif self.path.startswith("/meta"):
                body, kind = json.dumps({"stages": [s.name for s in STAGES]}).encode(), "application/json"
            else:
                body, kind = PAGE.read_bytes(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:  # quiet: the page asks every 2 seconds
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the live viewer of the runs.")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    args.runs.mkdir(parents=True, exist_ok=True)
    server = make_server(args.runs, args.port)
    print(f"life viewer on http://localhost:{args.port} (runs: {args.runs})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
