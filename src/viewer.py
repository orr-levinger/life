"""The viewer: a page that redraws every 2 seconds from the runs' scoreboards. Read-only.

    python -m src.viewer [--runs runs] [--port 8765]        then open http://localhost:8765

Live shows the runs in progress and says plainly when nothing is running; History keeps every past run.
Open it before a long run starts, so the run is watched as graphs while it goes.
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .scoreboard import ROW_KEYS, read_rows

PAGE = Path(__file__).with_name("viewer.html")
RUNNING_WITHIN = 600.0  # seconds: a run with no end that wrote its scoreboard this recently is still running


def runs(folder: Path) -> List[Dict]:
    """Every run under `folder`, newest first: its record (seed, kind, steps asked for, when it started), its rows as
    numbers (an empty cell is None), the seconds it took once finished, and its status: "running" while it keeps
    writing, "finished" once it has ended, "stopped" when it ended without finishing."""
    found = []
    for run in folder.iterdir():
        scoreboard, record, timing = run / "scoreboard.csv", run / "run.json", run / "time.txt"
        if not scoreboard.exists() or not record.exists():
            continue
        rows = read_rows(scoreboard)
        if rows and tuple(rows[0]) != ROW_KEYS:
            continue  # a scoreboard of an older design
        if timing.exists():
            status = "finished"
        elif time.time() - scoreboard.stat().st_mtime <= RUNNING_WITHIN:
            status = "running"
        else:
            status = "stopped"
        found.append({
            **json.loads(record.read_text()), "name": run.name, "status": status,
            "rows": [{key: _number(row[key]) for key in ROW_KEYS} for row in rows],
            "seconds": float(timing.read_text()) if timing.exists() else None,
        })
    return sorted(found, key=lambda run: run["started"], reverse=True)


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
