"""The owner's Mac sends its worlds' reports to their own S3 bucket (the owner, 2026-11-27: "a shared S3 bucket on my
personal AWS account"), so Claude can read what happens there without screenshots.

`Uploader`: every `EVERY` seconds, each run whose report changed is copied with the AWS command-line tool (`aws s3
sync` of its `report/`, `aws s3 cp` of its `species.json`), and the page server's own log; the owner's profile holds
the keys, never this code. A failure (no network, no tool, no profile) is written to the server's log and tried again
next time: it never stops the page or a world.

`ServerLog`: the page server's own log (`runs/server.log`): what it prints and every error, each line with its time,
trimmed to its last `keep` lines.
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional


class ServerLog:
    KEEP = 5000

    def __init__(self, path: Path, keep: int = KEEP) -> None:
        self.path, self.keep = Path(path), keep
        self._lock = threading.Lock()
        self._lines: List[str] = self.path.read_text(encoding="utf-8").splitlines()[-keep:] if self.path.exists() else []

    def write(self, text: str) -> None:
        """Each line of `text`, with its time; the oldest dropped past `keep`."""
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        new = [f"{stamp} {line}" for line in text.splitlines() if line.strip()]
        if not new:
            return
        with self._lock:
            self._lines = (self._lines + new)[-self.keep:]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("\n".join(self._lines) + "\n", encoding="utf-8")

    def flush(self) -> None:  # a stream's
        pass


class Uploader:
    EVERY = 120.0  # seconds between two looks

    def __init__(self, runs: Path, target: str, profile: str, run: Callable = subprocess.run,
                 log: Optional[ServerLog] = None) -> None:
        self.runs, self.target, self.profile = Path(runs), target.rstrip("/"), profile
        self.run, self.log = run, log
        self._sent: Dict[str, float] = {}  # each file's time of change when it was last sent

    def _aws(self, *words: str) -> None:
        command = ["aws", "s3", *words, "--profile", self.profile, "--only-show-errors"]
        try:
            done = self.run(command, capture_output=True, text=True, timeout=300)
            if done.returncode != 0:
                raise RuntimeError(done.stderr.strip() or f"exit {done.returncode}")
        except Exception as problem:  # no network, no tool, no profile: said, and tried again next time
            if self.log is not None:
                self.log.write(f"upload failed: {' '.join(command[:4])}: {problem}")
            raise _Failed() from problem

    def _changed(self, path: Path) -> bool:
        if not path.exists():
            return False
        latest = max((p.stat().st_mtime for p in path.rglob("*")), default=0.0) if path.is_dir() else path.stat().st_mtime
        return latest > self._sent.get(str(path), -1.0)

    def _send(self, path: Path, *words: str) -> None:
        latest = max((p.stat().st_mtime for p in path.rglob("*")), default=0.0) if path.is_dir() else path.stat().st_mtime
        try:
            self._aws(*words)
            self._sent[str(path)] = latest
        except _Failed:
            pass

    def once(self) -> None:
        """One look: every changed report, species book and the server's log, sent."""
        for folder in sorted(p for p in self.runs.iterdir() if p.is_dir()):
            report, book = folder / "report", folder / "species.json"
            if self._changed(report):
                self._send(report, "sync", str(report), f"{self.target}/{folder.name}/report")
            if self._changed(book):
                self._send(book, "cp", str(book), f"{self.target}/{folder.name}/species.json")
        log = self.runs / "server.log"
        if self._changed(log):
            self._send(log, "cp", str(log), f"{self.target}/server.log")

    def start(self) -> threading.Thread:
        def forever() -> None:
            while True:
                try:
                    self.once()
                except Exception as problem:  # never the page's end
                    if self.log is not None:
                        self.log.write(f"upload failed: {problem}")
                time.sleep(self.EVERY)
        thread = threading.Thread(target=forever, daemon=True, name="upload")
        thread.start()
        return thread


class _Failed(Exception):
    """An upload that did not go (already written to the log)."""
