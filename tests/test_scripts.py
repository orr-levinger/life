"""The owner's simple scripts for the Mac (2026-11-27: "Give me simple scripts to run ... I will run scripts with maybe
basic variable names"): each one is run with bash, with stand-in `python`, `aws`, `git` and `open` that write down how
they were called; nothing real is installed, sent or pulled."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ("setup", "connect-s3", "start", "update")
GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")  # on Windows, Git's bash (not another bash on the PATH)
BASH = str(GIT_BASH) if sys.platform == "win32" else shutil.which("bash")


def stand_in(folder: Path, name: str) -> None:
    """A command that writes its name and words to calls.txt and succeeds."""
    path = folder / name
    path.write_text(f'#!/bin/bash\necho "{name} $*" >> "$CALLS"\n', newline="\n")
    path.chmod(0o755)


class TestScripts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.life, fake = Path(self.tmp.name) / "life", Path(self.tmp.name) / "fake"
        self.life.mkdir()
        fake.mkdir()
        for name in SCRIPTS:
            shutil.copy(ROOT / name, self.life / name)
        for name in ("python", "python3.12", "aws", "git", "open"):
            stand_in(fake, name)
        self.calls = Path(self.tmp.name) / "calls.txt"
        self.env = {**os.environ, "CALLS": str(self.calls), "FAKE": str(fake)}

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, *words):
        first = 'export PATH="$(cd "$FAKE" && pwd):$PATH"; exec bash "$0" "$@"'  # the stand-ins before every other
        done = subprocess.run([BASH, "-c", first, str(self.life / words[0]), *words[1:]], env=self.env,
                              capture_output=True, text=True, timeout=30)
        said = self.calls.read_text().splitlines() if self.calls.exists() else []
        return done, said

    def a_venv(self):
        (self.life / ".venv" / "bin").mkdir(parents=True)
        (self.life / ".venv" / "bin" / "activate").write_text("", newline="\n")  # a .venv that changes nothing

    def test_connect_saves_and_checks_the_bucket(self):
        done, said = self.run_script("connect-s3", "my-bucket")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(said, ["aws s3 ls s3://my-bucket --profile orr-personal"])
        self.assertEqual((self.life / ".life-bucket").read_text().strip(), "my-bucket")
        self.assertIn("connected", done.stdout)

    def test_start_reports_to_the_saved_bucket(self):
        self.a_venv()
        (self.life / ".life-bucket").write_text("my-bucket\n")
        done, said = self.run_script("start")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("python -m src.viewer --report-to s3://my-bucket/mac", said)

    def test_start_without_a_bucket(self):
        self.a_venv()
        done, said = self.run_script("start")
        self.assertIn("python -m src.viewer", said)
        self.assertFalse(any("--report-to" in line for line in said))
        (self.life / ".venv" / "bin" / "activate").unlink()
        done, _ = self.run_script("start")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("./setup", done.stdout)  # no .venv: it says what to run

    def test_update_pulls(self):
        done, said = self.run_script("update")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(said, ["git pull origin evolve-from-scratch"])
        self.assertIn("Update code", done.stdout)

    def test_setup_makes_the_venv(self):
        (self.life / ".venv" / "bin").mkdir(parents=True)
        stand_in(self.life / ".venv" / "bin", "pip")  # the .venv's own pip, as python3.12 -m venv makes it
        done, said = self.run_script("setup")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(said, ["python3.12 -m venv .venv", "pip install -r requirements.txt"])


if __name__ == "__main__":
    unittest.main()
