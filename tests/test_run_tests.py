import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RUNNER = Path(__file__).resolve().parent.parent / "scripts" / "run_tests.py"


def run_on(tests: dict, limit: float) -> subprocess.CompletedProcess:
    """The runner on a folder of the given test files, with a time limit in seconds."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "tests"
        folder.mkdir()
        (folder / "__init__.py").write_text("")
        for name, code in tests.items():
            (folder / f"{name}.py").write_text(code)
        return subprocess.run([sys.executable, str(RUNNER), "--tests", str(folder), "--limit", str(limit)],
                              capture_output=True, text=True, cwd=tmp, timeout=60)


PASSES = "import unittest\nclass T(unittest.TestCase):\n    def test_ok(self):\n        pass\n"
FAILS = "import unittest\nclass T(unittest.TestCase):\n    def test_no(self):\n        self.fail('no')\n"
SLOW = "import time, unittest\nclass T(unittest.TestCase):\n    def test_slow(self):\n        time.sleep(30)\n"


class TestRunTests(unittest.TestCase):
    def test_all_passing_within_the_limit_is_ok(self):
        done = run_on({"test_a": PASSES, "test_b": PASSES}, 30)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("OK", done.stdout)

    def test_a_failing_test_fails_the_run(self):
        done = run_on({"test_a": PASSES, "test_b": FAILS}, 30)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("test_b", done.stdout)

    def test_over_the_limit_fails_the_run(self):
        done = run_on({"test_a": PASSES, "test_slow": SLOW}, 2)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("over the limit", done.stdout)
        self.assertIn("test_slow", done.stdout)  # which file was still running


if __name__ == "__main__":
    unittest.main()
