"""Every test file, a few at a time, under a hard time limit (the owner, 2026-11-26: "hard limits for tests! 5 minutes!
1 second above it should fail the tests!"). The run fails when a test fails or when the whole run passes the limit.

    python scripts/run_tests.py [--tests tests] [--limit 300] [--parallel 4]
"""
import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", type=Path, default=Path("tests"))
    parser.add_argument("--limit", type=float, default=300.0, help="seconds the whole run may take")
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()
    root = args.tests.resolve().parent
    waiting = sorted(args.tests.glob("test_*.py"))
    running, failed, started = {}, [], time.perf_counter()
    while waiting or running:
        while waiting and len(running) < args.parallel:
            path = waiting.pop(0)
            module = f"{args.tests.name}.{path.stem}"
            log = tempfile.TemporaryFile("w+")  # a file, not a pipe: a full pipe would stall the test
            running[path.stem] = (subprocess.Popen([sys.executable, "-m", "unittest", module], cwd=root, stdout=log,
                                                   stderr=subprocess.STDOUT, text=True), time.perf_counter(), log)
        for name, (process, began, log) in list(running.items()):
            if process.poll() is None:
                continue
            log.seek(0)
            output = log.read()
            log.close()
            del running[name]
            took = time.perf_counter() - began
            if process.returncode == 0:
                print(f"ok      {took:6.1f}s  {name}", flush=True)
            else:
                failed.append(name)
                print(f"FAILED  {took:6.1f}s  {name}\n{output[-4000:]}", flush=True)
        if time.perf_counter() - started > args.limit:
            for name, (process, _, _) in running.items():
                process.kill()
            print(f"FAILED: over the limit of {args.limit:.0f} s; still running: {', '.join(running)}", flush=True)
            return 1
        time.sleep(0.05)
    took = time.perf_counter() - started
    if failed:
        print(f"FAILED in {took:.1f} s: {', '.join(failed)}")
        return 1
    print(f"OK in {took:.1f} s (limit {args.limit:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
