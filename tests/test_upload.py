"""The owner's Mac sends its worlds' reports to their S3 bucket (the owner, 2026-11-27: "a shared S3 bucket on my
personal AWS account")."""
import tempfile
import unittest
from pathlib import Path

from src.upload import ServerLog, Uploader


class Recorder:
    """Stands in for the `aws` tool: keeps each command; fails when told to."""

    def __init__(self, fail=None):
        self.commands, self.fail = [], fail

    def __call__(self, command, **kwargs):
        if self.fail:
            raise self.fail
        self.commands.append(command)

        class Done:
            returncode, stderr = 0, ""
        return Done()


def a_run(runs: Path, name="20261127-094329-1-learning") -> Path:
    folder = runs / name
    (folder / "report").mkdir(parents=True)
    (folder / "report" / "state.jsonl").write_text('{"step": 500}\n')
    (folder / "species.json").write_text("{}")
    return folder


class TestUpload(unittest.TestCase):
    def test_the_upload_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            a_run(runs)
            (runs / "server.log").write_text("started\n")
            aws = Recorder()
            uploader = Uploader(runs, "s3://my-bucket/mac", "life-reports", run=aws)
            uploader.once()
            uploader.once()  # nothing changed since: nothing sent again
        name = "20261127-094329-1-learning"
        self.assertIn(["aws", "s3", "sync", str(runs / name / "report"), f"s3://my-bucket/mac/{name}/report",
                       "--profile", "life-reports", "--only-show-errors"], aws.commands)
        self.assertIn(["aws", "s3", "cp", str(runs / name / "species.json"), f"s3://my-bucket/mac/{name}/species.json",
                       "--profile", "life-reports", "--only-show-errors"], aws.commands)
        self.assertIn(["aws", "s3", "cp", str(runs / "server.log"), "s3://my-bucket/mac/server.log",
                       "--profile", "life-reports", "--only-show-errors"], aws.commands)
        self.assertEqual(len(aws.commands), 3)
        self.assertEqual(Uploader.EVERY, 120.0)  # every 2 minutes

    def test_a_failed_upload_is_logged_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            a_run(runs)
            log = ServerLog(runs / "server.log")
            Uploader(runs, "s3://my-bucket/mac", "life-reports", run=Recorder(fail=FileNotFoundError("aws")), log=log).once()
            said = (runs / "server.log").read_text()
        self.assertIn("upload failed", said)

    def test_the_server_keeps_its_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = ServerLog(Path(tmp) / "server.log", keep=5)
            for k in range(8):
                log.write(f"line {k}\n")
            kept = (Path(tmp) / "server.log").read_text().splitlines()
        self.assertEqual([line.split(" ", 1)[1] for line in kept], [f"line {k}" for k in range(3, 8)])  # time, then words
        from src.viewer import parse_args
        args = parse_args(["--report-to", "s3://b/mac", "--report-profile", "life-reports"])
        self.assertEqual((args.report_to, args.report_profile), ("s3://b/mac", "life-reports"))


if __name__ == "__main__":
    unittest.main()
