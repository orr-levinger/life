# Reports from your Mac to Claude, through your S3 bucket

Your Mac's page server copies each world's `report/` folder (its state every 500 steps, its events, its errors), its
tree of species and the server's own log to a private bucket in your AWS account every 2 minutes. Claude reads the
bucket from its box. Files are overwritten; the bucket deletes anything older than 7 days. You type every key yourself;
Claude never sees one.

Below, `life-reports-YOURNAME` is the bucket's name: pick your own (bucket names are global), and use it everywhere.

## 1. The bucket (AWS console, once)

1. S3 → **Create bucket**: name `life-reports-YOURNAME`, any region; keep **Block all public access** on; create.
2. The bucket → **Management** → **Create lifecycle rule**: name `expire-after-7-days`, apply to all objects, action
   **Expire current versions of objects** after **7** days (and, if versioning is on, **Permanently delete noncurrent
   versions** after 1 day); create.

## 2. Two users, each allowed only what it needs (IAM, once)

**The Mac's writer**: IAM → Users → **Create user** `life-mac-writer` (no console access) → **Attach policies
directly** → **Create policy** → JSON:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": "arn:aws:s3:::life-reports-YOURNAME",
     "Condition": {"StringLike": {"s3:prefix": ["mac/*", "mac"]}}},
    {"Effect": "Allow", "Action": ["s3:PutObject", "s3:DeleteObject"],
     "Resource": "arn:aws:s3:::life-reports-YOURNAME/mac/*"}
  ]
}
```

**Claude's reader**: `life-box-reader`, the same way, with:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": "arn:aws:s3:::life-reports-YOURNAME"},
    {"Effect": "Allow", "Action": "s3:GetObject", "Resource": "arn:aws:s3:::life-reports-YOURNAME/mac/*"}
  ]
}
```

For each user: **Security credentials** → **Create access key** → "Command Line Interface"; keep the two keys for the
next step (shown once).

## 3. The keys, typed by you

**On your Mac** (the AWS tool: `brew install awscli` if `aws --version` says it is missing):

```
aws configure --profile life-reports
```

Give the **writer's** two keys and the bucket's region; output format: `json`.

**On Claude's box** (the AWS tool is installed), in a terminal there:

```
aws configure --profile life-reports
```

Give the **reader's** two keys and the same region.

## 4. Run the page server with reports

On your Mac, in the `life` folder:

```
python -m src.viewer --report-to s3://life-reports-YOURNAME/mac --report-profile life-reports
```

It says `reports copied to s3://life-reports-YOURNAME/mac every 120 s`. A failed copy is written to `runs/server.log`
("upload failed: ...") and tried again 2 minutes later; the game never stops for it.

## 5. Tell Claude the bucket's name

Claude then reads with `aws s3 sync s3://life-reports-YOURNAME/mac <a folder> --profile life-reports` and tells you
what it sees: each world's state, its events (species splitting off and ending, Auto's nudges, starts, updates,
stops), any error with its traceback, the tree of species, and the server's log.

## What gets sent

| File | What | Kept |
|------|------|------|
| `<run>/report/state.jsonl` | every 500 steps: alive, species, generations, how smart, deaths by cause, Auto's level, every knob and cost, brain sizes | last 500 lines |
| `<run>/report/events.jsonl` | starts, resumes, updates, stops; species splits and endings; Auto's nudges | last 1,000 |
| `<run>/report/errors.jsonl` | every error in the world, with its traceback | last 100 |
| `<run>/species.json` | the tree of species | the latest |
| `server.log` | the page server's own log and errors | last 5,000 lines |
