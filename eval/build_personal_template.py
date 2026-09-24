"""Fetches emails from your inbox into eval/data/personal_test.csv with an
empty true_category column for you to fill in by hand.

    python eval/build_personal_template.py              # 200 emails
    python eval/build_personal_template.py --max 300
    python eval/build_personal_template.py --force      # overwrite existing file

The script refuses to overwrite an existing file unless you pass --force,
so your hand labels are never wiped by accident."""
import argparse
import csv
import os

from _paths import PERSONAL_TEST
from auth.gmail_auth import get_gmail_service
from auth.gmail_client import fetch_recent_emails

parser = argparse.ArgumentParser()
parser.add_argument("--max", type=int, default=200)
parser.add_argument("--query", default="in:inbox")
parser.add_argument("--out", default=PERSONAL_TEST)
parser.add_argument("--force", action="store_true")
args = parser.parse_args()

if os.path.exists(args.out) and not args.force:
    raise SystemExit(f"{args.out} already exists (it may contain your labels). Use --force or --out <new file>.")

service = get_gmail_service()
emails = fetch_recent_emails(service, query=args.query, max_results=args.max)

with open(args.out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["subject", "sender", "body", "true_category"])
    writer.writeheader()
    for email in emails:
        writer.writerow({
            "subject": email["subject"],
            "sender": email["sender"],
            "body": email["body"][:1000],
            "true_category": "",
        })

print(f"Wrote {len(emails)} rows to {args.out} — open it and fill in the true_category column for each row.")
