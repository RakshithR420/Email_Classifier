import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
from auth.gmail_auth import get_gmail_service
from auth.gmail_client import fetch_recent_emails

service = get_gmail_service()
emails = fetch_recent_emails(service, query="in:inbox", max_results=200)

with open("eval/data/personal_test.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["subject", "sender", "body", "true_category"])
    writer.writeheader()
    for email in emails:
        writer.writerow({
            "subject": email["subject"],
            "sender": email["sender"],
            "body": email["body"][:1000],
            "true_category": "",
        })

print("Wrote eval/data/personal_test.csv — open it and fill in the true_category column for each row")
