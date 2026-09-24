"""Agentic email classifier — entry point.

Examples:
    python main.py                  # dry run using config/settings.yaml
    python main.py --max 10         # only look at 10 emails
    python main.py --live           # actually label / archive in Gmail
    python main.py --query "in:inbox newer_than:2d"
"""

import argparse
import csv
import os
from collections import Counter
from datetime import datetime

from actions.actions import decide_and_act, get_actions_config
from auth.gmail_auth import get_gmail_service
from auth.gmail_client import fetch_recent_emails
from classifier.classifier import classify_email, get_categories, get_settings, load_model

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_FIELDS = [
    "timestamp", "email_id", "sender", "subject", "category", "confidence",
    "valid", "attempts", "action", "label", "reason", "reasoning", "dry_run",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Classify Gmail emails with a local LLM and act on them.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="really change Gmail (overrides settings.yaml)")
    mode.add_argument("--dry-run", action="store_true", help="only print what would happen (overrides settings.yaml)")
    parser.add_argument("--max", type=int, help="max emails to process (overrides fetch.max_results)")
    parser.add_argument("--query", help="Gmail search query (overrides fetch.query)")
    return parser.parse_args()


def check_config():
    """Warns about categories that have no action configured (and vice versa)."""
    categories = set(get_categories())
    actions = set(get_actions_config())
    for name in sorted(categories - actions):
        print(f"WARNING: category '{name}' has no entry in actions.yaml — it will get no action.")
    for name in sorted(actions - categories):
        print(f"WARNING: actions.yaml has '{name}' but categories.yaml does not.")


def open_log(log_dir):
    log_dir = os.path.join(PROJECT_DIR, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"run_{datetime.now():%Y%m%d_%H%M%S}.csv")
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
    writer.writeheader()
    return path, f, writer


def main():
    args = parse_args()
    settings = get_settings()
    run_cfg = settings.get("run", {})
    fetch_cfg = settings.get("fetch", {})
    classify_cfg = settings.get("classify", {})

    dry_run = run_cfg.get("dry_run", True)
    if args.live:
        dry_run = False
    elif args.dry_run:
        dry_run = True

    query = args.query or fetch_cfg.get("query", "in:inbox")
    max_results = args.max or fetch_cfg.get("max_results", 50)
    threshold = float(classify_cfg.get("confidence_threshold", 0.0))
    review_label = classify_cfg.get("low_confidence_label") or None
    processed_label = run_cfg.get("processed_label") or None

    check_config()

    print("Authenticating with Gmail...")
    service = get_gmail_service()

    print("Loading model (first run can take a minute)...")
    load_model()

    print(f"Fetching up to {max_results} emails matching: {query}")
    emails = fetch_recent_emails(service, query=query, max_results=max_results)
    print(f"Fetched {len(emails)} emails. dry_run={dry_run}\n")
    if not emails:
        return

    log_path, log_file, log_writer = open_log(run_cfg.get("log_dir", "logs"))
    categories_seen, actions_taken, failures = Counter(), Counter(), 0

    try:
        for i, email in enumerate(emails, 1):
            try:
                result = classify_email(email["subject"], email["sender"], email["body"] or email.get("snippet", ""))
                print(
                    f"[{i}/{len(emails)}] {email['subject']!r}\n"
                    f"  -> {result['category']} (confidence={result['confidence']}, "
                    f"attempts={result['attempts']}) {result['reasoning']}"
                )
                plan = decide_and_act(
                    service, email, result["category"], dry_run=dry_run,
                    confidence=result["confidence"], valid=result["valid"],
                    confidence_threshold=threshold, low_confidence_label=review_label,
                    processed_label=processed_label,
                )
            except Exception as e:  # one bad email should not stop the whole run
                failures += 1
                print(f"[{i}/{len(emails)}] ERROR on {email.get('subject')!r}: {e}")
                result = {"category": "", "confidence": "", "valid": False, "attempts": "", "reasoning": ""}
                plan = {"action": "error", "label": None, "reason": str(e)}

            categories_seen[result["category"] or "ERROR"] += 1
            actions_taken[plan["action"]] += 1
            log_writer.writerow({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "email_id": email["id"],
                "sender": email["sender"],
                "subject": email["subject"],
                "category": result["category"],
                "confidence": result["confidence"],
                "valid": result["valid"],
                "attempts": result["attempts"],
                "action": plan["action"],
                "label": plan.get("label") or "",
                "reason": plan["reason"],
                "reasoning": result["reasoning"],
                "dry_run": dry_run,
            })
            log_file.flush()
    finally:
        log_file.close()

    print("\n=== Summary ===")
    print("Categories: " + ", ".join(f"{k}={v}" for k, v in categories_seen.most_common()))
    print("Actions:    " + ", ".join(f"{k}={v}" for k, v in actions_taken.most_common()))
    if failures:
        print(f"Errors:     {failures}")
    print(f"Log saved to {log_path}")
    if dry_run:
        print("This was a dry run — nothing in Gmail was changed. Use --live to apply.")


if __name__ == "__main__":
    main()
