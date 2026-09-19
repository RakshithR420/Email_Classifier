from actions.actions import decide_and_act
from auth.gmail_auth import get_gmail_service
from auth.gmail_client import fetch_recent_emails
from classifier.classifier import classify_email, get_settings, load_model


def main():
    settings = get_settings()
    dry_run = settings.get("run", {}).get("dry_run", True)
    fetch_cfg = settings.get("fetch", {})

    print("Authenticating with Gmail...")
    service = get_gmail_service()

    print("Loading model (first run can take a minute)...")
    load_model()

    print("Fetching emails...")
    emails = fetch_recent_emails(
        service,
        query=fetch_cfg.get("query", "in:inbox"),
        max_results=fetch_cfg.get("max_results", 50),
    )
    print(f"Fetched {len(emails)} emails. dry_run={dry_run}\n")

    for email in emails:
        result = classify_email(email["subject"], email["sender"], email["body"])
        category = result["category"]
        print(
            f"{email['subject']!r} -> {category} "
            f"(confidence={result.get('confidence')}, reason={result.get('reasoning')})"
        )
        decide_and_act(service, email, category, dry_run=dry_run)
        print()


if __name__ == "__main__":
    main()