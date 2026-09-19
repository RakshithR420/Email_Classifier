import base64
import re
import time

from googleapiclient.errors import HttpError

def _get_header(headers, name):
    """Helper function to extract a specific header value from the Gmail API response."""
    for header in headers:
        if header["name"].lower() == name.lower():
            return header["value"]
    return ""

def _decode(data):
    """Helper function to decode base64url-encoded data from the Gmail API."""
    return base64.urlsafe_b64decode(data).decode("utf-8")

def _strip_html(html):
    """Helper function to remove HTML tags from a string."""
    return re.sub(r"<[^>]+>", "", html)

def _extract_body(payload):
    """Helper function to extract the email body from the Gmail API payload."""
    if "parts" in payload:
        for part in payload["parts"]:
            if part["mimeType"] == "text/plain":
                return _decode(part["body"]["data"])
            elif part["mimeType"] == "text/html":
                return _strip_html(_decode(part["body"]["data"]))
    elif "body" in payload and "data" in payload["body"]:
        return _decode(payload["body"]["data"])
    return ""


def fetch_recent_emails(service, query, max_results):
    response = service.users().messages().list(userId="me", q=query, maxResults=max_results).execute()

    message_refs = response.get("messages", [])
    emails = []

    for ref in message_refs:
        message = _get_message_with_retry(service, ref["id"])
        headers = message["payload"].get("headers", [])
        emails.append(
            {
                "id": message["id"],
                "sender": _get_header(headers, "From"),
                "subject": _get_header(headers, "Subject"),
                "body": _extract_body(message["payload"]),
            }
        )
        time.sleep(0.5)

    return emails

_label_id_cache = {}

def _get_message_with_retry(service, message_id, max_attempts=7):
    """Fetches a single message, retrying with exponential backoff if Gmail
    rate-limits us (HTTP 403/429 rateLimitExceeded). Backoff is capped at 60s
    so it can survive a full per-minute quota window."""
    for attempt in range(max_attempts):
        try:
            return (
                service.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )
        except HttpError as e:
            is_rate_limit = e.resp.status in (403, 429) and "rateLimitExceeded" in str(e)
            if is_rate_limit and attempt < max_attempts - 1:
                wait_time = min(2 ** attempt, 60)
                print(f"Rate limited, retrying in {wait_time}s (attempt {attempt + 1}/{max_attempts})...")
                time.sleep(wait_time)
            else:
                raise

def _get_or_create_label(service, label_name):
    if label_name in _label_id_cache:
        return _label_id_cache[label_name]

    labels = service.users().labels().list(userId="me").execute().get("labels", [])
    for label in labels:
        if label["name"] == label_name:
            _label_id_cache[label_name] = label["id"]
            return label["id"]

    created = (
        service.users()
        .labels()
        .create(
            userId="me",
            body={
                "name": label_name,
                "labelListVisibility": "labelShow",
                "messageListVisibility": "show",
            },
        )
        .execute()
    )
    _label_id_cache[label_name] = created["id"]
    return created["id"]


def apply_label(service, message_id, label_name):
    label_id = _get_or_create_label(service, label_name)
    service.users().messages().modify(
        userId="me", id=message_id, body={"addLabelIds": [label_id]}
    ).execute()


def archive_message(service, message_id):
    service.users().messages().modify(
        userId="me", id=message_id, body={"removeLabelIds": ["INBOX"]}
    ).execute()
