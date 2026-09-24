import base64
import html as html_lib
import re
import time

from googleapiclient.errors import HttpError

# Gmail rate-limit errors we should retry on (the rest are raised immediately).
_RETRYABLE_REASONS = ("rateLimitExceeded", "userRateLimitExceeded", "backendError")

_label_id_cache = {}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _get_header(headers, name):
    """Returns the value of header `name` (case-insensitive), or "" if missing."""
    for header in headers:
        if header["name"].lower() == name.lower():
            return header["value"]
    return ""


def _decode(data):
    """Decodes base64url data from the Gmail API. Tolerates missing padding and
    non-UTF-8 bytes (common in marketing mail) instead of crashing."""
    if not data:
        return ""
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")


def _strip_html(raw_html):
    """Turns an HTML email into readable plain text."""
    text = re.sub(r"(?is)<(script|style|head)[^>]*>.*?</\1>", " ", raw_html)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def _collect_parts(payload, plain, html):
    """Walks a (possibly nested) MIME tree and collects text/plain and
    text/html bodies. Real emails often nest multipart/alternative inside
    multipart/mixed, so a single level of `parts` is not enough."""
    mime = payload.get("mimeType", "")
    data = payload.get("body", {}).get("data")

    if mime == "text/plain" and data:
        plain.append(_decode(data))
    elif mime == "text/html" and data:
        html.append(_decode(data))

    for part in payload.get("parts", []) or []:
        _collect_parts(part, plain, html)


def _extract_body(payload):
    """Returns the email body as plain text. Prefers text/plain, falls back
    to stripped text/html, then to the payload's own body."""
    plain, html = [], []
    _collect_parts(payload, plain, html)

    if plain and any(p.strip() for p in plain):
        return "\n".join(plain).strip()
    if html:
        return _strip_html("\n".join(html))

    data = payload.get("body", {}).get("data")
    return _decode(data).strip() if data else ""


def parse_message(message):
    """Converts a raw Gmail API message into the simple dict the rest of the
    app uses."""
    payload = message.get("payload", {})
    headers = payload.get("headers", [])
    return {
        "id": message["id"],
        "thread_id": message.get("threadId"),
        "sender": _get_header(headers, "From"),
        "subject": _get_header(headers, "Subject"),
        "date": _get_header(headers, "Date"),
        "snippet": html_lib.unescape(message.get("snippet", "")),
        "label_ids": message.get("labelIds", []),
        "body": _extract_body(payload),
    }


# ---------------------------------------------------------------------------
# API calls with retry
# ---------------------------------------------------------------------------

def _is_retryable(error):
    status = getattr(error.resp, "status", None)
    if status in (500, 502, 503, 504):
        return True
    return status in (403, 429) and any(r in str(error) for r in _RETRYABLE_REASONS)


def _execute_with_retry(request_fn, max_attempts=7):
    """Runs `request_fn()` (which must build and .execute() a request),
    retrying with exponential backoff on rate limits / transient server
    errors. Backoff is capped at 60s so it can survive a full per-minute
    quota window."""
    for attempt in range(max_attempts):
        try:
            return request_fn()
        except HttpError as e:
            if _is_retryable(e) and attempt < max_attempts - 1:
                wait_time = min(2 ** attempt, 60)
                print(f"Gmail API busy, retrying in {wait_time}s (attempt {attempt + 1}/{max_attempts})...")
                time.sleep(wait_time)
            else:
                raise


def _get_message_with_retry(service, message_id, max_attempts=7):
    return _execute_with_retry(
        lambda: service.users().messages().get(userId="me", id=message_id, format="full").execute(),
        max_attempts=max_attempts,
    )


def list_message_ids(service, query, max_results):
    """Returns up to `max_results` message ids matching `query`, following
    pagination (the API returns at most 500 per page)."""
    ids, page_token = [], None
    while len(ids) < max_results:
        page_size = min(500, max_results - len(ids))
        response = _execute_with_retry(
            lambda: service.users()
            .messages()
            .list(userId="me", q=query, maxResults=page_size, pageToken=page_token)
            .execute()
        )
        ids.extend(m["id"] for m in response.get("messages", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return ids[:max_results]


def fetch_recent_emails(service, query="in:inbox", max_results=50, delay_seconds=0.2):
    """Fetches and parses up to `max_results` emails matching `query`."""
    emails = []
    for message_id in list_message_ids(service, query, max_results):
        message = _get_message_with_retry(service, message_id)
        emails.append(parse_message(message))
        if delay_seconds:
            time.sleep(delay_seconds)
    return emails


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def get_or_create_label(service, label_name):
    """Returns the id of Gmail label `label_name`, creating it if needed.
    Names containing "/" become nested labels (e.g. AI/Orders)."""
    if label_name in _label_id_cache:
        return _label_id_cache[label_name]

    labels = _execute_with_retry(
        lambda: service.users().labels().list(userId="me").execute()
    ).get("labels", [])
    for label in labels:
        _label_id_cache[label["name"]] = label["id"]
    if label_name in _label_id_cache:
        return _label_id_cache[label_name]

    created = _execute_with_retry(
        lambda: service.users()
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


# Kept for backwards compatibility with older imports.
_get_or_create_label = get_or_create_label


def modify_labels(service, message_id, add=None, remove=None):
    body = {"addLabelIds": list(add or []), "removeLabelIds": list(remove or [])}
    _execute_with_retry(
        lambda: service.users().messages().modify(userId="me", id=message_id, body=body).execute()
    )


def apply_label(service, message_id, label_name):
    modify_labels(service, message_id, add=[get_or_create_label(service, label_name)])


def archive_message(service, message_id):
    modify_labels(service, message_id, remove=["INBOX"])
