import base64

from auth.gmail_client import _extract_body, _strip_html, list_message_ids, parse_message


def b64(s):
    return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")  # Gmail omits padding


def test_nested_multipart_prefers_plain_text():
    payload = {
        "mimeType": "multipart/mixed",
        "parts": [
            {"mimeType": "multipart/alternative", "parts": [
                {"mimeType": "text/plain", "body": {"data": b64("Hello plain")}},
                {"mimeType": "text/html", "body": {"data": b64("<p>Hello html</p>")}},
            ]},
            {"mimeType": "application/pdf", "filename": "a.pdf", "body": {"attachmentId": "x"}},
        ],
    }
    assert _extract_body(payload) == "Hello plain"


def test_html_only_is_stripped():
    payload = {"mimeType": "multipart/alternative", "parts": [
        {"mimeType": "text/html", "body": {"data": b64("<style>.a{}</style><p>Hi &amp; bye</p><br>next")}},
    ]}
    assert _extract_body(payload) == "Hi & bye\nnext"


def test_single_part_and_missing_data():
    assert _extract_body({"mimeType": "text/plain", "body": {"data": b64("Just text")}}) == "Just text"
    assert _extract_body({"mimeType": "text/plain", "body": {"size": 0}}) == ""
    assert _extract_body({"mimeType": "multipart/mixed", "parts": [{"mimeType": "text/plain", "body": {}}]}) == ""


def test_bad_utf8_does_not_crash():
    data = base64.urlsafe_b64encode(b"caf\xe9").decode()
    assert _extract_body({"mimeType": "text/plain", "body": {"data": data}}).startswith("caf")


def test_strip_html_removes_scripts():
    assert _strip_html("<script>alert(1)</script><b>Deal</b>") == "Deal"


def test_parse_message():
    msg = {"id": "m1", "threadId": "t1", "snippet": "Hi &amp; hello", "labelIds": ["INBOX"],
           "payload": {"mimeType": "text/plain", "body": {"data": b64("Body")},
                       "headers": [{"name": "From", "value": "a@b.com"}, {"name": "subject", "value": "Yo"}]}}
    e = parse_message(msg)
    assert e["sender"] == "a@b.com" and e["subject"] == "Yo" and e["body"] == "Body" and e["snippet"] == "Hi & hello"


def test_list_message_ids_paginates():
    pages = {None: {"messages": [{"id": "1"}, {"id": "2"}], "nextPageToken": "p2"},
             "p2": {"messages": [{"id": "3"}]}}

    class Req:
        def __init__(self, token):
            self.token = token

        def execute(self):
            return pages[self.token]

    class Messages:
        def list(self, userId, q, maxResults, pageToken=None):
            return Req(pageToken)

    class Users:
        def messages(self):
            return Messages()

    class Service:
        def users(self):
            return Users()

    assert list_message_ids(Service(), "in:inbox", 10) == ["1", "2", "3"]
    assert list_message_ids(Service(), "in:inbox", 2) == ["1", "2"]
