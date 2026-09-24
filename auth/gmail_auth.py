import os
import pickle

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# gmail.modify = read + label + archive. It does NOT allow permanent deletion.
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

_AUTH_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_PATH = os.path.join(_AUTH_DIR, "credentials.json")
TOKEN_PATH = os.path.join(_AUTH_DIR, "token.pickle")


def _run_consent_flow():
    if not os.path.exists(CREDENTIALS_PATH):
        raise FileNotFoundError(
            f"Credentials file not found at {CREDENTIALS_PATH}. Download OAuth client "
            "credentials (Desktop app) from Google Cloud Console and save them there."
        )
    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
    return flow.run_local_server(port=0)


def get_gmail_service():
    """Returns an authenticated Gmail API client. The first run opens a
    browser for consent; after that the cached token is reused/refreshed."""
    creds = None

    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                # Token was revoked or expired for good (common for apps in
                # "Testing" mode, whose refresh tokens expire after 7 days).
                print("Saved Gmail token is no longer valid — asking for consent again...")
                creds = _run_consent_flow()
        else:
            creds = _run_consent_flow()

        with open(TOKEN_PATH, "wb") as token:
            pickle.dump(creds, token)

    return build("gmail", "v1", credentials=creds, cache_discovery=False)


if __name__ == "__main__":
    service = get_gmail_service()
    profile = service.users().getProfile(userId="me").execute()
    print(f"Authenticated as: {profile.get('emailAddress')}")
