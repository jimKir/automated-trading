#!/usr/bin/env python3
"""
One-time helper to obtain a Gmail OAuth2 refresh token.

Prerequisites:
    pip install google-auth-oauthlib

Usage:
    1. Download your OAuth credentials.json from Google Cloud Console
       (APIs & Services → Credentials → OAuth 2.0 Client IDs → Download)
    2. Place it in this directory as credentials.json
    3. Run: python get_refresh_token.py
    4. Sign in with kiritsis.di@gmail.com in the browser
    5. Copy the output JSON into AWS SSM Parameter Store

The script requests only the gmail.compose scope (create drafts).
"""
import json
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/gmail.compose"]
CREDS_FILE = Path(__file__).parent / "credentials.json"


def main():
    if not CREDS_FILE.exists():
        print(f"ERROR: {CREDS_FILE} not found.")
        print("Download it from Google Cloud Console → APIs & Services → Credentials")
        print("(OAuth 2.0 Client IDs → Download JSON)")
        raise SystemExit(1)

    flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_FILE), SCOPES)
    creds = flow.run_local_server(port=0)

    # Read client_id and client_secret from the credentials file
    with open(CREDS_FILE) as f:
        client_config = json.load(f)

    # Handle both "installed" and "web" credential types
    config = client_config.get("installed") or client_config.get("web", {})

    output = {
        "client_id": config["client_id"],
        "client_secret": config["client_secret"],
        "refresh_token": creds.refresh_token,
        "token_uri": config.get("token_uri", "https://oauth2.googleapis.com/token"),
    }

    print("\n" + "=" * 60)
    print("SUCCESS — Store this JSON in SSM Parameter Store:")
    print("=" * 60)
    print()
    print(json.dumps(output, indent=2))
    print()
    print("Command:")
    print(f"  aws ssm put-parameter \\")
    print(f"    --name /trading/gmail-oauth \\")
    print(f"    --type SecureString \\")
    print(f"    --value '{json.dumps(output)}'")
    print()


if __name__ == "__main__":
    main()
