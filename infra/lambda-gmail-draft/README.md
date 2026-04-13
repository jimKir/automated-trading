# Gmail Draft Lambda

Creates Gmail drafts from your daily trading summary and weekly scorecard
alerts — no SMTP credentials needed in GitHub Actions.

## Architecture

```
GitHub Actions (daily/weekly)
  └── Commits results/daily_summary.json + paper_monitor.json
        │
EventBridge Scheduler
  └── Triggers Lambda (Mon-Fri 22:00 UTC + Sun 19:00 UTC)
        │
Lambda (trading-gmail-draft)
  ├── Fetches JSON from GitHub raw content
  ├── Formats plain-text email
  └── Creates Gmail draft via Gmail API
        │
You review & forward from Gmail
```

## Prerequisites

- AWS CLI + [SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)
- A Google Cloud project with Gmail API enabled
- An OAuth 2.0 consent screen + credentials (Desktop app type)

## Setup

### 1. Gmail API OAuth Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
2. Create a project (or use an existing one)
3. Enable the **Gmail API** under APIs & Services → Library
4. Create an **OAuth 2.0 Client ID** (type: Desktop app)
5. Download the `credentials.json`
6. Run the one-time auth script to get a refresh token:

```bash
pip install google-auth-oauthlib
python get_refresh_token.py
```

This opens a browser — sign in with `kiritsis.di@gmail.com` and grant
"compose" permission. The script prints a JSON blob with your
`client_id`, `client_secret`, and `refresh_token`.

### 2. Store Secrets in AWS SSM Parameter Store

```bash
# Gmail OAuth credentials (JSON from step 1)
aws ssm put-parameter \
  --name /trading/gmail-oauth \
  --type SecureString \
  --value '{
    "client_id": "YOUR_CLIENT_ID",
    "client_secret": "YOUR_CLIENT_SECRET",
    "refresh_token": "YOUR_REFRESH_TOKEN",
    "token_uri": "https://oauth2.googleapis.com/token"
  }'

# GitHub PAT (fine-grained, repo Contents:read only)
aws ssm put-parameter \
  --name /trading/github-token \
  --type SecureString \
  --value "ghp_your_github_pat_here"
```

### 3. Deploy the Lambda

```bash
cd infra/lambda-gmail-draft

sam build
sam deploy \
  --stack-name trading-gmail-draft \
  --resolve-s3 \
  --capabilities CAPABILITY_NAMED_IAM \
  --region eu-north-1 \
  --parameter-overrides \
    GitHubRepo=jimKir/automated-trading \
    Recipients=kiritsis.di@gmail.com,o.zoumpou@gmail.com
```

### 4. Test

```bash
# Invoke manually
aws lambda invoke \
  --function-name trading-gmail-draft \
  --payload '{"source": "manual_test"}' \
  --region eu-north-1 \
  /dev/stdout
```

Check your Gmail drafts — you should see a new draft with the latest
daily summary.

## Schedules

| Schedule | UTC | EEST | What |
|----------|-----|------|------|
| Mon-Fri | 22:00 | 01:00+1 | Daily summary draft |
| Sunday | 19:00 | 22:00 | Scorecard alert draft (only if FAILING) |

## Cost

Lambda: ~$0.00/month (128MB × 30s × 30 invocations ≈ free tier).
EventBridge Scheduler: free. SSM: free for standard parameters.

## Files

| File | Description |
|------|-------------|
| `handler.py` | Lambda function code |
| `template.yaml` | SAM/CloudFormation template |
| `get_refresh_token.py` | One-time script to obtain Gmail refresh token |
| `README.md` | This file |
