# EC2 Paper Trading Deployment Guide

Quick reference for deploying the trading bot to AWS EC2 (eu-north-1).
Estimated time: 20 minutes from zero to running.

---

## Prerequisites

- AWS CLI installed and configured (`aws configure`)
- AWS account with EC2 permissions
- Alpaca paper trading API keys (rotated — old keys are compromised)
- S3 bucket already populated: `trading-data-380277571671-eu-north-1-an`

Verify AWS CLI works:
```bash
aws sts get-caller-identity
```

---

## Step 1 — Create SSH Key Pair (skip if you have one)

```bash
aws ec2 create-key-pair \
  --key-name trading-bot-key \
  --region eu-north-1 \
  --query 'KeyMaterial' \
  --output text > ~/.ssh/trading-bot-key.pem
chmod 400 ~/.ssh/trading-bot-key.pem
```

---

## Step 2 — Create Security Group

```bash
SG=$(aws ec2 create-security-group \
  --group-name trading-bot-sg \
  --description "Trading bot SSH access" \
  --region eu-north-1 \
  --query 'GroupId' --output text)

MY_IP=$(curl -s https://checkip.amazonaws.com)

aws ec2 authorize-security-group-ingress \
  --group-id $SG \
  --protocol tcp --port 22 \
  --cidr ${MY_IP}/32 \
  --region eu-north-1

echo "Security group: $SG"
echo "SSH allowed from: $MY_IP"
```

---

## Step 3 — Launch EC2 Instance

```bash
# Get latest Amazon Linux 2023 AMI
AMI=$(aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=al2023-ami-*-x86_64" \
            "Name=state,Values=available" \
  --region eu-north-1 \
  --query "sort_by(Images,&CreationDate)[-1].ImageId" \
  --output text)

echo "AMI: $AMI"

# Launch t3.small (~$17/month 24/7, ~$4/month market-hours only)
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $AMI \
  --instance-type t3.small \
  --key-name trading-bot-key \
  --security-group-ids $SG \
  --region eu-north-1 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=trading-bot}]' \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Instance ID: $INSTANCE_ID"

# Wait for running state
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region eu-north-1

# Get public IP
INSTANCE_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --region eu-north-1 \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "Instance IP: $INSTANCE_IP"
echo "Connect: ssh -i ~/.ssh/trading-bot-key.pem ec2-user@$INSTANCE_IP"
```

---

## Step 4 — Connect and Bootstrap

```bash
ssh -i ~/.ssh/trading-bot-key.pem ec2-user@$INSTANCE_IP
```

Once inside the EC2, run:

```bash
# Install Python 3.11
sudo dnf install python3.11 python3.11-pip git -y

# Clone repo
git clone https://github.com/jimKir/automated-trading.git
cd automated-trading

# Create credentials file
cat > .env << 'EOF'
ALPACA_API_KEY=your_rotated_key_here
ALPACA_API_SECRET=your_rotated_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_URL=https://data.alpaca.markets
DATA_SOURCE=s3
AWS_DEFAULT_REGION=eu-north-1
ALERT_EMAIL=kiritsis.di@gmail.com
EOF

# Sync historical data from S3 (seconds, not minutes)
aws s3 sync s3://trading-data-380277571671-eu-north-1-an/historical/daily/ \
  data/historical/daily/ --quiet
echo "Data synced: $(ls data/historical/daily/*.parquet | wc -l) parquet files"

# Set up Python environment
python3.11 -m venv venv
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
pip install --quiet -e .

# Run health check
python healthcheck.py

# Quick smoke test
python -c "from execution.live_engine import LiveEngine; print('Import OK')"
```

---

## Step 5 — Install as systemd Service

```bash
sudo tee /etc/systemd/system/trading-bot.service > /dev/null << 'EOF'
[Unit]
Description=Automated Trading Bot — Paper Mode
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/automated-trading
ExecStart=/home/ec2-user/automated-trading/venv/bin/python \
  execution/live_engine.py --mode paper --loop-interval 60
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=/home/ec2-user/automated-trading/.env

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot
sudo systemctl status trading-bot
```

---

## Step 6 — Verify Running

```bash
# Tail live logs
sudo journalctl -u trading-bot -f

# Expected output:
# [INFO] Starting PAPER trading engine
# [INFO] ChoppyDetector v4 initialized — 9 groups
# [INFO] AnomalyRegimeLayer initialized (4 sources)
# [INFO] DynamicUniverseScanner ready
# [INFO] Sleeping 60s...
```

---

## Daily Operations

### Check status from your Mac
```bash
cd ~/Documents/Claude/Projects/automated-trading
source venv/bin/activate
python scripts/status.py          # one-shot
python scripts/status.py --watch  # auto-refresh every 30s
```

### SSH into EC2 to check logs
```bash
# Get IP if you forgot it
INSTANCE_IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=trading-bot" \
            "Name=instance-state-name,Values=running" \
  --region eu-north-1 \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

ssh -i ~/.ssh/trading-bot-key.pem ec2-user@$INSTANCE_IP
sudo journalctl -u trading-bot -n 100   # last 100 lines
sudo journalctl -u trading-bot -f       # tail live
```

### Restart after code update
```bash
# On EC2:
cd ~/automated-trading
git pull origin main
sudo systemctl restart trading-bot
sudo systemctl status trading-bot
```

### Update historical data
```bash
# On EC2 (or set up daily cron):
source venv/bin/activate
python scripts/verify_data.py           # check what's stale
python daily_data_update.py             # fetch latest bars
aws s3 sync data/historical/daily/ \
  s3://trading-data-380277571671-eu-north-1-an/historical/daily/ --quiet
```

---

## Cost Control — Stop/Start by Market Hours

Paper trading only needs to run during US market hours (14:30–21:00 UTC / 17:30–00:00 EEST).

### Manual stop/start from Mac
```bash
# Get instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=trading-bot" \
  --region eu-north-1 \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

# Stop (saves ~70% cost)
aws ec2 stop-instances --instance-ids $INSTANCE_ID --region eu-north-1

# Start (before 14:30 UTC)
aws ec2 start-instances --instance-ids $INSTANCE_ID --region eu-north-1
```

### Automatic via EventBridge (optional, saves ~$13/month)
```bash
# Get instance ID first, then:
aws events put-rule \
  --name start-trading-bot \
  --schedule-expression "cron(25 14 ? * MON-FRI *)" \
  --region eu-north-1 \
  --state ENABLED

aws events put-rule \
  --name stop-trading-bot \
  --schedule-expression "cron(5 21 ? * MON-FRI *)" \
  --region eu-north-1 \
  --state ENABLED
```

Note: EventBridge start/stop requires attaching a Lambda or SSM target — see AWS docs for full setup. The manual commands above are simpler for paper trading.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | `source venv/bin/activate && pip install -e .` |
| `No Alpaca credentials` | Check `.env` file exists and has correct keys |
| `No SPY data` | `aws s3 sync s3://trading-data-380277571671-eu-north-1-an/historical/daily/ data/historical/daily/` |
| Service not starting | `sudo journalctl -u trading-bot -n 50` for error |
| SSH timeout | Re-add your IP: `aws ec2 authorize-security-group-ingress --group-id $SG --protocol tcp --port 22 --cidr $(curl -s https://checkip.amazonaws.com)/32 --region eu-north-1` |
| High memory usage | Upgrade to t3.medium: stop instance, change type, start |

---

## Go/No-Go for Live Capital

Paper trading threshold (your own criteria):
- **12 months** of paper trading
- OOS Sharpe **> 0.50** sustained
- At least **one drawdown episode** survived without hitting -15% MaxDD

Current baseline (Apr 2026): Sharpe +0.898 (ChoppyDetector v4, Sep 2025–Apr 2026).

---

## Key Tags in Repo

| Tag | Description |
|---|---|
| `v1.0.0-paper-baseline` | IS-validated regime weights |
| `choppy-v4` | ChoppyDetector with credit + order flow |
| `anomaly-layer-v1` | Multi-source anomaly detection |
| `dynamic-universe-v1` | Alpaca Screener integration |
| `prod-ready-v1` | Dress rehearsal passed |

