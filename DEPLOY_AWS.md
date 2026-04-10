# Deploying to Amazon Web Services

This guide covers deploying the automated trading system to AWS for 24/7 production operation.

## Recommended Architecture

```
+-----------------------------------------------------------+
|                    AWS Infrastructure                      |
|                                                            |
|  +----------------+    +----------------+                  |
|  |  EC2 t3.small  |    |  CloudWatch    |                  |
|  |  (LiveEngine)  |--->|  Logs+Alarms   |                  |
|  +-------+--------+    +----------------+                  |
|          |                                                 |
|  +-------v--------+    +----------------+                  |
|  |  AWS Secrets   |    |     S3         |                  |
|  |   Manager      |    |  (results/     |                  |
|  |  (API keys)    |    |   reports)     |                  |
|  +----------------+    +----------------+                  |
+-----------------------------------------------------------+
         |
         v
   Alpaca Paper/Live API
```

For ECS Fargate (containerized, recommended for production), see the detailed guide at [deploy/aws_setup.md](deploy/aws_setup.md).

## Prerequisites

- AWS account with billing enabled
- AWS CLI installed and configured (`aws configure`)
- Python 3.11+
- Docker (for ECS/Fargate option)

## Option A -- EC2 (Recommended for paper trading)

### 1. Launch EC2 instance

```bash
# t3.small is sufficient for paper trading (2 vCPU, 2GB RAM)
# Use t3.medium for live trading with H2O vol forecaster (needs more memory for Java)
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type t3.small \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxx \
  --subnet-id subnet-xxxxxxxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=trading-bot}]'
```

### 2. Security group rules

Minimum required inbound rules:

| Port | Protocol | Source | Purpose |
|---|---|---|---|
| 22 | TCP | Your IP only | SSH access |
| 8080 | TCP | VPC CIDR | Health check endpoint |

Outbound: allow all (needs to reach Alpaca API, yfinance, FRED, Databento, WebSocket streams).

### 3. Connect and set up

```bash
ssh -i your-key.pem ec2-user@<instance-ip>

# Install Python 3.11 + Java 17 (for H2O AutoML)
sudo dnf install python3.11 python3.11-pip git java-17-amazon-corretto -y

# Clone repo
git clone https://github.com/jimKir/automated-trading.git
cd automated-trading
pip3.11 install -r requirements.txt
```

### 4. Store credentials in AWS Secrets Manager

```bash
# Store Alpaca credentials
aws secretsmanager create-secret \
  --name trading/alpaca \
  --secret-string '{"ALPACA_API_KEY":"your_key","ALPACA_SECRET_KEY":"your_secret","ALPACA_BASE_URL":"https://paper-api.alpaca.markets"}'

# Store Databento key (optional, for microstructure signals)
aws secretsmanager create-secret \
  --name trading/databento \
  --secret-string '{"DATABENTO_API_KEY":"your_key"}'
```

Alternatively, use `.env` file on the instance (simpler but less secure):

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 5. Run as a systemd service (auto-restart on crash)

```bash
sudo tee /etc/systemd/system/trading-bot.service > /dev/null <<EOF
[Unit]
Description=Automated Trading Bot
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/automated-trading
ExecStart=/usr/bin/python3.11 main.py paper
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal
EnvironmentFile=/home/ec2-user/automated-trading/.env
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot

# Check status
sudo systemctl status trading-bot
sudo journalctl -u trading-bot -f  # tail logs
```

### 6. Schedule market-hours only (optional cost saving)

Use EventBridge to start/stop the instance on market days:

```bash
# Start at 09:00 ET (13:00 UTC) Mon-Fri
aws events put-rule \
  --name start-trading-bot \
  --schedule-expression "cron(0 13 ? * MON-FRI *)" \
  --state ENABLED

# Stop at 17:00 ET (21:00 UTC) Mon-Fri
aws events put-rule \
  --name stop-trading-bot \
  --schedule-expression "cron(0 21 ? * MON-FRI *)" \
  --state ENABLED
```

Note: If trading crypto (BTC-USD, ETH-USD, SOL-USD), the bot should run 24/7 since crypto markets never close.

---

## Option B -- ECS Fargate (Recommended for live capital)

Best option for live trading -- fully managed, auto-scaling, no server management.
See [deploy/aws_setup.md](deploy/aws_setup.md) for the complete step-by-step guide.

The repo already includes a Dockerfile and docker-compose.yml. Summary:

```bash
# Build and push to ECR
aws ecr create-repository --repository-name automated-trading
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker build -t automated-trading .
docker tag automated-trading:latest <account>.dkr.ecr.us-east-1.amazonaws.com/automated-trading:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/automated-trading:latest

# Or use the automated deployment script:
./deploy/deploy_aws.sh
```

The Dockerfile installs Java 17 (required for H2O AutoML), exposes port 8080 for health checks, and runs `python main.py paper` by default.

---

## Option C -- AWS Lambda + EventBridge (Serverless, lowest cost)

Suitable if you only need daily rebalance decisions (no persistent state or WebSocket streaming).

**Limitation:** Lambda has a 15-minute max execution time and cannot maintain persistent connections. The intraday shock detector (5-min polling) and order flow streaming require EC2 or ECS.

```bash
# Package the strategy
pip install -r requirements.txt -t ./package
cd package && zip -r ../deployment.zip . && cd ..
zip -g deployment.zip main.py strategy/ regime/ risk/ core/ config/ data/ utils/

# Create Lambda function
aws lambda create-function \
  --function-name trading-daily-rebalance \
  --runtime python3.11 \
  --handler main.lambda_handler \
  --zip-file fileb://deployment.zip \
  --role arn:aws:iam::ACCOUNT_ID:role/trading-lambda-role \
  --timeout 900 \
  --memory-size 512

# Trigger daily at 09:35 ET (13:35 UTC)
aws events put-rule \
  --name daily-rebalance \
  --schedule-expression "cron(35 13 ? * MON-FRI *)"
```

Note: This option cannot run the H2O vol forecaster (requires Java runtime) or persistent health checks.

---

## CloudWatch Monitoring

```bash
# Create alarm: alert if trading bot stops logging for 30 minutes
aws cloudwatch put-metric-alarm \
  --alarm-name trading-bot-dead \
  --alarm-description "Trading bot has stopped logging" \
  --metric-name IncomingLogEvents \
  --namespace AWS/Logs \
  --statistic Sum \
  --period 1800 \
  --threshold 1 \
  --comparison-operator LessThanThreshold \
  --evaluation-periods 1 \
  --alarm-actions arn:aws:sns:us-east-1:ACCOUNT_ID:trading-alerts

# Create SNS topic for alerts
aws sns create-topic --name trading-alerts
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:ACCOUNT_ID:trading-alerts \
  --protocol email \
  --notification-endpoint your@email.com
```

---

## Cost Estimates

| Option | Instance | Est. Monthly Cost | Notes |
|---|---|---|---|
| EC2 t3.small (24/7) | Always-on | ~$17/month | Simplest, full feature set |
| EC2 t3.small (market hours only) | 6.5h/day x 21 days | ~$4/month | EventBridge start/stop, equities only |
| Lambda | Serverless | ~$1/month | No WebSocket, no ISD, daily rebalance only |
| ECS Fargate (0.25 vCPU) | Managed | ~$12/month | Recommended for production |

All costs exclude data transfer and Secrets Manager (~$0.40/secret/month).

---

## Recommended Path

1. **Paper trading:** EC2 t3.small with market-hours EventBridge schedule (~$4/month) or ECS Fargate (~$12/month)
2. **Live capital:** ECS Fargate with CloudWatch alarms and Secrets Manager (see `deploy/aws_setup.md`)

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `ALPACA_API_KEY` | Yes | Alpaca API key (or `APCA_API_KEY_ID`) |
| `ALPACA_SECRET_KEY` | Yes | Alpaca secret key (or `APCA_API_SECRET_KEY`) |
| `ALPACA_BASE_URL` | Yes | `https://paper-api.alpaca.markets` or live URL |
| `TRADING_MODE` | No | `paper` (default) or `live` |
| `DATABENTO_API_KEY` | No | For microstructure signals (imbalance, options flow) |
| `ALERT_EMAIL` | No | Email for trade alerts and error notifications |
| `BINANCE_API_KEY` | No | Binance crypto trading |
| `BINANCE_API_SECRET` | No | Binance crypto trading |
| `IBKR_HOST` | No | Interactive Brokers TWS host (default 127.0.0.1) |
| `IBKR_PORT` | No | 7497 (TWS paper) / 7496 (TWS live) / 4002 (Gateway paper) |
| `AWS_REGION` | No | Required if using Secrets Manager or SES (default: us-east-1) |
| `S3_BUCKET` | No | S3 bucket for daily report uploads |
| `SES_SENDER` | No | SES sender email for daily reports |
| `SES_RECIPIENT` | No | SES recipient email for daily reports |

See `.env.example` for a complete template.

---

## Running Diagnostics on AWS

Use `deploy/run_diagnostic.sh` to run one-shot validation tasks (e.g., Databento signal validation) on ECS without affecting the running trading service. See the script for details.
