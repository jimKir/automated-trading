# AWS Deployment Guide — Paper Trading

This guide walks you through running the trading system **24/7 on AWS** using
**ECS Fargate** (serverless containers — no servers to manage).
Total estimated cost: **~$15–25/month** for a t3-equivalent Fargate task.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                      AWS Account                    │
│                                                     │
│  ECR Repo          ECS Cluster (Fargate)            │
│  (Docker image) ──▶  Task: trading_paper            │
│                       ├─ main.py paper (loop)       │
│                       └─ healthcheck :8080          │
│                                                     │
│  EventBridge ─────▶  ECS Task: daily_report         │
│  (cron 18:00 UTC)     (runs & exits)                │
│                                                     │
│  Secrets Manager ──▶  API keys injected as env vars │
│  S3 Bucket       ◀──  HTML reports uploaded daily   │
│  SES             ──▶  Email report to you           │
│  CloudWatch Logs ◀──  All container stdout/stderr   │
└─────────────────────────────────────────────────────┘
```

---

## Step 1 — Prerequisites

### 1a. Install tools locally

```bash
# AWS CLI
brew install awscli            # macOS
# or: pip install awscli

# Docker Desktop (needed to build & push image)
# https://www.docker.com/products/docker-desktop/

# Verify
aws --version
docker --version
```

### 1b. Create an AWS account + IAM user

1. Go to https://aws.amazon.com and create a free account
2. In **IAM → Users**, create a user `trading-deploy`
3. Attach policy: **AdministratorAccess** (or a scoped policy — see §9)
4. Create **Access Key** → download CSV
5. Configure locally:

```bash
aws configure
# AWS Access Key ID: (from CSV)
# AWS Secret Access Key: (from CSV)
# Default region: eu-west-1     ← or us-east-1, your choice
# Default output: json
```

---

## Step 2 — Push Docker Image to ECR

```bash
# 1. Set your region & account ID
REGION=eu-west-1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

# 2. Create ECR repository
aws ecr create-repository --repository-name trading-system --region $REGION

# 3. Authenticate Docker to ECR
aws ecr get-login-password --region $REGION \
  | docker login --username AWS \
    --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com

# 4. Build and push
cd /path/to/trading_system
docker build -t trading-system .
docker tag trading-system:latest \
  $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/trading-system:latest
docker push \
  $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/trading-system:latest
```

---

## Step 3 — Store Secrets in AWS Secrets Manager

Never put API keys in environment variables directly in ECS.
Use Secrets Manager — ECS injects them at runtime.

```bash
# Store each secret
aws secretsmanager create-secret \
  --name trading/alpaca_api_key \
  --secret-string "your_alpaca_key_here"

aws secretsmanager create-secret \
  --name trading/alpaca_api_secret \
  --secret-string "your_alpaca_secret_here"

aws secretsmanager create-secret \
  --name trading/binance_api_key \
  --secret-string "your_binance_testnet_key"

aws secretsmanager create-secret \
  --name trading/binance_api_secret \
  --secret-string "your_binance_testnet_secret"

# Repeat for any other secrets (SES, S3 bucket name, etc.)
```

---

## Step 4 — Create ECS Cluster

```bash
aws ecs create-cluster \
  --cluster-name trading-cluster \
  --region $REGION
```

---

## Step 5 — Create IAM Role for ECS Task

The container needs permissions to read Secrets Manager, write to S3, and send emails via SES.

```bash
# Create execution role (allows ECS to pull image + secrets)
aws iam create-role \
  --role-name ecsTaskExecutionRole \
  --assume-role-policy-document '{
    "Version":"2012-10-17",
    "Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]
  }'

aws iam attach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

aws iam attach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite

# Create task role (runtime permissions)
aws iam create-role \
  --role-name tradingTaskRole \
  --assume-role-policy-document '{
    "Version":"2012-10-17",
    "Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]
  }'

aws iam attach-role-policy \
  --role-name tradingTaskRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
  --role-name tradingTaskRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSESFullAccess
```

---

## Step 6 — Register ECS Task Definition

Save this as `deploy/task-definition.json`, fill in your `ACCOUNT` and `REGION`:

```json
{
  "family": "trading-paper",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn":      "arn:aws:iam::ACCOUNT:role/tradingTaskRole",
  "containerDefinitions": [
    {
      "name": "trading",
      "image": "ACCOUNT.dkr.ecr.REGION.amazonaws.com/trading-system:latest",
      "essential": true,
      "command": ["python", "main.py", "paper"],
      "environment": [
        { "name": "TRADING_MODE",  "value": "paper" },
        { "name": "TZ",            "value": "Europe/Athens" },
        { "name": "AWS_REGION",    "value": "REGION" },
        { "name": "S3_BUCKET",     "value": "your-trading-reports-bucket" },
        { "name": "SES_SENDER",    "value": "reports@yourdomain.com" },
        { "name": "SES_RECIPIENT", "value": "you@yourdomain.com" }
      ],
      "secrets": [
        { "name": "ALPACA_API_KEY",      "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT:secret:trading/alpaca_api_key" },
        { "name": "ALPACA_API_SECRET",   "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT:secret:trading/alpaca_api_secret" },
        { "name": "BINANCE_API_KEY",     "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT:secret:trading/binance_api_key" },
        { "name": "BINANCE_API_SECRET",  "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT:secret:trading/binance_api_secret" }
      ],
      "portMappings": [
        { "containerPort": 8080, "protocol": "tcp" }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group":         "/ecs/trading-paper",
          "awslogs-region":        "REGION",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command":     ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"],
        "interval":    30,
        "timeout":     5,
        "retries":     3,
        "startPeriod": 20
      }
    }
  ]
}
```

Register it:

```bash
# Create CloudWatch log group first
aws logs create-log-group --log-group-name /ecs/trading-paper --region $REGION

# Register task definition
aws ecs register-task-definition \
  --cli-input-json file://deploy/task-definition.json \
  --region $REGION
```

---

## Step 7 — Create ECS Service (keeps container running 24/7)

```bash
# Get your default VPC and subnets
VPC=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
      --query "Vpcs[0].VpcId" --output text)
SUBNETS=$(aws ec2 describe-subnets \
          --filters "Name=vpc-id,Values=$VPC" \
          --query "Subnets[*].SubnetId" --output text | tr '\t' ',')

# Create a security group allowing outbound + health check
SG=$(aws ec2 create-security-group \
       --group-name trading-sg \
       --description "Trading system SG" \
       --vpc-id $VPC \
       --query GroupId --output text)

aws ec2 authorize-security-group-ingress \
  --group-id $SG --protocol tcp --port 8080 --cidr 0.0.0.0/0

aws ec2 authorize-security-group-egress \
  --group-id $SG --protocol -1 --port -1 --cidr 0.0.0.0/0 2>/dev/null || true

# Create the ECS service
aws ecs create-service \
  --cluster trading-cluster \
  --service-name trading-paper \
  --task-definition trading-paper \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SG],assignPublicIp=ENABLED}" \
  --region $REGION
```

The service will automatically restart the container if it crashes.

---

## Step 8 — Schedule Daily Report (EventBridge)

This runs `daily_report.py` every day at 18:00 UTC (21:00 Athens time).

```bash
# Create the scheduled rule
aws events put-rule \
  --name trading-daily-report \
  --schedule-expression "cron(0 18 * * ? *)" \
  --state ENABLED \
  --region $REGION

# Get the task definition ARN
TASK_DEF=$(aws ecs describe-task-definition \
  --task-definition trading-paper \
  --query "taskDefinition.taskDefinitionArn" --output text)

# Create a role for EventBridge to run ECS tasks
aws iam create-role \
  --role-name eventsECSRole \
  --assume-role-policy-document '{
    "Version":"2012-10-17",
    "Statement":[{"Effect":"Allow","Principal":{"Service":"events.amazonaws.com"},"Action":"sts:AssumeRole"}]
  }'
aws iam attach-role-policy \
  --role-name eventsECSRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonECS_FullAccess

EVENTS_ROLE=$(aws iam get-role --role-name eventsECSRole \
              --query "Role.Arn" --output text)

# Add the target (daily_report task)
aws events put-targets \
  --rule trading-daily-report \
  --targets "[{
    \"Id\": \"daily-report\",
    \"Arn\": \"arn:aws:ecs:$REGION:$ACCOUNT:cluster/trading-cluster\",
    \"RoleArn\": \"$EVENTS_ROLE\",
    \"EcsParameters\": {
      \"TaskDefinitionArn\": \"$TASK_DEF\",
      \"TaskCount\": 1,
      \"LaunchType\": \"FARGATE\",
      \"NetworkConfiguration\": {
        \"awsvpcConfiguration\": {
          \"Subnets\": [\"$(echo $SUBNETS | cut -d, -f1)\"],
          \"SecurityGroups\": [\"$SG\"],
          \"AssignPublicIp\": \"ENABLED\"
        }
      },
      \"Overrides\": {
        \"ContainerOverrides\": [{
          \"Name\": \"trading\",
          \"Command\": [\"python\", \"daily_report.py\", \"--email\", \"--s3\"]
        }]
      }
    }
  }]" \
  --region $REGION
```

---

## Step 9 — Verify It's Running

```bash
# Check service status
aws ecs describe-services \
  --cluster trading-cluster \
  --services trading-paper \
  --query "services[0].{status:status,running:runningCount,desired:desiredCount}" \
  --region $REGION

# Get the public IP of the running task
TASK_ARN=$(aws ecs list-tasks \
  --cluster trading-cluster \
  --service-name trading-paper \
  --query "taskArns[0]" --output text --region $REGION)

ENI=$(aws ecs describe-tasks \
  --cluster trading-cluster \
  --tasks $TASK_ARN \
  --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" \
  --output text --region $REGION)

PUBLIC_IP=$(aws ec2 describe-network-interfaces \
  --network-interface-ids $ENI \
  --query "NetworkInterfaces[0].Association.PublicIp" \
  --output text --region $REGION)

echo "Health check: http://$PUBLIC_IP:8080/health"
echo "Status:       http://$PUBLIC_IP:8080/status"
echo "Signals:      http://$PUBLIC_IP:8080/signals"

# Test it
curl http://$PUBLIC_IP:8080/health
```

### View live logs

```bash
aws logs tail /ecs/trading-paper --follow --region $REGION
```

---

## Step 10 — Update After Code Changes

```bash
# Rebuild and push new image
docker build -t trading-system .
docker tag trading-system:latest \
  $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/trading-system:latest
docker push \
  $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/trading-system:latest

# Force ECS to pull the new image (rolling restart, zero downtime)
aws ecs update-service \
  --cluster trading-cluster \
  --service trading-paper \
  --force-new-deployment \
  --region $REGION
```

---

## Cost Estimate

| Service | Usage | Monthly Cost |
|---------|-------|-------------|
| ECS Fargate (0.25 vCPU, 0.5 GB) | 720 hrs/month | ~$9 |
| ECR storage | ~200 MB | ~$0.02 |
| CloudWatch Logs | ~500 MB/month | ~$0.25 |
| Secrets Manager | 4 secrets | ~$1.60 |
| SES | 30 emails/month | Free tier |
| S3 | 30 HTML files | ~$0.01 |
| **Total** | | **~$11–12/month** |

---

## Moving to Live Trading (When Ready)

1. Change `TRADING_MODE=live` in Secrets Manager / env
2. Update `settings.yaml`: `system.mode: live`
3. Switch Alpaca to `base_url: https://api.alpaca.markets`
4. Switch Binance `testnet: false`
5. Increase Fargate CPU/memory: `"cpu": "1024", "memory": "2048"`
6. Consider adding an ALB + HTTPS for the health endpoint

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Task keeps stopping | Check CloudWatch logs: `aws logs tail /ecs/trading-paper --follow` |
| `NoCredentialsError` | Ensure Secrets Manager ARNs in task definition are exact |
| Health check failing | Verify security group allows inbound TCP 8080 |
| No data fetched | yfinance may be rate-limited — add `time.sleep(1)` in `data/feed.py` |
| Image not found | Re-run `docker push` and `update-service --force-new-deployment` |
