#!/bin/bash
# ============================================================
#  Trading System v15b — AWS ECS Fargate Deployment Script
# ============================================================
# Usage:
#   chmod +x deploy/deploy_aws.sh
#   ./deploy/deploy_aws.sh
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - Docker installed and running
#   - Alpaca API keys ready
# ============================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────
REGION="${AWS_REGION:-eu-west-1}"
CLUSTER_NAME="trading-cluster"
SERVICE_NAME="trading-paper"
REPO_NAME="trading-system"
TASK_FAMILY="trading-paper"
LOG_GROUP="/ecs/trading-paper"

echo "============================================"
echo "  Trading System v15b — AWS Deployment"
echo "============================================"
echo "  Region:  $REGION"
echo ""

# ── Step 1: Get AWS Account ID ─────────────────────────────
echo "[1/8] Getting AWS account ID..."
ACCOUNT=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
if [ -z "$ACCOUNT" ]; then
    echo "ERROR: AWS CLI not configured. Run 'aws configure' first."
    exit 1
fi
echo "  Account: $ACCOUNT"

# ── Step 2: Create ECR repo (if not exists) ────────────────
echo "[2/8] Creating ECR repository..."
aws ecr describe-repositories --repository-names $REPO_NAME --region $REGION >/dev/null 2>&1 || \
    aws ecr create-repository --repository-name $REPO_NAME --region $REGION >/dev/null
ECR_URI="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME"
echo "  ECR: $ECR_URI"

# ── Step 3: Build and push Docker image ────────────────────
echo "[3/8] Building Docker image..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

docker build -t $REPO_NAME . --quiet
docker tag $REPO_NAME:latest $ECR_URI:latest

echo "  Pushing to ECR..."
aws ecr get-login-password --region $REGION | \
    docker login --username AWS --password-stdin $ECR_URI 2>/dev/null
docker push $ECR_URI:latest --quiet
echo "  Image pushed."

# ── Step 4: Store secrets ──────────────────────────────────
echo "[4/8] Setting up secrets..."
store_secret() {
    local name=$1 value=$2
    aws secretsmanager describe-secret --secret-id "$name" --region $REGION >/dev/null 2>&1 && \
        aws secretsmanager put-secret-value --secret-id "$name" --secret-string "$value" --region $REGION >/dev/null || \
        aws secretsmanager create-secret --name "$name" --secret-string "$value" --region $REGION >/dev/null
}

if [ -f .env ]; then
    source .env
fi

if [ -z "${ALPACA_API_KEY:-}" ] || [ "$ALPACA_API_KEY" = "your_alpaca_api_key_here" ]; then
    echo "  WARNING: ALPACA_API_KEY not set in .env — skipping secrets."
    echo "  Run this after setting your keys in .env"
else
    store_secret "trading/alpaca_api_key" "$ALPACA_API_KEY"
    store_secret "trading/alpaca_api_secret" "$ALPACA_API_SECRET"
    echo "  Alpaca secrets stored."
fi

# ── Step 5: Create ECS cluster ─────────────────────────────
echo "[5/8] Creating ECS cluster..."
aws ecs describe-clusters --clusters $CLUSTER_NAME --region $REGION \
    --query "clusters[?status=='ACTIVE'].clusterName" --output text | grep -q $CLUSTER_NAME || \
    aws ecs create-cluster --cluster-name $CLUSTER_NAME --region $REGION >/dev/null
echo "  Cluster: $CLUSTER_NAME"

# ── Step 6: Create IAM roles ──────────────────────────────
echo "[6/8] Setting up IAM roles..."
TRUST_POLICY='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

# Execution role
aws iam get-role --role-name ecsTaskExecutionRole >/dev/null 2>&1 || \
    aws iam create-role --role-name ecsTaskExecutionRole --assume-role-policy-document "$TRUST_POLICY" >/dev/null
aws iam attach-role-policy --role-name ecsTaskExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy 2>/dev/null || true
aws iam attach-role-policy --role-name ecsTaskExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite 2>/dev/null || true

# Task role
aws iam get-role --role-name tradingTaskRole >/dev/null 2>&1 || \
    aws iam create-role --role-name tradingTaskRole --assume-role-policy-document "$TRUST_POLICY" >/dev/null
aws iam attach-role-policy --role-name tradingTaskRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess 2>/dev/null || true

echo "  IAM roles configured."

# ── Step 7: Create log group + register task definition ────
echo "[7/8] Registering task definition..."
aws logs create-log-group --log-group-name $LOG_GROUP --region $REGION 2>/dev/null || true

# Generate task definition JSON
cat > /tmp/task-def.json <<TASKEOF
{
  "family": "$TASK_FAMILY",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::${ACCOUNT}:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::${ACCOUNT}:role/tradingTaskRole",
  "containerDefinitions": [
    {
      "name": "trading",
      "image": "${ECR_URI}:latest",
      "essential": true,
      "command": ["python", "main.py", "paper"],
      "environment": [
        { "name": "TRADING_MODE",  "value": "paper" },
        { "name": "TZ",            "value": "Europe/Athens" }
      ],
      "secrets": [
        { "name": "ALPACA_API_KEY",    "valueFrom": "arn:aws:secretsmanager:${REGION}:${ACCOUNT}:secret:trading/alpaca_api_key" },
        { "name": "ALPACA_API_SECRET", "valueFrom": "arn:aws:secretsmanager:${REGION}:${ACCOUNT}:secret:trading/alpaca_api_secret" }
      ],
      "portMappings": [
        { "containerPort": 8080, "protocol": "tcp" }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "${LOG_GROUP}",
          "awslogs-region": "${REGION}",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 30
      }
    }
  ]
}
TASKEOF

aws ecs register-task-definition --cli-input-json file:///tmp/task-def.json --region $REGION >/dev/null
echo "  Task definition registered."

# ── Step 8: Create/update ECS service ──────────────────────
echo "[8/8] Creating ECS service..."

# Get default VPC networking
VPC=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
      --query "Vpcs[0].VpcId" --output text --region $REGION)
SUBNETS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC" \
          --query "Subnets[*].SubnetId" --output text --region $REGION | tr '\t' ',')
FIRST_SUBNET=$(echo $SUBNETS | cut -d, -f1)

# Create security group (if not exists)
SG=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=trading-sg" "Name=vpc-id,Values=$VPC" \
     --query "SecurityGroups[0].GroupId" --output text --region $REGION 2>/dev/null)
if [ "$SG" = "None" ] || [ -z "$SG" ]; then
    SG=$(aws ec2 create-security-group --group-name trading-sg --description "Trading system" \
         --vpc-id $VPC --query GroupId --output text --region $REGION)
    aws ec2 authorize-security-group-ingress --group-id $SG --protocol tcp --port 8080 --cidr 0.0.0.0/0 --region $REGION 2>/dev/null || true
fi

# Check if service exists
EXISTING=$(aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME \
           --query "services[?status=='ACTIVE'].serviceName" --output text --region $REGION 2>/dev/null)

if [ -n "$EXISTING" ] && [ "$EXISTING" != "None" ]; then
    echo "  Service exists — forcing new deployment..."
    aws ecs update-service --cluster $CLUSTER_NAME --service $SERVICE_NAME \
        --force-new-deployment --region $REGION >/dev/null
else
    echo "  Creating new service..."
    aws ecs create-service \
        --cluster $CLUSTER_NAME \
        --service-name $SERVICE_NAME \
        --task-definition $TASK_FAMILY \
        --desired-count 1 \
        --launch-type FARGATE \
        --network-configuration "awsvpcConfiguration={subnets=[$FIRST_SUBNET],securityGroups=[$SG],assignPublicIp=ENABLED}" \
        --region $REGION >/dev/null
fi

echo ""
echo "============================================"
echo "  DEPLOYMENT COMPLETE"
echo "============================================"
echo ""
echo "  Monitor logs:"
echo "    aws logs tail $LOG_GROUP --follow --region $REGION"
echo ""
echo "  Check status:"
echo "    aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $REGION"
echo ""
echo "  Stop trading:"
echo "    aws ecs update-service --cluster $CLUSTER_NAME --service $SERVICE_NAME --desired-count 0 --region $REGION"
echo ""
echo "  Estimated cost: ~\$11-15/month"
echo "============================================"
