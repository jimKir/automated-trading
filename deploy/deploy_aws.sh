#!/bin/bash
# ============================================================
#  Trading System — AWS ECS Fargate Deployment
#  Region: eu-north-1 | Mode: paper trading
#  Uses: ECR + ECS Fargate + Secrets Manager + S3
# ============================================================
set -euo pipefail

REGION="${AWS_REGION:-eu-north-1}"
PROJECT="trading-bot"
ENV="${TRADING_ENV:-paper}"
CLUSTER="${PROJECT}-${ENV}-cluster"
SERVICE="${PROJECT}-${ENV}-service"
REPO="${PROJECT}"
LOG_GROUP="/ecs/${PROJECT}-${ENV}"
S3_DATA="trading-data-380277571671-eu-north-1-an"

echo "============================================================"
echo "  Trading System — AWS Deployment"
echo "  Region: $REGION | Mode: $ENV"
echo "============================================================"

# ── Preflight ─────────────────────────────────────────────
echo "[1/6] Preflight checks..."
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
echo "  Account: $ACCOUNT | Region: $REGION"

if [ -f .env ]; then source .env; fi

if [ -z "${ALPACA_API_KEY:-}" ] || [ "$ALPACA_API_KEY" = "your_new_alpaca_key_here" ]; then
  echo "  ERROR: Set ALPACA_API_KEY in .env first (rotate at app.alpaca.markets)"
  exit 1
fi

# ── Build & push Docker image ─────────────────────────────
echo "[2/6] Building and pushing Docker image..."
ECR_URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}"
aws ecr describe-repositories --repository-names $REPO --region $REGION &>/dev/null || \
  aws ecr create-repository --repository-name $REPO --region $REGION >/dev/null

aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $ECR_URI

IMAGE_TAG=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")
docker build -t ${REPO}:${IMAGE_TAG} .
docker tag ${REPO}:${IMAGE_TAG} ${ECR_URI}:${IMAGE_TAG}
docker tag ${REPO}:${IMAGE_TAG} ${ECR_URI}:latest
docker push ${ECR_URI}:${IMAGE_TAG}
docker push ${ECR_URI}:latest
echo "  Pushed: ${ECR_URI}:${IMAGE_TAG}"

# ── Sync data to S3 ───────────────────────────────────────
echo "[3/6] Syncing historical data to S3..."
if [ -d data/historical/daily ] && [ "$(ls data/historical/daily/*.parquet 2>/dev/null | wc -l)" -gt 0 ]; then
  aws s3 sync data/historical/daily/ \
    s3://${S3_DATA}/historical/daily/ --quiet
  echo "  Synced $(ls data/historical/daily/*.parquet | wc -l) parquet files"
else
  echo "  No local parquet files — ECS will sync from S3 on startup"
fi

# ── Store secrets ─────────────────────────────────────────
echo "[4/6] Storing credentials in Secrets Manager..."
store_secret() {
  local name=$1 val=$2
  aws secretsmanager describe-secret --secret-id "$name" --region $REGION &>/dev/null && \
    aws secretsmanager put-secret-value --secret-id "$name" --secret-string "$val" --region $REGION >/dev/null || \
    aws secretsmanager create-secret --name "$name" --secret-string "$val" --region $REGION >/dev/null
  echo "  Stored: $name"
}
store_secret "trading/alpaca_api_key"    "$ALPACA_API_KEY"
store_secret "trading/alpaca_api_secret" "$ALPACA_API_SECRET"

# ── Terraform apply ───────────────────────────────────────
echo "[5/6] Applying Terraform..."
cd infra/terraform/aws

if [ ! -f terraform.tfvars ]; then
  cat > terraform.tfvars << TFEOF
aws_region        = "$REGION"
project_name      = "$PROJECT"
environment       = "$ENV"
trading_mode      = "$ENV"
s3_data_bucket    = "$S3_DATA"
alpaca_api_key    = "$ALPACA_API_KEY"
alpaca_api_secret = "$ALPACA_API_SECRET"
TFEOF
fi

terraform init -upgrade -input=false
terraform apply -auto-approve -input=false

ECR_URL=$(terraform output -raw ecr_repository_url)
CLUSTER_NAME=$(terraform output -raw ecs_cluster_name)
SERVICE_NAME=$(terraform output -raw ecs_service_name)
LOG_GROUP_NAME=$(terraform output -raw cloudwatch_log_group)

cd ../../..

# ── Start service ─────────────────────────────────────────
echo "[6/6] Starting ECS service..."
aws ecs update-service \
  --cluster $CLUSTER_NAME \
  --service $SERVICE_NAME \
  --desired-count 1 \
  --force-new-deployment \
  --region $REGION >/dev/null

echo ""
echo "============================================================"
echo "  DEPLOYMENT COMPLETE"
echo "============================================================"
echo ""
echo "  Tail logs:"
echo "    aws logs tail $LOG_GROUP_NAME --follow --region $REGION"
echo ""
echo "  Check service:"
echo "    aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $REGION --query 'services[0].{Status:status,Running:runningCount,Desired:desiredCount}'"
echo ""
echo "  Stop trading:"
echo "    aws ecs update-service --cluster $CLUSTER_NAME --service $SERVICE_NAME --desired-count 0 --region $REGION"
echo ""
echo "  Portfolio status (from Mac):"
echo "    python scripts/status.py"
echo ""
echo "  Estimated cost: ~\$11/month (market hours only via EventBridge)"
echo "============================================================"
