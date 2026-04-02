#!/usr/bin/env bash
# ============================================================
# Run Databento signal validation as a one-shot ECS Fargate task
# Uses your EXISTING cluster, image, and IAM roles.
# Takes ~20-30 min, logs stream to CloudWatch.
# ============================================================
set -euo pipefail

REGION="${AWS_DEFAULT_REGION:-eu-west-1}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
CLUSTER="trading-cluster"
LOG_GROUP="/ecs/trading-diagnostics"

echo "Account: $ACCOUNT  Region: $REGION"

# ── Step 1: Add DATABENTO_KEY to Secrets Manager ────────────────
echo ""
echo "Step 1/5: Storing Databento key in Secrets Manager..."
aws secretsmanager create-secret \
  --name trading/databento_key \
  --secret-string "db-SpVxiQLLTdDe9iD3sLwTpiqgBjtxk" \
  --region $REGION 2>/dev/null \
|| aws secretsmanager update-secret \
  --secret-id trading/databento_key \
  --secret-string "db-SpVxiQLLTdDe9iD3sLwTpiqgBjtxk" \
  --region $REGION
echo "  ✅ Secret stored at trading/databento_key"

# ── Step 2: Create CloudWatch log group ─────────────────────────
echo ""
echo "Step 2/5: Creating CloudWatch log group..."
aws logs create-log-group \
  --log-group-name $LOG_GROUP \
  --region $REGION 2>/dev/null || echo "  (already exists)"
echo "  ✅ Log group: $LOG_GROUP"

# ── Step 3: Get networking info from existing service ───────────
echo ""
echo "Step 3/5: Reading VPC / subnets / security group from existing service..."

# Pull from the existing trading-paper service
SERVICE_INFO=$(aws ecs describe-services \
  --cluster $CLUSTER \
  --services trading-paper \
  --query "services[0].networkConfiguration.awsvpcConfiguration" \
  --region $REGION --output json 2>/dev/null)

SUBNET=$(echo $SERVICE_INFO | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['subnets'][0])")
SG=$(echo $SERVICE_INFO | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['securityGroups'][0])")
echo "  Subnet: $SUBNET  SG: $SG"

# ── Step 4: Register one-shot task definition ───────────────────
echo ""
echo "Step 4/5: Registering diagnostic task definition..."

IMAGE="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/trading-system:latest"
EXEC_ROLE="arn:aws:iam::$ACCOUNT:role/ecsTaskExecutionRole"
TASK_ROLE="arn:aws:iam::$ACCOUNT:role/tradingTaskRole"

cat > /tmp/diagnostic-task.json <<TASKDEF
{
  "family": "trading-diagnostic",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "8192",
  "executionRoleArn": "$EXEC_ROLE",
  "taskRoleArn": "$TASK_ROLE",
  "containerDefinitions": [
    {
      "name": "diagnostic",
      "image": "$IMAGE",
      "essential": true,
      "command": [
        "python",
        "diagnostics/validate_databento_signals.py"
      ],
      "environment": [
        { "name": "TZ",           "value": "Europe/Athens" },
        { "name": "AWS_REGION",   "value": "$REGION" },
        { "name": "PYTHONUNBUFFERED", "value": "1" }
      ],
      "secrets": [
        {
          "name": "ALPACA_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:$REGION:$ACCOUNT:secret:trading/alpaca_api_key"
        },
        {
          "name": "ALPACA_API_SECRET",
          "valueFrom": "arn:aws:secretsmanager:$REGION:$ACCOUNT:secret:trading/alpaca_api_secret"
        },
        {
          "name": "DATABENTO_KEY",
          "valueFrom": "arn:aws:secretsmanager:$REGION:$ACCOUNT:secret:trading/databento_key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group":         "$LOG_GROUP",
          "awslogs-region":        "$REGION",
          "awslogs-stream-prefix": "diagnostic"
        }
      }
    }
  ]
}
TASKDEF

aws ecs register-task-definition \
  --cli-input-json file:///tmp/diagnostic-task.json \
  --region $REGION \
  --query "taskDefinition.taskDefinitionArn" --output text
echo "  ✅ Task definition registered"

# ── Step 5: Run the task ─────────────────────────────────────────
echo ""
echo "Step 5/5: Launching diagnostic task..."
TASK_ARN=$(aws ecs run-task \
  --cluster $CLUSTER \
  --task-definition trading-diagnostic \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNET],securityGroups=[$SG],assignPublicIp=ENABLED}" \
  --region $REGION \
  --query "tasks[0].taskArn" --output text)

echo "  ✅ Task launched: $TASK_ARN"
echo ""
echo "================================================================"
echo "  Diagnostic is running. Takes ~20-30 minutes."
echo ""
echo "  Watch logs in real time:"
echo "  aws logs tail $LOG_GROUP --follow --region $REGION"
echo ""
echo "  Check task status:"
echo "  aws ecs describe-tasks --cluster $CLUSTER --tasks $TASK_ARN --region $REGION --query 'tasks[0].lastStatus'"
echo ""
echo "  Task ARN: $TASK_ARN"
echo "================================================================"
