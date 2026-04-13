#!/usr/bin/env bash
# ============================================================
#  Import pre-existing AWS resources into Terraform state
#  Run once per workspace after 'terraform init' + workspace select.
#
#  Usage: ./import_existing.sh <environment>
#  Example: ./import_existing.sh paper
#
#  Safe to re-run — skips resources already in state.
# ============================================================
set -euo pipefail

ENV="${1:?Usage: $0 <environment>}"
PROJECT="trading-bot"
REGION="${AWS_DEFAULT_REGION:-eu-north-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
PREFIX="${PROJECT}-${ENV}"

echo "Importing existing resources for workspace: ${ENV}"
echo "Account: ${ACCOUNT_ID}, Region: ${REGION}"

# Helper: import only if not already in state
safe_import() {
  local addr="$1"
  local id="$2"
  if terraform state show "$addr" &>/dev/null; then
    echo "  ✓ ${addr} already in state — skipping"
  else
    echo "  → Importing ${addr} ..."
    terraform import -var-file="envs/${ENV}.tfvars" "$addr" "$id" || {
      echo "  ✗ Failed to import ${addr} (may not exist yet — will be created)"
    }
  fi
}

# ── ECR ──────────────────────────────────────────────────────────────────────
safe_import "aws_ecr_repository.trading" "${PROJECT}"
safe_import "aws_ecr_lifecycle_policy.trading" "${PROJECT}"

# ── Secrets Manager ──────────────────────────────────────────────────────────
for secret_name in "trading/alpaca_api_key" "trading/alpaca_api_secret"; do
  if [[ "$secret_name" == *"api_key"* ]]; then
    resource="aws_secretsmanager_secret.alpaca_key"
  else
    resource="aws_secretsmanager_secret.alpaca_secret"
  fi
  arn=$(aws secretsmanager describe-secret --secret-id "$secret_name" --region "$REGION" --query ARN --output text 2>/dev/null || echo "")
  if [[ -n "$arn" && "$arn" != "None" ]]; then
    safe_import "$resource" "$arn"
  else
    echo "  ○ Secret ${secret_name} not found in AWS — will be created"
  fi
done

# ── IAM Roles + Policies ────────────────────────────────────────────────────
safe_import "aws_iam_role.ecs_execution" "${PREFIX}-ecs-execution"
safe_import "aws_iam_role_policy_attachment.ecs_execution_basic" "${PREFIX}-ecs-execution/arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
safe_import "aws_iam_role_policy.ecs_execution_secrets" "${PREFIX}-ecs-execution:secrets-read"

safe_import "aws_iam_role.ecs_task" "${PREFIX}-ecs-task"
safe_import "aws_iam_role_policy.ecs_task_s3" "${PREFIX}-ecs-task:s3-data-access"
safe_import "aws_iam_role_policy_attachment.ecs_task_cloudwatch" "${PREFIX}-ecs-task/arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"

# ── Security Group ──────────────────────────────────────────────────────────
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=${PREFIX}-sg" \
  --region "$REGION" --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")
if [[ "$SG_ID" != "None" && -n "$SG_ID" ]]; then
  safe_import "aws_security_group.trading" "$SG_ID"
else
  echo "  ○ Security group ${PREFIX}-sg not found — will be created"
fi

# ── CloudWatch ──────────────────────────────────────────────────────────────
safe_import "aws_cloudwatch_log_group.trading" "/ecs/${PREFIX}"
safe_import "aws_cloudwatch_metric_alarm.bot_dead" "${PREFIX}-bot-dead"

# ── ECS Cluster ─────────────────────────────────────────────────────────────
CLUSTER_ARN=$(aws ecs describe-clusters --clusters "${PREFIX}-cluster" --region "$REGION" \
  --query 'clusters[0].clusterArn' --output text 2>/dev/null || echo "None")
if [[ "$CLUSTER_ARN" != "None" && -n "$CLUSTER_ARN" ]]; then
  safe_import "aws_ecs_cluster.trading" "${PREFIX}-cluster"
  safe_import "aws_ecs_cluster_capacity_providers.trading" "${PREFIX}-cluster"
else
  echo "  ○ ECS cluster ${PREFIX}-cluster not found — will be created"
fi

# ── ECS Task Definition ─────────────────────────────────────────────────────
# Import the latest active revision
TASK_ARN=$(aws ecs describe-task-definition --task-definition "${PREFIX}-task" --region "$REGION" \
  --query 'taskDefinition.taskDefinitionArn' --output text 2>/dev/null || echo "None")
if [[ "$TASK_ARN" != "None" && -n "$TASK_ARN" ]]; then
  safe_import "aws_ecs_task_definition.trading" "${TASK_ARN}"
else
  echo "  ○ ECS task definition ${PREFIX}-task not found — will be created"
fi

# ── ECS Service ─────────────────────────────────────────────────────────────
SERVICE_ARN=$(aws ecs describe-services --cluster "${PREFIX}-cluster" --services "${PREFIX}-service" \
  --region "$REGION" --query 'services[?status==`ACTIVE`].serviceArn | [0]' --output text 2>/dev/null || echo "None")
if [[ "$SERVICE_ARN" != "None" && -n "$SERVICE_ARN" ]]; then
  safe_import "aws_ecs_service.trading" "${PREFIX}-cluster/${PREFIX}-service"
else
  echo "  ○ ECS service ${PREFIX}-service not found — will be created"
fi

# ── EventBridge Schedules (production only) ─────────────────────────────────
if [[ "$ENV" == "production" ]]; then
  safe_import 'aws_iam_role.eventbridge[0]' "${PREFIX}-eventbridge"
  safe_import 'aws_iam_role_policy.eventbridge_ecs[0]' "${PREFIX}-eventbridge:ecs-scale"

  START_ARN=$(aws scheduler get-schedule --name "${PREFIX}-start" --region "$REGION" \
    --query 'Arn' --output text 2>/dev/null || echo "None")
  if [[ "$START_ARN" != "None" && -n "$START_ARN" ]]; then
    safe_import 'aws_scheduler_schedule.start_trading[0]' "default/${PREFIX}-start"
    safe_import 'aws_scheduler_schedule.stop_trading[0]' "default/${PREFIX}-stop"
  else
    echo "  ○ EventBridge schedules not found — will be created"
  fi
fi

echo ""
echo "Import complete. Run 'terraform plan' to verify."
