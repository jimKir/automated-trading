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

# ECR (shared, not env-prefixed)
safe_import "aws_ecr_repository.trading" "${PROJECT}"

# Secrets Manager (shared, not env-prefixed)
# Import by ARN — need to look up the full ARN with random suffix
for secret_name in "trading/alpaca_api_key" "trading/alpaca_api_secret"; do
  resource="aws_secretsmanager_secret.$(echo "$secret_name" | sed 's|trading/alpaca_||; s|_.*||')"
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

# IAM Roles
safe_import "aws_iam_role.ecs_execution" "${PREFIX}-ecs-execution"
safe_import "aws_iam_role.ecs_task" "${PREFIX}-ecs-task"

# Security Group — need to look up by name to get the ID
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=${PREFIX}-sg" \
  --region "$REGION" --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")
if [[ "$SG_ID" != "None" && -n "$SG_ID" ]]; then
  safe_import "aws_security_group.trading" "$SG_ID"
else
  echo "  ○ Security group ${PREFIX}-sg not found — will be created"
fi

# CloudWatch Log Group
safe_import "aws_cloudwatch_log_group.trading" "/ecs/${PREFIX}"

echo ""
echo "Import complete. Run 'terraform plan' to verify."
