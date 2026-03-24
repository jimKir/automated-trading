#!/usr/bin/env bash
# Setup infrastructure for Market Data Platform
# Usage: ./scripts/setup_infra.sh [environment]

set -euo pipefail

ENVIRONMENT="${1:-dev}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="${SCRIPT_DIR}/../infra/terraform"

echo "=== Market Data Platform Infrastructure Setup ==="
echo "Environment: ${ENVIRONMENT}"
echo ""

# Check prerequisites
command -v terraform >/dev/null 2>&1 || { echo "Error: terraform not found"; exit 1; }
command -v az >/dev/null 2>&1 || { echo "Error: Azure CLI (az) not found"; exit 1; }

# Check Azure login
if ! az account show >/dev/null 2>&1; then
    echo "Not logged into Azure. Running 'az login'..."
    az login
fi

echo "Using Azure subscription: $(az account show --query name -o tsv)"
echo ""

# Initialize Terraform
cd "${INFRA_DIR}"
echo "Initializing Terraform..."
terraform init

# Plan
echo ""
echo "Planning infrastructure changes..."
terraform plan \
    -var "environment=${ENVIRONMENT}" \
    -out="tfplan.${ENVIRONMENT}"

# Apply (with confirmation)
echo ""
read -rp "Apply changes? (y/N): " confirm
if [[ "${confirm}" =~ ^[Yy]$ ]]; then
    terraform apply "tfplan.${ENVIRONMENT}"
    echo ""
    echo "=== Infrastructure deployed successfully ==="
    echo ""
    echo "Outputs:"
    terraform output
else
    echo "Aborted."
fi
