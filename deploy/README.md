# deploy/

AWS deployment scripts and guides.

## `aws_setup.md`

Complete step-by-step guide for deploying to AWS ECS Fargate (~$12/month). Covers ECR image push, Secrets Manager, ECS cluster/service creation, IAM roles, health checks, EventBridge daily reports.

## `deploy_aws.sh`

Automated deployment script. Runs all 8 steps from `aws_setup.md` in sequence:
1. Get AWS account ID
2. Create ECR repository
3. Build and push Docker image
4. Set up Secrets Manager entries (reads from `.env`)
5. Create ECS cluster
6. Create IAM roles (execution + task)
7. Register task definition
8. Create ECS service

Idempotent -- checks if resources exist before creating.

## `run_diagnostic.sh`

Runs a one-shot diagnostic task on AWS (e.g., Databento signal validation). Launches a separate ECS task using the existing cluster and image. ~20-30 minute duration, output to CloudWatch logs.

See also [DEPLOY_AWS.md](../DEPLOY_AWS.md) at the repo root for a higher-level deployment overview covering EC2, ECS, and Lambda options.
