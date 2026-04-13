# ============================================================
#  Trading System — AWS Infrastructure (Terraform)
#  Region: eu-north-1 (Stockholm)
#  Resources: ECR, ECS Fargate, S3, IAM, Security Group,
#             Secrets Manager, CloudWatch, EventBridge
# ============================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # State stored in existing S3 bucket
  backend "s3" {
    bucket = "trading-data-380277571671-eu-north-1-an"
    key    = "terraform/trading-bot.tfstate"
    region = "eu-north-1"
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  name_prefix = "${var.project_name}-${var.environment}"
  common_tags = merge(var.tags, {
    environment = var.environment
    managed_by  = "terraform"
    project     = var.project_name
  })
}

# ── ECR Repository ────────────────────────────────────────────────────────────
resource "aws_ecr_repository" "trading" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = local.common_tags
}

resource "aws_ecr_lifecycle_policy" "trading" {
  repository = aws_ecr_repository.trading.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 5 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = { type = "expire" }
    }]
  })
}

# ── Secrets Manager ──────────────────────────────────────────────────────────
resource "aws_secretsmanager_secret" "alpaca_key" {
  name                    = "trading/alpaca_api_key"
  recovery_window_in_days = 0
  tags                    = local.common_tags
}

resource "aws_secretsmanager_secret_version" "alpaca_key" {
  secret_id     = aws_secretsmanager_secret.alpaca_key.id
  secret_string = var.alpaca_api_key
}

resource "aws_secretsmanager_secret" "alpaca_secret" {
  name                    = "trading/alpaca_api_secret"
  recovery_window_in_days = 0
  tags                    = local.common_tags
}

resource "aws_secretsmanager_secret_version" "alpaca_secret" {
  secret_id     = aws_secretsmanager_secret.alpaca_secret.id
  secret_string = var.alpaca_api_secret
}

# ── IAM Roles ─────────────────────────────────────────────────────────────────
data "aws_iam_policy_document" "ecs_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# Execution role — ECS agent permissions
resource "aws_iam_role" "ecs_execution" {
  name               = "${local.name_prefix}-ecs-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_trust.json
  tags               = local.common_tags
}

resource "aws_iam_role_policy_attachment" "ecs_execution_basic" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_execution_secrets" {
  name = "secrets-read"
  role = aws_iam_role.ecs_execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = [
        aws_secretsmanager_secret.alpaca_key.arn,
        aws_secretsmanager_secret.alpaca_secret.arn,
      ]
    }]
  })
}

# Task role — application permissions (S3, CloudWatch)
resource "aws_iam_role" "ecs_task" {
  name               = "${local.name_prefix}-ecs-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_trust.json
  tags               = local.common_tags
}

resource "aws_iam_role_policy" "ecs_task_s3" {
  name = "s3-data-access"
  role = aws_iam_role.ecs_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject", "s3:PutObject",
        "s3:ListBucket", "s3:DeleteObject"
      ]
      Resource = [
        "arn:aws:s3:::${var.s3_data_bucket}",
        "arn:aws:s3:::${var.s3_data_bucket}/*"
      ]
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_cloudwatch" {
  role       = aws_iam_role.ecs_task.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

# ── Networking ────────────────────────────────────────────────────────────────
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "aws_security_group" "trading" {
  name        = "${local.name_prefix}-sg"
  description = "Trading bot outbound only"
  vpc_id      = data.aws_vpc.default.id

  # Outbound: all (needs to reach Alpaca API, yfinance, Kalshi)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

# ── CloudWatch Log Group ──────────────────────────────────────────────────────
resource "aws_cloudwatch_log_group" "trading" {
  name              = "/ecs/${local.name_prefix}"
  retention_in_days = 30
  tags              = local.common_tags
}

# ── CloudWatch Alarm — bot stopped logging ───────────────────────────────────
resource "aws_cloudwatch_metric_alarm" "bot_dead" {
  alarm_name          = "${local.name_prefix}-bot-dead"
  alarm_description   = "Trading bot has stopped logging for 30 minutes"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  metric_name         = "IncomingLogEvents"
  namespace           = "AWS/Logs"
  period              = 1800
  statistic           = "Sum"
  threshold           = 1
  treat_missing_data  = "breaching"

  dimensions = {
    LogGroupName = aws_cloudwatch_log_group.trading.name
  }

  tags = local.common_tags
}

# ── ECS Cluster ───────────────────────────────────────────────────────────────
resource "aws_ecs_cluster" "trading" {
  name = "${local.name_prefix}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = local.common_tags
}

resource "aws_ecs_cluster_capacity_providers" "trading" {
  cluster_name       = aws_ecs_cluster.trading.name
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 1
  }
}

# ── ECS Task Definition ───────────────────────────────────────────────────────
resource "aws_ecs_task_definition" "trading" {
  family                   = "${local.name_prefix}-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name      = "trading"
    image     = "${aws_ecr_repository.trading.repository_url}:latest"
    essential = true

    command = [
      "python", "execution/live_engine.py",
      "--mode", var.trading_mode,
      "--loop-interval", "60"
    ]

    environment = [
      { name = "TRADING_MODE",        value = var.trading_mode },
      { name = "DATA_SOURCE",         value = "s3" },
      { name = "AWS_DEFAULT_REGION",  value = var.aws_region },
      { name = "S3_DATA_BUCKET",      value = var.s3_data_bucket },
      { name = "TZ",                  value = "Europe/Athens" },
      { name = "PYTHONUNBUFFERED",    value = "1" },
    ]

    secrets = [
      {
        name      = "ALPACA_API_KEY"
        valueFrom = aws_secretsmanager_secret.alpaca_key.arn
      },
      {
        name      = "ALPACA_API_SECRET"
        valueFrom = aws_secretsmanager_secret.alpaca_secret.arn
      },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.trading.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }

    healthCheck = {
      command     = ["CMD-SHELL", "pgrep -f live_engine.py || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }
  }])

  tags = local.common_tags
}

# ── ECS Service ───────────────────────────────────────────────────────────────
resource "aws_ecs_service" "trading" {
  name                               = "${local.name_prefix}-service"
  cluster                            = aws_ecs_cluster.trading.id
  task_definition                    = aws_ecs_task_definition.trading.arn
  desired_count                      = var.desired_count
  launch_type                        = "FARGATE"
  platform_version                   = "LATEST"
  health_check_grace_period_seconds  = 60

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.trading.id]
    assign_public_ip = true
  }

  # Ignore desired_count changes (EventBridge scales this)
  lifecycle {
    ignore_changes = [desired_count]
  }

  tags = local.common_tags
}

# ── EventBridge — Start at market open (Mon-Fri 14:25 UTC = 09:25 ET) ────────
resource "aws_iam_role" "eventbridge" {
  name = "${local.name_prefix}-eventbridge"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "scheduler.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
  tags = local.common_tags
}

resource "aws_iam_role_policy" "eventbridge_ecs" {
  name = "ecs-scale"
  role = aws_iam_role.eventbridge.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["ecs:UpdateService"]
      Resource = [aws_ecs_service.trading.id]
    }]
  })
}

resource "aws_scheduler_schedule" "start_trading" {
  name       = "${local.name_prefix}-start"
  group_name = "default"

  flexible_time_window { mode = "OFF" }

  # Mon-Fri 14:25 UTC (09:25 ET — 5 min before open)
  schedule_expression          = "cron(25 14 ? * MON-FRI *)"
  schedule_expression_timezone = "UTC"

  target {
    arn      = "arn:aws:scheduler:::aws-sdk:ecs:updateService"
    role_arn = aws_iam_role.eventbridge.arn
    input = jsonencode({
      Cluster      = aws_ecs_cluster.trading.name
      Service      = aws_ecs_service.trading.name
      DesiredCount = 1
    })
  }
}

resource "aws_scheduler_schedule" "stop_trading" {
  name       = "${local.name_prefix}-stop"
  group_name = "default"

  flexible_time_window { mode = "OFF" }

  # Mon-Fri 21:05 UTC (17:05 ET — 5 min after close)
  schedule_expression          = "cron(5 21 ? * MON-FRI *)"
  schedule_expression_timezone = "UTC"

  target {
    arn      = "arn:aws:scheduler:::aws-sdk:ecs:updateService"
    role_arn = aws_iam_role.eventbridge.arn
    input = jsonencode({
      Cluster      = aws_ecs_cluster.trading.name
      Service      = aws_ecs_service.trading.name
      DesiredCount = 0
    })
  }
}
