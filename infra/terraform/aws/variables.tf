variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "eu-north-1"
}

variable "project_name" {
  description = "Project name prefix for all resources"
  type        = string
  default     = "trading-bot"
}

variable "environment" {
  description = "Deployment environment (production = always active, paper = on-demand testing)"
  type        = string
  default     = "paper"
  validation {
    condition     = contains(["production", "paper"], var.environment)
    error_message = "Must be 'production' or 'paper'."
  }
}

variable "trading_mode" {
  description = "Engine trading mode passed to live_engine.py --mode"
  type        = string
  default     = "paper"
}

variable "s3_data_bucket" {
  description = "S3 bucket containing historical market data"
  type        = string
  default     = "trading-data-380277571671-eu-north-1-an"
}

variable "task_cpu" {
  description = "Fargate task CPU units (512=0.5vCPU)"
  type        = string
  default     = "512"
}

variable "task_memory" {
  description = "Fargate task memory (MB)"
  type        = string
  default     = "1024"
}

variable "desired_count" {
  description = "Initial ECS service desired count (production=1, paper=0)"
  type        = number
  default     = 0
}

variable "enable_schedules" {
  description = "Create EventBridge market-hours schedules (true for production, false for paper)"
  type        = bool
  default     = false
}

variable "alpaca_api_key" {
  description = "Alpaca API key (stored in Secrets Manager)"
  type        = string
  sensitive   = true
}

variable "alpaca_api_secret" {
  description = "Alpaca API secret (stored in Secrets Manager)"
  type        = string
  sensitive   = true
}

variable "tags" {
  description = "Tags applied to all resources"
  type        = map(string)
  default = {
    project    = "automated-trading"
    managed_by = "terraform"
  }
}
