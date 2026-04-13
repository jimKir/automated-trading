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
  description = "Deployment environment"
  type        = string
  default     = "paper"
  validation {
    condition     = contains(["paper", "live"], var.environment)
    error_message = "Must be paper or live."
  }
}

variable "trading_mode" {
  description = "Engine trading mode"
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
  description = "Initial ECS service desired count"
  type        = number
  default     = 0
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
