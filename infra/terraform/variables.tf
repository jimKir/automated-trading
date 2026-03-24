variable "project_name" {
  description = "Project name prefix for all resources"
  type        = string
  default     = "market-data"
}

variable "environment" {
  description = "Deployment environment (dev/staging/prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "location" {
  description = "Azure region for resource deployment"
  type        = string
  default     = "eastus2"
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default = {
    project   = "market-data-platform"
    managed   = "terraform"
    component = "data-infra"
  }
}

variable "container_cpu" {
  description = "CPU cores for the container instance"
  type        = number
  default     = 2
}

variable "container_memory" {
  description = "Memory (GB) for the container instance"
  type        = number
  default     = 4
}

variable "databento_api_key" {
  description = "Databento API key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "alpaca_api_key" {
  description = "Alpaca API key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "alpaca_secret_key" {
  description = "Alpaca secret key"
  type        = string
  sensitive   = true
  default     = ""
}
