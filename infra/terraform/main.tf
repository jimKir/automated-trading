terraform {
  required_version = ">= 1.5.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.80"
    }
  }

  backend "azurerm" {
    resource_group_name  = "market-data-tfstate"
    storage_account_name = "marketdatatfstate"
    container_name       = "tfstate"
    key                  = "market-data-platform.tfstate"
  }
}

provider "azurerm" {
  features {}
}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "${var.project_name}-${var.environment}-rg"
  location = var.location
  tags     = var.tags
}

# Storage Account for data lake
resource "azurerm_storage_account" "datalake" {
  name                     = "${replace(var.project_name, "-", "")}${var.environment}dl"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = var.environment == "prod" ? "GRS" : "LRS"
  account_kind             = "StorageV2"
  is_hns_enabled           = true # Enable hierarchical namespace for Data Lake Gen2

  blob_properties {
    versioning_enabled = true

    delete_retention_policy {
      days = 30
    }

    container_delete_retention_policy {
      days = 7
    }
  }

  tags = var.tags
}

# Blob containers
resource "azurerm_storage_container" "raw" {
  name                  = "raw"
  storage_account_name  = azurerm_storage_account.datalake.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "analytics" {
  name                  = "analytics"
  storage_account_name  = azurerm_storage_account.datalake.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "features" {
  name                  = "features"
  storage_account_name  = azurerm_storage_account.datalake.name
  container_access_type = "private"
}

# Azure Container Registry
resource "azurerm_container_registry" "acr" {
  name                = "${replace(var.project_name, "-", "")}${var.environment}acr"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = var.environment == "prod" ? "Premium" : "Basic"
  admin_enabled       = true

  tags = var.tags
}

# Azure Container Instance for the platform
resource "azurerm_container_group" "platform" {
  name                = "${var.project_name}-${var.environment}-aci"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  ip_address_type     = "Public"
  os_type             = "Linux"

  container {
    name   = "market-data"
    image  = "${azurerm_container_registry.acr.login_server}/market-data-platform:latest"
    cpu    = var.container_cpu
    memory = var.container_memory

    ports {
      port     = 8000
      protocol = "TCP"
    }

    environment_variables = {
      STORAGE_PROVIDER           = "azure"
      AZURE_STORAGE_ACCOUNT_NAME = azurerm_storage_account.datalake.name
      AZURE_CONTAINER_NAME       = "raw"
    }

    secure_environment_variables = {
      AZURE_STORAGE_ACCOUNT_KEY = azurerm_storage_account.datalake.primary_access_key
      DATABENTO_API_KEY         = var.databento_api_key
      ALPACA_API_KEY            = var.alpaca_api_key
      ALPACA_SECRET_KEY         = var.alpaca_secret_key
    }
  }

  tags = var.tags
}

# Key Vault for secrets
resource "azurerm_key_vault" "main" {
  name                = "${replace(var.project_name, "-", "")}${var.environment}kv"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"

  purge_protection_enabled = var.environment == "prod"

  tags = var.tags
}

data "azurerm_client_config" "current" {}

# Log Analytics Workspace for monitoring
resource "azurerm_log_analytics_workspace" "main" {
  name                = "${var.project_name}-${var.environment}-logs"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "PerGB2018"
  retention_in_days   = var.environment == "prod" ? 90 : 30

  tags = var.tags
}
