output "resource_group_name" {
  description = "Name of the resource group"
  value       = azurerm_resource_group.main.name
}

output "storage_account_name" {
  description = "Name of the storage account"
  value       = azurerm_storage_account.datalake.name
}

output "storage_account_primary_key" {
  description = "Primary access key for the storage account"
  value       = azurerm_storage_account.datalake.primary_access_key
  sensitive   = true
}

output "container_registry_login_server" {
  description = "Login server for the container registry"
  value       = azurerm_container_registry.acr.login_server
}

output "container_group_fqdn" {
  description = "FQDN of the container group"
  value       = azurerm_container_group.platform.fqdn
}

output "key_vault_uri" {
  description = "URI of the Key Vault"
  value       = azurerm_key_vault.main.vault_uri
}

output "log_analytics_workspace_id" {
  description = "ID of the Log Analytics Workspace"
  value       = azurerm_log_analytics_workspace.main.id
}
