output "resource_group_name" {
  value = azurerm_resource_group.rg.name
}

output "workspace_name" {
  value = azurerm_machine_learning_workspace.ml_workspace.name
}

output "storage_account_name" {
  value = azurerm_storage_account.storage.name
}

output "acr_name" {
  value = azurerm_container_registry.acr.name
}

output "keyvault_name" {
  value = azurerm_key_vault.kv.name
}

output "compute_cluster_name" {
  value = azurerm_machine_learning_compute_cluster.cpu_cluster.name
}
