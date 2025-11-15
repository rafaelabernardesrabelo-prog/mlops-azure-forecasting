resource "azurerm_machine_learning_workspace" "ml_workspace" {
  name                = var.workspace_name
  location            = var.location
  resource_group_name = azurerm_resource_group.rg.name

  # O Workspace exige estes três IDs:
  key_vault_id            = azurerm_key_vault.kv.id
  storage_account_id      = azurerm_storage_account.storage.id
  application_insights_id = azurerm_application_insights.appi.id

  identity {
    type = "SystemAssigned"
  }

  tags = {
    environment = "dev"
  }
}
