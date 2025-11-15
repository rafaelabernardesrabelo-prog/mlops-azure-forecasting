resource "azurerm_machine_learning_workspace" "ml_workspace" {
  name                    = "${var.prefix}-mlw"
  location                = var.location
  resource_group_name     = azurerm_resource_group.rg.name

  application_insights_id = azurerm_application_insights.appi.id
  key_vault_id            = azurerm_key_vault.kv.id
  storage_account_id      = azurerm_storage_account.storage.id

  tags = {
    environment = "dev"
  }
}
