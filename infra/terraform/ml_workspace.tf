resource "azurerm_machine_learning_workspace" "ml_workspace" {
  name                = var.workspace_name
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  identity {
    type = "SystemAssigned"
  }

  sku_name = "Basic"

  tags = {
    environment = "dev"
  }
}
