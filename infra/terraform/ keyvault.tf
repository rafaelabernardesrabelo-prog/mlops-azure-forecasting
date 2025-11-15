resource "azurerm_key_vault" "kv" {
  name                = replace("${var.prefix}kv", "-", "")
  location            = var.location
  resource_group_name = azurerm_resource_group.rg.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"

  tags = {
    environment = "dev"
  }
}

data "azurerm_client_config" "current" {}
