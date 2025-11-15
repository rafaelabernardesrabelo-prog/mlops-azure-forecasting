resource "azurerm_application_insights" "appi" {
  name                = var.appi_name
  location            = var.location
  resource_group_name = azurerm_resource_group.rg.name
  application_type    = "web"

  tags = {
    environment = "dev"
  }

  lifecycle {
    ignore_changes = [
      workspace_id
    ]
  }
}
