resource "azurerm_application_insights" "appi" {
  name                = "${var.prefix}-appi"
  location            = var.location
  resource_group_name = azurerm_resource_group.rg.name
  application_type    = "web"

  tags = {
    environment = "dev"
  }
}
