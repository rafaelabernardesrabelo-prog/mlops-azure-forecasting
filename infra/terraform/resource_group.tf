resource "azurerm_resource_group" "rg" {
  name     = "${var.prefix}-rg"
  location = var.location

  tags = {
    environment = "dev"
    project     = var.prefix
  }
}
