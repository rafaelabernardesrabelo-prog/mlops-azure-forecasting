resource "azurerm_machine_learning_compute_cluster" "cpu_cluster" {
  name                = "${var.prefix}-cpu"
  location            = var.location
  resource_group_name = azurerm_resource_group.rg.name
  workspace_name      = azurerm_machine_learning_workspace.ml_workspace.name

  vm_size             = "Standard_DS11_v2"
  min_nodes           = 0
  max_nodes           = 2
  idle_time_before_scale_down = "PT10M"

  tags = {
    environment = "dev"
  }
}
