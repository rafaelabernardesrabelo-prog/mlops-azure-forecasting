resource "azurerm_machine_learning_compute_cluster" "cpu_cluster" {
  name                          = "cpu-cluster"
  location                      = azurerm_resource_group.rg.location
  vm_size                       = "Standard_DS11_v2"
  machine_learning_workspace_id = azurerm_machine_learning_workspace.ml_workspace.id
  vm_priority                   = "Dedicated" # ou "LowPriority"

  scale_settings {
    min_node_count                       = 0
    max_node_count                       = 2
    scale_down_nodes_after_idle_duration = "PT10M"
  }

  identity {
    type = "SystemAssigned"
  }
}
