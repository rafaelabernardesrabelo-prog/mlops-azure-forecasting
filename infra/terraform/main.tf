# main.tf

terraform {
  backend "local" {}
}

# Aqui você importa os módulos necessários (exemplo):
module "resource_group" {
  source = "./resource_group"
}

module "storage" {
  source = "./storage"
}

module "keyvault" {
  source = "./keyvault"
}

module "ml_workspace" {
  source = "./ml_workspace"
}

module "compute_cluster" {
  source = "./compute_cluster"
}

module "appinsights" {
  source = "./appinsights"
}

module "acr" {
  source = "./acr"
}
