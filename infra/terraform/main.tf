terraform {
  backend "local" {}
}

module "resource_group" {
  source = "./"
}
