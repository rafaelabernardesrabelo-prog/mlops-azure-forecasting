variable "prefix" {
  description = "Prefix for all resource names"
  type        = string
  default     = "forecasting-demo"
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "eastus"
}

variable "workspace_name" {
  type        = string
  description = "Name of the Azure Machine Learning workspace"
  default     = "forecasting-demo-mlw"
}