terraform {
  required_version = ">= 1.6"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }
}

provider "azurerm" {
  features {}
  # Fourni par terraform.tfvars (ignoré par git) : pas d'identifiant dans le code.
  subscription_id = var.subscription_id
}
