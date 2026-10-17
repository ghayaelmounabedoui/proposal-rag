terraform {
  required_version = ">= 1.6"

  # État stocké dans Azure (chiffré, versionné, verrouillé pendant un apply).
  # Le compte est hors de ce Terraform : il survit à un terraform destroy.
  backend "azurerm" {
    resource_group_name  = "rg-proposalrag-tfstate"
    storage_account_name = "tfstateprag44386"
    container_name       = "tfstate"
    key                  = "proposal-rag-dev.tfstate"
    use_azuread_auth     = true
  }

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
