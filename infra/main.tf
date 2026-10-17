locals {
  rg_name = "rg-${var.name_prefix}-dev"
  tags = {
    project = "proposal-rag"
    env     = "dev"
  }
  backend_image  = "${var.image_prefix}-backend:${var.image_tag}"
  frontend_image = "${var.image_prefix}-frontend:${var.image_tag}"
}

resource "azurerm_resource_group" "main" {
  name     = local.rg_name
  location = var.location
  tags     = local.tags
}

# Pas de Log Analytics ni de registre Azure : les images viennent de GHCR (coût fixe ~0 EUR).
resource "azurerm_container_app_environment" "main" {
  name                = "cae-${var.name_prefix}-dev"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  tags                = local.tags
}

# ---- Données persistantes -----------------------------------------------------
# Le compte de stockage vit dans un autre groupe de ressources, hors de ce Terraform :
# un terraform destroy ne le supprime donc pas (et un verrou l'en empêche).

data "azurerm_storage_account" "data" {
  name                = var.data_storage_account
  resource_group_name = var.data_resource_group
}

resource "azurerm_container_app_environment_storage" "data" {
  for_each                     = toset(["chroma", "raw"])
  name                         = each.key
  container_app_environment_id = azurerm_container_app_environment.main.id
  account_name                 = data.azurerm_storage_account.data.name
  share_name                   = each.key
  access_key                   = data.azurerm_storage_account.data.primary_access_key
  access_mode                  = "ReadWrite"
}

resource "azurerm_container_app" "backend" {
  name                         = "ca-backend"
  resource_group_name          = azurerm_resource_group.main.name
  container_app_environment_id = azurerm_container_app_environment.main.id
  revision_mode                = "Single"
  tags                         = local.tags


  secret {
    name  = "groq-api-key"
    value = var.groq_api_key
  }
  secret {
    name  = "api-key"
    value = var.api_key
  }

  # Interne : joignable seulement depuis le frontend, jamais depuis Internet.
  ingress {
    external_enabled = false
    target_port      = 8000
    transport        = "http"
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    min_replicas = 0 # 0 = aucun coût quand personne n'utilise l'application
    max_replicas = 1

    # SQLite (utilisé par Chroma) sur un partage SMB : nobrl évite les erreurs de verrouillage.
    # uid/gid 1000 = utilisateur "appuser" du conteneur.
    dynamic "volume" {
      for_each = azurerm_container_app_environment_storage.data
      content {
        name          = volume.key
        storage_type  = "AzureFile"
        storage_name  = volume.value.name
        mount_options = "uid=1000,gid=1000,nobrl,mfsymlinks,cache=strict"
      }
    }

    container {
      name   = "backend"
      image  = local.backend_image
      cpu    = 1.0
      memory = "2Gi"

      volume_mounts {
        name = "chroma"
        path = "/app/var/chroma"
      }
      volume_mounts {
        name = "raw"
        path = "/app/data/raw"
      }

      env {
        name        = "GROQ_API_KEY"
        secret_name = "groq-api-key"
      }
      env {
        name        = "API_KEY"
        secret_name = "api-key"
      }
      env {
        name  = "PYTHONPATH"
        value = "/app/src"
      }
      env {
        name  = "LOG_LEVEL"
        value = "INFO"
      }

      startup_probe {
        transport               = "HTTP"
        path                    = "/health"
        port                    = 8000
        interval_seconds        = 10
        failure_count_threshold = 30 # chargement des modèles : jusqu'à ~5 min
      }
    }
  }
}

resource "azurerm_container_app" "frontend" {
  name                         = "ca-frontend"
  resource_group_name          = azurerm_resource_group.main.name
  container_app_environment_id = azurerm_container_app_environment.main.id
  revision_mode                = "Single"
  tags                         = local.tags


  secret {
    name  = "api-key"
    value = var.api_key
  }
  secret {
    name  = "aad-client-secret"
    value = azuread_application_password.frontend.value
  }

  ingress {
    external_enabled = true
    target_port      = 8501
    transport        = "http"

    # Restriction par IP facultative, en plus de la connexion Entra ID.
    dynamic "ip_security_restriction" {
      for_each = var.allowed_ip == "" ? [] : [var.allowed_ip]
      content {
        name             = "mon-ip"
        action           = "Allow"
        ip_address_range = "${ip_security_restriction.value}/32"
      }
    }

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    min_replicas = 0
    max_replicas = 1

    container {
      name   = "frontend"
      image  = local.frontend_image
      cpu    = 0.5
      memory = "1Gi"

      env {
        name  = "PROPOSAL_RAG_API_URL"
        value = "http://${azurerm_container_app.backend.name}"
      }
      env {
        name        = "PROPOSAL_RAG_API_KEY"
        secret_name = "api-key"
      }
    }
  }
}

# Droit de déploiement pour GitHub Actions, limité à ce groupe de ressources.
# Défini ici (et non à la main) pour être recréé après chaque terraform destroy.
resource "azurerm_role_assignment" "ci_deploy" {
  count                = var.deploy_principal_id == "" ? 0 : 1
  scope                = azurerm_resource_group.main.id
  role_definition_name = "Contributor"
  principal_id         = var.deploy_principal_id
}

# ---- Connexion par compte (Microsoft Entra ID) --------------------------------

data "azuread_client_config" "current" {}

locals {
  # Adresse du frontend, calculable avant sa création (nom de l'app + domaine de l'environnement).
  frontend_fqdn = "ca-frontend.${azurerm_container_app_environment.main.default_domain}"
}

resource "azuread_application" "frontend" {
  display_name     = "proposal-rag-frontend"
  sign_in_audience = "AzureADMyOrg" # uniquement les comptes de votre annuaire
  owners           = [data.azuread_client_config.current.object_id]

  web {
    redirect_uris = ["https://${local.frontend_fqdn}/.auth/login/aad/callback"]
    implicit_grant {
      id_token_issuance_enabled = true
    }
  }
}

resource "azuread_service_principal" "frontend" {
  client_id                    = azuread_application.frontend.client_id
  app_role_assignment_required = true # seuls les utilisateurs affectés peuvent se connecter
  owners                       = [data.azuread_client_config.current.object_id]
}

# Vous, utilisateur autorisé (rôle par défaut "User").
resource "azuread_app_role_assignment" "me" {
  app_role_id         = "00000000-0000-0000-0000-000000000000"
  principal_object_id = data.azuread_client_config.current.object_id
  resource_object_id  = azuread_service_principal.frontend.object_id
}

resource "azuread_application_password" "frontend" {
  application_id = azuread_application.frontend.id
  display_name   = "container-apps-auth"
  end_date       = timeadd(timestamp(), "2160h") # ~90 jours

  lifecycle {
    ignore_changes = [end_date] # ne pas régénérer le secret à chaque plan
  }
}

# Active la connexion Entra ID devant le frontend (aucun code à modifier dans l'application).
resource "azapi_resource" "frontend_auth" {
  type      = "Microsoft.App/containerApps/authConfigs@2024-03-01"
  name      = "current"
  parent_id = azurerm_container_app.frontend.id

  body = {
    properties = {
      platform = { enabled = true }
      globalValidation = {
        unauthenticatedClientAction = "RedirectToLoginPage"
        redirectToProvider          = "azureactivedirectory"
      }
      identityProviders = {
        azureActiveDirectory = {
          enabled = true
          registration = {
            openIdIssuer            = "https://login.microsoftonline.com/${data.azuread_client_config.current.tenant_id}/v2.0"
            clientId                = azuread_application.frontend.client_id
            clientSecretSettingName = "aad-client-secret"
          }
        }
      }
      login = { preserveUrlFragmentsForLogins = false }
    }
  }

  depends_on = [azuread_app_role_assignment.me]
}
