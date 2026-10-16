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

    container {
      name   = "backend"
      image  = local.backend_image
      cpu    = 1.0
      memory = "2Gi"

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

  ingress {
    external_enabled = true
    target_port      = 8501
    transport        = "http"

    # Pas d'authentification applicative pour l'instant : accès limité à votre IP.
    ip_security_restriction {
      name             = "mon-ip"
      action           = "Allow"
      ip_address_range = "${var.allowed_ip}/32"
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
