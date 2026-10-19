variable "subscription_id" {
  description = "Identifiant de l'abonnement Azure (az account show --query id -o tsv)."
  type        = string
}

variable "location" {
  description = "Région Azure."
  type        = string
  default     = "francecentral"
}

variable "name_prefix" {
  description = "Préfixe des ressources (minuscules/chiffres, court)."
  type        = string
  default     = "proposalrag"
}

variable "image_prefix" {
  description = "Préfixe des images GHCR, sans tag (ex. ghcr.io/utilisateur/proposal-rag)."
  type        = string
}

variable "image_tag" {
  description = "Tag publié par la CI (ex. sha-abc123...)."
  type        = string
}

variable "allowed_ip" {
  description = "Optionnel : restreindre l'accès à une IPv4 (ex. 203.0.113.7). Vide = pas de restriction (la connexion Entra ID protège le site)."
  type        = string
  default     = ""
}

variable "deploy_principal_id" {
  description = "Object ID du principal de service utilisé par GitHub Actions (vide = aucun droit de déploiement créé)."
  type        = string
  default     = ""
}

variable "data_resource_group" {
  description = "Groupe de ressources du stockage persistant (créé hors Terraform, protégé par un verrou)."
  type        = string
  default     = "rg-proposalrag-data"
}

variable "data_storage_account" {
  description = "Compte de stockage contenant les partages chroma et raw."
  type        = string
  default     = "dataprag46571"
}

variable "key_vault_name" {
  description = "Key Vault contenant les secrets groq-api-key et api-key (créé hors Terraform, dans le groupe de données)."
  type        = string
  default     = "kv-prag-30420"
}

variable "alert_email" {
  description = "Adresse qui reçoit les alertes de supervision."
  type        = string
}

variable "apps_identity_name" {
  description = "Identité managée des applications (créée hors Terraform, avec accès en lecture au Key Vault)."
  type        = string
  default     = "id-prag-apps"
}
