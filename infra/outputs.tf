output "resource_group" {
  value = azurerm_resource_group.main.name
}

output "frontend_url" {
  value = "https://${azurerm_container_app.frontend.ingress[0].fqdn}"
}
