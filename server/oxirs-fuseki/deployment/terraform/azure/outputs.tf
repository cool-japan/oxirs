/**
 * Outputs for OxiRS Fuseki Azure Infrastructure
 */

output "resource_group_name" {
  description = "Resource Group Name"
  value       = azurerm_resource_group.oxirs.name
}

output "location" {
  description = "Azure Region"
  value       = azurerm_resource_group.oxirs.location
}

output "cluster_name" {
  description = "AKS Cluster Name"
  value       = azurerm_kubernetes_cluster.oxirs.name
}

output "cluster_id" {
  description = "AKS Cluster ID"
  value       = azurerm_kubernetes_cluster.oxirs.id
}

output "cluster_fqdn" {
  description = "AKS Cluster FQDN"
  value       = azurerm_kubernetes_cluster.oxirs.fqdn
}

output "kube_config" {
  description = "Kubernetes Configuration"
  value       = azurerm_kubernetes_cluster.oxirs.kube_config_raw
  sensitive   = true
}

output "client_certificate" {
  description = "Kubernetes Client Certificate"
  value       = azurerm_kubernetes_cluster.oxirs.kube_config[0].client_certificate
  sensitive   = true
}

output "client_key" {
  description = "Kubernetes Client Key"
  value       = azurerm_kubernetes_cluster.oxirs.kube_config[0].client_key
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "Kubernetes Cluster CA Certificate"
  value       = azurerm_kubernetes_cluster.oxirs.kube_config[0].cluster_ca_certificate
  sensitive   = true
}

output "host" {
  description = "Kubernetes Host"
  value       = azurerm_kubernetes_cluster.oxirs.kube_config[0].host
  sensitive   = true
}

output "aks_identity_principal_id" {
  description = "AKS Identity Principal ID"
  value       = azurerm_user_assigned_identity.aks.principal_id
}

output "workload_identity_client_id" {
  description = "Workload Identity Client ID"
  value       = azurerm_user_assigned_identity.oxirs_workload.client_id
}

output "storage_account_name" {
  description = "Storage Account Name"
  value       = azurerm_storage_account.oxirs.name
}

output "storage_account_primary_key" {
  description = "Storage Account Primary Key"
  value       = azurerm_storage_account.oxirs.primary_access_key
  sensitive   = true
}

output "storage_account_connection_string" {
  description = "Storage Account Connection String"
  value       = azurerm_storage_account.oxirs.primary_connection_string
  sensitive   = true
}

output "backups_container_name" {
  description = "Backups Container Name"
  value       = azurerm_storage_container.backups.name
}

output "azure_files_share_name" {
  description = "Azure Files Share Name"
  value       = azurerm_storage_share.oxirs_data.name
}

output "postgresql_server_name" {
  description = "PostgreSQL Server Name"
  value       = var.enable_postgresql ? azurerm_postgresql_flexible_server.oxirs[0].name : null
}

output "postgresql_server_fqdn" {
  description = "PostgreSQL Server FQDN"
  value       = var.enable_postgresql ? azurerm_postgresql_flexible_server.oxirs[0].fqdn : null
}

output "postgresql_database_name" {
  description = "PostgreSQL Database Name"
  value       = var.enable_postgresql ? azurerm_postgresql_flexible_server_database.oxirs[0].name : null
}

output "key_vault_id" {
  description = "Key Vault ID"
  value       = azurerm_key_vault.oxirs.id
}

output "key_vault_uri" {
  description = "Key Vault URI"
  value       = azurerm_key_vault.oxirs.vault_uri
}

output "log_analytics_workspace_id" {
  description = "Log Analytics Workspace ID"
  value       = azurerm_log_analytics_workspace.oxirs.id
}

output "log_analytics_workspace_primary_key" {
  description = "Log Analytics Workspace Primary Key"
  value       = azurerm_log_analytics_workspace.oxirs.primary_shared_key
  sensitive   = true
}

output "application_gateway_public_ip" {
  description = "Application Gateway Public IP"
  value       = var.enable_application_gateway ? azurerm_public_ip.appgw[0].ip_address : null
}

output "kubernetes_namespace" {
  description = "Kubernetes Namespace"
  value       = kubernetes_namespace.oxirs.metadata[0].name
}

output "oidc_issuer_url" {
  description = "OIDC Issuer URL for Workload Identity"
  value       = azurerm_kubernetes_cluster.oxirs.oidc_issuer_url
}

# Connection information
output "az_aks_get_credentials_command" {
  description = "Command to configure kubectl"
  value       = "az aks get-credentials --resource-group ${azurerm_resource_group.oxirs.name} --name ${azurerm_kubernetes_cluster.oxirs.name}"
}

output "az_login_command" {
  description = "Command to login to Azure"
  value       = "az login"
}

# Cost estimation information
output "estimated_monthly_cost" {
  description = "Estimated monthly cost breakdown (USD)"
  value = {
    aks_control_plane = "~$73/month"
    system_nodes      = "~$${var.system_node_count * 70}/month for ${var.system_node_count} ${var.system_node_vm_size} nodes"
    application_nodes = "~$${var.min_node_count * 140}/month for ${var.min_node_count} ${var.node_vm_size} nodes (min)"
    storage_account   = "~$25/month for GRS storage"
    azure_files       = "~$${var.azure_files_quota_gb * 0.06}/month for ${var.azure_files_quota_gb}GB"
    postgresql        = var.enable_postgresql ? "~$150/month for ${var.postgresql_sku_name}" : "Not enabled"
    log_analytics     = "~$20-50/month depending on ingestion"
    app_gateway       = var.enable_application_gateway ? "~$180/month for WAF_v2" : "Not enabled"
    network           = "~$30-70/month depending on egress"
    total_min         = "~$${var.system_node_count * 70 + var.min_node_count * 140 + 25 + var.azure_files_quota_gb * 0.06 + (var.enable_postgresql ? 150 : 0) + 20 + (var.enable_application_gateway ? 180 : 0) + 30 + 73}/month"
  }
}

# Deployment information
output "deployment_info" {
  description = "Deployment information and next steps"
  value = {
    step_1 = "Configure kubectl: ${output.az_aks_get_credentials_command.value}"
    step_2 = "Verify cluster: kubectl cluster-info"
    step_3 = "Deploy OxiRS Fuseki: kubectl apply -f ../../kubernetes/"
    step_4 = "Check pods: kubectl get pods -n ${var.k8s_namespace}"
    step_5 = "Access service: kubectl get svc -n ${var.k8s_namespace}"
  }
}
