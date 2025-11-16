/**
 * OxiRS Fuseki - Azure Infrastructure with AKS
 *
 * This Terraform configuration provisions a production-ready AKS cluster
 * for running OxiRS Fuseki SPARQL servers with high availability,
 * monitoring, and persistent storage.
 *
 * Components:
 * - AKS cluster with auto-scaling node pools
 * - Virtual Network with private subnets
 * - Azure Database for PostgreSQL for metadata
 * - Azure Storage Account for backups
 * - Azure Files for shared storage
 * - Azure Monitor and Log Analytics
 * - Managed identities and RBAC
 * - Application Gateway for ingress
 */

terraform {
  required_version = ">= 1.9"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
    azuread = {
      source  = "hashicorp/azuread"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.33"
    }
  }

  backend "azurerm" {
    # Configure Azure Storage backend for state
    # resource_group_name  = "terraform-state-rg"
    # storage_account_name = "oxirsterraformstate"
    # container_name       = "tfstate"
    # key                  = "fuseki.terraform.tfstate"
  }
}

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy = false
    }
    resource_group {
      prevent_deletion_if_contains_resources = true
    }
  }
}

provider "azuread" {}

# Data sources
data "azurerm_client_config" "current" {}

# Resource Group
resource "azurerm_resource_group" "oxirs" {
  name     = "${var.cluster_name}-rg"
  location = var.location

  tags = merge(
    var.tags,
    {
      environment = var.environment
      managed_by  = "terraform"
    }
  )
}

# Virtual Network
resource "azurerm_virtual_network" "oxirs" {
  name                = "${var.cluster_name}-vnet"
  location            = azurerm_resource_group.oxirs.location
  resource_group_name = azurerm_resource_group.oxirs.name
  address_space       = ["10.0.0.0/16"]

  tags = var.tags
}

# Subnet for AKS
resource "azurerm_subnet" "aks" {
  name                 = "${var.cluster_name}-aks-subnet"
  resource_group_name  = azurerm_resource_group.oxirs.name
  virtual_network_name = azurerm_virtual_network.oxirs.name
  address_prefixes     = ["10.0.1.0/24"]
}

# Subnet for Application Gateway
resource "azurerm_subnet" "appgw" {
  name                 = "${var.cluster_name}-appgw-subnet"
  resource_group_name  = azurerm_resource_group.oxirs.name
  virtual_network_name = azurerm_virtual_network.oxirs.name
  address_prefixes     = ["10.0.2.0/24"]
}

# Subnet for PostgreSQL
resource "azurerm_subnet" "postgresql" {
  name                 = "${var.cluster_name}-postgresql-subnet"
  resource_group_name  = azurerm_resource_group.oxirs.name
  virtual_network_name = azurerm_virtual_network.oxirs.name
  address_prefixes     = ["10.0.3.0/24"]

  delegation {
    name = "postgresql-delegation"
    service_delegation {
      name = "Microsoft.DBforPostgreSQL/flexibleServers"
      actions = [
        "Microsoft.Network/virtualNetworks/subnets/join/action",
      ]
    }
  }
}

# Network Security Group for AKS
resource "azurerm_network_security_group" "aks" {
  name                = "${var.cluster_name}-aks-nsg"
  location            = azurerm_resource_group.oxirs.location
  resource_group_name = azurerm_resource_group.oxirs.name

  security_rule {
    name                       = "AllowHTTPS"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "AllowHTTP"
    priority                   = 110
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "80"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  tags = var.tags
}

resource "azurerm_subnet_network_security_group_association" "aks" {
  subnet_id                 = azurerm_subnet.aks.id
  network_security_group_id = azurerm_network_security_group.aks.id
}

# Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "oxirs" {
  name                = "${var.cluster_name}-logs"
  location            = azurerm_resource_group.oxirs.location
  resource_group_name = azurerm_resource_group.oxirs.name
  sku                 = "PerGB2018"
  retention_in_days   = var.log_retention_days

  tags = var.tags
}

# User Assigned Identity for AKS
resource "azurerm_user_assigned_identity" "aks" {
  name                = "${var.cluster_name}-aks-identity"
  location            = azurerm_resource_group.oxirs.location
  resource_group_name = azurerm_resource_group.oxirs.name

  tags = var.tags
}

# Role assignments for AKS identity
resource "azurerm_role_assignment" "aks_network_contributor" {
  scope                = azurerm_virtual_network.oxirs.id
  role_definition_name = "Network Contributor"
  principal_id         = azurerm_user_assigned_identity.aks.principal_id
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "oxirs" {
  name                = var.cluster_name
  location            = azurerm_resource_group.oxirs.location
  resource_group_name = azurerm_resource_group.oxirs.name
  dns_prefix          = var.cluster_name
  kubernetes_version  = var.kubernetes_version

  # Private cluster configuration
  private_cluster_enabled = var.private_cluster_enabled

  default_node_pool {
    name                = "system"
    node_count          = var.system_node_count
    vm_size             = var.system_node_vm_size
    vnet_subnet_id      = azurerm_subnet.aks.id
    enable_auto_scaling = true
    min_count           = var.system_node_count
    max_count           = var.system_node_count * 2
    max_pods            = 110
    os_disk_size_gb     = 100
    os_disk_type        = "Managed"

    node_labels = {
      "workload" = "system"
    }

    tags = var.tags
  }

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.aks.id]
  }

  network_profile {
    network_plugin    = "azure"
    network_policy    = "azure"
    load_balancer_sku = "standard"
    service_cidr      = "10.1.0.0/16"
    dns_service_ip    = "10.1.0.10"
  }

  oms_agent {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.oxirs.id
  }

  azure_active_directory_role_based_access_control {
    azure_rbac_enabled = true
    tenant_id          = data.azurerm_client_config.current.tenant_id
  }

  key_vault_secrets_provider {
    secret_rotation_enabled = true
  }

  workload_identity_enabled = true
  oidc_issuer_enabled       = true

  maintenance_window_auto_upgrade {
    frequency   = "Weekly"
    interval    = 1
    duration    = 4
    day_of_week = "Sunday"
    start_time  = "03:00"
  }

  maintenance_window_node_os {
    frequency   = "Weekly"
    interval    = 1
    duration    = 4
    day_of_week = "Sunday"
    start_time  = "05:00"
  }

  tags = var.tags

  depends_on = [
    azurerm_role_assignment.aks_network_contributor
  ]
}

# Application Node Pool for OxiRS Fuseki
resource "azurerm_kubernetes_cluster_node_pool" "application" {
  name                  = "oxirs"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.oxirs.id
  vm_size               = var.node_vm_size
  vnet_subnet_id        = azurerm_subnet.aks.id
  enable_auto_scaling   = true
  min_count             = var.min_node_count
  max_count             = var.max_node_count
  max_pods              = 110
  os_disk_size_gb       = var.node_disk_size_gb
  os_disk_type          = "Managed"

  node_labels = {
    "workload"    = "application"
    "app"         = "oxirs-fuseki"
    "environment" = var.environment
  }

  node_taints = []

  tags = var.tags
}

# Storage Account for backups
resource "azurerm_storage_account" "oxirs" {
  name                     = replace("${var.cluster_name}storage", "-", "")
  resource_group_name      = azurerm_resource_group.oxirs.name
  location                 = azurerm_resource_group.oxirs.location
  account_tier             = "Standard"
  account_replication_type = var.storage_replication_type
  account_kind             = "StorageV2"

  blob_properties {
    versioning_enabled = true

    delete_retention_policy {
      days = var.backup_retention_days
    }
  }

  tags = var.tags
}

# Blob container for backups
resource "azurerm_storage_container" "backups" {
  name                  = "oxirs-backups"
  storage_account_name  = azurerm_storage_account.oxirs.name
  container_access_type = "private"
}

# Azure Files share for persistent data
resource "azurerm_storage_share" "oxirs_data" {
  name                 = "oxirs-data"
  storage_account_name = azurerm_storage_account.oxirs.name
  quota                = var.azure_files_quota_gb
}

# Private DNS Zone for PostgreSQL
resource "azurerm_private_dns_zone" "postgresql" {
  count               = var.enable_postgresql ? 1 : 0
  name                = "privatelink.postgres.database.azure.com"
  resource_group_name = azurerm_resource_group.oxirs.name

  tags = var.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "postgresql" {
  count                 = var.enable_postgresql ? 1 : 0
  name                  = "${var.cluster_name}-postgresql-dns-link"
  private_dns_zone_name = azurerm_private_dns_zone.postgresql[0].name
  resource_group_name   = azurerm_resource_group.oxirs.name
  virtual_network_id    = azurerm_virtual_network.oxirs.id

  tags = var.tags
}

# PostgreSQL Flexible Server
resource "azurerm_postgresql_flexible_server" "oxirs" {
  count               = var.enable_postgresql ? 1 : 0
  name                = "${var.cluster_name}-postgresql"
  resource_group_name = azurerm_resource_group.oxirs.name
  location            = azurerm_resource_group.oxirs.location
  version             = "15"

  delegated_subnet_id = azurerm_subnet.postgresql.id
  private_dns_zone_id = azurerm_private_dns_zone.postgresql[0].id

  administrator_login    = var.postgresql_admin_username
  administrator_password = var.postgresql_admin_password

  storage_mb = var.postgresql_storage_mb
  sku_name   = var.postgresql_sku_name

  backup_retention_days        = var.postgresql_backup_retention_days
  geo_redundant_backup_enabled = var.environment == "production"

  high_availability {
    mode = var.environment == "production" ? "ZoneRedundant" : "Disabled"
  }

  tags = var.tags

  depends_on = [
    azurerm_private_dns_zone_virtual_network_link.postgresql
  ]
}

# PostgreSQL Database
resource "azurerm_postgresql_flexible_server_database" "oxirs" {
  count     = var.enable_postgresql ? 1 : 0
  name      = "oxirs_fuseki"
  server_id = azurerm_postgresql_flexible_server.oxirs[0].id
  collation = "en_US.utf8"
  charset   = "utf8"
}

# Key Vault for secrets
resource "azurerm_key_vault" "oxirs" {
  name                       = "${var.cluster_name}-kv"
  location                   = azurerm_resource_group.oxirs.location
  resource_group_name        = azurerm_resource_group.oxirs.name
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  soft_delete_retention_days = 7
  purge_protection_enabled   = var.environment == "production"

  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id

    secret_permissions = [
      "Get", "List", "Set", "Delete", "Purge", "Recover"
    ]
  }

  tags = var.tags
}

# Managed Identity for OxiRS Fuseki workload
resource "azurerm_user_assigned_identity" "oxirs_workload" {
  name                = "${var.cluster_name}-workload-identity"
  location            = azurerm_resource_group.oxirs.location
  resource_group_name = azurerm_resource_group.oxirs.name

  tags = var.tags
}

# Role assignments for workload identity
resource "azurerm_role_assignment" "oxirs_storage_blob_contributor" {
  scope                = azurerm_storage_account.oxirs.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_user_assigned_identity.oxirs_workload.principal_id
}

resource "azurerm_role_assignment" "oxirs_key_vault_reader" {
  scope                = azurerm_key_vault.oxirs.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_user_assigned_identity.oxirs_workload.principal_id
}

# Federated identity credential for workload identity
resource "azurerm_federated_identity_credential" "oxirs" {
  name                = "${var.cluster_name}-federated-identity"
  resource_group_name = azurerm_resource_group.oxirs.name
  parent_id           = azurerm_user_assigned_identity.oxirs_workload.id
  audience            = ["api://AzureADTokenExchange"]
  issuer              = azurerm_kubernetes_cluster.oxirs.oidc_issuer_url
  subject             = "system:serviceaccount:${var.k8s_namespace}:oxirs-fuseki"
}

# Public IP for Application Gateway
resource "azurerm_public_ip" "appgw" {
  count               = var.enable_application_gateway ? 1 : 0
  name                = "${var.cluster_name}-appgw-pip"
  location            = azurerm_resource_group.oxirs.location
  resource_group_name = azurerm_resource_group.oxirs.name
  allocation_method   = "Static"
  sku                 = "Standard"

  tags = var.tags
}

# Application Gateway (optional)
resource "azurerm_application_gateway" "oxirs" {
  count               = var.enable_application_gateway ? 1 : 0
  name                = "${var.cluster_name}-appgw"
  location            = azurerm_resource_group.oxirs.location
  resource_group_name = azurerm_resource_group.oxirs.name

  sku {
    name     = "WAF_v2"
    tier     = "WAF_v2"
    capacity = 2
  }

  gateway_ip_configuration {
    name      = "gateway-ip-config"
    subnet_id = azurerm_subnet.appgw.id
  }

  frontend_port {
    name = "https-port"
    port = 443
  }

  frontend_port {
    name = "http-port"
    port = 80
  }

  frontend_ip_configuration {
    name                 = "frontend-ip-config"
    public_ip_address_id = azurerm_public_ip.appgw[0].id
  }

  backend_address_pool {
    name = "oxirs-backend-pool"
  }

  backend_http_settings {
    name                  = "backend-http-settings"
    cookie_based_affinity = "Disabled"
    port                  = 3030
    protocol              = "Http"
    request_timeout       = 300
  }

  http_listener {
    name                           = "http-listener"
    frontend_ip_configuration_name = "frontend-ip-config"
    frontend_port_name             = "http-port"
    protocol                       = "Http"
  }

  request_routing_rule {
    name                       = "routing-rule"
    rule_type                  = "Basic"
    http_listener_name         = "http-listener"
    backend_address_pool_name  = "oxirs-backend-pool"
    backend_http_settings_name = "backend-http-settings"
    priority                   = 100
  }

  waf_configuration {
    enabled          = true
    firewall_mode    = "Prevention"
    rule_set_type    = "OWASP"
    rule_set_version = "3.2"
  }

  tags = var.tags
}

# Azure Monitor action group for alerts
resource "azurerm_monitor_action_group" "oxirs" {
  name                = "${var.cluster_name}-alerts"
  resource_group_name = azurerm_resource_group.oxirs.name
  short_name          = "oxirs"

  dynamic "email_receiver" {
    for_each = var.alert_email_receivers
    content {
      name          = "email-${email_receiver.key}"
      email_address = email_receiver.value
    }
  }

  tags = var.tags
}

# Metric alert for high CPU
resource "azurerm_monitor_metric_alert" "cpu" {
  name                = "${var.cluster_name}-high-cpu"
  resource_group_name = azurerm_resource_group.oxirs.name
  scopes              = [azurerm_kubernetes_cluster.oxirs.id]
  description         = "Alert when CPU usage is high"
  severity            = 2

  criteria {
    metric_namespace = "Microsoft.ContainerService/managedClusters"
    metric_name      = "node_cpu_usage_percentage"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = 80
  }

  action {
    action_group_id = azurerm_monitor_action_group.oxirs.id
  }

  tags = var.tags
}

# Metric alert for high memory
resource "azurerm_monitor_metric_alert" "memory" {
  name                = "${var.cluster_name}-high-memory"
  resource_group_name = azurerm_resource_group.oxirs.name
  scopes              = [azurerm_kubernetes_cluster.oxirs.id]
  description         = "Alert when memory usage is high"
  severity            = 2

  criteria {
    metric_namespace = "Microsoft.ContainerService/managedClusters"
    metric_name      = "node_memory_working_set_percentage"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = 80
  }

  action {
    action_group_id = azurerm_monitor_action_group.oxirs.id
  }

  tags = var.tags
}

# Configure kubectl provider
provider "kubernetes" {
  host                   = azurerm_kubernetes_cluster.oxirs.kube_config[0].host
  client_certificate     = base64decode(azurerm_kubernetes_cluster.oxirs.kube_config[0].client_certificate)
  client_key             = base64decode(azurerm_kubernetes_cluster.oxirs.kube_config[0].client_key)
  cluster_ca_certificate = base64decode(azurerm_kubernetes_cluster.oxirs.kube_config[0].cluster_ca_certificate)
}

# Create Kubernetes namespace
resource "kubernetes_namespace" "oxirs" {
  metadata {
    name = var.k8s_namespace

    labels = {
      name        = var.k8s_namespace
      environment = var.environment
    }
  }

  depends_on = [azurerm_kubernetes_cluster.oxirs]
}
