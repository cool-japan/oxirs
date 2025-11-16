/**
 * Variables for OxiRS Fuseki Azure Infrastructure
 */

variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "eastus"
}

variable "cluster_name" {
  description = "Name of the AKS cluster"
  type        = string
  default     = "oxirs-fuseki"
}

variable "environment" {
  description = "Environment name (production, staging, development)"
  type        = string
  default     = "production"

  validation {
    condition     = contains(["production", "staging", "development"], var.environment)
    error_message = "Environment must be production, staging, or development."
  }
}

variable "k8s_namespace" {
  description = "Kubernetes namespace for OxiRS Fuseki"
  type        = string
  default     = "oxirs-fuseki"
}

variable "kubernetes_version" {
  description = "Kubernetes version for AKS"
  type        = string
  default     = "1.29"
}

# AKS System Node Pool Configuration
variable "system_node_count" {
  description = "Number of system nodes"
  type        = number
  default     = 1
}

variable "system_node_vm_size" {
  description = "VM size for system nodes"
  type        = string
  default     = "Standard_D2s_v3"
}

# AKS Application Node Pool Configuration
variable "node_vm_size" {
  description = "VM size for application nodes"
  type        = string
  default     = "Standard_D4s_v3"
}

variable "node_disk_size_gb" {
  description = "Disk size for application nodes in GB"
  type        = number
  default     = 100
}

variable "min_node_count" {
  description = "Minimum number of application nodes"
  type        = number
  default     = 1
}

variable "max_node_count" {
  description = "Maximum number of application nodes"
  type        = number
  default     = 10
}

# Cluster Configuration
variable "private_cluster_enabled" {
  description = "Enable private AKS cluster"
  type        = bool
  default     = false
}

# Storage Configuration
variable "storage_replication_type" {
  description = "Storage account replication type"
  type        = string
  default     = "GRS"

  validation {
    condition     = contains(["LRS", "GRS", "RAGRS", "ZRS", "GZRS", "RAGZRS"], var.storage_replication_type)
    error_message = "Storage replication type must be one of: LRS, GRS, RAGRS, ZRS, GZRS, RAGZRS."
  }
}

variable "azure_files_quota_gb" {
  description = "Azure Files share quota in GB"
  type        = number
  default     = 1024
}

variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 30
}

# PostgreSQL Configuration
variable "enable_postgresql" {
  description = "Enable Azure Database for PostgreSQL"
  type        = bool
  default     = true
}

variable "postgresql_admin_username" {
  description = "PostgreSQL administrator username"
  type        = string
  default     = "oxirsadmin"
}

variable "postgresql_admin_password" {
  description = "PostgreSQL administrator password"
  type        = string
  sensitive   = true
}

variable "postgresql_sku_name" {
  description = "PostgreSQL SKU name"
  type        = string
  default     = "GP_Standard_D2s_v3"
}

variable "postgresql_storage_mb" {
  description = "PostgreSQL storage in MB"
  type        = number
  default     = 131072 # 128GB
}

variable "postgresql_backup_retention_days" {
  description = "PostgreSQL backup retention in days"
  type        = number
  default     = 7
}

# Application Gateway Configuration
variable "enable_application_gateway" {
  description = "Enable Application Gateway for ingress"
  type        = bool
  default     = false
}

# Monitoring Configuration
variable "log_retention_days" {
  description = "Log Analytics workspace retention in days"
  type        = number
  default     = 30
}

variable "alert_email_receivers" {
  description = "List of email addresses for alerts"
  type        = list(string)
  default     = []
}

# Tags
variable "tags" {
  description = "Additional tags to apply to resources"
  type        = map(string)
  default = {
    managed_by  = "terraform"
    application = "oxirs-fuseki"
  }
}
