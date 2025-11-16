/**
 * Variables for OxiRS Fuseki GCP Infrastructure
 */

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "availability_zones" {
  description = "List of availability zones for the GKE cluster"
  type        = list(string)
  default     = ["us-central1-a", "us-central1-b", "us-central1-c"]
}

variable "cluster_name" {
  description = "Name of the GKE cluster"
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

# GKE Node Pool Configuration
variable "node_machine_type" {
  description = "Machine type for GKE nodes"
  type        = string
  default     = "n2-standard-4"
}

variable "node_disk_size_gb" {
  description = "Disk size for GKE nodes in GB"
  type        = number
  default     = 100
}

variable "min_node_count" {
  description = "Minimum number of nodes per zone"
  type        = number
  default     = 1
}

variable "max_node_count" {
  description = "Maximum number of nodes per zone"
  type        = number
  default     = 10
}

# Storage Configuration
variable "filestore_capacity_gb" {
  description = "Capacity of Cloud Filestore in GB"
  type        = number
  default     = 1024
}

variable "backup_retention_days" {
  description = "Number of days to retain backups in Cloud Storage"
  type        = number
  default     = 30
}

# Cloud SQL Configuration
variable "enable_cloud_sql" {
  description = "Enable Cloud SQL for metadata storage"
  type        = bool
  default     = true
}

variable "cloud_sql_tier" {
  description = "Machine type for Cloud SQL instance"
  type        = string
  default     = "db-n1-standard-2"
}

variable "cloud_sql_disk_size" {
  description = "Disk size for Cloud SQL in GB"
  type        = number
  default     = 100
}

# Network Configuration
variable "master_authorized_networks" {
  description = "List of CIDR blocks authorized to access the GKE master"
  type = list(object({
    cidr_block   = string
    display_name = string
  }))
  default = [
    {
      cidr_block   = "0.0.0.0/0"
      display_name = "All"
    }
  ]
}

# Monitoring Configuration
variable "notification_channels" {
  description = "List of notification channel IDs for alerts"
  type        = list(string)
  default     = []
}

variable "enable_monitoring_dashboards" {
  description = "Enable custom Cloud Monitoring dashboards"
  type        = bool
  default     = true
}

# Tags
variable "tags" {
  description = "Additional tags to apply to resources"
  type        = map(string)
  default = {
    managed_by = "terraform"
    application = "oxirs-fuseki"
  }
}
