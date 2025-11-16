/**
 * OxiRS Fuseki - GCP Infrastructure with GKE
 *
 * This Terraform configuration provisions a production-ready GKE cluster
 * for running OxiRS Fuseki SPARQL servers with high availability,
 * monitoring, and persistent storage.
 *
 * Components:
 * - GKE cluster with auto-scaling node pools
 * - VPC network with private subnets
 * - Cloud SQL for metadata storage
 * - Cloud Storage for backups
 * - Cloud Filestore for shared storage
 * - Cloud Monitoring and Logging
 * - IAM roles and service accounts
 * - Load balancing and auto-scaling
 */

terraform {
  required_version = ">= 1.9"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 6.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.33"
    }
  }

  backend "gcs" {
    # Configure GCS backend for state storage
    # bucket = "oxirs-terraform-state"
    # prefix = "fuseki"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# Enable required GCP APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "compute.googleapis.com",
    "container.googleapis.com",
    "servicenetworking.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "sqladmin.googleapis.com",
    "storage-api.googleapis.com",
    "file.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "cloudkms.googleapis.com",
    "iam.googleapis.com",
  ])

  project = var.project_id
  service = each.value

  disable_dependent_services = true
  disable_on_destroy         = false
}

# VPC Network
resource "google_compute_network" "oxirs_vpc" {
  name                    = "${var.cluster_name}-vpc"
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"

  depends_on = [google_project_service.required_apis]
}

# Subnet for GKE cluster
resource "google_compute_subnetwork" "oxirs_subnet" {
  name          = "${var.cluster_name}-subnet"
  ip_cidr_range = "10.0.0.0/20"
  region        = var.region
  network       = google_compute_network.oxirs_vpc.id

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.4.0.0/14"
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.8.0.0/20"
  }

  private_ip_google_access = true
}

# Cloud Router for NAT
resource "google_compute_router" "oxirs_router" {
  name    = "${var.cluster_name}-router"
  region  = var.region
  network = google_compute_network.oxirs_vpc.id
}

# Cloud NAT for private nodes
resource "google_compute_router_nat" "oxirs_nat" {
  name                               = "${var.cluster_name}-nat"
  router                             = google_compute_router.oxirs_router.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# Service Account for GKE nodes
resource "google_service_account" "gke_nodes" {
  account_id   = "${var.cluster_name}-gke-sa"
  display_name = "GKE Node Service Account for ${var.cluster_name}"
}

# IAM roles for GKE service account
resource "google_project_iam_member" "gke_node_roles" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/storage.objectViewer",
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

# GKE Cluster
resource "google_container_cluster" "oxirs_gke" {
  name     = var.cluster_name
  location = var.region

  # Regional cluster with multiple zones
  node_locations = var.availability_zones

  # Use private cluster for security
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }

  # Network configuration
  network    = google_compute_network.oxirs_vpc.name
  subnetwork = google_compute_subnetwork.oxirs_subnet.name

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Master authorized networks
  master_authorized_networks_config {
    dynamic "cidr_blocks" {
      for_each = var.master_authorized_networks
      content {
        cidr_block   = cidr_blocks.value.cidr_block
        display_name = cidr_blocks.value.display_name
      }
    }
  }

  # Remove default node pool (we'll create custom ones)
  remove_default_node_pool = true
  initial_node_count       = 1

  # Cluster features
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  addons_config {
    http_load_balancing {
      disabled = false
    }
    horizontal_pod_autoscaling {
      disabled = false
    }
    network_policy_config {
      disabled = false
    }
    gcp_filestore_csi_driver_config {
      enabled = true
    }
  }

  network_policy {
    enabled  = true
    provider = "PROVIDER_UNSPECIFIED"
  }

  maintenance_policy {
    daily_maintenance_window {
      start_time = "03:00"
    }
  }

  # Logging and monitoring
  logging_service    = "logging.googleapis.com/kubernetes"
  monitoring_service = "monitoring.googleapis.com/kubernetes"

  # Security
  enable_shielded_nodes = true

  binary_authorization {
    evaluation_mode = "PROJECT_SINGLETON_POLICY_ENFORCE"
  }

  depends_on = [
    google_project_service.required_apis,
    google_compute_subnetwork.oxirs_subnet
  ]
}

# Primary node pool for OxiRS Fuseki
resource "google_container_node_pool" "oxirs_primary" {
  name       = "${var.cluster_name}-primary-pool"
  location   = var.region
  cluster    = google_container_cluster.oxirs_gke.name
  node_count = var.min_node_count

  autoscaling {
    min_node_count = var.min_node_count
    max_node_count = var.max_node_count
  }

  node_config {
    machine_type = var.node_machine_type
    disk_size_gb = var.node_disk_size_gb
    disk_type    = "pd-ssd"

    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      environment = var.environment
      app         = "oxirs-fuseki"
    }

    tags = ["gke-node", "${var.cluster_name}-node"]

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# Cloud Storage bucket for backups
resource "google_storage_bucket" "oxirs_backups" {
  name     = "${var.project_id}-${var.cluster_name}-backups"
  location = var.region

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = var.backup_retention_days
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age                   = 7
      with_state            = "NONCURRENT_VERSION"
      num_newer_versions    = 3
    }
    action {
      type = "Delete"
    }
  }

  encryption {
    default_kms_key_name = google_kms_crypto_key.oxirs_key.id
  }

  depends_on = [google_project_service.required_apis]
}

# Cloud Filestore for shared persistent storage
resource "google_filestore_instance" "oxirs_filestore" {
  name     = "${var.cluster_name}-filestore"
  location = var.availability_zones[0]
  tier     = "BASIC_HDD"

  file_shares {
    capacity_gb = var.filestore_capacity_gb
    name        = "oxirs_data"
  }

  networks {
    network = google_compute_network.oxirs_vpc.name
    modes   = ["MODE_IPV4"]
  }

  depends_on = [google_project_service.required_apis]
}

# Cloud SQL instance for metadata
resource "google_sql_database_instance" "oxirs_metadata" {
  count            = var.enable_cloud_sql ? 1 : 0
  name             = "${var.cluster_name}-metadata"
  database_version = "POSTGRES_15"
  region           = var.region

  settings {
    tier              = var.cloud_sql_tier
    availability_type = "REGIONAL"
    disk_size         = var.cloud_sql_disk_size
    disk_type         = "PD_SSD"
    disk_autoresize   = true

    backup_configuration {
      enabled                        = true
      start_time                     = "02:00"
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
      backup_retention_settings {
        retained_backups = 30
      }
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.oxirs_vpc.id
      require_ssl     = true
    }

    maintenance_window {
      day          = 7
      hour         = 3
      update_track = "stable"
    }

    insights_config {
      query_insights_enabled  = true
      query_string_length     = 1024
      record_application_tags = true
    }
  }

  deletion_protection = var.environment == "production"

  depends_on = [google_project_service.required_apis]
}

# Cloud SQL database
resource "google_sql_database" "oxirs_db" {
  count    = var.enable_cloud_sql ? 1 : 0
  name     = "oxirs_fuseki"
  instance = google_sql_database_instance.oxirs_metadata[0].name
}

# KMS keyring for encryption
resource "google_kms_key_ring" "oxirs_keyring" {
  name     = "${var.cluster_name}-keyring"
  location = var.region

  depends_on = [google_project_service.required_apis]
}

# KMS crypto key
resource "google_kms_crypto_key" "oxirs_key" {
  name            = "${var.cluster_name}-key"
  key_ring        = google_kms_key_ring.oxirs_keyring.id
  rotation_period = "7776000s" # 90 days

  lifecycle {
    prevent_destroy = true
  }
}

# Service account for workload identity
resource "google_service_account" "oxirs_workload" {
  account_id   = "${var.cluster_name}-workload-sa"
  display_name = "OxiRS Fuseki Workload Identity Service Account"
}

# IAM roles for workload identity
resource "google_project_iam_member" "oxirs_workload_roles" {
  for_each = toset([
    "roles/storage.objectAdmin",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.oxirs_workload.email}"
}

# Workload identity binding
resource "google_service_account_iam_member" "oxirs_workload_identity" {
  service_account_id = google_service_account.oxirs_workload.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${var.k8s_namespace}/oxirs-fuseki]"
}

# Cloud Monitoring alert policy for high CPU
resource "google_monitoring_alert_policy" "high_cpu" {
  display_name = "${var.cluster_name} - High CPU Usage"
  combiner     = "OR"

  conditions {
    display_name = "CPU usage above 80%"

    condition_threshold {
      filter          = "resource.type = \"k8s_container\" AND resource.labels.cluster_name = \"${var.cluster_name}\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.8

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = var.notification_channels

  depends_on = [google_project_service.required_apis]
}

# Cloud Monitoring alert policy for pod restarts
resource "google_monitoring_alert_policy" "pod_restarts" {
  display_name = "${var.cluster_name} - Frequent Pod Restarts"
  combiner     = "OR"

  conditions {
    display_name = "Pod restart count above threshold"

    condition_threshold {
      filter          = "resource.type = \"k8s_pod\" AND resource.labels.cluster_name = \"${var.cluster_name}\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 5

      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields      = ["resource.namespace_name", "resource.pod_name"]
      }
    }
  }

  notification_channels = var.notification_channels

  depends_on = [google_project_service.required_apis]
}

# Firewall rule for health checks
resource "google_compute_firewall" "health_check" {
  name    = "${var.cluster_name}-allow-health-check"
  network = google_compute_network.oxirs_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["80", "443", "3030"]
  }

  source_ranges = ["35.191.0.0/16", "130.211.0.0/22"]
  target_tags   = ["gke-node"]
}

# Output cluster connection info
data "google_client_config" "default" {}

provider "kubernetes" {
  host                   = "https://${google_container_cluster.oxirs_gke.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.oxirs_gke.master_auth[0].cluster_ca_certificate)
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

  depends_on = [google_container_cluster.oxirs_gke]
}
