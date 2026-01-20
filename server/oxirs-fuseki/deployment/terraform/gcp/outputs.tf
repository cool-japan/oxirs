/**
 * Outputs for OxiRS Fuseki GCP Infrastructure
 */

output "project_id" {
  description = "GCP Project ID"
  value       = var.project_id
}

output "region" {
  description = "GCP Region"
  value       = var.region
}

output "cluster_name" {
  description = "GKE Cluster Name"
  value       = google_container_cluster.oxirs_gke.name
}

output "cluster_endpoint" {
  description = "GKE Cluster Endpoint"
  value       = google_container_cluster.oxirs_gke.endpoint
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "GKE Cluster CA Certificate"
  value       = google_container_cluster.oxirs_gke.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "cluster_location" {
  description = "GKE Cluster Location"
  value       = google_container_cluster.oxirs_gke.location
}

output "vpc_network" {
  description = "VPC Network Name"
  value       = google_compute_network.oxirs_vpc.name
}

output "subnet_name" {
  description = "Subnet Name"
  value       = google_compute_subnetwork.oxirs_subnet.name
}

output "subnet_cidr" {
  description = "Subnet CIDR Range"
  value       = google_compute_subnetwork.oxirs_subnet.ip_cidr_range
}

output "gke_service_account" {
  description = "GKE Node Service Account Email"
  value       = google_service_account.gke_nodes.email
}

output "workload_identity_service_account" {
  description = "Workload Identity Service Account Email"
  value       = google_service_account.oxirs_workload.email
}

output "storage_bucket_name" {
  description = "Cloud Storage Bucket Name for Backups"
  value       = google_storage_bucket.oxirs_backups.name
}

output "storage_bucket_url" {
  description = "Cloud Storage Bucket URL"
  value       = google_storage_bucket.oxirs_backups.url
}

output "filestore_instance_name" {
  description = "Cloud Filestore Instance Name"
  value       = google_filestore_instance.oxirs_filestore.name
}

output "filestore_ip_address" {
  description = "Cloud Filestore IP Address"
  value       = google_filestore_instance.oxirs_filestore.networks[0].ip_addresses[0]
}

output "filestore_file_share_name" {
  description = "Cloud Filestore File Share Name"
  value       = google_filestore_instance.oxirs_filestore.file_shares[0].name
}

output "cloud_sql_instance_name" {
  description = "Cloud SQL Instance Name"
  value       = var.enable_cloud_sql ? google_sql_database_instance.oxirs_metadata[0].name : null
}

output "cloud_sql_connection_name" {
  description = "Cloud SQL Connection Name"
  value       = var.enable_cloud_sql ? google_sql_database_instance.oxirs_metadata[0].connection_name : null
}

output "cloud_sql_private_ip" {
  description = "Cloud SQL Private IP Address"
  value       = var.enable_cloud_sql ? google_sql_database_instance.oxirs_metadata[0].private_ip_address : null
  sensitive   = true
}

output "kms_keyring_id" {
  description = "KMS Keyring ID"
  value       = google_kms_key_ring.oxirs_keyring.id
}

output "kms_crypto_key_id" {
  description = "KMS Crypto Key ID"
  value       = google_kms_crypto_key.oxirs_key.id
}

output "kubernetes_namespace" {
  description = "Kubernetes Namespace"
  value       = kubernetes_namespace.oxirs.metadata[0].name
}

# Connection information
output "kubectl_config_command" {
  description = "Command to configure kubectl"
  value       = "gcloud container clusters get-credentials ${google_container_cluster.oxirs_gke.name} --region ${var.region} --project ${var.project_id}"
}

output "gcloud_connect_command" {
  description = "Command to connect to the cluster"
  value       = "gcloud compute ssh --zone=${var.availability_zones[0]} --project=${var.project_id}"
}

# Cost estimation information
output "estimated_monthly_cost" {
  description = "Estimated monthly cost breakdown (USD)"
  value = {
    gke_cluster   = "~$73/month (zonal) or ~$219/month (regional control plane)"
    gke_nodes     = "~$${var.min_node_count * 3 * 95}/month for ${var.min_node_count * 3} ${var.node_machine_type} nodes (min)"
    filestore     = "~$${var.filestore_capacity_gb * 0.20}/month for ${var.filestore_capacity_gb}GB"
    cloud_sql     = var.enable_cloud_sql ? "~$150/month for ${var.cloud_sql_tier}" : "Not enabled"
    storage       = "~$20/month for 1TB of backups"
    network       = "~$50-100/month depending on egress"
    monitoring    = "~$10-30/month"
    total_min     = "~$${var.min_node_count * 3 * 95 + var.filestore_capacity_gb * 0.20 + (var.enable_cloud_sql ? 150 : 0) + 20 + 50 + 10 + 219}/month"
  }
}

# Deployment information
output "deployment_info" {
  description = "Deployment information and next steps"
  value = {
    step_1 = "Configure kubectl: ${output.kubectl_config_command.value}"
    step_2 = "Verify cluster: kubectl cluster-info"
    step_3 = "Deploy OxiRS Fuseki: kubectl apply -f ../../kubernetes/"
    step_4 = "Check pods: kubectl get pods -n ${var.k8s_namespace}"
    step_5 = "Access service: kubectl get svc -n ${var.k8s_namespace}"
  }
}
