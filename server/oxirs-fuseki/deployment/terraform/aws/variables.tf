# OxiRS Fuseki - AWS Terraform Variables

variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string

  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be dev, staging, or production."
  }
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "oxirs-fuseki"
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

# EKS Configuration
variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "node_instance_type" {
  description = "EC2 instance type for EKS nodes"
  type        = string
  default     = "t3.xlarge"
}

variable "node_count_min" {
  description = "Minimum number of EKS nodes"
  type        = number
  default     = 2
}

variable "node_count_max" {
  description = "Maximum number of EKS nodes"
  type        = number
  default     = 10
}

variable "node_count_desired" {
  description = "Desired number of EKS nodes"
  type        = number
  default     = 3
}

variable "node_disk_size" {
  description = "Disk size for EKS nodes (GB)"
  type        = number
  default     = 100
}

# Storage Configuration
variable "enable_efs" {
  description = "Enable EFS for shared storage"
  type        = bool
  default     = true
}

variable "enable_rds" {
  description = "Enable RDS for metadata storage"
  type        = bool
  default     = false
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "rds_allocated_storage" {
  description = "RDS allocated storage (GB)"
  type        = number
  default     = 100
}

variable "rds_username" {
  description = "RDS master username"
  type        = string
  default     = "fuseki"
  sensitive   = true
}

variable "rds_password" {
  description = "RDS master password"
  type        = string
  sensitive   = true
}

# Backup Configuration
variable "backup_retention_days" {
  description = "S3 backup retention period (days)"
  type        = number
  default     = 90
}

variable "log_retention_days" {
  description = "CloudWatch log retention period (days)"
  type        = number
  default     = 30
}

# Fuseki Configuration
variable "fuseki_image" {
  description = "OxiRS Fuseki Docker image"
  type        = string
  default     = "oxirs/fuseki:latest"
}

variable "fuseki_replicas" {
  description = "Number of Fuseki replicas"
  type        = number
  default     = 3
}

variable "fuseki_cpu_request" {
  description = "CPU request for Fuseki pods"
  type        = string
  default     = "1000m"
}

variable "fuseki_memory_request" {
  description = "Memory request for Fuseki pods"
  type        = string
  default     = "2Gi"
}

variable "fuseki_cpu_limit" {
  description = "CPU limit for Fuseki pods"
  type        = string
  default     = "4000m"
}

variable "fuseki_memory_limit" {
  description = "Memory limit for Fuseki pods"
  type        = string
  default     = "8Gi"
}

# Monitoring Configuration
variable "enable_prometheus" {
  description = "Enable Prometheus monitoring"
  type        = bool
  default     = true
}

variable "enable_grafana" {
  description = "Enable Grafana dashboards"
  type        = bool
  default     = true
}

# Tags
variable "additional_tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}
