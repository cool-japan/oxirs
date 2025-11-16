# OxiRS Fuseki - AWS Terraform Outputs

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = module.vpc.private_subnets
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = module.vpc.public_subnets
}

output "efs_file_system_id" {
  description = "EFS file system ID"
  value       = var.enable_efs ? aws_efs_file_system.fuseki_data[0].id : null
}

output "efs_dns_name" {
  description = "EFS DNS name"
  value       = var.enable_efs ? aws_efs_file_system.fuseki_data[0].dns_name : null
}

output "rds_endpoint" {
  description = "RDS endpoint"
  value       = var.enable_rds ? aws_db_instance.fuseki_metadata[0].endpoint : null
}

output "rds_database_name" {
  description = "RDS database name"
  value       = var.enable_rds ? aws_db_instance.fuseki_metadata[0].db_name : null
}

output "backup_bucket_name" {
  description = "S3 backup bucket name"
  value       = aws_s3_bucket.backups.id
}

output "backup_bucket_arn" {
  description = "S3 backup bucket ARN"
  value       = aws_s3_bucket.backups.arn
}

output "fuseki_pod_role_arn" {
  description = "IAM role ARN for Fuseki pods"
  value       = aws_iam_role.fuseki_pod.arn
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group name"
  value       = aws_cloudwatch_log_group.fuseki.name
}

# Configure kubectl command
output "configure_kubectl" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}
