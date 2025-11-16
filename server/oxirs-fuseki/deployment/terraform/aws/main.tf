# OxiRS Fuseki - AWS Terraform Module
# Deploys OxiRS Fuseki on AWS EKS with complete infrastructure

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }

  backend "s3" {
    # Configure with -backend-config flags or environment variables
    # bucket = "your-terraform-state-bucket"
    # key    = "oxirs-fuseki/terraform.tfstate"
    # region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "OxiRS-Fuseki"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC Module
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.project_name}-vpc-${var.environment}"
  cidr = var.vpc_cidr

  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  enable_nat_gateway   = true
  single_nat_gateway   = var.environment != "production"
  enable_dns_hostnames = true
  enable_dns_support   = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
  }
}

# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "${var.project_name}-${var.environment}"
  cluster_version = var.kubernetes_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_private_access = true
  cluster_endpoint_public_access  = true

  # Enable IRSA for pod-level IAM
  enable_irsa = true

  # Cluster addons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  # EKS Managed Node Groups
  eks_managed_node_groups = {
    fuseki_nodes = {
      name = "fuseki-nodes"

      instance_types = [var.node_instance_type]
      capacity_type  = "ON_DEMAND"

      min_size     = var.node_count_min
      max_size     = var.node_count_max
      desired_size = var.node_count_desired

      disk_size = var.node_disk_size

      labels = {
        role = "fuseki"
      }

      tags = {
        NodeGroup = "fuseki"
      }
    }
  }
}

# EFS for shared storage (optional, for RWX volumes)
resource "aws_efs_file_system" "fuseki_data" {
  count = var.enable_efs ? 1 : 0

  creation_token = "${var.project_name}-data-${var.environment}"
  encrypted      = true

  performance_mode = "generalPurpose"
  throughput_mode  = "bursting"

  lifecycle_policy {
    transition_to_ia = "AFTER_30_DAYS"
  }

  tags = {
    Name = "${var.project_name}-data-${var.environment}"
  }
}

resource "aws_efs_mount_target" "fuseki_data" {
  count = var.enable_efs ? length(module.vpc.private_subnets) : 0

  file_system_id  = aws_efs_file_system.fuseki_data[0].id
  subnet_id       = module.vpc.private_subnets[count.index]
  security_groups = [aws_security_group.efs[0].id]
}

# Security group for EFS
resource "aws_security_group" "efs" {
  count = var.enable_efs ? 1 : 0

  name_prefix = "${var.project_name}-efs-${var.environment}"
  description = "Security group for EFS mount targets"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 2049
    to_port     = 2049
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-efs-${var.environment}"
  }
}

# RDS for metadata (optional)
resource "aws_db_subnet_group" "fuseki" {
  count = var.enable_rds ? 1 : 0

  name       = "${var.project_name}-${var.environment}"
  subnet_ids = module.vpc.private_subnets

  tags = {
    Name = "${var.project_name}-db-subnet-group"
  }
}

resource "aws_db_instance" "fuseki_metadata" {
  count = var.enable_rds ? 1 : 0

  identifier = "${var.project_name}-metadata-${var.environment}"

  engine               = "postgres"
  engine_version       = "15.3"
  instance_class       = var.rds_instance_class
  allocated_storage    = var.rds_allocated_storage
  storage_encrypted    = true
  storage_type         = "gp3"
  db_subnet_group_name = aws_db_subnet_group.fuseki[0].name
  vpc_security_group_ids = [aws_security_group.rds[0].id]

  db_name  = "fuseki"
  username = var.rds_username
  password = var.rds_password

  backup_retention_period = var.environment == "production" ? 30 : 7
  backup_window           = "03:00-04:00"
  maintenance_window      = "Mon:04:00-Mon:05:00"

  skip_final_snapshot       = var.environment != "production"
  final_snapshot_identifier = "${var.project_name}-final-${var.environment}-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"

  tags = {
    Name = "${var.project_name}-metadata-${var.environment}"
  }
}

# Security group for RDS
resource "aws_security_group" "rds" {
  count = var.enable_rds ? 1 : 0

  name_prefix = "${var.project_name}-rds-${var.environment}"
  description = "Security group for RDS instance"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-rds-${var.environment}"
  }
}

# S3 bucket for backups
resource "aws_s3_bucket" "backups" {
  bucket = "${var.project_name}-backups-${var.environment}-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name = "${var.project_name}-backups"
  }
}

resource "aws_s3_bucket_versioning" "backups" {
  bucket = aws_s3_bucket.backups.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id

  rule {
    id     = "backup-lifecycle"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = var.backup_retention_days
    }
  }
}

# IAM role for Fuseki pods
resource "aws_iam_role" "fuseki_pod" {
  name = "${var.project_name}-pod-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRoleWithWebIdentity"
      Effect = "Allow"
      Principal = {
        Federated = module.eks.oidc_provider_arn
      }
      Condition = {
        StringEquals = {
          "${module.eks.oidc_provider}:sub" = "system:serviceaccount:oxirs:fuseki-sa"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "fuseki_s3" {
  name = "s3-backup-access"
  role = aws_iam_role.fuseki_pod.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ]
      Resource = [
        aws_s3_bucket.backups.arn,
        "${aws_s3_bucket.backups.arn}/*"
      ]
    }]
  })
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "fuseki" {
  name              = "/aws/eks/${var.project_name}-${var.environment}/fuseki"
  retention_in_days = var.log_retention_days

  tags = {
    Name = "${var.project_name}-logs"
  }
}
