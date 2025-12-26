# Cloud Deployment Guide for OxiRS Cluster

**Version:** 0.1.0-rc.1
**Last Updated:** December 4, 2025

## Overview

This guide provides step-by-step instructions for deploying OxiRS Cluster on major cloud platforms (AWS, Google Cloud, Azure) with best practices for production workloads.

## Table of Contents

1. [AWS Deployment](#aws-deployment)
2. [Google Cloud Deployment](#google-cloud-deployment)
3. [Azure Deployment](#azure-deployment)
4. [Multi-Cloud Deployment](#multi-cloud-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Cost Optimization](#cost-optimization)

---

## AWS Deployment

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     AWS Cloud                            │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   us-east-1  │  │  us-west-2   │  │  eu-west-1   │  │
│  │              │  │              │  │              │  │
│  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │  │
│  │ │ AZ-1a    │ │  │ │ AZ-2a    │ │  │ │ AZ-1a    │ │  │
│  │ │ • Node 1 │ │  │ │ • Node 4 │ │  │ │ • Node 7 │ │  │
│  │ │ • Node 2 │ │  │ │ • Node 5 │ │  │ │ • Node 8 │ │  │
│  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │  │
│  │              │  │              │  │              │  │
│  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │  │
│  │ │ AZ-1b    │ │  │ │ AZ-2b    │ │  │ │ AZ-1b    │ │  │
│  │ │ • Node 3 │ │  │ │ • Node 6 │ │  │ │ • Node 9 │ │  │
│  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                           │
│  S3 Buckets: oxirs-data-{region}                        │
│  VPC: 10.0.0.0/16 with Transit Gateway                  │
└─────────────────────────────────────────────────────────┘
```

### Prerequisites

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure

# Install Terraform (optional)
brew install terraform  # macOS
```

### Step 1: Create VPC and Networking

**Terraform (infrastructure/aws/vpc.tf):**

```hcl
# VPC Configuration
resource "aws_vpc" "oxirs_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "oxirs-cluster-vpc"
    Environment = "production"
  }
}

# Public Subnets (for NAT gateways)
resource "aws_subnet" "public" {
  count             = 2
  vpc_id            = aws_vpc.oxirs_vpc.id
  cidr_block        = "10.0.${count.index}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "oxirs-public-${count.index}"
  }
}

# Private Subnets (for cluster nodes)
resource "aws_subnet" "private" {
  count             = 3
  vpc_id            = aws_vpc.oxirs_vpc.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index % 2]

  tags = {
    Name = "oxirs-private-${count.index}"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.oxirs_vpc.id

  tags = {
    Name = "oxirs-igw"
  }
}

# NAT Gateway (for private subnet internet access)
resource "aws_nat_gateway" "main" {
  count         = 2
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = {
    Name = "oxirs-nat-${count.index}"
  }
}
```

### Step 2: Launch EC2 Instances

**Terraform (infrastructure/aws/ec2.tf):**

```hcl
# Security Group
resource "aws_security_group" "oxirs_cluster" {
  name        = "oxirs-cluster-sg"
  description = "Security group for OxiRS cluster nodes"
  vpc_id      = aws_vpc.oxirs_vpc.id

  # Cluster communication (Raft)
  ingress {
    from_port   = 8080
    to_port     = 8089
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  # SPARQL endpoint
  ingress {
    from_port   = 3030
    to_port     = 3030
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # GraphQL endpoint
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # SSH
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Restrict to your IP in production
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# EC2 Instances
resource "aws_instance" "oxirs_node" {
  count         = 9  # 3 nodes per AZ, 3 AZs
  ami           = "ami-0c55b159cbfafe1f0"  # Amazon Linux 2023
  instance_type = "t3.xlarge"  # 4 vCPU, 16 GB RAM

  subnet_id              = aws_subnet.private[count.index % 3].id
  vpc_security_group_ids = [aws_security_group.oxirs_cluster.id]

  key_name = aws_key_pair.oxirs.key_name

  user_data = templatefile("${path.module}/user_data.sh", {
    node_id = count.index + 1
    region  = var.aws_region
  })

  root_block_device {
    volume_type = "gp3"
    volume_size = 100
    iops        = 3000
    throughput  = 125
  }

  tags = {
    Name        = "oxirs-node-${count.index + 1}"
    Role        = "cluster-node"
    Environment = "production"
  }
}
```

### Step 3: Configure S3 Storage

```hcl
# S3 Bucket for backups
resource "aws_s3_bucket" "oxirs_backups" {
  bucket = "oxirs-cluster-backups-${var.aws_region}"

  tags = {
    Name        = "OxiRS Cluster Backups"
    Environment = "production"
  }
}

# Enable versioning
resource "aws_s3_bucket_versioning" "backups" {
  bucket = aws_s3_bucket.oxirs_backups.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Lifecycle policy
resource "aws_s3_bucket_lifecycle_configuration" "backups" {
  bucket = aws_s3_bucket.oxirs_backups.id

  rule {
    id     = "transition-to-glacier"
    status = "Enabled"

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = 365
    }
  }
}
```

### Step 4: Deploy OxiRS Cluster

**User Data Script (user_data.sh):**

```bash
#!/bin/bash
set -e

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Install dependencies
yum update -y
yum install -y git gcc openssl-devel pkg-config

# Clone OxiRS repository
cd /opt
git clone https://github.com/cool-japan/oxirs.git
cd oxirs/storage/oxirs-cluster

# Build release binary
cargo build --release --features cuda

# Create systemd service
cat > /etc/systemd/system/oxirs-cluster.service <<EOF
[Unit]
Description=OxiRS Cluster Node
After=network.target

[Service]
Type=simple
User=oxirs
WorkingDirectory=/opt/oxirs/storage/oxirs-cluster
ExecStart=/opt/oxirs/storage/oxirs-cluster/target/release/oxirs-cluster \
    --config /etc/oxirs/cluster.toml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create configuration
mkdir -p /etc/oxirs
cat > /etc/oxirs/cluster.toml <<EOF
[cluster]
node_id = ${node_id}
region = "${region}"
data_dir = "/var/lib/oxirs"

[storage]
backend = "s3"
bucket = "oxirs-cluster-backups-${region}"
default_tier = "hot"

[network]
bind_address = "0.0.0.0:8080"
advertise_address = "$(ec2-metadata --local-ipv4 | cut -d' ' -f2):8080"

[replication]
replication_factor = 3
sync_writes = true
EOF

# Start service
systemctl enable oxirs-cluster
systemctl start oxirs-cluster
```

### Step 5: Configure Application Load Balancer

```hcl
# Application Load Balancer
resource "aws_lb" "oxirs" {
  name               = "oxirs-cluster-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  tags = {
    Name = "oxirs-cluster-alb"
  }
}

# Target Group
resource "aws_lb_target_group" "oxirs_sparql" {
  name     = "oxirs-sparql-tg"
  port     = 3030
  protocol = "HTTP"
  vpc_id   = aws_vpc.oxirs_vpc.id

  health_check {
    path                = "/health"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 2
  }
}

# Listener
resource "aws_lb_listener" "sparql" {
  load_balancer_arn = aws_lb.oxirs.arn
  port              = 443
  protocol          = "HTTPS"
  certificate_arn   = aws_acm_certificate.oxirs.arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.oxirs_sparql.arn
  }
}

# Target Group Attachments
resource "aws_lb_target_group_attachment" "oxirs_nodes" {
  count            = length(aws_instance.oxirs_node)
  target_group_arn = aws_lb_target_group.oxirs_sparql.arn
  target_id        = aws_instance.oxirs_node[count.index].id
  port             = 3030
}
```

### Step 6: Deploy with Terraform

```bash
cd infrastructure/aws

# Initialize Terraform
terraform init

# Plan deployment
terraform plan -out=tfplan

# Apply configuration
terraform apply tfplan

# Get cluster endpoints
terraform output
```

---

## Google Cloud Deployment

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│              Google Cloud Platform                       │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ us-central1  │  │  us-west1    │  │ europe-west1 │  │
│  │              │  │              │  │              │  │
│  │ • GKE Cluster│  │ • GKE Cluster│  │ • GKE Cluster│  │
│  │ • 3 nodes    │  │ • 3 nodes    │  │ • 3 nodes    │  │
│  │ • n2-std-4   │  │ • n2-std-4   │  │ • n2-std-4   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                           │
│  GCS Buckets: oxirs-data-{region}                       │
│  VPC: Custom network with global routing                │
└─────────────────────────────────────────────────────────┘
```

### Prerequisites

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### Step 1: Create GKE Cluster

**Terraform (infrastructure/gcp/gke.tf):**

```hcl
# GKE Cluster
resource "google_container_cluster" "oxirs" {
  name     = "oxirs-cluster"
  location = "us-central1"

  # Multi-zonal cluster
  node_locations = [
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
  ]

  # Networking
  network    = google_compute_network.oxirs_vpc.name
  subnetwork = google_compute_subnetwork.oxirs_subnet.name

  # Workload Identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Initial node pool
  remove_default_node_pool = true
  initial_node_count       = 1
}

# Node Pool
resource "google_container_node_pool" "oxirs_nodes" {
  name       = "oxirs-node-pool"
  cluster    = google_container_cluster.oxirs.name
  location   = google_container_cluster.oxirs.location
  node_count = 3

  node_config {
    machine_type = "n2-standard-4"  # 4 vCPU, 16 GB RAM
    disk_size_gb = 100
    disk_type    = "pd-ssd"

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    labels = {
      app = "oxirs-cluster"
    }

    tags = ["oxirs-cluster"]
  }

  autoscaling {
    min_node_count = 3
    max_node_count = 10
  }
}
```

### Step 2: Deploy to GKE

**Kubernetes Manifest (k8s/oxirs-deployment.yaml):**

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: oxirs-cluster
  namespace: oxirs
spec:
  serviceName: oxirs-cluster
  replicas: 9
  selector:
    matchLabels:
      app: oxirs-cluster
  template:
    metadata:
      labels:
        app: oxirs-cluster
    spec:
      containers:
      - name: oxirs
        image: gcr.io/YOUR_PROJECT/oxirs-cluster:latest
        ports:
        - containerPort: 8080
          name: raft
        - containerPort: 3030
          name: sparql
        - containerPort: 8000
          name: graphql
        env:
        - name: NODE_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: REGION
          value: "us-central1"
        - name: GCS_BUCKET
          value: "oxirs-cluster-backups"
        volumeMounts:
        - name: data
          mountPath: /var/lib/oxirs
        resources:
          requests:
            cpu: "2"
            memory: "8Gi"
          limits:
            cpu: "4"
            memory: "16Gi"
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: "pd-ssd"
      resources:
        requests:
          storage: 100Gi
---
apiVersion: v1
kind: Service
metadata:
  name: oxirs-cluster
  namespace: oxirs
spec:
  type: LoadBalancer
  ports:
  - port: 3030
    targetPort: 3030
    name: sparql
  - port: 8000
    targetPort: 8000
    name: graphql
  selector:
    app: oxirs-cluster
```

**Deploy:**

```bash
# Create namespace
kubectl create namespace oxirs

# Apply manifests
kubectl apply -f k8s/oxirs-deployment.yaml

# Check status
kubectl get pods -n oxirs
kubectl get svc -n oxirs
```

---

## Azure Deployment

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  Microsoft Azure                         │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  East US     │  │  West US 2   │  │ North Europe │  │
│  │              │  │              │  │              │  │
│  │ • AKS Cluster│  │ • AKS Cluster│  │ • AKS Cluster│  │
│  │ • 3 nodes    │  │ • 3 nodes    │  │ • 3 nodes    │  │
│  │ • D4s_v3     │  │ • D4s_v3     │  │ • D4s_v3     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                           │
│  Storage Accounts: oxirs{region}                        │
│  VNet: 10.0.0.0/8 with Global VNet Peering             │
└─────────────────────────────────────────────────────────┘
```

### Prerequisites

```bash
# Install Azure CLI
brew install azure-cli  # macOS

# Login
az login
az account set --subscription YOUR_SUBSCRIPTION_ID
```

### Step 1: Create AKS Cluster

**Terraform (infrastructure/azure/aks.tf):**

```hcl
# Resource Group
resource "azurerm_resource_group" "oxirs" {
  name     = "oxirs-cluster-rg"
  location = "East US"
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "oxirs" {
  name                = "oxirs-cluster"
  location            = azurerm_resource_group.oxirs.location
  resource_group_name = azurerm_resource_group.oxirs.name
  dns_prefix          = "oxirs"

  default_node_pool {
    name                = "default"
    node_count          = 3
    vm_size             = "Standard_D4s_v3"  # 4 vCPU, 16 GB RAM
    availability_zones  = ["1", "2", "3"]
    enable_auto_scaling = true
    min_count           = 3
    max_count           = 10
    os_disk_size_gb     = 100
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin    = "azure"
    load_balancer_sku = "standard"
  }

  tags = {
    Environment = "production"
  }
}
```

### Step 2: Configure Azure Blob Storage

```hcl
# Storage Account
resource "azurerm_storage_account" "oxirs" {
  name                     = "oxirsclusterbackups"
  resource_group_name      = azurerm_resource_group.oxirs.name
  location                 = azurerm_resource_group.oxirs.location
  account_tier             = "Standard"
  account_replication_type = "GRS"  # Geo-redundant

  blob_properties {
    versioning_enabled = true

    delete_retention_policy {
      days = 90
    }
  }
}

# Container
resource "azurerm_storage_container" "backups" {
  name                  = "backups"
  storage_account_name  = azurerm_storage_account.oxirs.name
  container_access_type = "private"
}
```

---

## Multi-Cloud Deployment

### Global Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Multi-Cloud OxiRS                          │
│                                                                │
│  ┌───────────┐     ┌───────────┐     ┌───────────┐          │
│  │    AWS    │────│    GCP    │────│   Azure   │          │
│  │           │     │           │     │           │          │
│  │ 3 regions │     │ 3 regions │     │ 3 regions │          │
│  │ 9 nodes   │     │ 9 nodes   │     │ 9 nodes   │          │
│  └───────────┘     └───────────┘     └───────────┘          │
│                                                                │
│  Global Traffic Manager / Route 53 / Cloud DNS               │
│  Cross-cloud VPN / SD-WAN                                    │
└──────────────────────────────────────────────────────────────┘
```

### Configuration

**oxirs.toml (Multi-cloud):**

```toml
[cluster]
multi_cloud = true
regions = [
    { provider = "aws", region = "us-east-1" },
    { provider = "gcp", region = "us-central1" },
    { provider = "azure", region = "eastus" },
]

[replication]
cross_cloud_replication = true
replication_factor = 9  # 3 per cloud provider
sync_writes = false  # Async for cross-cloud latency

[disaster_recovery]
multi_cloud_failover = true
rto_seconds = 300
rpo_seconds = 60
```

---

## Monitoring and Observability

### Prometheus + Grafana

**Kubernetes Manifest:**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'oxirs-cluster'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
          - oxirs
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: oxirs-cluster
      - source_labels: [__address__]
        action: replace
        target_label: __address__
        regex: ([^:]+)(?::\d+)?
        replacement: $1:9090
```

### CloudWatch (AWS)

```bash
# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/amazon_linux/amd64/latest/amazon-cloudwatch-agent.rpm
sudo rpm -U ./amazon-cloudwatch-agent.rpm

# Configure
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config \
    -m ec2 \
    -s \
    -c file:/opt/aws/amazon-cloudwatch-agent/etc/config.json
```

---

## Cost Optimization

### Auto-Scaling Configuration

```toml
[auto_scaling]
enabled = true
min_nodes = 9   # 3 per region
max_nodes = 27  # 9 per region

# Scale up
scale_up_threshold_cpu = 0.75
scale_up_threshold_memory = 0.80

# Scale down
scale_down_threshold_cpu = 0.30
scale_down_threshold_memory = 0.40
scale_down_cooldown_minutes = 10

# Predictive scaling
predictive_scaling = true
forecast_horizon_minutes = 60
```

### Spot/Preemptible Instances

**AWS:**
```hcl
resource "aws_instance" "oxirs_spot" {
  instance_market_options {
    market_type = "spot"
    spot_options {
      max_price = "0.50"  # Max $0.50/hour
    }
  }
}
```

**GCP:**
```hcl
resource "google_container_node_pool" "oxirs_preemptible" {
  node_config {
    preemptible  = true
    machine_type = "n2-standard-4"
  }
}
```

### Storage Tiering

```rust
// Automatic data lifecycle management
let lifecycle_policy = LifecyclePolicy {
    hot_to_warm_days: 30,
    warm_to_cold_days: 90,
    cold_to_archive_days: 365,
};
```

---

## Additional Resources

- [AWS Best Practices](https://docs.aws.amazon.com/wellarchitected/)
- [GCP Architecture Center](https://cloud.google.com/architecture)
- [Azure Architecture Center](https://docs.microsoft.com/en-us/azure/architecture/)
- [OxiRS Performance Tuning](./PERFORMANCE_TUNING.md)

---

**Last Updated:** December 4, 2025
**Maintainer:** OxiRS Team
