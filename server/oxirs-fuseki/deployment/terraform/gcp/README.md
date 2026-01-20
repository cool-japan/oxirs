# OxiRS Fuseki - GCP Terraform Module

Terraform configuration for deploying OxiRS Fuseki on Google Cloud Platform (GCP) using Google Kubernetes Engine (GKE).

## Architecture

This Terraform module provisions:

### Compute
- **GKE Regional Cluster** with private nodes and Workload Identity
- **Auto-scaling Node Pools** with n2-standard-4 instances
- **Shielded GKE Nodes** for enhanced security

### Networking
- **VPC Network** with custom subnets
- **Cloud Router & Cloud NAT** for private node internet access
- **Firewall Rules** for health checks and monitoring
- **Private GKE Master** with authorized networks

### Storage
- **Cloud Filestore** (1TB) for shared persistent RDF dataset storage
- **Cloud Storage Bucket** for automated backups with versioning
- **Cloud SQL (PostgreSQL 15)** (optional) for metadata and state

### Security
- **Workload Identity** for pod-level IAM permissions
- **Cloud KMS** for encryption at rest
- **Binary Authorization** for container image validation
- **Private GKE Nodes** without public IPs
- **Service Accounts** with least-privilege IAM roles

### Monitoring
- **Cloud Monitoring** integration for cluster metrics
- **Cloud Logging** for centralized log management
- **Alert Policies** for high CPU and pod restarts
- **Custom Dashboards** (optional)

## Prerequisites

1. **GCP Account** with billing enabled
2. **Project** created in GCP Console
3. **Terraform** 1.9+ installed
4. **gcloud CLI** installed and authenticated:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```
5. **kubectl** installed for Kubernetes management

## Quick Start

### 1. Configure GCP

```bash
# Set your project
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Enable billing (if not already enabled)
gcloud beta billing projects link $PROJECT_ID --billing-account=YOUR_BILLING_ACCOUNT_ID
```

### 2. Initialize Terraform

```bash
cd deployment/terraform/gcp

# Initialize Terraform
terraform init
```

### 3. Configure Variables

Create a `terraform.tfvars` file:

```hcl
project_id   = "your-project-id"
region       = "us-central1"
cluster_name = "oxirs-fuseki-prod"
environment  = "production"

# Node pool configuration
node_machine_type = "n2-standard-4"
min_node_count    = 1
max_node_count    = 10

# Storage configuration
filestore_capacity_gb = 1024
backup_retention_days = 30

# Enable Cloud SQL for metadata
enable_cloud_sql = true
cloud_sql_tier   = "db-n1-standard-2"

# Network security
master_authorized_networks = [
  {
    cidr_block   = "YOUR_OFFICE_IP/32"
    display_name = "Office"
  }
]

# Monitoring
notification_channels = ["projects/YOUR_PROJECT/notificationChannels/YOUR_CHANNEL_ID"]
```

### 4. Plan and Apply

```bash
# Preview changes
terraform plan

# Apply configuration
terraform apply

# Note the outputs
terraform output
```

### 5. Configure kubectl

```bash
# Get cluster credentials
gcloud container clusters get-credentials oxirs-fuseki-prod \
  --region us-central1 \
  --project your-project-id

# Verify connection
kubectl cluster-info
kubectl get nodes
```

### 6. Deploy OxiRS Fuseki

```bash
# Deploy to Kubernetes
kubectl apply -f ../../kubernetes/

# Check deployment
kubectl get pods -n oxirs-fuseki
kubectl get svc -n oxirs-fuseki

# Watch deployment progress
kubectl rollout status deployment/oxirs-fuseki -n oxirs-fuseki
```

## Configuration

### Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `project_id` | string | - | **Required.** GCP Project ID |
| `region` | string | `us-central1` | GCP region |
| `availability_zones` | list(string) | `["us-central1-a", "us-central1-b", "us-central1-c"]` | Availability zones |
| `cluster_name` | string | `oxirs-fuseki` | GKE cluster name |
| `environment` | string | `production` | Environment name |
| `node_machine_type` | string | `n2-standard-4` | GKE node machine type |
| `min_node_count` | number | `1` | Minimum nodes per zone |
| `max_node_count` | number | `10` | Maximum nodes per zone |
| `filestore_capacity_gb` | number | `1024` | Cloud Filestore capacity |
| `enable_cloud_sql` | bool | `true` | Enable Cloud SQL |
| `cloud_sql_tier` | string | `db-n1-standard-2` | Cloud SQL machine type |

### Environments

Different configurations for each environment:

#### Production
```hcl
environment       = "production"
node_machine_type = "n2-standard-4"
min_node_count    = 2
max_node_count    = 10
enable_cloud_sql  = true
```

#### Staging
```hcl
environment       = "staging"
node_machine_type = "n2-standard-2"
min_node_count    = 1
max_node_count    = 5
enable_cloud_sql  = true
```

#### Development
```hcl
environment       = "development"
node_machine_type = "n2-standard-2"
min_node_count    = 1
max_node_count    = 3
enable_cloud_sql  = false
```

## Outputs

After applying, Terraform provides:

```bash
# View all outputs
terraform output

# Specific outputs
terraform output cluster_endpoint
terraform output kubectl_config_command
terraform output storage_bucket_name
terraform output filestore_ip_address
```

## Cost Estimation

### Minimum Configuration (Development)
- **GKE Control Plane**: ~$73/month (zonal) or ~$219/month (regional)
- **GKE Nodes** (1 per zone, 3 total): ~$285/month (n2-standard-4)
- **Cloud Filestore** (1TB): ~$200/month
- **Cloud Storage**: ~$20/month
- **Networking**: ~$50/month
- **Monitoring**: ~$10/month
- **Total**: ~$638-784/month

### Production Configuration
- **GKE Control Plane**: ~$219/month (regional HA)
- **GKE Nodes** (2 per zone, 6 total): ~$570/month
- **Cloud SQL** (db-n1-standard-2): ~$150/month
- **Cloud Filestore** (1TB): ~$200/month
- **Cloud Storage**: ~$20/month
- **Networking**: ~$100/month
- **Monitoring**: ~$30/month
- **Total**: ~$1,289/month

*Note: Actual costs may vary based on usage, egress, and specific configuration.*

## Security Best Practices

### Network Security
- Private GKE nodes without public IPs
- Master authorized networks to restrict API access
- Firewall rules for specific services only
- Cloud NAT for outbound internet access

### Identity and Access
- Workload Identity for pod-level permissions
- Service accounts with least-privilege IAM roles
- No service account keys (use Workload Identity)
- Binary Authorization for container images

### Data Protection
- Encryption at rest with Cloud KMS
- Encrypted connections (TLS)
- Backup retention policies
- Point-in-time recovery for Cloud SQL

### Monitoring and Compliance
- Cloud Audit Logs enabled
- Alert policies for anomalies
- Regular security scanning
- Compliance with organization policies

## Maintenance

### Cluster Upgrades

GKE automatically manages control plane upgrades. For node upgrades:

```bash
# Check available versions
gcloud container get-server-config --region us-central1

# Upgrade control plane (if needed)
gcloud container clusters upgrade oxirs-fuseki-prod \
  --region us-central1 \
  --master \
  --cluster-version VERSION

# Upgrade node pools
gcloud container clusters upgrade oxirs-fuseki-prod \
  --region us-central1 \
  --node-pool oxirs-fuseki-prod-primary-pool
```

### Backups

Backups are automatically stored in Cloud Storage:

```bash
# List backups
gsutil ls gs://YOUR_PROJECT-oxirs-fuseki-backups/

# Download backup
gsutil cp gs://YOUR_PROJECT-oxirs-fuseki-backups/backup-YYYYMMDD.tar.gz .

# Restore from backup (see restoration guide)
```

### Scaling

```bash
# Manual scaling
gcloud container clusters resize oxirs-fuseki-prod \
  --region us-central1 \
  --node-pool oxirs-fuseki-prod-primary-pool \
  --num-nodes 5

# Or use kubectl
kubectl scale deployment oxirs-fuseki --replicas=5 -n oxirs-fuseki
```

## Monitoring

### Access Cloud Monitoring

```bash
# Open monitoring console
echo "https://console.cloud.google.com/monitoring?project=$PROJECT_ID"
```

### View Metrics

- **CPU Usage**: GKE cluster and pod CPU utilization
- **Memory Usage**: Memory consumption per pod
- **Disk I/O**: Filestore and persistent disk operations
- **Network**: Ingress/egress traffic
- **SPARQL Queries**: Custom metrics from OxiRS Fuseki

### Alerts

Pre-configured alerts:
- High CPU usage (>80% for 5 minutes)
- Frequent pod restarts (>5 in 5 minutes)
- Disk space warnings
- Node health issues

## Troubleshooting

### Common Issues

#### Issue: Cannot connect to cluster
```bash
# Re-authenticate
gcloud auth login
gcloud container clusters get-credentials CLUSTER_NAME --region REGION

# Check cluster status
gcloud container clusters describe CLUSTER_NAME --region REGION
```

#### Issue: Pods not starting
```bash
# Check pod status
kubectl describe pod POD_NAME -n oxirs-fuseki

# View logs
kubectl logs POD_NAME -n oxirs-fuseki

# Check events
kubectl get events -n oxirs-fuseki --sort-by='.lastTimestamp'
```

#### Issue: Storage mount failures
```bash
# Check Filestore status
gcloud filestore instances describe oxirs-fuseki-filestore --zone ZONE

# Verify PV/PVC
kubectl get pv
kubectl get pvc -n oxirs-fuseki
```

### Support

For issues and questions:
- **GitHub Issues**: https://github.com/cool-japan/oxirs/issues
- **Documentation**: https://github.com/cool-japan/oxirs/tree/main/docs
- **GCP Support**: https://cloud.google.com/support

## Cleanup

To destroy all resources:

```bash
# Warning: This will delete all data!
terraform destroy

# Confirm when prompted
```

## Advanced Configuration

### Custom Machine Types

```hcl
node_machine_type = "n2-custom-4-16384"  # 4 vCPUs, 16GB RAM
```

### Multiple Node Pools

Add additional node pools for specific workloads in `main.tf`.

### VPC Peering

Configure VPC peering for hybrid connectivity.

### Private Service Connect

Enable Private Service Connect for fully private architecture.

## License

See the main OxiRS project for licensing information.
