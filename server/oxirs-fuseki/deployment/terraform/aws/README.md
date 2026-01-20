# OxiRS Fuseki - AWS Terraform Deployment

This Terraform module deploys OxiRS Fuseki on AWS EKS with complete infrastructure including networking, storage, backups, and monitoring.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          AWS Account                            │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                        VPC                                 │ │
│  │                                                            │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │ │
│  │  │   Public     │  │   Public     │  │   Public     │   │ │
│  │  │   Subnet 1   │  │   Subnet 2   │  │   Subnet 3   │   │ │
│  │  │   (NAT GW)   │  │   (NAT GW)   │  │   (NAT GW)   │   │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │ │
│  │         │                  │                  │          │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │ │
│  │  │   Private    │  │   Private    │  │   Private    │   │ │
│  │  │   Subnet 1   │  │   Subnet 2   │  │   Subnet 3   │   │ │
│  │  │  (EKS Nodes) │  │  (EKS Nodes) │  │  (EKS Nodes) │   │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │ │
│  │                                                            │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                    EKS Cluster                            │ │
│  │                                                            │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐               │ │
│  │  │  Fuseki  │  │  Fuseki  │  │  Fuseki  │               │ │
│  │  │   Pod 1  │  │   Pod 2  │  │   Pod 3  │               │ │
│  │  └──────────┘  └──────────┘  └──────────┘               │ │
│  │                                                            │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │     EFS     │  │     RDS     │  │     S3      │           │
│  │  (Shared)   │  │ (Metadata)  │  │  (Backups)  │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- AWS CLI configured with appropriate credentials
- Terraform >= 1.5.0
- kubectl
- aws-iam-authenticator

## Quick Start

### 1. Initialize Terraform

```bash
cd deployment/terraform/aws
terraform init
```

### 2. Configure Variables

Create a `terraform.tfvars` file:

```hcl
aws_region   = "us-east-1"
environment  = "production"
project_name = "oxirs-fuseki"

# EKS Configuration
kubernetes_version  = "1.28"
node_instance_type  = "t3.xlarge"
node_count_min      = 3
node_count_max      = 10
node_count_desired  = 3

# Storage
enable_efs = true
enable_rds = false

# Fuseki Configuration
fuseki_replicas        = 3
fuseki_cpu_request     = "1000m"
fuseki_memory_request  = "2Gi"
fuseki_cpu_limit       = "4000m"
fuseki_memory_limit    = "8Gi"

# Monitoring
enable_prometheus = true
enable_grafana    = true

# Backups
backup_retention_days = 90
log_retention_days    = 30
```

### 3. Plan Deployment

```bash
terraform plan -out=tfplan
```

### 4. Apply Configuration

```bash
terraform apply tfplan
```

### 5. Configure kubectl

```bash
aws eks update-kubeconfig --region us-east-1 --name oxirs-fuseki-production
```

### 6. Verify Deployment

```bash
kubectl get nodes
kubectl get pods -n oxirs
```

## Configuration

### Environment Variables

Required environment variables:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION` (or use var)

Optional for RDS:

- `TF_VAR_rds_password` - RDS master password

### Terraform Backend

Configure S3 backend for state management:

```bash
terraform init \
  -backend-config="bucket=your-terraform-state" \
  -backend-config="key=oxirs-fuseki/terraform.tfstate" \
  -backend-config="region=us-east-1"
```

## Components

### VPC

- CIDR: 10.0.0.0/16 (configurable)
- 3 public subnets with NAT gateways
- 3 private subnets for EKS nodes
- DNS enabled

### EKS Cluster

- Managed Kubernetes service
- IRSA enabled for pod-level IAM
- Addons: CoreDNS, kube-proxy, VPC CNI, EBS CSI
- Auto-scaling node groups

### Storage

**EBS (Persistent Volumes)**
- GP3 volumes via EBS CSI driver
- Per-pod persistent storage

**EFS (Shared Storage)**
- Optional shared file system
- Accessible from all pods
- Automatic transitions to IA storage class

**RDS (Metadata)**
- Optional PostgreSQL database
- Automated backups
- Multi-AZ for production

**S3 (Backups)**
- Versioning enabled
- Server-side encryption
- Lifecycle policies (Standard → IA → Glacier → Delete)

### Monitoring

- CloudWatch for logs
- Optional Prometheus for metrics
- Optional Grafana for dashboards

## Costs

Estimated monthly costs (us-east-1, production):

| Component | Instance Type | Quantity | Monthly Cost |
|-----------|--------------|----------|--------------|
| EKS Cluster | - | 1 | $73 |
| EC2 Nodes | t3.xlarge | 3 | $300 |
| NAT Gateway | - | 1-3 | $33-99 |
| EFS | Standard | 100GB | $30 |
| S3 Backups | Standard | 500GB | $11 |
| Data Transfer | - | 1TB | $90 |
| **Total** | | | **$537-607** |

Costs can be reduced:
- Dev/staging: Use single NAT gateway, smaller instances
- Production: Savings Plans, Reserved Instances

## Scaling

### Horizontal Pod Autoscaler

Automatically scales Fuseki pods based on CPU/memory:

```yaml
minReplicas: 3
maxReplicas: 10
targetCPUUtilizationPercentage: 70
```

### Cluster Autoscaler

Automatically adds/removes nodes based on pod requirements.

## Disaster Recovery

### Backup Strategy

- **S3 Backups**: Automated daily backups
- **Lifecycle**: 30 days Standard → 90 days IA → 365 days Glacier
- **Retention**: Configurable (default 90 days)

### RDS Backups

- Automated daily snapshots
- 30-day retention (production)
- Point-in-time recovery

### Recovery Procedure

1. **Restore from S3**:
   ```bash
   aws s3 cp s3://oxirs-fuseki-backups-production/backup-20250101.tar.gz .
   kubectl cp backup-20250101.tar.gz fuseki-pod:/data/restore/
   ```

2. **Apply to pod**:
   ```bash
   kubectl exec -it fuseki-pod -- /restore-backup.sh
   ```

## Security

### IAM Roles

- **EKS Cluster Role**: Manages cluster resources
- **Node Role**: EC2 permissions for nodes
- **Pod Role**: Fine-grained S3 access via IRSA

### Network Security

- Private subnets for all pods
- Security groups restrict access
- No direct internet access for pods

### Encryption

- EBS volumes encrypted at rest
- EFS encrypted at rest
- RDS encrypted at rest
- S3 server-side encryption

## Monitoring & Logging

### CloudWatch Logs

All pod logs are sent to CloudWatch:

```bash
aws logs tail /aws/eks/oxirs-fuseki-production/fuseki --follow
```

### Metrics

View cluster metrics:

```bash
kubectl top nodes
kubectl top pods -n oxirs
```

## Troubleshooting

### Pod Not Starting

```bash
kubectl describe pod <pod-name> -n oxirs
kubectl logs <pod-name> -n oxirs --previous
```

### EFS Mount Issues

```bash
kubectl get pvc -n oxirs
kubectl describe pvc <pvc-name> -n oxirs
```

### Network Issues

```bash
kubectl run -it --rm debug --image=busybox --restart=Never -- sh
# Test DNS
nslookup fuseki-service.oxirs.svc.cluster.local
```

## Cleanup

To destroy all resources:

```bash
terraform destroy
```

**Warning**: This will delete all data including backups (unless S3 versioning prevents it).

## Support

- Documentation: https://docs.oxirs.org
- Issues: https://github.com/cool-japan/oxirs/issues
- Terraform Registry: https://registry.terraform.io/modules/oxirs/fuseki-aws

## License

Apache 2.0
