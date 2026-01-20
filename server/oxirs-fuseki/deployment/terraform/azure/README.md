# OxiRS Fuseki - Azure Terraform Module

Terraform configuration for deploying OxiRS Fuseki on Microsoft Azure using Azure Kubernetes Service (AKS).

## Architecture

This Terraform module provisions:

### Compute
- **AKS Cluster** with system and application node pools
- **Auto-scaling Node Pools** with Standard_D4s_v3 instances
- **Azure Monitor** integration for metrics and logs

### Networking
- **Virtual Network** with multiple subnets
- **Network Security Groups** for traffic control
- **Private DNS Zones** for PostgreSQL
- **Application Gateway** (optional) with WAF

### Storage
- **Azure Files** (1TB) for shared persistent storage
- **Azure Storage Account** for backups with GRS replication
- **Azure Database for PostgreSQL** (optional) for metadata

### Security
- **Workload Identity** with federated credentials
- **Azure Key Vault** for secrets management
- **Managed Identities** for Azure resource access
- **Azure RBAC** for role-based access control
- **Private AKS** option for enhanced security

### Monitoring
- **Log Analytics Workspace** for centralized logging
- **Azure Monitor** for metrics and alerts
- **Action Groups** for alert notifications
- **Metric Alerts** for CPU and memory thresholds

## Prerequisites

1. **Azure Account** with active subscription
2. **Azure CLI** installed and authenticated:
   ```bash
   az login
   az account set --subscription "your-subscription-id"
   ```
3. **Terraform** 1.9+ installed
4. **kubectl** installed for Kubernetes management

## Quick Start

### 1. Configure Azure

```bash
# Login to Azure
az login

# Select subscription
az account list --output table
az account set --subscription "your-subscription-id"

# Create resource group for Terraform state (optional)
az group create --name terraform-state-rg --location eastus

# Create storage account for Terraform state (optional)
az storage account create \
  --name oxirsterraformstate \
  --resource-group terraform-state-rg \
  --location eastus \
  --sku Standard_LRS

# Create blob container
az storage container create \
  --name tfstate \
  --account-name oxirsterraformstate
```

### 2. Initialize Terraform

```bash
cd deployment/terraform/azure

# Initialize Terraform
terraform init
```

### 3. Configure Variables

Create a `terraform.tfvars` file:

```hcl
location     = "eastus"
cluster_name = "oxirs-fuseki-prod"
environment  = "production"

# Node pool configuration
node_vm_size      = "Standard_D4s_v3"
min_node_count    = 1
max_node_count    = 10
node_disk_size_gb = 100

# Storage configuration
storage_replication_type = "GRS"
azure_files_quota_gb     = 1024
backup_retention_days    = 30

# PostgreSQL configuration
enable_postgresql               = true
postgresql_admin_username       = "oxirsadmin"
postgresql_admin_password       = "YourSecurePassword123!"
postgresql_sku_name             = "GP_Standard_D2s_v3"
postgresql_storage_mb           = 131072
postgresql_backup_retention_days = 7

# Networking
private_cluster_enabled = false

# Application Gateway (optional)
enable_application_gateway = false

# Monitoring
log_retention_days = 30
alert_email_receivers = [
  "ops@example.com",
  "admin@example.com"
]

# Tags
tags = {
  environment = "production"
  managed_by  = "terraform"
  application = "oxirs-fuseki"
  team        = "data-engineering"
}
```

### 4. Plan and Apply

```bash
# Preview changes
terraform plan

# Apply configuration
terraform apply

# Note the outputs
terraform output
terraform output -json > outputs.json
```

### 5. Configure kubectl

```bash
# Get AKS credentials
az aks get-credentials \
  --resource-group oxirs-fuseki-prod-rg \
  --name oxirs-fuseki-prod

# Verify connection
kubectl cluster-info
kubectl get nodes
kubectl get namespaces
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
| `location` | string | `eastus` | Azure region |
| `cluster_name` | string | `oxirs-fuseki` | AKS cluster name |
| `environment` | string | `production` | Environment name |
| `kubernetes_version` | string | `1.29` | Kubernetes version |
| `node_vm_size` | string | `Standard_D4s_v3` | Application node VM size |
| `min_node_count` | number | `1` | Minimum application nodes |
| `max_node_count` | number | `10` | Maximum application nodes |
| `azure_files_quota_gb` | number | `1024` | Azure Files quota |
| `enable_postgresql` | bool | `true` | Enable PostgreSQL |
| `private_cluster_enabled` | bool | `false` | Enable private AKS |

### Environments

Different configurations for each environment:

#### Production
```hcl
environment              = "production"
node_vm_size             = "Standard_D4s_v3"
min_node_count           = 2
max_node_count           = 10
enable_postgresql        = true
private_cluster_enabled  = true
storage_replication_type = "GRS"
```

#### Staging
```hcl
environment              = "staging"
node_vm_size             = "Standard_D2s_v3"
min_node_count           = 1
max_node_count           = 5
enable_postgresql        = true
private_cluster_enabled  = false
storage_replication_type = "LRS"
```

#### Development
```hcl
environment              = "development"
node_vm_size             = "Standard_D2s_v3"
min_node_count           = 1
max_node_count           = 3
enable_postgresql        = false
private_cluster_enabled  = false
storage_replication_type = "LRS"
```

## Outputs

After applying, Terraform provides:

```bash
# View all outputs
terraform output

# Specific outputs
terraform output kube_config > ~/.kube/config-oxirs
terraform output storage_account_name
terraform output postgresql_server_fqdn
terraform output workload_identity_client_id
```

## Cost Estimation

### Minimum Configuration (Development)
- **AKS Control Plane**: ~$73/month
- **System Nodes** (1x Standard_D2s_v3): ~$70/month
- **Application Nodes** (1x Standard_D4s_v3): ~$140/month
- **Azure Files** (1TB): ~$60/month
- **Storage Account**: ~$25/month
- **Log Analytics**: ~$20/month
- **Total**: ~$388/month

### Production Configuration
- **AKS Control Plane**: ~$73/month
- **System Nodes** (1x Standard_D2s_v3): ~$70/month
- **Application Nodes** (2x Standard_D4s_v3): ~$280/month
- **PostgreSQL** (GP_Standard_D2s_v3): ~$150/month
- **Azure Files** (1TB): ~$60/month
- **Storage Account** (GRS): ~$25/month
- **Application Gateway** (WAF_v2): ~$180/month (if enabled)
- **Log Analytics**: ~$50/month
- **Networking**: ~$50/month
- **Total**: ~$758/month (without App Gateway) or ~$938/month (with App Gateway)

*Note: Actual costs may vary based on usage, data transfer, and specific configuration.*

## Security Best Practices

### Network Security
- Use Network Security Groups to control traffic
- Enable private AKS cluster for production
- Use Application Gateway with WAF for ingress
- Implement Azure Private Link for database connections

### Identity and Access
- Use Workload Identity for pod-level permissions
- Enable Azure RBAC for Kubernetes authorization
- Use Managed Identities instead of service principals
- Store secrets in Azure Key Vault

### Data Protection
- Use Azure Disk Encryption for node disks
- Enable encryption at rest for storage accounts
- Use geo-redundant storage (GRS) for production
- Enable soft delete and versioning for backups

### Monitoring and Compliance
- Enable Azure Monitor for comprehensive visibility
- Configure alert rules for critical metrics
- Use Azure Policy for governance
- Enable Azure Security Center

## Maintenance

### Cluster Upgrades

AKS handles control plane upgrades automatically during maintenance windows. For node upgrades:

```bash
# Check available versions
az aks get-versions --location eastus --output table

# Upgrade control plane
az aks upgrade \
  --resource-group oxirs-fuseki-prod-rg \
  --name oxirs-fuseki-prod \
  --kubernetes-version 1.29.0

# Upgrade node pools
az aks nodepool upgrade \
  --resource-group oxirs-fuseki-prod-rg \
  --cluster-name oxirs-fuseki-prod \
  --name oxirs \
  --kubernetes-version 1.29.0
```

### Backups

Backups are automatically stored in Azure Storage:

```bash
# List backups
az storage blob list \
  --account-name YOUR_STORAGE_ACCOUNT \
  --container-name oxirs-backups \
  --output table

# Download backup
az storage blob download \
  --account-name YOUR_STORAGE_ACCOUNT \
  --container-name oxirs-backups \
  --name backup-YYYYMMDD.tar.gz \
  --file ./backup.tar.gz
```

### Scaling

```bash
# Manual scaling
az aks scale \
  --resource-group oxirs-fuseki-prod-rg \
  --name oxirs-fuseki-prod \
  --nodepool-name oxirs \
  --node-count 5

# Or use kubectl
kubectl scale deployment oxirs-fuseki --replicas=5 -n oxirs-fuseki
```

### Monitoring

#### Access Azure Monitor

```bash
# Open Azure Portal
echo "https://portal.azure.com/#@/resource/subscriptions/YOUR_SUB_ID/resourceGroups/oxirs-fuseki-prod-rg/providers/Microsoft.ContainerService/managedClusters/oxirs-fuseki-prod/overview"
```

#### View Metrics

- **Cluster Health**: Overall cluster status
- **Node Health**: Individual node status
- **CPU Usage**: Per-pod and per-node CPU utilization
- **Memory Usage**: Memory consumption metrics
- **Network**: Ingress/egress traffic
- **Storage**: Disk I/O and capacity

#### View Logs

```bash
# Using kubectl
kubectl logs -f deployment/oxirs-fuseki -n oxirs-fuseki

# Using Azure CLI
az monitor log-analytics query \
  --workspace YOUR_WORKSPACE_ID \
  --analytics-query "ContainerLog | where ContainerName contains 'oxirs-fuseki' | top 100 by TimeGenerated"
```

## Troubleshooting

### Common Issues

#### Issue: Cannot connect to cluster
```bash
# Re-authenticate
az login
az aks get-credentials --resource-group RG_NAME --name CLUSTER_NAME --overwrite-existing

# Check cluster status
az aks show --resource-group RG_NAME --name CLUSTER_NAME --output table
```

#### Issue: Pods not starting
```bash
# Check pod status
kubectl describe pod POD_NAME -n oxirs-fuseki

# View logs
kubectl logs POD_NAME -n oxirs-fuseki --previous

# Check events
kubectl get events -n oxirs-fuseki --sort-by='.lastTimestamp'
```

#### Issue: Storage mount failures
```bash
# Check Azure Files status
az storage share list --account-name STORAGE_ACCOUNT --output table

# Verify PV/PVC
kubectl get pv
kubectl get pvc -n oxirs-fuseki

# Check storage class
kubectl get storageclass
```

#### Issue: Workload Identity not working
```bash
# Verify federated identity credential
az identity federated-credential list \
  --resource-group RG_NAME \
  --identity-name IDENTITY_NAME

# Check service account annotation
kubectl get sa oxirs-fuseki -n oxirs-fuseki -o yaml
```

### Support

For issues and questions:
- **GitHub Issues**: https://github.com/cool-japan/oxirs/issues
- **Documentation**: https://github.com/cool-japan/oxirs/tree/main/docs
- **Azure Support**: https://portal.azure.com/#blade/Microsoft_Azure_Support/HelpAndSupportBlade

## Cleanup

To destroy all resources:

```bash
# Warning: This will delete all data!
terraform destroy

# Confirm when prompted

# Or delete resource group directly (faster)
az group delete --name oxirs-fuseki-prod-rg --yes --no-wait
```

## Advanced Configuration

### Private AKS Cluster

For maximum security, enable private cluster:

```hcl
private_cluster_enabled = true
```

This requires:
- VPN or ExpressRoute connection to Azure
- Jump box or bastion host for kubectl access
- Private DNS zone for API server

### Application Gateway Ingress

Enable Application Gateway for advanced ingress:

```hcl
enable_application_gateway = true
```

Benefits:
- Web Application Firewall (WAF)
- SSL/TLS termination
- URL-based routing
- Multi-site hosting

### Azure Policy Integration

Add Azure Policy for governance:

```hcl
resource "azurerm_policy_assignment" "aks_policy" {
  name                 = "aks-policy"
  scope                = azurerm_kubernetes_cluster.oxirs.id
  policy_definition_id = "/providers/Microsoft.Authorization/policyDefinitions/..."
}
```

### Azure Firewall Integration

Add Azure Firewall for egress control:

```hcl
resource "azurerm_firewall" "oxirs" {
  name                = "${var.cluster_name}-firewall"
  location            = azurerm_resource_group.oxirs.location
  resource_group_name = azurerm_resource_group.oxirs.name
  sku_name            = "AZFW_VNet"
  sku_tier            = "Standard"

  ip_configuration {
    name                 = "configuration"
    subnet_id            = azurerm_subnet.firewall.id
    public_ip_address_id = azurerm_public_ip.firewall.id
  }
}
```

## License

See the main OxiRS project for licensing information.
