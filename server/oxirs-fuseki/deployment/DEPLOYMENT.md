# OxiRS Fuseki - Deployment Guide

This guide covers deploying OxiRS Fuseki using Docker Compose and Kubernetes.

## Table of Contents

- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Configuration](#configuration)
- [Monitoring and Observability](#monitoring-and-observability)
- [Security](#security)
- [Troubleshooting](#troubleshooting)

## Docker Deployment

### Prerequisites

- Docker 24.0+
- Docker Compose 2.20+
- 4GB+ RAM available
- 20GB+ disk space

### Quick Start

1. **Production Deployment**

```bash
cd deployment/docker

# Start the full stack
docker-compose up -d

# View logs
docker-compose logs -f oxirs-fuseki

# Check health
curl http://localhost:3030/health
```

2. **Development Deployment**

```bash
cd deployment/docker

# Start minimal development stack
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f
```

### Services

The production stack includes:

- **OxiRS Fuseki** (`:3030`) - SPARQL server
- **Prometheus** (`:9091`) - Metrics collection
- **Grafana** (`:3000`) - Visualization dashboards
- **Redis** (`:6379`) - Session storage
- **NGINX** (`:80`, `:443`) - Reverse proxy
- **OpenTelemetry Collector** (`:4317`, `:4318`) - Tracing
- **Jaeger** (`:16686`) - Trace visualization

### Configuration

Create `deployment/docker/config/oxirs.toml`:

```toml
[server]
host = "0.0.0.0"
port = 3030
admin_ui = true

[security]
auth_required = true

[datasets.default]
name = "default"
type = "memory"
persistent = true
```

### Scaling

```bash
# Scale to 3 replicas
docker-compose up -d --scale oxirs-fuseki=3

# Behind NGINX load balancer
docker-compose up -d
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes 1.28+
- kubectl configured
- 8GB+ RAM per node
- Persistent volume provisioner
- (Optional) cert-manager for TLS
- (Optional) Prometheus Operator

### Quick Start

1. **Create Namespace**

```bash
kubectl apply -f deployment/kubernetes/namespace.yaml
```

2. **Deploy Infrastructure**

```bash
# RBAC
kubectl apply -f deployment/kubernetes/rbac.yaml

# ConfigMap
kubectl apply -f deployment/kubernetes/configmap.yaml

# Secrets (create manually or use cert-manager)
kubectl create secret tls oxirs-fuseki-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key \
  -n oxirs

# Storage
kubectl apply -f deployment/kubernetes/persistentvolume.yaml
```

3. **Deploy Application**

```bash
# Deployment
kubectl apply -f deployment/kubernetes/deployment.yaml

# Services
kubectl apply -f deployment/kubernetes/service.yaml

# Ingress
kubectl apply -f deployment/kubernetes/ingress.yaml

# HPA (optional)
kubectl apply -f deployment/kubernetes/hpa.yaml

# ServiceMonitor (if using Prometheus Operator)
kubectl apply -f deployment/kubernetes/servicemonitor.yaml
```

4. **Verify Deployment**

```bash
# Check pods
kubectl get pods -n oxirs

# Check services
kubectl get svc -n oxirs

# Check logs
kubectl logs -f -n oxirs -l app=oxirs-fuseki

# Port-forward for testing
kubectl port-forward -n oxirs svc/oxirs-fuseki 3030:3030
```

### Architecture

```
Internet
    ↓
Ingress (TLS termination)
    ↓
Service (LoadBalancer)
    ↓
Pods (3+ replicas)
    ↓
PersistentVolume (RWO/RWX)
```

### Scaling

The deployment includes Horizontal Pod Autoscaler (HPA):

```bash
# View HPA status
kubectl get hpa -n oxirs

# Manual scaling
kubectl scale deployment oxirs-fuseki --replicas=5 -n oxirs
```

**HPA Triggers:**
- CPU > 70%
- Memory > 80%
- SPARQL queries/sec > 100

### High Availability

For production HA setup:

1. **Multiple Replicas**: Deploy 3+ replicas across zones
2. **Pod Anti-Affinity**: Configured in deployment.yaml
3. **Topology Spread**: Distributes pods across zones
4. **Health Checks**: Liveness, readiness, startup probes
5. **Rolling Updates**: Zero-downtime deployments

### Storage

**Single Node:**
```yaml
accessModes: [ReadWriteOnce]
storageClassName: fast-ssd
```

**Multi-Node (Shared):**
```yaml
accessModes: [ReadWriteMany]
storageClassName: shared-nfs
```

## Configuration

### Environment Variables

```bash
RUST_LOG=info                    # Log level
OXIRS_CONFIG=/etc/oxirs/oxirs.toml  # Config file
OXIRS_DATA=/data/oxirs           # Data directory
OXIRS_LOG_DIR=/var/log/oxirs     # Log directory
```

### ConfigMap Updates

```bash
# Edit ConfigMap
kubectl edit configmap oxirs-fuseki-config -n oxirs

# Restart pods to apply changes
kubectl rollout restart deployment oxirs-fuseki -n oxirs
```

### Secrets Management

**Using kubectl:**
```bash
kubectl create secret generic oxirs-secrets \
  --from-literal=jwt-secret=YOUR_SECRET \
  --from-literal=admin-password=ADMIN_PASS \
  -n oxirs
```

**Using External Secrets Operator:**
```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: oxirs-fuseki-secrets
spec:
  secretStoreRef:
    name: aws-secrets-manager
  target:
    name: oxirs-secrets
  data:
  - secretKey: jwt-secret
    remoteRef:
      key: oxirs/jwt-secret
```

## Monitoring and Observability

### Metrics

**Prometheus:**
- Endpoint: http://oxirs-fuseki:9090/metrics
- Scrape interval: 15s

**Key Metrics:**
- `sparql_query_duration_seconds`
- `sparql_queries_total`
- `http_requests_total`
- `cache_hit_ratio`
- `active_connections`

### Grafana Dashboards

Access Grafana at http://localhost:3000 (Docker) or via Ingress (K8s).

**Default credentials:** admin / oxirs123

**Pre-configured dashboards:**
- OxiRS Fuseki Overview
- SPARQL Query Performance
- System Resources
- Cache Statistics

### Tracing

**Jaeger UI:** http://localhost:16686

**OpenTelemetry:** Sends traces to Jaeger via OTLP.

### Logs

**Docker:**
```bash
docker-compose logs -f oxirs-fuseki
```

**Kubernetes:**
```bash
kubectl logs -f -n oxirs -l app=oxirs-fuseki

# Stern (multi-pod logs)
stern -n oxirs oxirs-fuseki
```

## Security

### TLS/SSL

**Docker:** Configure in `config/oxirs.toml`:
```toml
[server.tls]
cert_path = "/etc/oxirs/tls/tls.crt"
key_path = "/etc/oxirs/tls/tls.key"
```

**Kubernetes:** Use cert-manager:
```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### Authentication

**JWT:** Configure in ConfigMap:
```toml
[security.jwt]
secret = "${JWT_SECRET}"
issuer = "oxirs-fuseki"
expiration_secs = 3600
```

**OAuth2:** Enable in configuration:
```toml
[security.oauth]
enabled = true
client_id = "YOUR_CLIENT_ID"
```

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: oxirs-fuseki
spec:
  podSelector:
    matchLabels:
      app: oxirs-fuseki
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - port: 3030
```

## Troubleshooting

### Pod Not Starting

```bash
# Check events
kubectl describe pod <pod-name> -n oxirs

# Check logs
kubectl logs <pod-name> -n oxirs --previous

# Check resource limits
kubectl top pod <pod-name> -n oxirs
```

### Performance Issues

```bash
# Check HPA
kubectl get hpa -n oxirs

# Check metrics
kubectl top nodes
kubectl top pods -n oxirs

# Review slow queries
kubectl logs -n oxirs -l app=oxirs-fuseki | grep "slow query"
```

### Storage Issues

```bash
# Check PVC
kubectl get pvc -n oxirs
kubectl describe pvc oxirs-fuseki-data -n oxirs

# Check disk usage
kubectl exec -it <pod-name> -n oxirs -- df -h
```

### Connection Issues

```bash
# Test internal connectivity
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -n oxirs \
  -- curl http://oxirs-fuseki:3030/health

# Test from outside
kubectl port-forward -n oxirs svc/oxirs-fuseki 3030:3030
curl http://localhost:3030/health
```

## Backup and Recovery

### Manual Backup

```bash
# Docker
docker-compose exec oxirs-fuseki tar czf /tmp/backup.tar.gz /data/oxirs
docker cp <container-id>:/tmp/backup.tar.gz ./backup-$(date +%Y%m%d).tar.gz

# Kubernetes
kubectl exec -n oxirs <pod-name> -- tar czf /tmp/backup.tar.gz /data/oxirs
kubectl cp oxirs/<pod-name>:/tmp/backup.tar.gz ./backup-$(date +%Y%m%d).tar.gz
```

### Automated Backup (Kubernetes)

Use Velero for automated backups:

```bash
velero backup create oxirs-backup --include-namespaces oxirs
velero restore create --from-backup oxirs-backup
```

## Performance Tuning

### Resource Limits

Adjust based on workload:

```yaml
resources:
  requests:
    cpu: "1000m"
    memory: "2Gi"
  limits:
    cpu: "4000m"
    memory: "8Gi"
```

### Connection Pool

```toml
[performance.connection_pool]
max_connections = 200
min_connections = 20
```

### Caching

```toml
[performance.caching]
enabled = true
max_size = 50000
ttl_secs = 1200
```

## Support

For issues and questions:
- GitHub: https://github.com/cool-japan/oxirs
- Documentation: https://docs.oxirs.org
- Issues: https://github.com/cool-japan/oxirs/issues
