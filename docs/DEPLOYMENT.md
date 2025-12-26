# OxiRS Deployment Guide

**Version**: 0.1.0-rc.1
**Date**: December 21, 2025
**Status**: Production Deployment Ready

## 🚀 Quick Start

### Prerequisites
- **Rust**: 1.90+ (MSRV)
- **Docker**: 20.10+ (for containerized deployment)
- **Kubernetes**: 1.25+ (for orchestrated deployment)
- **Memory**: 4GB minimum, 16GB recommended
- **CPU**: 4 cores minimum, 8+ recommended
- **Disk**: 10GB minimum, 100GB+ for production

---

## 📦 Installation Methods

### Method 1: Binary Installation (Recommended)

```bash
# Download latest release
curl -LO https://github.com/cool-japan/oxirs/releases/download/v0.1.0-rc.1/oxirs-x86_64-linux.tar.gz

# Extract
tar xzf oxirs-x86_64-linux.tar.gz

# Install
sudo mv oxirs /usr/local/bin/
sudo mv oxirs-fuseki /usr/local/bin/

# Verify
oxirs --version
oxirs-fuseki --version
```

### Method 2: Build from Source

```bash
# Clone repository
git clone https://github.com/cool-japan/oxirs.git
cd oxirs

# Build release binaries
cargo build --release --workspace

# Binaries located at:
# - target/release/oxirs (CLI)
# - target/release/oxirs-fuseki (Server)

# Install (optional)
cargo install --path tools/oxirs
cargo install --path server/oxirs-fuseki
```

### Method 3: Docker (Production)

```bash
# Pull latest image
docker pull ghcr.io/cool-japan/oxirs-fuseki:v0.1.0-rc.1

# Run with default configuration
docker run -p 3030:3030 ghcr.io/cool-japan/oxirs-fuseki:v0.1.0-rc.1

# Run with custom configuration
docker run -p 3030:3030 \
  -v $(pwd)/oxirs.toml:/etc/oxirs/oxirs.toml:ro \
  -v oxirs-data:/var/lib/oxirs \
  ghcr.io/cool-japan/oxirs-fuseki:v0.1.0-rc.1
```

### Method 4: Kubernetes (Cloud)

```bash
# Apply Kubernetes manifests
kubectl apply -f deployments/kubernetes/

# Verify deployment
kubectl get pods -n oxirs
kubectl get services -n oxirs
```

---

## 🐳 Docker Deployment

### Dockerfile

```dockerfile
# /Users/kitasan/work/oxirs/Dockerfile

FROM rust:1.90-slim as builder

WORKDIR /build

# Copy source
COPY . .

# Build release
RUN cargo build --release --bin oxirs-fuseki

# Runtime image
FROM debian:bookworm-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create oxirs user
RUN useradd -m -u 1000 oxirs

# Copy binary
COPY --from=builder /build/target/release/oxirs-fuseki /usr/local/bin/

# Create data directory
RUN mkdir -p /var/lib/oxirs && chown oxirs:oxirs /var/lib/oxirs

# Switch to oxirs user
USER oxirs

# Expose ports
EXPOSE 3030

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3030/$/ping || exit 1

# Run server
CMD ["oxirs-fuseki", "--config", "/etc/oxirs/oxirs.toml"]
```

### docker-compose.yml

```yaml
# /Users/kitasan/work/oxirs/docker-compose.yml

version: '3.8'

services:
  oxirs-fuseki:
    image: ghcr.io/cool-japan/oxirs-fuseki:v0.1.0-rc.1
    container_name: oxirs-fuseki
    ports:
      - "3030:3030"
    volumes:
      - ./oxirs.toml:/etc/oxirs/oxirs.toml:ro
      - oxirs-data:/var/lib/oxirs
    environment:
      - RUST_LOG=info
      - OXIRS_HOST=0.0.0.0
      - OXIRS_PORT=3030
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3030/$/ping"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 10s

  prometheus:
    image: prom/prometheus:latest
    container_name: oxirs-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./monitoring/prometheus/alert-rules.yml:/etc/prometheus/alert-rules.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: oxirs-grafana
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/grafana:/etc/grafana/provisioning:ro
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    restart: unless-stopped
    depends_on:
      - prometheus

volumes:
  oxirs-data:
  prometheus-data:
  grafana-data:
```

### Docker Commands

```bash
# Build custom image
docker build -t oxirs-fuseki:local .

# Run with docker-compose (full stack)
docker-compose up -d

# View logs
docker-compose logs -f oxirs-fuseki

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

---

## ☸️ Kubernetes Deployment

### Namespace

```yaml
# deployments/kubernetes/namespace.yaml

apiVersion: v1
kind: Namespace
metadata:
  name: oxirs
  labels:
    name: oxirs
    environment: production
```

### ConfigMap

```yaml
# deployments/kubernetes/configmap.yaml

apiVersion: v1
kind: ConfigMap
metadata:
  name: oxirs-config
  namespace: oxirs
data:
  oxirs.toml: |
    [general]
    default_format = "turtle"
    timeout = 300
    log_level = "info"

    [server]
    host = "0.0.0.0"
    port = 3030
    enable_cors = true
    enable_admin_ui = true

    [server.auth]
    enabled = true
    method = "jwt"

    [[datasets]]
    name = "default"
    type = "tdb2"
    location = "/var/lib/oxirs/datasets/default"
```

### Secret

```yaml
# deployments/kubernetes/secret.yaml

apiVersion: v1
kind: Secret
metadata:
  name: oxirs-secret
  namespace: oxirs
type: Opaque
data:
  jwt-secret: <base64-encoded-jwt-secret>
  oauth-client-id: <base64-encoded-client-id>
  oauth-client-secret: <base64-encoded-client-secret>
```

### PersistentVolumeClaim

```yaml
# deployments/kubernetes/pvc.yaml

apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: oxirs-data
  namespace: oxirs
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
```

### Deployment

```yaml
# deployments/kubernetes/deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: oxirs-fuseki
  namespace: oxirs
  labels:
    app: oxirs-fuseki
spec:
  replicas: 3
  selector:
    matchLabels:
      app: oxirs-fuseki
  template:
    metadata:
      labels:
        app: oxirs-fuseki
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "3030"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: oxirs-fuseki
        image: ghcr.io/cool-japan/oxirs-fuseki:v0.1.0-rc.1
        imagePullPolicy: IfNotPresent
        ports:
        - name: http
          containerPort: 3030
          protocol: TCP
        env:
        - name: RUST_LOG
          value: "info"
        - name: OXIRS_CONFIG
          value: "/etc/oxirs/oxirs.toml"
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: oxirs-secret
              key: jwt-secret
        volumeMounts:
        - name: config
          mountPath: /etc/oxirs
          readOnly: true
        - name: data
          mountPath: /var/lib/oxirs
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
          limits:
            memory: "16Gi"
            cpu: "8000m"
        livenessProbe:
          httpGet:
            path: /$/ping
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 3
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /$/ping
            port: http
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
      volumes:
      - name: config
        configMap:
          name: oxirs-config
      - name: data
        persistentVolumeClaim:
          claimName: oxirs-data
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - oxirs-fuseki
              topologyKey: kubernetes.io/hostname
```

### Service

```yaml
# deployments/kubernetes/service.yaml

apiVersion: v1
kind: Service
metadata:
  name: oxirs-fuseki
  namespace: oxirs
  labels:
    app: oxirs-fuseki
spec:
  type: ClusterIP
  ports:
  - port: 3030
    targetPort: http
    protocol: TCP
    name: http
  selector:
    app: oxirs-fuseki
```

### Ingress

```yaml
# deployments/kubernetes/ingress.yaml

apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: oxirs-fuseki
  namespace: oxirs
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - sparql.example.com
    secretName: oxirs-tls
  rules:
  - host: sparql.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: oxirs-fuseki
            port:
              number: 3030
```

### HorizontalPodAutoscaler

```yaml
# deployments/kubernetes/hpa.yaml

apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: oxirs-fuseki
  namespace: oxirs
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: oxirs-fuseki
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 2
        periodSeconds: 30
      selectPolicy: Max
```

### Apply Kubernetes Manifests

```bash
# Apply all manifests
kubectl apply -f deployments/kubernetes/namespace.yaml
kubectl apply -f deployments/kubernetes/configmap.yaml
kubectl apply -f deployments/kubernetes/secret.yaml
kubectl apply -f deployments/kubernetes/pvc.yaml
kubectl apply -f deployments/kubernetes/deployment.yaml
kubectl apply -f deployments/kubernetes/service.yaml
kubectl apply -f deployments/kubernetes/ingress.yaml
kubectl apply -f deployments/kubernetes/hpa.yaml

# Or apply all at once
kubectl apply -f deployments/kubernetes/

# Verify deployment
kubectl get all -n oxirs
kubectl describe deployment oxirs-fuseki -n oxirs
kubectl logs -f deployment/oxirs-fuseki -n oxirs
```

---

## 🔧 Configuration

### Production Configuration

```toml
# /etc/oxirs/oxirs.toml

[general]
default_format = "turtle"
timeout = 300
log_level = "info"
max_query_results = 10000

[server]
host = "0.0.0.0"
port = 3030
enable_cors = true
enable_admin_ui = false  # Disable in production
graphql_path = "/graphql"
sparql_query_path = "/query"
sparql_update_path = "/update"

[server.tls]
enabled = true
cert_path = "/etc/ssl/certs/server.crt"
key_path = "/etc/ssl/private/server.key"

[server.auth]
enabled = true
method = "jwt"
jwt_secret_env = "JWT_SECRET"
token_expiry = 3600

[server.cors]
enabled = true
allowed_origins = ["https://app.example.com"]
allowed_methods = ["GET", "POST", "OPTIONS"]
allowed_headers = ["Content-Type", "Authorization"]
max_age = 3600

[server.rate_limit]
enabled = true
requests_per_second = 100
burst_size = 200

[[datasets]]
name = "production"
type = "tdb2"
location = "/var/lib/oxirs/datasets/production"
read_only = false

[datasets.production.options]
cache_size = 10000
buffer_pool_size = 1000
enable_transactions = true
sync_on_commit = true

[monitoring]
enabled = true
prometheus_port = 9090
health_check_interval = 30

[logging]
format = "json"
output = "stdout"
level = "info"
include_timestamps = true
include_location = false
```

---

## 🔒 Security Hardening

### TLS Configuration

```bash
# Generate self-signed certificate (development only)
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout server.key \
  -out server.crt \
  -days 365 \
  -subj "/CN=sparql.example.com"

# Production: Use Let's Encrypt with cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

### JWT Secret Generation

```bash
# Generate secure JWT secret
openssl rand -base64 32

# Set as Kubernetes secret
kubectl create secret generic oxirs-secret \
  --from-literal=jwt-secret=$(openssl rand -base64 32) \
  -n oxirs
```

### Firewall Rules

```bash
# Allow only necessary ports
sudo ufw allow 3030/tcp  # SPARQL endpoint
sudo ufw allow 9090/tcp  # Prometheus metrics
sudo ufw enable
```

### Network Policies (Kubernetes)

```yaml
# deployments/kubernetes/networkpolicy.yaml

apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: oxirs-fuseki
  namespace: oxirs
spec:
  podSelector:
    matchLabels:
      app: oxirs-fuseki
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 3030
  egress:
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 53  # DNS
  - to:
    - podSelector:
        matchLabels:
          app: postgres  # If using external DB
    ports:
    - protocol: TCP
      port: 5432
```

---

## 📊 Monitoring Setup

### Prometheus Configuration

```yaml
# monitoring/prometheus/prometheus.yml

global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
  - static_configs:
    - targets:
      - alertmanager:9093

rule_files:
  - /etc/prometheus/alert-rules.yml

scrape_configs:
  - job_name: 'oxirs-fuseki'
    static_configs:
    - targets: ['oxirs-fuseki:3030']
    metrics_path: /metrics
```

### Grafana Dashboard Import

```bash
# Access Grafana
kubectl port-forward svc/grafana 3000:3000 -n monitoring

# Navigate to http://localhost:3000
# Login: admin / <password-from-secret>
# Import dashboard: monitoring/grafana/oxirs-dashboard.json
```

---

## 🔄 Backup & Recovery

### Backup Script

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/oxirs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATA_DIR="/var/lib/oxirs/datasets"

# Create backup
tar czf "${BACKUP_DIR}/oxirs-backup-${TIMESTAMP}.tar.gz" "${DATA_DIR}"

# Verify backup
tar tzf "${BACKUP_DIR}/oxirs-backup-${TIMESTAMP}.tar.gz" > /dev/null

# Cleanup old backups (keep last 7 days)
find "${BACKUP_DIR}" -name "oxirs-backup-*.tar.gz" -mtime +7 -delete

echo "Backup completed: oxirs-backup-${TIMESTAMP}.tar.gz"
```

### Restore Script

```bash
#!/bin/bash
# restore.sh

BACKUP_FILE=$1
DATA_DIR="/var/lib/oxirs/datasets"

if [ -z "$BACKUP_FILE" ]; then
  echo "Usage: $0 <backup-file>"
  exit 1
fi

# Stop OxiRS
systemctl stop oxirs-fuseki

# Restore backup
rm -rf "${DATA_DIR}"
tar xzf "${BACKUP_FILE}" -C /

# Start OxiRS
systemctl start oxirs-fuseki

echo "Restore completed from: ${BACKUP_FILE}"
```

### Kubernetes Backup (Using Velero)

```bash
# Install Velero
velero install \
  --provider aws \
  --bucket oxirs-backups \
  --secret-file ./credentials-velero

# Create backup
velero backup create oxirs-backup --include-namespaces oxirs

# Restore backup
velero restore create --from-backup oxirs-backup
```

---

## 🎯 Performance Tuning

### System Limits

```bash
# /etc/security/limits.conf

oxirs soft nofile 65536
oxirs hard nofile 65536
oxirs soft nproc 32768
oxirs hard nproc 32768
```

### Kernel Parameters

```bash
# /etc/sysctl.conf

# TCP tuning
net.core.somaxconn = 32768
net.ipv4.tcp_max_syn_backlog = 8192
net.ipv4.tcp_tw_reuse = 1

# Memory
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
```

### JVM-style Tuning (Rust equivalent)

```bash
# Set environment variables
export RUST_BACKTRACE=1
export RUST_LOG=info
export MALLOC_ARENA_MAX=2  # Limit memory allocator arenas
```

---

## 🚦 Health Checks

### Endpoint Health Check

```bash
# Check if server is running
curl http://localhost:3030/$/ping

# Expected response: 200 OK
```

### Readiness Check

```bash
# Check if server is ready to accept traffic
curl http://localhost:3030/$/ready

# Expected response: 200 OK with status JSON
```

### Liveness Check

```bash
# Check if server process is alive
curl http://localhost:3030/$/alive

# Expected response: 200 OK
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Find process using port 3030
lsof -i :3030

# Kill process
kill -9 <PID>
```

#### 2. Permission Denied
```bash
# Fix data directory permissions
sudo chown -R oxirs:oxirs /var/lib/oxirs
sudo chmod -R 755 /var/lib/oxirs
```

#### 3. Out of Memory
```bash
# Check memory usage
free -h

# Increase swap
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 4. Kubernetes Pod CrashLoopBackOff
```bash
# Check pod logs
kubectl logs -f <pod-name> -n oxirs

# Describe pod for events
kubectl describe pod <pod-name> -n oxirs

# Check resource limits
kubectl top pod <pod-name> -n oxirs
```

### Logs

```bash
# Docker logs
docker logs oxirs-fuseki --tail 100 -f

# Kubernetes logs
kubectl logs -f deployment/oxirs-fuseki -n oxirs

# Systemd logs
journalctl -u oxirs-fuseki -f
```

---

## 📋 Checklist

### Pre-Deployment
- [ ] Configuration file created and validated
- [ ] TLS certificates obtained and installed
- [ ] JWT secret generated securely
- [ ] Firewall rules configured
- [ ] Backup strategy implemented
- [ ] Monitoring configured (Prometheus + Grafana)
- [ ] Alert rules tested

### Deployment
- [ ] Application deployed successfully
- [ ] Health checks passing
- [ ] Metrics being collected
- [ ] Logs being aggregated
- [ ] Backups running automatically

### Post-Deployment
- [ ] Load testing completed
- [ ] Security scan performed
- [ ] Documentation updated
- [ ] Team trained on operations
- [ ] Incident response plan documented

---

## 🔗 Additional Resources

- **GitHub**: https://github.com/cool-japan/oxirs
- **Documentation**: https://docs.oxirs.io
- **Discord**: https://discord.gg/oxirs
- **Security**: security@oxirs.io

---

*Deployment Guide - October 12, 2025*
*Production-ready deployment for v0.1.0-rc.1*
