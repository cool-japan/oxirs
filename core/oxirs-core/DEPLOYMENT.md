# OxiRS Core - Deployment Handbook

This handbook provides comprehensive guidance for deploying OxiRS Core in production environments, from single-server deployments to distributed clusters.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Single-Server Deployment](#single-server-deployment)
5. [Clustered Deployment](#clustered-deployment)
6. [Cloud Deployment](#cloud-deployment)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)
8. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **CPU**: 2 cores (4+ recommended)
- **RAM**: 4GB (8GB+ recommended)
- **Storage**: SSD with 10GB free space (minimum)
- **OS**: Linux (Ubuntu 20.04+, RHEL 8+), macOS 11+, Windows Server 2019+
- **Rust**: 1.75+ (for building from source)

### Recommended Requirements for Production

- **CPU**: 8+ cores (16+ for high-traffic deployments)
- **RAM**: 32GB+ (adjust based on dataset size)
- **Storage**: NVMe SSD with 100GB+ free space
- **Network**: 1Gbps+ network interface
- **OS**: Linux (Ubuntu 22.04 LTS recommended)

### Dataset Size Guidelines

| Dataset Size | RAM Requirement | Storage | Configuration |
|--------------|----------------|---------|---------------|
| < 1M triples | 2GB | 1GB | In-memory store |
| 1M - 10M triples | 8GB | 10GB | Memory-mapped store |
| 10M - 100M triples | 32GB | 100GB | Mmap + caching |
| 100M - 1B triples | 64GB+ | 500GB+ | Clustered deployment |
| > 1B triples | 128GB+ per node | 1TB+ | Multi-node cluster |

## Installation

### From Prebuilt Binaries

```bash
# Download latest release
curl -L https://github.com/cool-japan/oxirs/releases/download/v0.1.0-rc.2/oxirs-core-x86_64-linux.tar.gz -o oxirs-core.tar.gz

# Extract
tar -xzf oxirs-core.tar.gz

# Install
sudo mv oxirs-core /usr/local/bin/

# Verify installation
oxirs-core --version
```

### From Source

```bash
# Clone repository
git clone https://github.com/cool-japan/oxirs.git
cd oxirs/core/oxirs-core

# Build with optimizations
cargo build --release --all-features

# Install
sudo cp target/release/oxirs-core /usr/local/bin/

# Verify
oxirs-core --version
```

### Using Docker

```bash
# Pull official image
docker pull ghcr.io/cool-japan/oxirs-core:latest

# Run container
docker run -d \
  --name oxirs-core \
  -p 5000:5000 \
  -v /data/oxirs:/data \
  ghcr.io/cool-japan/oxirs-core:latest
```

## Configuration

### Configuration File

Create `/etc/oxirs/config.toml`:

```toml
[server]
host = "0.0.0.0"
port = 5000
workers = 8  # Number of worker threads

[storage]
type = "mmap"  # Options: memory, mmap, cluster
path = "/var/lib/oxirs/data"
max_size_gb = 100

[cache]
enabled = true
max_entries = 100000
ttl_seconds = 300

[transactions]
wal_enabled = true
wal_path = "/var/lib/oxirs/wal"
checkpoint_interval = 1000
isolation_level = "snapshot"  # Options: read_uncommitted, read_committed, repeatable_read, snapshot, serializable

[performance]
batch_size = 1000
enable_simd = true
parallel_queries = true
num_query_threads = 4

[monitoring]
metrics_enabled = true
metrics_port = 9090  # Prometheus metrics
health_check_enabled = true
health_check_port = 8080
profiling_sample_rate = 0.01  # Profile 1% of queries

[security]
max_query_time_seconds = 30
max_query_memory_mb = 100
max_results_per_query = 10000

[logging]
level = "info"  # Options: trace, debug, info, warn, error
format = "json"  # Options: json, text
output = "/var/log/oxirs/oxirs.log"
```

### Environment Variables

```bash
# Override config with environment variables
export OXIRS_HOST="0.0.0.0"
export OXIRS_PORT="5000"
export OXIRS_STORAGE_PATH="/data/oxirs"
export OXIRS_LOG_LEVEL="info"
export OXIRS_METRICS_ENABLED="true"
```

## Single-Server Deployment

### Systemd Service (Linux)

Create `/etc/systemd/system/oxirs-core.service`:

```ini
[Unit]
Description=OxiRS Core RDF/SPARQL Server
After=network.target

[Service]
Type=simple
User=oxirs
Group=oxirs
WorkingDirectory=/var/lib/oxirs
ExecStart=/usr/local/bin/oxirs-core --config /etc/oxirs/config.toml
Restart=always
RestartSec=10
StandardOutput=append:/var/log/oxirs/stdout.log
StandardError=append:/var/log/oxirs/stderr.log

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/oxirs /var/log/oxirs

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
```

**Setup and start:**

```bash
# Create user and directories
sudo useradd -r -s /bin/false oxirs
sudo mkdir -p /var/lib/oxirs /var/log/oxirs /etc/oxirs
sudo chown -R oxirs:oxirs /var/lib/oxirs /var/log/oxirs

# Install service
sudo systemctl daemon-reload
sudo systemctl enable oxirs-core
sudo systemctl start oxirs-core

# Check status
sudo systemctl status oxirs-core

# View logs
sudo journalctl -u oxirs-core -f
```

### Nginx Reverse Proxy

```nginx
upstream oxirs_backend {
    server 127.0.0.1:5000;
    keepalive 32;
}

server {
    listen 80;
    server_name sparql.example.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name sparql.example.com;

    ssl_certificate /etc/letsencrypt/live/sparql.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/sparql.example.com/privkey.pem;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # SPARQL endpoint
    location /sparql {
        proxy_pass http://oxirs_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;

        # Rate limiting
        limit_req zone=api burst=20 nodelay;
    }

    # Metrics endpoint (restrict access)
    location /metrics {
        proxy_pass http://127.0.0.1:9090/metrics;
        allow 10.0.0.0/8;  # Internal network only
        deny all;
    }

    # Health check
    location /health {
        proxy_pass http://127.0.0.1:8080/health;
        access_log off;
    }
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
```

## Clustered Deployment

### Architecture

```
┌─────────────┐
│ Load        │
│ Balancer    │
└─────┬───────┘
      │
   ┌──┴──┬──────┬──────┐
   │     │      │      │
┌──▼─┐ ┌─▼─┐ ┌──▼─┐ ┌──▼─┐
│Node│ │Node│ │Node│ │Node│
│ 1  │ │ 2  │ │ 3  │ │ 4  │
└────┘ └───┘ └────┘ └────┘
```

### Cluster Configuration

Create `/etc/oxirs/cluster.toml`:

```toml
[cluster]
node_id = "node1"
cluster_name = "oxirs-prod"

[cluster.discovery]
method = "static"  # Options: static, consul, etcd, kubernetes
nodes = [
    "node1.internal:5000",
    "node2.internal:5000",
    "node3.internal:5000",
]

[cluster.replication]
factor = 3  # Number of replicas per data partition
consistency = "quorum"  # Options: one, quorum, all

[cluster.partitioning]
strategy = "hash"  # Options: hash, range, custom
partitions = 16

[cluster.consensus]
algorithm = "raft"
election_timeout_ms = 1000
heartbeat_interval_ms = 100
```

### Node Deployment

On each node:

```bash
# Node 1
sudo OXIRS_CLUSTER_NODE_ID=node1 \
     OXIRS_CLUSTER_NODES="node1.internal:5000,node2.internal:5000,node3.internal:5000" \
     systemctl start oxirs-core

# Node 2
sudo OXIRS_CLUSTER_NODE_ID=node2 \
     OXIRS_CLUSTER_NODES="node1.internal:5000,node2.internal:5000,node3.internal:5000" \
     systemctl start oxirs-core

# Node 3
sudo OXIRS_CLUSTER_NODE_ID=node3 \
     OXIRS_CLUSTER_NODES="node1.internal:5000,node2.internal:5000,node3.internal:5000" \
     systemctl start oxirs-core
```

### Load Balancer (HAProxy)

```haproxy
global
    maxconn 50000
    log /dev/log local0
    user haproxy
    group haproxy
    daemon

defaults
    mode http
    log global
    option httplog
    option dontlognull
    timeout connect 5000ms
    timeout client 300000ms
    timeout server 300000ms

frontend sparql_frontend
    bind *:80
    default_backend sparql_backend

backend sparql_backend
    balance leastconn
    option httpchk GET /health
    http-check expect status 200

    server node1 node1.internal:5000 check inter 2000ms rise 2 fall 3
    server node2 node2.internal:5000 check inter 2000ms rise 2 fall 3
    server node3 node3.internal:5000 check inter 2000ms rise 2 fall 3

listen stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 30s
```

## Cloud Deployment

### AWS Deployment

#### Using EC2

```bash
# Launch EC2 instance
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \  # Ubuntu 22.04 LTS
  --instance-type c5.2xlarge \
  --key-name your-key \
  --security-group-ids sg-xxxxxxxx \
  --subnet-id subnet-xxxxxxxx \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --user-data file://install-oxirs.sh
```

**install-oxirs.sh:**

```bash
#!/bin/bash
set -e

# Update system
apt-get update && apt-get upgrade -y

# Install dependencies
apt-get install -y curl build-essential

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Install OxiRS Core
cd /opt
git clone https://github.com/cool-japan/oxirs.git
cd oxirs/core/oxirs-core
cargo build --release --all-features

# Setup systemd service
cp target/release/oxirs-core /usr/local/bin/
# ... (copy service file and configuration)

# Start service
systemctl enable oxirs-core
systemctl start oxirs-core
```

#### Using ECS/Fargate

**task-definition.json:**

```json
{
  "family": "oxirs-core",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "4096",
  "memory": "16384",
  "containerDefinitions": [
    {
      "name": "oxirs-core",
      "image": "ghcr.io/cool-japan/oxirs-core:latest",
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "OXIRS_STORAGE_TYPE",
          "value": "mmap"
        },
        {
          "name": "OXIRS_LOG_LEVEL",
          "value": "info"
        }
      ],
      "mountPoints": [
        {
          "sourceVolume": "oxirs-data",
          "containerPath": "/data"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/oxirs-core",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ],
  "volumes": [
    {
      "name": "oxirs-data",
      "efsVolumeConfiguration": {
        "fileSystemId": "fs-xxxxxxxx",
        "transitEncryption": "ENABLED"
      }
    }
  ]
}
```

### Google Cloud Platform

#### Using GKE

**deployment.yaml:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: oxirs-core
  labels:
    app: oxirs-core
spec:
  replicas: 3
  selector:
    matchLabels:
      app: oxirs-core
  template:
    metadata:
      labels:
        app: oxirs-core
    spec:
      containers:
      - name: oxirs-core
        image: ghcr.io/cool-japan/oxirs-core:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "8Gi"
            cpu: "2000m"
          limits:
            memory: "16Gi"
            cpu: "4000m"
        volumeMounts:
        - name: oxirs-data
          mountPath: /data
        env:
        - name: OXIRS_STORAGE_PATH
          value: "/data"
        - name: OXIRS_LOG_LEVEL
          value: "info"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
      volumes:
      - name: oxirs-data
        persistentVolumeClaim:
          claimName: oxirs-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: oxirs-core-service
spec:
  type: LoadBalancer
  selector:
    app: oxirs-core
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
```

## Monitoring and Maintenance

### Prometheus Metrics

Configure Prometheus to scrape OxiRS metrics:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'oxirs'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Grafana Dashboard

Import the OxiRS dashboard (ID: oxirs-core-dashboard) or create custom panels:

**Key Metrics to Monitor:**

- Query rate (queries/second)
- Query latency (p50, p95, p99)
- Cache hit rate
- Storage size
- Memory usage
- CPU usage
- Transaction throughput
- Error rate

### Log Aggregation

#### Using ELK Stack

```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/oxirs/*.log
  json.keys_under_root: true
  json.add_error_key: true

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "oxirs-%{+yyyy.MM.dd}"

setup.kibana:
  host: "kibana:5601"
```

### Backup Strategy

```bash
#!/bin/bash
# /usr/local/bin/backup-oxirs.sh

BACKUP_DIR="/backup/oxirs"
DATE=$(date +%Y%m%d_%H%M%S)
FULL_BACKUP_DAY=0  # Sunday

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Determine backup type
if [ $(date +%u) -eq $FULL_BACKUP_DAY ]; then
  # Full backup on Sunday
  oxirs-cli backup --type full --output "$BACKUP_DIR/full_$DATE.db"
else
  # Incremental backup on other days
  oxirs-cli backup --type incremental --output "$BACKUP_DIR/incr_$DATE.db"
fi

# Compress
gzip "$BACKUP_DIR"/*.db

# Upload to S3
aws s3 sync "$BACKUP_DIR" s3://your-bucket/oxirs-backups/

# Clean old backups (keep 30 days)
find "$BACKUP_DIR" -name "*.gz" -mtime +30 -delete
```

**Schedule with cron:**

```bash
# Daily backup at 2 AM
0 2 * * * /usr/local/bin/backup-oxirs.sh
```

## Troubleshooting

### High Memory Usage

```bash
# Check current memory usage
free -h

# Check OxiRS memory usage
ps aux | grep oxirs-core

# Solutions:
# 1. Reduce cache size in config
# 2. Enable compaction
oxirs-cli compact
# 3. Switch to memory-mapped storage
# 4. Add more RAM or use clustering
```

### Slow Queries

```bash
# Enable query profiling
oxirs-cli profiling enable

# Identify slow queries
oxirs-cli profiling slow-queries --threshold 1000  # >1s

# Solutions:
# 1. Add indexes
# 2. Optimize query patterns
# 3. Enable query result caching
# 4. Increase query thread pool
```

### Disk Space Issues

```bash
# Check disk usage
df -h /var/lib/oxirs

# Clean WAL logs
oxirs-cli wal clean --keep-days 7

# Compact database
oxirs-cli compact

# Check for deleted triples
oxirs-cli stats --deleted-ratio
```

### Connection Issues

```bash
# Check if service is running
systemctl status oxirs-core

# Check listening ports
netstat -tlnp | grep 5000

# Check firewall
sudo ufw status
sudo ufw allow 5000/tcp

# Test connection
curl http://localhost:5000/health
```

### Performance Degradation

```bash
# Check system resources
top
iostat -x 5

# Check OxiRS statistics
oxirs-cli stats

# Analyze query patterns
oxirs-cli analyze-queries

# Solutions:
# 1. Increase cache size
# 2. Add query result caching
# 3. Optimize indexes
# 4. Scale horizontally
```

---

## Additional Resources

- [Tutorial](TUTORIAL.md) - Getting started guide
- [Best Practices](BEST_PRACTICES.md) - Production best practices
- [Architecture](ARCHITECTURE.md) - System architecture details
- [Performance Guide](PERFORMANCE_GUIDE.md) - Optimization strategies

For support, visit [GitHub Issues](https://github.com/cool-japan/oxirs/issues) or join our community discussions.
