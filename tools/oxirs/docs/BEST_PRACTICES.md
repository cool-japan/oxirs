# OxiRS Best Practices Guide

**Version**: 0.1.0-beta.2
**Last Updated**: December 21, 2025
**Status**: Production-Ready

## Overview

This guide provides best practices, tips, and recommendations for using OxiRS CLI effectively in development and production environments.

## Table of Contents

- [General Guidelines](#general-guidelines)
- [Dataset Management](#dataset-management)
- [Query Optimization](#query-optimization)
- [Performance Tuning](#performance-tuning)
- [Security Best Practices](#security-best-practices)
- [Production Deployment](#production-deployment)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Troubleshooting](#troubleshooting)
- [Common Patterns](#common-patterns)

---

## General Guidelines

### 1. Follow Naming Conventions

✅ **DO**:
```bash
# Use alphanumeric, underscore, hyphen
oxirs init my_dataset
oxirs init my-dataset
oxirs init myDataset2025
```

❌ **DON'T**:
```bash
# Avoid dots, spaces, special characters
oxirs init my.dataset      # Error: dots not allowed
oxirs init my dataset      # Error: spaces not allowed
oxirs init my@dataset      # Error: special chars not allowed
```

### 2. Use Descriptive Dataset Names

✅ **DO**:
```bash
oxirs init product_catalog_2025
oxirs init user_profiles_prod
oxirs init geo_spatial_dev
```

❌ **DON'T**:
```bash
oxirs init data
oxirs init db1
oxirs init temp
```

### 3. Organize by Environment

```bash
# Development
oxirs init myapp_dev

# Staging
oxirs init myapp_staging

# Production
oxirs init myapp_prod

# Testing
oxirs init myapp_test
```

---

## Dataset Management

### Initialize Datasets Properly

```bash
# Development: Use memory for fast iteration
oxirs init dev_dataset --format memory

# Production: Use TDB2 for persistence
oxirs init prod_dataset --format tdb2 --location /var/lib/oxirs/prod
```

### Regular Maintenance

```bash
# Compact dataset monthly to reclaim space
oxirs tdbcompact ./data/mydata --delete-old

# Check statistics weekly
oxirs tdbstats ./data/mydata --detailed

# Backup before major operations
oxirs tdbbackup ./data/mydata ./backups/mydata-$(date +%Y%m%d)
```

### Dataset Lifecycle

```bash
# 1. Initialize
oxirs init myapp

# 2. Import initial data
oxirs import myapp schema.ttl

# 3. Bulk load data
oxirs tdbloader ./data/myapp data/*.nt --progress --stats

# 4. Regular updates via SPARQL UPDATE
oxirs update myapp "INSERT DATA { ... }"

# 5. Periodic optimization
oxirs tdbcompact ./data/myapp

# 6. Regular backups
oxirs tdbbackup ./data/myapp ./backups/myapp-weekly --compress
```

---

## Query Optimization

### 1. Use LIMIT for Exploration

✅ **DO**:
```sparql
# Always use LIMIT for exploratory queries
SELECT * WHERE { ?s ?p ?o } LIMIT 100

# Pagination
SELECT * WHERE { ?s ?p ?o }
ORDER BY ?s
LIMIT 100 OFFSET 0
```

❌ **DON'T**:
```sparql
# Avoid unbounded queries
SELECT * WHERE { ?s ?p ?o }  # Can return millions of results
```

### 2. Filter Early

✅ **DO**:
```sparql
# Apply filters first to reduce dataset size
SELECT ?name ?email WHERE {
  ?person a foaf:Person .           # Type filter first
  ?person foaf:name ?name .
  ?person foaf:mbox ?email .
  FILTER(LANG(?name) = "en")        # Language filter
}
```

❌ **DON'T**:
```sparql
# Avoid filtering after collecting all data
SELECT ?name ?email WHERE {
  ?person foaf:name ?name .
  ?person foaf:mbox ?email .
  ?person a ?type .
  FILTER(?type = foaf:Person)       # Filter too late
}
```

### 3. Use Property Paths Wisely

✅ **DO**:
```sparql
# Simple paths are efficient
SELECT ?friend WHERE {
  :alice foaf:knows/foaf:knows ?friend .  # 2-hop path
}

# Bounded paths
SELECT ?ancestor WHERE {
  :person foaf:parent{1,3} ?ancestor .    # Max 3 levels
}
```

❌ **DON'T**:
```sparql
# Avoid unbounded paths
SELECT ?connection WHERE {
  :alice foaf:knows+ ?connection .  # Can be very expensive
}
```

### 4. Analyze Before Executing

```bash
# Explain query plan
oxirs explain mydata query.sparql --mode full

# Profile execution
oxirs performance profile "SELECT ..." --dataset mydata

# Benchmark different approaches
oxirs benchmark mydata --suite custom-queries
```

### 5. Reuse Common Queries with Templates

```bash
# Create template
oxirs template create person-search "
  PREFIX foaf: <http://xmlns.com/foaf/0.1/>
  SELECT ?person ?name WHERE {
    ?person a foaf:Person .
    ?person foaf:name ?name .
    FILTER(CONTAINS(LCASE(?name), LCASE('{{search_term}}')))
  } LIMIT {{limit}}
"

# Use template
oxirs template render person-search \
  --param search_term="alice" \
  --param limit=100
```

---

## Performance Tuning

### For Large Datasets (>1M triples)

#### 1. Batch Import

```bash
# Use batch import with parallel processing
oxirs batch import \
  --dataset mydata \
  --files data/*.nt \
  --parallel 8 \
  --format ntriples

# Alternative: Use tdbloader for bulk loading
oxirs tdbloader ./data/mydata \
  data/*.nt \
  --progress \
  --stats
```

#### 2. Streaming Operations

```bash
# Stream large exports to avoid memory issues
oxirs export mydata output.nq --format nquads | gzip > output.nq.gz

# Migrate formats with streaming
oxirs migrate \
  --source huge-file.ttl \
  --target huge-file.nt \
  --from turtle \
  --to ntriples
```

#### 3. Query Performance

```bash
# Analyze query before execution
oxirs explain mydata complex-query.sparql --file --mode full

# Use appropriate output format
oxirs query mydata query.sparql --file --output json  # Fastest
oxirs query mydata query.sparql --file --output table # Human-readable

# Enable query caching (in configuration)
```

#### 4. Server Configuration

```toml
# oxirs.toml - High-performance configuration
[server]
workers = 16
keep_alive = 120

[query]
optimize = true
parallel = true
threads = 16
simd = true
cache = true
cache_size = 4096  # 4GB cache

[connection_pool]
max_size = 128

[[datasets]]
name = "large-dataset"
location = "/fast-ssd/oxirs/data"
type = "tdb2"

[datasets.cache]
result_cache = 4096  # 4GB
pattern_cache = 2048  # 2GB

[datasets.performance]
batch_size = 100000
parallel = true
threads = 16
```

---

## Security Best Practices

### 1. Credential Management

✅ **DO**:
```bash
# Use environment variables for sensitive data
export OXIRS_ADMIN_PASSWORD="$(openssl rand -base64 32)"
export JWT_SECRET="$(openssl rand -base64 64)"

# Use in configuration
# oxirs.toml
[auth.jwt]
secret = "${JWT_SECRET}"
```

❌ **DON'T**:
```toml
# Don't hardcode credentials in config files
[auth]
admin_password = "admin123"  # NEVER DO THIS
```

### 2. Configuration File Permissions

```bash
# Restrict config file permissions
chmod 600 oxirs.toml

# Verify permissions
ls -l oxirs.toml
# Should show: -rw------- (owner read/write only)
```

### 3. Input Validation

✅ **DO**:
```bash
# Always validate input files before import
oxirs rdfparse untrusted.ttl --format turtle

# Then import if valid
oxirs import mydata untrusted.ttl --format turtle
```

### 4. TLS/SSL in Production

```toml
# oxirs.toml
[server.tls]
enabled = true
cert = "/etc/letsencrypt/live/example.com/fullchain.pem"
key = "/etc/letsencrypt/live/example.com/privkey.pem"
version = "tls13"  # Use latest TLS version
```

### 5. Rate Limiting

```toml
# oxirs.toml
[rate_limit]
enabled = true
requests_per_minute = 1000
burst = 100
by_ip = true

# Whitelist trusted IPs
whitelist = [
    "10.0.0.0/8",      # Internal network
    "192.168.0.0/16",  # Private network
]
```

### 6. DDoS Protection

```toml
# oxirs.toml
[ddos_protection]
enabled = true
max_connections_per_ip = 100
request_rate_threshold = 1000
ban_duration = 3600
auto_ban = true
```

---

## Production Deployment

### 1. Systemd Service

Create `/etc/systemd/system/oxirs.service`:

```ini
[Unit]
Description=OxiRS SPARQL Server
After=network.target

[Service]
Type=simple
User=oxirs
Group=oxirs
WorkingDirectory=/var/lib/oxirs
ExecStart=/usr/local/bin/oxirs serve --config /etc/oxirs/production.toml
Restart=always
RestartSec=10

# Security hardening
PrivateTmp=true
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/oxirs /var/log/oxirs

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

# Environment
Environment="RUST_LOG=warn"
EnvironmentFile=-/etc/oxirs/environment

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable oxirs
sudo systemctl start oxirs
sudo systemctl status oxirs
```

### 2. Reverse Proxy (Nginx)

```nginx
# /etc/nginx/sites-available/oxirs
upstream oxirs {
    server localhost:3030;
    keepalive 32;
}

server {
    listen 443 ssl http2;
    server_name sparql.example.com;

    ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://oxirs;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";

        # Timeouts
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    # Metrics endpoint (restrict access)
    location /metrics {
        allow 10.0.0.0/8;
        deny all;
        proxy_pass http://oxirs;
    }
}
```

### 3. Docker Deployment

```dockerfile
# Dockerfile
FROM rust:1.70-slim as builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin oxirs

FROM debian:bullseye-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/oxirs /usr/local/bin/
COPY oxirs.toml /etc/oxirs/oxirs.toml
VOLUME /data
EXPOSE 3030
CMD ["oxirs", "serve", "--config", "/etc/oxirs/oxirs.toml"]
```

```bash
# Build
docker build -t oxirs:beta.1 .

# Run
docker run -d \
  --name oxirs-server \
  -p 3030:3030 \
  -v $(pwd)/data:/data \
  -v $(pwd)/oxirs.toml:/etc/oxirs/oxirs.toml:ro \
  -e JWT_SECRET="$(openssl rand -base64 64)" \
  --restart unless-stopped \
  oxirs:beta.1
```

### 4. Kubernetes Deployment

```yaml
# oxirs-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: oxirs-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: oxirs
  template:
    metadata:
      labels:
        app: oxirs
    spec:
      containers:
      - name: oxirs
        image: oxirs:beta.1
        ports:
        - containerPort: 3030
        env:
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: oxirs-secrets
              key: jwt-secret
        volumeMounts:
        - name: data
          mountPath: /data
        - name: config
          mountPath: /etc/oxirs
        livenessProbe:
          httpGet:
            path: /health/live
            port: 3030
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 3030
          initialDelaySeconds: 10
          periodSeconds: 5
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: oxirs-data-pvc
      - name: config
        configMap:
          name: oxirs-config

---
apiVersion: v1
kind: Service
metadata:
  name: oxirs-service
spec:
  selector:
    app: oxirs
  ports:
  - protocol: TCP
    port: 3030
    targetPort: 3030
  type: LoadBalancer
```

---

## Monitoring & Maintenance

### 1. Health Checks

```bash
# Server health
curl http://localhost:3030/health/live

# Check dataset statistics
oxirs tdbstats mydata --format json | jq '.triples_count'

# Monitor response time
time curl -X POST http://localhost:3030/sparql \
  -H "Content-Type: application/sparql-query" \
  -d "SELECT * WHERE { ?s ?p ?o } LIMIT 1"
```

### 2. Metrics Collection

```bash
# Prometheus metrics
curl http://localhost:3030/metrics

# Custom metric queries
oxirs query mydata "
  SELECT (COUNT(*) as ?triples) WHERE {?s ?p ?o}
" --output json | jq '.results.bindings[0].triples.value'
```

### 3. Logging

```bash
# Enable structured logging
RUST_LOG=info oxirs serve --config oxirs.toml

# Debug specific modules
RUST_LOG=oxirs_core=debug,oxirs_arq=trace oxirs serve --config oxirs.toml

# Log to file with rotation
# Set in oxirs.toml:
[logging]
output = "file"
file = "/var/log/oxirs/server.log"

[logging.rotation]
enabled = true
max_size = 100  # MB
max_age = 30    # days
max_backups = 10
compress = true
```

### 4. Backup Strategy

```bash
#!/bin/bash
# backup-oxirs.sh - Daily backup script

DATE=$(date +%Y%m%d)
DATASET="production"
BACKUP_DIR="/backups/oxirs"
RETENTION_DAYS=90

# Create backup
oxirs tdbbackup \
  ./data/$DATASET \
  $BACKUP_DIR/$DATASET-$DATE \
  --compress

# Export to N-Quads (for disaster recovery)
oxirs export $DATASET \
  $BACKUP_DIR/$DATASET-$DATE.nq \
  --format nquads

# Verify backup
if oxirs tdbstats $BACKUP_DIR/$DATASET-$DATE > /dev/null 2>&1; then
  echo "✅ Backup successful: $DATE"

  # Cleanup old backups
  find $BACKUP_DIR -name "$DATASET-*" -mtime +$RETENTION_DAYS -delete
else
  echo "❌ Backup failed: $DATE"
  # Send alert
  exit 1
fi
```

Make it executable and add to cron:
```bash
chmod +x backup-oxirs.sh

# Add to crontab (daily at 2 AM)
crontab -e
# 0 2 * * * /usr/local/bin/backup-oxirs.sh >> /var/log/oxirs/backup.log 2>&1
```

---

## Troubleshooting

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| "Dataset not found" | Dataset not initialized | Run `oxirs init <name>` |
| "Format not recognized" | Missing format flag | Specify `--format` explicitly |
| "Permission denied" | Wrong directory permissions | `chmod 755 <dir>`, check ownership |
| "Port already in use" | Port conflict | Use different port with `--port` |
| "Out of memory" | Large query result | Use LIMIT, enable streaming, add RAM |
| "Invalid SPARQL syntax" | Query syntax error | Use `oxirs qparse` to validate |
| "Connection timeout" | Slow query | Increase timeout, optimize query |
| "Authentication failed" | Wrong credentials | Check JWT_SECRET, verify auth config |

### Debug Mode

```bash
# Enable verbose logging
oxirs --verbose query mydata "SELECT * WHERE { ?s ?p ?o }"

# Check configuration
oxirs config validate oxirs.toml

# Test query syntax
oxirs qparse "SELECT * WHERE { ?s ?p ?o }"

# Explain query plan
oxirs explain mydata query.sparql --file --mode full
```

---

## Common Patterns

### Pattern 1: Daily Data Pipeline

```bash
#!/bin/bash
# daily-pipeline.sh

DATASET="analytics"
DATE=$(date +%Y%m%d)

# 1. Backup yesterday's data
oxirs tdbbackup ./data/$DATASET ./backups/$DATASET-$DATE

# 2. Import new data
oxirs batch import \
  --dataset $DATASET \
  --files /incoming/$DATE/*.nt \
  --parallel 8

# 3. Run analytics queries
oxirs query $DATASET analytics.sparql \
  --file \
  --output json > /results/$DATE-analytics.json

# 4. Compact dataset
oxirs tdbcompact ./data/$DATASET

# 5. Export summary
oxirs query $DATASET summary.sparql \
  --file \
  --output csv > /reports/$DATE-summary.csv
```

### Pattern 2: Multi-Stage ETL

```bash
#!/bin/bash
# etl-pipeline.sh

# Extract: Import from multiple sources
oxirs batch import --dataset staging \
  --files /sources/system-a/*.ttl \
  --graph http://example.org/system-a

oxirs batch import --dataset staging \
  --files /sources/system-b/*.nt \
  --graph http://example.org/system-b

# Transform: Run SPARQL CONSTRUCT queries
oxirs query staging transform-a.sparql \
  --file \
  --output turtle > /tmp/transformed-a.ttl

oxirs query staging transform-b.sparql \
  --file \
  --output turtle > /tmp/transformed-b.ttl

# Load: Import transformed data to production
oxirs import production /tmp/transformed-a.ttl \
  --graph http://example.org/canonical

oxirs import production /tmp/transformed-b.ttl \
  --graph http://example.org/canonical

# Cleanup
rm /tmp/transformed-*.ttl
```

### Pattern 3: Real-time Updates

```bash
#!/bin/bash
# watch-and-update.sh

DATASET="realtime"
WATCH_DIR="/incoming/updates"

# Watch for new files and import them
inotifywait -m -e create "$WATCH_DIR" --format '%f' | while read FILE; do
  echo "Processing $FILE..."

  # Validate
  if oxirs rdfparse "$WATCH_DIR/$FILE"; then
    # Import
    oxirs import $DATASET "$WATCH_DIR/$FILE"

    # Archive
    mv "$WATCH_DIR/$FILE" "/archive/$FILE"

    echo "✅ Imported $FILE"
  else
    echo "❌ Invalid RDF: $FILE"
    mv "$WATCH_DIR/$FILE" "/errors/$FILE"
  fi
done
```

---

## See Also

- [COMMAND_REFERENCE.md](COMMAND_REFERENCE.md) - Full command reference
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration guide
- [INTERACTIVE.md](INTERACTIVE.md) - Interactive mode guide
- [README.md](../README.md) - Getting started
- [QUICKSTART.md](../QUICKSTART.md) - Quick start guide

---

**OxiRS v0.1.0-beta.2** - Production-ready best practices for semantic web operations
