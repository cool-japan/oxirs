# OxiRS Federate - Deployment Guide

**Version**: 0.1.0-rc.2
**Last Updated**: December 6, 2025
**Status**: Production-Ready

This comprehensive guide covers deploying oxirs-federate in production environments, from basic setups to advanced enterprise configurations.

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Deployment Architectures](#deployment-architectures)
6. [Performance Tuning](#performance-tuning)
7. [Monitoring & Observability](#monitoring--observability)
8. [Security](#security)
9. [High Availability](#high-availability)
10. [Troubleshooting](#troubleshooting)
11. [Production Checklist](#production-checklist)

---

## Quick Start

### Minimal Deployment (Development)

```bash
# 1. Add dependency to Cargo.toml
[dependencies]
oxirs-federate = { version = "0.1.0-rc.2", features = ["default"] }

# 2. Basic server setup
use oxirs_federate::FederationEngine;

#[tokio::main]
async fn main() -> Result<()> {
    let engine = FederationEngine::new();

    // Register SPARQL endpoints
    engine.register_service(sparql_service).await?;

    // Execute federated queries
    let results = engine.execute_sparql(query).await?;

    Ok(())
}
```

### Production Deployment (Docker)

```bash
# Clone and build
git clone https://github.com/cool-japan/oxirs
cd oxirs/stream/oxirs-federate

# Build release binary
cargo build --release --all-features

# Run with configuration
./target/release/oxirs-federate --config /etc/oxirs/config.toml
```

---

## System Requirements

### Minimum Requirements

- **CPU**: 2 cores, 2.0 GHz
- **RAM**: 4 GB
- **Disk**: 10 GB SSD
- **OS**: Linux (Ubuntu 20.04+), macOS (12+), Windows Server 2019+
- **Rust**: 1.90.0 or later

### Recommended for Production

- **CPU**: 8+ cores, 3.0 GHz (with AVX2 for SIMD acceleration)
- **RAM**: 16-32 GB (64 GB+ for large-scale federations)
- **Disk**: 100+ GB NVMe SSD
- **Network**: 10 Gbps
- **OS**: Ubuntu 22.04 LTS (Kernel 5.15+)

### Optional Accelerators (v0.2.0 Features)

- **GPU**: NVIDIA (CUDA 11+), AMD (ROCm 5+), or Apple Silicon (Metal)
- **GPU RAM**: 8+ GB VRAM for ML models
- **SIMD**: CPU with AVX2 (x86_64) or NEON (ARM64)

---

## Installation

### From Crates.io

```bash
cargo install oxirs-federate --features all
```

### From Source

```bash
# Clone repository
git clone https://github.com/cool-japan/oxirs
cd oxirs/stream/oxirs-federate

# Build with all features
cargo build --release --all-features

# Install systemd service (Linux)
sudo cp target/release/oxirs-federate /usr/local/bin/
sudo cp deploy/oxirs-federate.service /etc/systemd/system/
sudo systemctl enable oxirs-federate
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM rust:1.90 as builder
WORKDIR /build
COPY . .
RUN cargo build --release --all-features

FROM ubuntu:22.04
RUN apt-get update && apt-get install -y ca-certificates libssl3 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/oxirs-federate /usr/local/bin/
COPY config/production.toml /etc/oxirs/config.toml
EXPOSE 8080 8081
CMD ["oxirs-federate", "--config", "/etc/oxirs/config.toml"]
```

```bash
# Build and run
docker build -t oxirs-federate:latest .
docker run -d -p 8080:8080 -p 8081:8081 \
  -v /data/oxirs:/data \
  -v /etc/oxirs:/etc/oxirs:ro \
  --name oxirs-federate \
  oxirs-federate:latest
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: oxirs-federate
spec:
  replicas: 3
  selector:
    matchLabels:
      app: oxirs-federate
  template:
    metadata:
      labels:
        app: oxirs-federate
    spec:
      containers:
      - name: oxirs-federate
        image: oxirs-federate:latest
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 8081
          name: metrics
        env:
        - name: RUST_LOG
          value: "info"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        volumeMounts:
        - name: config
          mountPath: /etc/oxirs
          readOnly: true
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: oxirs-config
---
apiVersion: v1
kind: Service
metadata:
  name: oxirs-federate
spec:
  selector:
    app: oxirs-federate
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: metrics
    port: 8081
    targetPort: 8081
  type: LoadBalancer
```

---

## Configuration

### Basic Configuration (`config.toml`)

```toml
[server]
host = "0.0.0.0"
port = 8080
metrics_port = 8081
workers = 8
max_connections = 1000

[federation]
# Query timeout in seconds
query_timeout = 300
# Maximum number of concurrent subqueries
max_concurrent_queries = 50
# Enable query result caching
enable_caching = true
cache_size_mb = 1024

[logging]
level = "info"
format = "json"
output = "/var/log/oxirs/federate.log"

[monitoring]
enable_metrics = true
enable_tracing = true
# Prometheus metrics endpoint
metrics_endpoint = "/metrics"
# OpenTelemetry collector endpoint
otlp_endpoint = "http://localhost:4317"
```

### Advanced Configuration (v0.2.0 Features)

```toml
[gpu_acceleration]
enabled = true
backend = "metal"  # Options: cuda, metal, rocm, opencl, webgpu
device_id = 0
max_batch_size = 10000
fallback_to_cpu = true

[simd_optimization]
enabled = true
# Auto-detect: avx2 (x86_64) or neon (arm64)
auto_detect = true
batch_size = 1024

[jit_compilation]
enabled = true
cache_size = 100
optimization_level = "aggressive"  # Options: none, basic, aggressive
enable_profiling = true

[ml_serving]
enabled = true
max_models = 10
enable_warmup = true
auto_rollback_threshold = 0.1

[security]
enable_auth = true
auth_methods = ["oauth2", "mtls", "api_key"]
rate_limiting = true
rate_limit_requests_per_minute = 1000

[high_availability]
enable_clustering = true
cluster_size = 3
consensus_protocol = "raft"
heartbeat_interval_ms = 1000
```

---

## Deployment Architectures

### 1. Single-Node Deployment

**Use Case**: Development, small deployments (<100 queries/sec)

```
┌─────────────────────────┐
│   OxiRS Federate        │
│   ┌─────────────┐       │
│   │ Federation  │       │
│   │   Engine    │       │
│   └─────────────┘       │
│         ↓               │
│   ┌─────────────┐       │
│   │   Cache     │       │
│   └─────────────┘       │
└─────────────────────────┘
          ↓
    External SPARQL
      Endpoints
```

**Configuration**:
- 1 instance
- Local caching
- In-memory storage

### 2. Load-Balanced Deployment

**Use Case**: Medium-scale (<1000 queries/sec)

```
           ┌─────────────┐
           │ Load Balancer│
           └──────┬───────┘
        ┌─────────┼──────────┐
        ↓         ↓          ↓
    ┌───────┐ ┌───────┐ ┌───────┐
    │ Node 1│ │ Node 2│ │ Node 3│
    └───────┘ └───────┘ └───────┘
        ↓         ↓          ↓
      ┌───────────────────────┐
      │    Shared Redis Cache │
      └───────────────────────┘
```

**Configuration**:
- 3-5 instances
- Redis cache
- Health checks
- Session affinity

### 3. Clustered Deployment (High Availability)

**Use Case**: Enterprise, mission-critical (>1000 queries/sec)

```
         ┌─────────────────┐
         │   API Gateway   │
         └────────┬─────────┘
                  │
         ┌────────┴─────────┐
         │ Service Mesh     │
         │  (Istio/Linkerd) │
         └────────┬─────────┘
        ┌─────────┼──────────┐
        ↓         ↓          ↓
   [OxiRS-1]  [OxiRS-2]  [OxiRS-3]
        ↓         ↓          ↓
    ┌─────────────────────────┐
    │  Raft Consensus Layer   │
    └─────────────────────────┘
              ↓
    ┌─────────────────────────┐
    │  Distributed Cache      │
    │  (Redis Cluster)        │
    └─────────────────────────┘
              ↓
    ┌─────────────────────────┐
    │  Federated SPARQL       │
    │  Endpoints              │
    └─────────────────────────┘
```

**Features**:
- Automatic failover
- Data replication
- Consensus-based coordination
- Distributed caching
- Service mesh integration

---

## Performance Tuning

### 1. Memory Optimization

```toml
[memory]
# Enable memory-mapped datasets for large RDF graphs
use_mmap = true
# LRU cache size (MB)
cache_size_mb = 4096
# Garbage collection interval (seconds)
gc_interval = 300
```

### 2. Query Optimization

```toml
[query_optimization]
# Enable query result caching
enable_cache = true
# Cache TTL (seconds)
cache_ttl = 3600
# Enable query plan caching
enable_plan_cache = true
# Maximum query complexity (prevents DoS)
max_complexity = 10000
```

### 3. Connection Pooling

```toml
[connection_pool]
# Pool size per endpoint
pool_size = 20
# Connection timeout (seconds)
timeout = 30
# Keep-alive interval (seconds)
keep_alive = 60
# Maximum retries
max_retries = 3
```

### 4. GPU Acceleration Tuning

```toml
[gpu]
# Batch size (adjust based on GPU memory)
batch_size = 10000
# Enable memory prefetching
prefetch = true
# Use tensor cores (if available)
use_tensor_cores = true
```

### 5. SIMD Optimization

```rust
// Code-level optimization
use oxirs_federate::simd_optimized_joins::{JoinAlgorithm, SimdJoinConfig};

let config = SimdJoinConfig {
    algorithm: JoinAlgorithm::Hash,
    enable_simd: true,
    batch_size: 2048,  // Optimal for AVX2
    ..Default::default()
};
```

---

## Monitoring & Observability

### Metrics

**Prometheus Integration**:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'oxirs-federate'
    static_configs:
      - targets: ['localhost:8081']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

**Key Metrics**:
- `oxirs_queries_total`: Total queries executed
- `oxirs_query_duration_seconds`: Query execution time
- `oxirs_cache_hit_rate`: Cache hit percentage
- `oxirs_service_availability`: Service health status
- `oxirs_gpu_utilization`: GPU usage (if enabled)
- `oxirs_simd_speedup`: SIMD acceleration factor

### Logging

```toml
[logging]
level = "info"  # Options: trace, debug, info, warn, error
format = "json"
output = "stdout"  # Or file path

# Log rotation
rotate = true
max_size_mb = 100
max_backups = 10
```

### Distributed Tracing

```rust
// OpenTelemetry integration
use oxirs_federate::distributed_tracing::TracingConfig;

let tracing_config = TracingConfig {
    service_name: "oxirs-federate".to_string(),
    otlp_endpoint: "http://jaeger:4317".to_string(),
    sample_rate: 0.1,  // 10% of queries
};
```

### Health Checks

```bash
# Liveness probe
curl http://localhost:8080/health
# Response: {"status": "healthy", "services": 15, "cache_hit_rate": 0.85}

# Readiness probe
curl http://localhost:8080/ready
# Response: {"ready": true, "version": "0.2.0-rc.1"}
```

---

## Security

### 1. Authentication

```toml
[auth]
# API Key authentication
api_key_header = "X-API-Key"
api_keys_file = "/etc/oxirs/api-keys.json"

# OAuth2
oauth2_issuer = "https://auth.example.com"
oauth2_audience = "oxirs-federate"

# mTLS
mtls_enabled = true
ca_cert = "/etc/oxirs/ca.crt"
server_cert = "/etc/oxirs/server.crt"
server_key = "/etc/oxirs/server.key"
```

### 2. Rate Limiting

```toml
[rate_limiting]
enabled = true
# Requests per minute per client
rpm = 1000
# Burst allowance
burst = 100
# Rate limit by IP
by_ip = true
# Rate limit by API key
by_api_key = true
```

### 3. Encryption

```toml
[encryption]
# TLS for external connections
tls_enabled = true
tls_cert = "/etc/oxirs/tls.crt"
tls_key = "/etc/oxirs/tls.key"

# Encrypt cache data
encrypt_cache = true
encryption_key_file = "/etc/oxirs/cache-key.bin"
```

### 4. Network Security

```bash
# Firewall rules (iptables)
sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT  # HTTP
sudo iptables -A INPUT -p tcp --dport 8081 -j ACCEPT  # Metrics (internal only)
sudo iptables -A INPUT -j DROP

# Or use UFW
sudo ufw allow 8080/tcp
sudo ufw allow from 10.0.0.0/8 to any port 8081  # Metrics from internal network only
```

---

## High Availability

### 1. Clustering

```toml
[cluster]
enabled = true
# Cluster nodes
nodes = ["node1.example.com:8082", "node2.example.com:8082", "node3.example.com:8082"]
# This node's ID
node_id = "node1"
# Consensus protocol
consensus = "raft"
# Leader election timeout (ms)
election_timeout = 5000
```

### 2. Failover

```toml
[failover]
enabled = true
# Health check interval (seconds)
health_check_interval = 10
# Failure threshold (consecutive failures)
failure_threshold = 3
# Automatic failover
auto_failover = true
```

### 3. Backup & Recovery

```bash
# Backup configuration
./oxirs-federate backup --output /backup/oxirs-$(date +%Y%m%d).tar.gz

# Restore from backup
./oxirs-federate restore --input /backup/oxirs-20251206.tar.gz

# Automated backups (cron)
0 2 * * * /usr/local/bin/oxirs-federate backup --output /backup/oxirs-$(date +\%Y\%m\%d).tar.gz
```

---

## Troubleshooting

### Common Issues

#### 1. High Memory Usage

**Symptoms**: Memory > 80%, OOM kills

**Solutions**:
```toml
# Reduce cache size
[memory]
cache_size_mb = 2048  # Reduce from 4096

# Enable memory limits
[limits]
max_memory_mb = 6144
```

#### 2. Slow Query Performance

**Symptoms**: Query latency > 5s

**Diagnostics**:
```bash
# Check query plan
curl -X POST http://localhost:8080/explain -d '{"query": "SELECT ..."}'

# Enable profiling
RUST_LOG=oxirs_federate=debug ./oxirs-federate
```

**Solutions**:
- Enable JIT compilation
- Use GPU acceleration for large datasets
- Optimize join algorithms (hash vs merge)

#### 3. Service Unavailability

**Symptoms**: 503 errors, service timeouts

**Diagnostics**:
```bash
# Check service health
curl http://localhost:8080/services

# Check logs
tail -f /var/log/oxirs/federate.log | grep ERROR
```

**Solutions**:
- Increase timeouts
- Enable retry logic
- Check network connectivity

### Debug Mode

```bash
# Run with verbose logging
RUST_LOG=oxirs_federate=trace ./oxirs-federate --config config.toml

# Enable CPU profiling
cargo run --release --features profiling

# Enable memory profiling
MALLOC_TRACE=oxirs.trace ./oxirs-federate
```

---

## Production Checklist

### Pre-Deployment

- [ ] Review and test configuration
- [ ] Set up monitoring (Prometheus, Grafana)
- [ ] Configure logging aggregation
- [ ] Set up distributed tracing
- [ ] Enable authentication
- [ ] Configure rate limiting
- [ ] Set up TLS certificates
- [ ] Plan backup strategy
- [ ] Document runbooks
- [ ] Load test with production-like data

### Deployment

- [ ] Deploy to staging first
- [ ] Run smoke tests
- [ ] Verify health checks
- [ ] Check metrics dashboards
- [ ] Test failover scenarios
- [ ] Gradual rollout (canary/blue-green)
- [ ] Monitor error rates
- [ ] Have rollback plan ready

### Post-Deployment

- [ ] Monitor for 24 hours
- [ ] Review performance metrics
- [ ] Check error logs
- [ ] Verify cache hit rates
- [ ] Test backup/restore
- [ ] Update documentation
- [ ] Schedule capacity review
- [ ] Plan for scaling

---

## Additional Resources

- **Documentation**: https://docs.rs/oxirs-federate
- **Examples**: `/examples` directory
- **Benchmarks**: `/benches` directory
- **Issue Tracker**: https://github.com/cool-japan/oxirs/issues
- **Community**: https://github.com/cool-japan/oxirs/discussions

---

## Support

For production support inquiries:
- **GitHub Issues**: https://github.com/cool-japan/oxirs/issues
- **Email**: support@cool-japan.org
- **Documentation**: https://oxirs.dev/docs

---

**Last Updated**: December 6, 2025
**Version**: 0.2.0-rc.1
