# OxiRS Embed - Production Deployment Guide

**Version**: 0.1.0
**Last Updated**: 2026-01-06
**Status**: Production Ready ✅

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Deployment Architectures](#deployment-architectures)
4. [Installation & Setup](#installation--setup)
5. [Configuration](#configuration)
6. [Performance Optimization](#performance-optimization)
7. [Scaling Strategies](#scaling-strategies)
8. [Monitoring & Observability](#monitoring--observability)
9. [Security Best Practices](#security-best-practices)
10. [Troubleshooting](#troubleshooting)
11. [Production Checklist](#production-checklist)

---

## Overview

This guide provides comprehensive instructions for deploying oxirs-embed in production environments, from single-server setups to distributed multi-region deployments.

### Deployment Options

- **Standalone Server**: Single instance for small-scale applications
- **Load-Balanced Cluster**: Multiple instances behind a load balancer
- **Distributed System**: Federation across multiple data centers
- **Cloud-Native**: Kubernetes deployment with auto-scaling
- **Edge Deployment**: Quantized models for edge devices

---

## System Requirements

### Minimum Requirements (Small Scale: <100K entities)

```yaml
CPU: 4 cores (x86_64 or ARM64)
RAM: 8 GB
Storage: 10 GB SSD
Network: 1 Gbps
OS: Linux (Ubuntu 22.04+, RHEL 8+), macOS 12+, Windows Server 2019+
```

### Recommended Requirements (Medium Scale: 100K-1M entities)

```yaml
CPU: 16 cores (x86_64 with AVX2)
RAM: 32 GB
Storage: 100 GB NVMe SSD
Network: 10 Gbps
OS: Linux (Ubuntu 22.04 LTS)
GPU: Optional - NVIDIA Tesla T4 or better (8GB+ VRAM)
```

### High-Performance Requirements (Large Scale: >1M entities)

```yaml
CPU: 32+ cores (AMD EPYC or Intel Xeon)
RAM: 128+ GB
Storage: 500 GB+ NVMe SSD (RAID 10)
Network: 25+ Gbps
GPU: NVIDIA A100 (40GB+) or H100
```

### Software Dependencies

```toml
# Cargo.toml
[dependencies]
oxirs-embed = { version = "0.1.0", features = ["all"] }
tokio = { version = "1.48", features = ["full"] }
tracing-subscriber = "0.3"
```

---

## Deployment Architectures

### 1. Standalone Deployment

**Use Case**: Development, testing, small-scale applications

```
┌─────────────────────────┐
│   Client Applications   │
└───────────┬─────────────┘
            │
            │ HTTP/GraphQL
            ▼
┌─────────────────────────┐
│  OxiRS Embed Server     │
│  - Inference Engine     │
│  - Model Cache          │
│  - Vector Search Index  │
└─────────────────────────┘
            │
            ▼
┌─────────────────────────┐
│  Persistent Storage     │
│  (Model files, Cache)   │
└─────────────────────────┘
```

**Configuration**:

```toml
# oxirs.toml
[server]
host = "0.0.0.0"
port = 8080
workers = 4

[inference]
cache_size = 10000
batch_size = 100
max_concurrent = 10

[storage]
model_path = "/var/lib/oxirs/models"
cache_path = "/var/lib/oxirs/cache"
```

### 2. Load-Balanced Cluster

**Use Case**: High availability, horizontal scaling

```
┌─────────────────────────┐
│   Client Applications   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Load Balancer (HAProxy/Nginx)   │
└────┬──────────┬─────────┘
     │          │
     ▼          ▼
┌─────────┐ ┌─────────┐
│ Server 1│ │ Server 2│
└────┬────┘ └────┬────┘
     │           │
     └───────┬───┘
             ▼
┌─────────────────────────┐
│  Shared Model Storage   │
│  (NFS/S3/GCS)          │
└─────────────────────────┘
```

**HAProxy Configuration**:

```haproxy
frontend oxirs_frontend
    bind *:80
    mode http
    default_backend oxirs_servers

backend oxirs_servers
    mode http
    balance roundrobin
    option httpchk GET /health
    server oxirs1 10.0.1.10:8080 check
    server oxirs2 10.0.1.11:8080 check
    server oxirs3 10.0.1.12:8080 check
```

### 3. Kubernetes Deployment

**Use Case**: Cloud-native, auto-scaling, multi-region

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: oxirs-embed
  labels:
    app: oxirs-embed
spec:
  replicas: 3
  selector:
    matchLabels:
      app: oxirs-embed
  template:
    metadata:
      labels:
        app: oxirs-embed
    spec:
      containers:
      - name: oxirs-embed
        image: ghcr.io/cool-japan/oxirs-embed:0.1.0
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: RUST_LOG
          value: "info"
        - name: OXIRS_WORKERS
          value: "8"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
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
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /var/lib/oxirs/models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: oxirs-models-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: oxirs-embed-service
spec:
  type: LoadBalancer
  selector:
    app: oxirs-embed
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
    name: http
  - protocol: TCP
    port: 9090
    targetPort: 9090
    name: metrics

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: oxirs-embed-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: oxirs-embed
  minReplicas: 3
  maxReplicas: 20
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
```

### 4. Edge Deployment (Quantized Models)

**Use Case**: IoT, mobile, resource-constrained environments

```rust
use oxirs_embed::{
    quantization::{QuantizationConfig, QuantizationMethod, ModelQuantizer},
    TransE, EmbeddingModel,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Load and quantize model
    let mut model = TransE::load("model.bin")?;

    let quant_config = QuantizationConfig {
        method: QuantizationMethod::Int8,
        symmetric: true,
        per_channel: false,
        calibration_samples: 100,
    };

    let quantizer = ModelQuantizer::new(quant_config);
    // Quantized model is 4x smaller and 2-3x faster on CPU

    // Deploy to edge device
    model.save("model_quantized.bin")?;

    Ok(())
}
```

---

## Installation & Setup

### Option 1: Pre-built Binaries

```bash
# Download latest release
wget https://github.com/cool-japan/oxirs/releases/download/v0.1.0/oxirs-embed-linux-x86_64.tar.gz

# Extract
tar -xzf oxirs-embed-linux-x86_64.tar.gz

# Install
sudo mv oxirs-embed /usr/local/bin/
sudo chmod +x /usr/local/bin/oxirs-embed

# Verify installation
oxirs-embed --version
```

### Option 2: Build from Source

```bash
# Clone repository
git clone https://github.com/cool-japan/oxirs.git
cd oxirs/ai/oxirs-embed

# Build optimized binary
cargo build --release --features all

# Install
sudo cp target/release/oxirs-embed /usr/local/bin/
```

### Option 3: Docker

```dockerfile
# Dockerfile
FROM rust:1.80-slim as builder

WORKDIR /usr/src/oxirs
COPY . .

RUN cargo build --release --features all

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/src/oxirs/target/release/oxirs-embed /usr/local/bin/

EXPOSE 8080 9090

CMD ["oxirs-embed", "serve"]
```

```bash
# Build Docker image
docker build -t oxirs-embed:0.1.0 .

# Run container
docker run -d \
  --name oxirs-embed \
  -p 8080:8080 \
  -p 9090:9090 \
  -v /path/to/models:/var/lib/oxirs/models \
  oxirs-embed:0.1.0
```

---

## Configuration

### Environment Variables

```bash
# Server configuration
export OXIRS_HOST="0.0.0.0"
export OXIRS_PORT="8080"
export OXIRS_WORKERS="8"

# Logging
export RUST_LOG="info,oxirs_embed=debug"
export RUST_BACKTRACE="1"

# Performance
export OXIRS_CACHE_SIZE="100000"
export OXIRS_BATCH_SIZE="200"
export OXIRS_MAX_CONCURRENT="20"

# GPU (if available)
export CUDA_VISIBLE_DEVICES="0,1"
export OXIRS_GPU_MEMORY_FRACTION="0.8"

# Storage
export OXIRS_MODEL_PATH="/var/lib/oxirs/models"
export OXIRS_CACHE_PATH="/var/lib/oxirs/cache"
```

### Configuration File (oxirs.toml)

```toml
[server]
host = "0.0.0.0"
port = 8080
workers = 8
request_timeout_seconds = 300
keep_alive_timeout_seconds = 75

[inference]
cache_size = 100000
batch_size = 200
max_concurrent = 20
cache_ttl_seconds = 3600
enable_caching = true
warm_up_cache = true

[performance]
use_mixed_precision = true
use_gpu = true
gpu_memory_fraction = 0.8
num_gpu_streams = 4

[storage]
model_path = "/var/lib/oxirs/models"
cache_path = "/var/lib/oxirs/cache"
backup_path = "/var/backups/oxirs"

[monitoring]
enable_metrics = true
metrics_port = 9090
enable_tracing = true
trace_sample_rate = 0.1

[security]
enable_tls = true
cert_path = "/etc/oxirs/certs/server.crt"
key_path = "/etc/oxirs/certs/server.key"
enable_auth = true
jwt_secret_path = "/etc/oxirs/secrets/jwt.key"

[limits]
max_request_size_mb = 100
max_entities_per_query = 10000
max_batch_size = 1000
rate_limit_per_second = 1000
```

---

## Performance Optimization

### 1. Mixed Precision Training

```rust
use oxirs_embed::mixed_precision::{MixedPrecisionConfig, MixedPrecisionTrainer};

let mp_config = MixedPrecisionConfig {
    enabled: true,
    loss_scale: 1024.0,
    dynamic_loss_scaling: true,
    gradient_clip_value: Some(1.0),
    ..Default::default()
};

let mp_trainer = MixedPrecisionTrainer::new(mp_config);
// 2x faster training, 50% less memory
```

### 2. Model Quantization

```rust
use oxirs_embed::quantization::{QuantizationConfig, QuantizationMethod};

let quant_config = QuantizationConfig {
    method: QuantizationMethod::Int8,
    symmetric: true,
    per_channel: false,
    calibration_samples: 1000,
};

// 4x model compression, 2-3x faster inference
```

### 3. Batch Processing

```rust
use oxirs_embed::inference::InferenceEngine;

let mut engine = InferenceEngine::new(model, config);

// Batch multiple requests for better throughput
let results = engine.predict_batch(&queries).await?;
// 5-10x throughput improvement
```

### 4. GPU Acceleration

```rust
use oxirs_embed::gpu_acceleration::{GpuConfig, GpuAccelerator};

let gpu_config = GpuConfig {
    device_id: 0,
    memory_fraction: 0.8,
    num_streams: 4,
    enable_tensor_cores: true,
};

// 10-100x faster embedding computation
```

### 5. Caching Strategy

```rust
let cache_config = InferenceConfig {
    cache_size: 100000,      // Cache most frequently used embeddings
    cache_ttl: 3600,          // 1 hour TTL
    enable_caching: true,
    warm_up_cache: true,     // Pre-load frequently used embeddings
    ..Default::default()
};

// 100-1000x faster for repeated queries
```

---

## Scaling Strategies

### Horizontal Scaling

**Stateless Design**: Each server instance is stateless
- Share model files via NFS/S3/GCS
- Use Redis for distributed caching
- Sticky sessions not required

**Auto-Scaling Rules**:
```yaml
# Scale up when:
- CPU > 70% for 2 minutes
- Memory > 80% for 2 minutes
- Request latency p95 > 500ms for 5 minutes

# Scale down when:
- CPU < 30% for 10 minutes
- Memory < 50% for 10 minutes
- Request count < 100/min for 15 minutes
```

### Vertical Scaling

**When to scale vertically**:
- Single large model (>10M entities)
- GPU acceleration required
- Complex graph computations

**Scaling Limits**:
- CPU: Up to 128 cores cost-effective
- RAM: Up to 1TB practical
- GPU: Up to 8x A100/H100 per node

---

## Monitoring & Observability

### Metrics (Prometheus)

```prometheus
# Request metrics
http_requests_total
http_request_duration_seconds
http_requests_in_flight

# Inference metrics
inference_requests_total
inference_latency_seconds
inference_cache_hit_rate
inference_batch_size

# Model metrics
model_memory_bytes
model_parameters_total
embedding_dimensions

# Resource metrics
cpu_usage_percent
memory_usage_bytes
gpu_utilization_percent
gpu_memory_usage_bytes
```

### Logging (Structured JSON)

```json
{
  "timestamp": "2025-12-25T10:30:45Z",
  "level": "INFO",
  "target": "oxirs_embed::inference",
  "message": "Batch inference completed",
  "fields": {
    "batch_size": 150,
    "latency_ms": 45,
    "cache_hit_rate": 0.85,
    "model": "hole_biomedical_v1"
  }
}
```

### Distributed Tracing (Jaeger/Zipkin)

```rust
use tracing_subscriber::layer::SubscriberExt;

tracing_subscriber::registry()
    .with(tracing_subscriber::fmt::layer())
    .with(tracing_jaeger::layer("oxirs-embed"))
    .init();
```

### Health Checks

```rust
// GET /health
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime_seconds": 86400,
  "cache_hit_rate": 0.85,
  "models_loaded": 3
}

// GET /ready
{
  "ready": true,
  "models_ready": true,
  "cache_ready": true,
  "storage_ready": true
}
```

---

## Security Best Practices

### 1. TLS/SSL Encryption

```toml
[security]
enable_tls = true
cert_path = "/etc/oxirs/certs/server.crt"
key_path = "/etc/oxirs/certs/server.key"
min_tls_version = "1.3"
```

### 2. Authentication & Authorization

```rust
// JWT-based authentication
use jsonwebtoken::{encode, decode, Header, Validation};

// API key authentication
let api_key = env::var("OXIRS_API_KEY")?;
```

### 3. Rate Limiting

```toml
[limits]
rate_limit_per_second = 1000
rate_limit_per_ip = 100
burst_size = 50
```

### 4. Input Validation

- Sanitize all user inputs
- Limit request sizes (<100MB)
- Validate entity/relation IRIs
- Prevent injection attacks

### 5. Network Security

- Use VPC/private networks
- Firewall rules (allow only necessary ports)
- DDoS protection (Cloudflare, AWS Shield)

---

## Troubleshooting

### High Latency

**Symptoms**: p95 latency > 500ms

**Solutions**:
1. Enable caching (`enable_caching = true`)
2. Increase batch size
3. Use GPU acceleration
4. Add more replicas

### Memory Issues

**Symptoms**: OOM errors, high memory usage

**Solutions**:
1. Reduce cache size
2. Use quantized models
3. Enable memory-efficient embeddings
4. Increase instance RAM

### GPU Errors

**Symptoms**: CUDA out of memory, slow GPU inference

**Solutions**:
1. Reduce `gpu_memory_fraction`
2. Decrease batch size
3. Use mixed precision
4. Check GPU compatibility

### Model Loading Failures

**Symptoms**: Cannot load model files

**Solutions**:
1. Check file permissions
2. Verify model format (bincode)
3. Ensure sufficient disk space
4. Check model compatibility

---

## Production Checklist

### Pre-Deployment
- [ ] Load testing completed (>10K RPS sustained)
- [ ] Security audit passed
- [ ] Backup & disaster recovery plan
- [ ] Monitoring & alerting configured
- [ ] Documentation reviewed
- [ ] TLS certificates valid (>30 days)

### Deployment
- [ ] Health checks passing
- [ ] Metrics being collected
- [ ] Logs being aggregated
- [ ] Auto-scaling configured
- [ ] Load balancer configured
- [ ] DNS records updated

### Post-Deployment
- [ ] Smoke tests passed
- [ ] Performance within SLA
- [ ] No error spikes
- [ ] Cache warm-up complete
- [ ] Team notified
- [ ] Runbook available

---

## Support & Resources

- **Documentation**: https://docs.oxirs.dev
- **GitHub**: https://github.com/cool-japan/oxirs
- **Issues**: https://github.com/cool-japan/oxirs/issues
- **Discord**: https://discord.gg/oxirs
- **Email**: support@oxirs.dev

---

**License**: MIT
**Maintainers**: OxiRS Team
**Version**: 0.1.0 (Production Ready)
