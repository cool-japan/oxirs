# Phase C Deployment Guide

**Production deployment of GraphRAG, DID/VC, and WASM features**

This guide covers deploying all three Phase C features in a production environment with high availability, security, and performance.

## Quick Start (Development)

```bash
# Clone and build
git clone https://github.com/cool-japan/oxirs.git
cd oxirs
cargo build --release -p oxirs-fuseki \
    --features "graphrag,did,wasm"

# Run server
./target/release/oxirs-fuseki \
    --config oxirs-phase-c.toml \
    --port 3030
```

---

## Production Deployment

### 1. GraphRAG Deployment

#### Configuration

```toml
# oxirs-phase-c.toml

[graphrag]
enabled = true

# Retrieval settings
top_k = 20
expansion_hops = 2
max_subgraph_size = 500
max_seeds = 10

# Fusion strategy
vector_weight = 0.7
keyword_weight = 0.3
fusion_strategy = "ReciprocalRankFusion"
rrf_k = 60.0

# Community detection
enable_communities = true
community_algorithm = "Louvain"
min_community_size = 2
max_community_level = 3

# Performance
cache_enabled = true
cache_size = 1000
cache_ttl_secs = 3600

# Context building
triple_format = "NaturalLanguage"
max_context_length = 4000
include_community_summaries = true

[graphrag.vector_index]
type = "hnsw"
dim = 384
m = 16
ef_construction = 200
ef_search = 50

[graphrag.embeddings]
model = "sentence-transformers/all-MiniLM-L6-v2"
batch_size = 32
device = "cuda"  # or "cpu"

[graphrag.llm]
provider = "openai"  # or "local"
model = "gpt-4-turbo"
max_tokens = 1000
temperature = 0.7
```

#### Resource Requirements

| Component | CPU | RAM | GPU | Storage |
|-----------|-----|-----|-----|---------|
| Vector Index (1M vectors) | 4 cores | 8GB | Optional | 4GB |
| Embedding Model | 2 cores | 4GB | 4GB VRAM (CUDA) | 500MB |
| Community Detection | 2 cores | 2GB | - | - |
| LLM (local) | 8 cores | 16GB | 24GB VRAM | 10GB |
| **Total (with local LLM)** | 16 cores | 30GB | 28GB VRAM | 15GB |
| **Total (with API)** | 8 cores | 14GB | 4GB VRAM | 5GB |

#### Scaling Recommendations

```yaml
# Kubernetes Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: oxirs-graphrag
spec:
  replicas: 3
  selector:
    matchLabels:
      app: oxirs-graphrag
  template:
    metadata:
      labels:
        app: oxirs-graphrag
    spec:
      containers:
      - name: oxirs
        image: oxirs:0.1.0-rc.1
        resources:
          requests:
            cpu: "4"
            memory: "8Gi"
            nvidia.com/gpu: "1"
          limits:
            cpu: "8"
            memory: "16Gi"
            nvidia.com/gpu: "1"
        env:
        - name: GRAPHRAG_CACHE_SIZE
          value: "2000"
        - name: GRAPHRAG_EXPANSION_HOPS
          value: "2"
```

---

### 2. DID/VC Deployment

#### Configuration

```toml
[did]
enabled = true

# DID Resolution
cache_enabled = true
cache_ttl_secs = 300
cache_size = 10000

# Supported methods
methods = ["key", "web"]

# did:web resolution
[did.web]
enabled = true
timeout_secs = 30
max_redirects = 3
verify_tls = true

# Verification settings
[did.verification]
strict_mode = true
check_expiration = true
check_not_before = true
require_proof_purpose = true

# Key management
[did.keystore]
type = "file"  # or "hsm", "kms"
path = "/var/lib/oxirs/keys"
encryption = "aes-256-gcm"
backup_enabled = true
backup_path = "/var/backups/oxirs/keys"

# Audit logging
[did.audit]
enabled = true
log_all_verifications = true
log_path = "/var/log/oxirs/did-audit.log"
retention_days = 90
```

#### Security Hardening

```bash
# 1. Generate secure keys
openssl rand -base64 32 > /etc/oxirs/keystore.key

# 2. Set proper permissions
chmod 600 /var/lib/oxirs/keys/*
chown oxirs:oxirs /var/lib/oxirs/keys

# 3. Enable SELinux/AppArmor
# SELinux policy for key access
semodule -i oxirs-did.pp

# 4. Use Hardware Security Module (HSM)
# For production, integrate with AWS KMS, Azure Key Vault, or YubiHSM
```

#### High Availability Setup

```yaml
# HA Configuration with Redis backing
[did.cache]
backend = "redis"
redis_url = "redis://redis-cluster:6379"
redis_cluster_enabled = true
redis_replicas = 3

[did.keystore]
type = "distributed"
backend = "vault"  # HashiCorp Vault
vault_address = "https://vault.internal:8200"
vault_path = "secret/oxirs/keys"
```

---

### 3. WASM Deployment

#### Build for Production

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build optimized WASM
cd platforms/oxirs-wasm
wasm-pack build --target web --release

# Optimize with wasm-opt
wasm-opt -O3 \
    --enable-mutable-globals \
    --enable-simd \
    pkg/oxirs_wasm_bg.wasm \
    -o pkg/oxirs_wasm_bg.wasm

# Check size
ls -lh pkg/oxirs_wasm_bg.wasm
# Target: <300KB
```

#### CDN Deployment

```bash
# Option 1: NPM + unpkg
npm publish

# Users can then use:
# https://unpkg.com/oxirs-wasm@0.1.0/pkg/oxirs_wasm.js

# Option 2: Self-hosted CDN
aws s3 cp pkg/ s3://cdn.oxirs.io/wasm/0.1.0/ --recursive
aws cloudfront create-invalidation --distribution-id E123 --paths "/wasm/*"

# Option 3: GitHub Pages
cp -r pkg/ docs/wasm/
git add docs/wasm/
git commit -m "Deploy WASM v0.1.0"
git push origin gh-pages
```

#### Browser Caching Strategy

```html
<!-- index.html -->
<script type="module">
    import init from 'https://cdn.oxirs.io/wasm/0.1.0/oxirs_wasm.js';

    // Service Worker for offline support
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/sw.js');
    }

    // Initialize with caching
    const wasmModule = await init();

    // Store in IndexedDB for offline use
    const db = await openDB('oxirs-wasm', 1, {
        upgrade(db) {
            db.createObjectStore('graphs');
        }
    });

    // Cache frequently used graphs
    await db.put('graphs', graphData, 'research-2025-q1');
</script>
```

```javascript
// sw.js - Service Worker
self.addEventListener('fetch', (event) => {
    if (event.request.url.includes('/wasm/')) {
        event.respondWith(
            caches.open('oxirs-wasm-v1').then((cache) => {
                return cache.match(event.request).then((response) => {
                    return response || fetch(event.request).then((response) => {
                        cache.put(event.request, response.clone());
                        return response;
                    });
                });
            })
        );
    }
});
```

---

## Monitoring & Observability

### Prometheus Metrics

```toml
[monitoring]
enabled = true
metrics_endpoint = "/metrics"
port = 9090

[monitoring.graphrag]
track_latency = true
track_cache_hit_rate = true
track_community_count = true
track_sources_used = true

[monitoring.did]
track_verifications = true
track_cache_hit_rate = true
track_resolution_failures = true

[monitoring.wasm]
track_downloads = true
track_init_time = true
```

### Grafana Dashboards

```bash
# Import pre-built dashboards
curl -X POST http://grafana:3000/api/dashboards/import \
  -H "Content-Type: application/json" \
  -d @monitoring/dashboards/oxirs-phase-c.json
```

### Logging Configuration

```toml
[logging]
level = "info"
format = "json"
output = "stdout"

[logging.graphrag]
level = "debug"
log_queries = true
log_retrieval_scores = true
log_community_detection = false

[logging.did]
level = "info"
log_verifications = true
log_cache_operations = false
audit_log_path = "/var/log/oxirs/did-audit.jsonl"
```

---

## Performance Tuning

### GraphRAG Optimization

```toml
[graphrag.performance]
# Vector search
vector_index_parallel = true
vector_batch_size = 100

# SPARQL expansion
expansion_parallel = true
expansion_timeout_ms = 5000
expansion_max_results = 1000

# Community detection
community_cache_enabled = true
community_cache_size = 500
skip_communities_for_small_graphs = true
small_graph_threshold = 50

# LLM
llm_timeout_ms = 10000
llm_retry_attempts = 3
llm_batch_requests = false
```

### DID/VC Optimization

```toml
[did.performance]
# Caching
cache_backend = "redis"  # or "in-memory"
cache_ttl_secs = 300
cache_size = 10000
prefetch_enabled = true

# Resolution
parallel_resolution = true
resolution_timeout_ms = 5000
max_concurrent_resolutions = 100

# Verification
batch_verification = true
batch_size = 50
signature_verification_threads = 4
```

### WASM Optimization

```toml
[wasm]
# Compression
serve_compressed = true
compression_level = 9
brotli_enabled = true

# Caching
cache_control = "public, max-age=31536000, immutable"
enable_etag = true

# Performance
lazy_compilation = true
streaming_instantiation = true
```

---

## Security Configuration

### TLS/HTTPS Setup

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name api.pharma-research.example;

    ssl_certificate /etc/ssl/certs/pharma.crt;
    ssl_certificate_key /etc/ssl/private/pharma.key;
    ssl_protocols TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # GraphRAG endpoint
    location /api/graphrag {
        proxy_pass http://oxirs-server:3030;
        proxy_set_header X-Real-IP $remote_addr;

        # Rate limiting
        limit_req zone=graphrag burst=5 nodelay;
    }

    # DID/VC endpoints
    location /api/did {
        proxy_pass http://oxirs-server:3030;
        limit_req zone=did burst=100;
    }

    # WASM assets with caching
    location /wasm/ {
        root /usr/share/nginx/html;
        expires 1y;
        add_header Cache-Control "public, immutable";
        gzip_static on;
        brotli_static on;
    }
}

# Rate limiting zones
http {
    limit_req_zone $binary_remote_addr zone=graphrag:10m rate=5r/s;
    limit_req_zone $binary_remote_addr zone=did:10m rate=100r/s;
}
```

### CORS Configuration

```toml
[security.cors]
enabled = true
allowed_origins = [
    "https://research.pharma-corp.com",
    "https://app.pharma-corp.com"
]
allowed_methods = ["GET", "POST", "OPTIONS"]
allowed_headers = ["Content-Type", "Authorization"]
max_age = 3600
```

---

## Backup & Recovery

### Database Backup

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR=/var/backups/oxirs/$(date +%Y%m%d)
mkdir -p $BACKUP_DIR

# Backup RDF store
oxirs-fuseki dump \
    --dataset research \
    --output $BACKUP_DIR/rdf-dump.nq.gz

# Backup keystore
tar czf $BACKUP_DIR/keystore.tar.gz \
    /var/lib/oxirs/keys/

# Backup audit logs
cp /var/log/oxirs/did-audit.jsonl $BACKUP_DIR/

# Upload to S3
aws s3 sync $BACKUP_DIR s3://backups.pharma/oxirs/$(date +%Y%m%d)/

echo "Backup completed: $BACKUP_DIR"
```

### Disaster Recovery

```bash
#!/bin/bash
# restore.sh

BACKUP_DATE=$1
BACKUP_DIR=s3://backups.pharma/oxirs/$BACKUP_DATE

# Download backup
aws s3 sync $BACKUP_DIR /tmp/restore/

# Restore RDF store
oxirs-fuseki load \
    --dataset research \
    --input /tmp/restore/rdf-dump.nq.gz

# Restore keystore
tar xzf /tmp/restore/keystore.tar.gz -C /

# Restart services
systemctl restart oxirs-fuseki
systemctl restart oxirs-graphrag

echo "Restore completed from $BACKUP_DATE"
```

---

## Health Checks

### Kubernetes Probes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 3030
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /ready
    port: 3030
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 2

startupProbe:
  httpGet:
    path: /startup
    port: 3030
  initialDelaySeconds: 0
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 30
```

### Health Check Endpoints

```rust
// Health check implementation
use axum::{Json, http::StatusCode};

async fn health_check(
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<HealthStatus>) {
    let mut status = HealthStatus {
        healthy: true,
        components: HashMap::new(),
    };

    // Check GraphRAG
    status.components.insert("graphrag".to_string(), ComponentHealth {
        healthy: state.graphrag.is_healthy().await,
        message: None,
    });

    // Check DID resolver
    status.components.insert("did_resolver".to_string(), ComponentHealth {
        healthy: state.did_resolver.is_healthy().await,
        message: None,
    });

    // Check database
    status.components.insert("database".to_string(), ComponentHealth {
        healthy: state.db.ping().await.is_ok(),
        message: None,
    });

    let all_healthy = status.components.values().all(|c| c.healthy);
    status.healthy = all_healthy;

    let status_code = if all_healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (status_code, Json(status))
}
```

---

## Performance Benchmarking

### Load Testing

```bash
# Install k6
brew install k6

# Run load test
k6 run --vus 10 --duration 30s scripts/graphrag_load_test.js
```

```javascript
// scripts/graphrag_load_test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
    vus: 10,
    duration: '30s',
    thresholds: {
        http_req_duration: ['p(95)<500'],  // 95% under 500ms
        http_req_failed: ['rate<0.01'],     // <1% failures
    },
};

export default function() {
    const payload = JSON.stringify({
        question: 'Which compounds target ACE2?'
    });

    const params = {
        headers: { 'Content-Type': 'application/json' },
    };

    const res = http.post('http://localhost:3030/api/graphrag', payload, params);

    check(res, {
        'status is 200': (r) => r.status === 200,
        'response has answer': (r) => JSON.parse(r.body).answer !== undefined,
        'latency < 500ms': (r) => r.timings.duration < 500,
    });

    sleep(1);
}
```

### Benchmark Results (Expected)

```
GraphRAG Query (1M triples):
  ✓ p50: 250ms
  ✓ p95: 450ms
  ✓ p99: 800ms
  ✓ Success rate: 99.9%

DID Resolution (cached):
  ✓ p50: 0.5ms
  ✓ p95: 2ms
  ✓ p99: 5ms
  ✓ Cache hit rate: 95%

VC Verification:
  ✓ p50: 5ms
  ✓ p95: 15ms
  ✓ p99: 30ms
  ✓ Success rate: 99.99%
```

---

## Troubleshooting

### Common Issues

#### GraphRAG Slow Queries

```bash
# Check vector index performance
curl http://localhost:9090/metrics | grep graphrag_vector_search_duration

# Diagnose:
# 1. HNSW parameters too high (ef_search)
# 2. Large subgraphs (reduce expansion_hops)
# 3. Too many communities (increase min_community_size)

# Fix:
# Reduce expansion_hops from 2 to 1
# Increase min_community_size from 2 to 5
# Reduce ef_search from 50 to 20
```

#### DID Resolution Failures

```bash
# Check logs
tail -f /var/log/oxirs/did-audit.log | grep "resolution_failed"

# Common causes:
# 1. did:web network timeout
# 2. Invalid DID format
# 3. Missing verification method

# Fix:
# Increase timeout: did.web.timeout_secs = 60
# Enable retry: did.web.retry_attempts = 3
# Fall back to cached: did.cache.serve_stale_on_error = true
```

#### WASM Loading Errors

```javascript
// Check browser console
console.error('WASM error:', error);

// Common causes:
// 1. CORS issues
// 2. WASM file too large
// 3. Browser compatibility

// Fix:
// Add CORS headers
// Use streaming instantiation
// Check browser support: WebAssembly.validate(bytes)
```

---

## Migration Guide

### From Existing OxiRS Deployment

```bash
# 1. Backup current data
./scripts/backup.sh

# 2. Update Cargo.toml
cargo add oxirs-graphrag@0.1.0-rc.1
cargo add oxirs-did@0.1.0-rc.1
cargo add oxirs-wasm@0.1.0-rc.1

# 3. Update configuration
cat >> oxirs.toml << EOF
[graphrag]
enabled = true
# ... configuration ...

[did]
enabled = true
# ... configuration ...
EOF

# 4. Rebuild
cargo build --release --features "graphrag,did,wasm"

# 5. Test in staging
cargo test --workspace

# 6. Deploy
systemctl restart oxirs-fuseki
```

### Feature Flags

```toml
# Incremental adoption via feature flags

# Stage 1: Enable DID/VC only
cargo build --release --features "did"

# Stage 2: Add GraphRAG
cargo build --release --features "did,graphrag"

# Stage 3: Add WASM
cargo build --release --features "did,graphrag,wasm"

# Full deployment
cargo build --release --all-features
```

---

## Production Checklist

### Pre-Deployment

- [ ] All tests passing (cargo test --workspace)
- [ ] Zero warnings (cargo clippy --all)
- [ ] Security audit completed
- [ ] Performance benchmarks run
- [ ] Documentation reviewed
- [ ] Backup strategy defined
- [ ] Monitoring configured
- [ ] Health checks verified
- [ ] TLS certificates installed
- [ ] Rate limiting configured

### GraphRAG Specific

- [ ] Vector index built and tested
- [ ] Embedding model loaded
- [ ] LLM API key configured (if using API)
- [ ] SPARQL engine connection verified
- [ ] Cache Redis instance running
- [ ] Community detection parameters tuned

### DID/VC Specific

- [ ] Keystore encrypted and backed up
- [ ] DID methods tested (key, web)
- [ ] Audit logging configured
- [ ] Compliance requirements reviewed
- [ ] Key rotation plan defined
- [ ] HSM integration tested (if applicable)

### WASM Specific

- [ ] WASM binary optimized (<300KB)
- [ ] CDN configured and tested
- [ ] Browser compatibility verified
- [ ] Service Worker deployed
- [ ] CORS headers configured
- [ ] TypeScript types published

---

## Scaling Guidelines

### Horizontal Scaling

```yaml
# Load balancer configuration
upstream oxirs_cluster {
    least_conn;
    server oxirs-1:3030 weight=1;
    server oxirs-2:3030 weight=1;
    server oxirs-3:3030 weight=1;

    # Health check
    check interval=3000 rise=2 fall=3 timeout=1000;
}

# Session affinity for GraphRAG caching
sticky cookie oxirs_route expires=1h;
```

### Vertical Scaling

| Traffic Level | CPU | RAM | GPU | Replicas |
|---------------|-----|-----|-----|----------|
| <100 QPS | 8 cores | 16GB | 8GB | 2 |
| 100-500 QPS | 16 cores | 32GB | 16GB | 4 |
| 500-1K QPS | 32 cores | 64GB | 24GB | 8 |
| >1K QPS | 64 cores | 128GB | 48GB | 16 |

---

## Cost Estimation

### AWS Deployment (monthly, us-east-1)

| Component | Instance | Cost |
|-----------|----------|------|
| OxiRS Server (x2) | r6i.2xlarge (8vCPU, 64GB) | $800 |
| GPU (GraphRAG) | g5.xlarge (4vCPU, 16GB, 1xA10G) | $600 |
| Redis Cache | cache.r6g.large (2vCPU, 13GB) | $180 |
| RDS (PostgreSQL) | db.r6i.large (2vCPU, 16GB) | $250 |
| S3 (1TB) | Standard | $23 |
| Data Transfer (5TB) | - | $450 |
| **Total** | - | **~$2,300/month** |

### Cost Optimization

```toml
# Use local LLM instead of API
[graphrag.llm]
provider = "local"
model_path = "/models/llama-2-7b"
# Saves: ~$500-1000/month in API costs

# Use in-memory cache instead of Redis
[did.cache]
backend = "in-memory"
# Saves: ~$180/month

# Optimize WASM CDN
# Use GitHub Pages (free) or Cloudflare Pages (free)
# Saves: ~$50/month vs AWS CloudFront
```

---

## Compliance & Audit

### GDPR Compliance

```rust
// Implement right to erasure
async fn delete_user_data(user_did: &Did) -> Result<(), DidError> {
    // 1. Revoke credentials
    revocation_list.add(user_did, RevocationReason::UserRequest).await?;

    // 2. Delete from cache
    cache.delete_all_for_did(user_did).await?;

    // 3. Audit log
    audit_log::record("user_data_deleted", user_did).await?;

    Ok(())
}

// Data minimization
let subject = CredentialSubject::new(Some(&user_did))
    .with_claim("role", "researcher")  // ✓ Necessary
    .with_claim("age_range", "30-40")  // ✓ Generalized
    // NOT: .with_claim("full_name", "...")  // ✗ PII
    // NOT: .with_claim("exact_age", 35)     // ✗ Too specific
```

### SOC 2 Compliance

```toml
[compliance.soc2]
audit_all_accesses = true
encrypt_at_rest = true
encrypt_in_transit = true
key_rotation_days = 90
access_log_retention_days = 365
vulnerability_scan_enabled = true
```

---

## Migration Path

### Phase 1: DID/VC (Week 1-2)

1. Deploy oxirs-did in parallel with existing system
2. Start issuing VCs for new datasets
3. Gradually verify existing datasets
4. Monitor performance impact

### Phase 2: GraphRAG (Week 3-4)

1. Build vector index from existing RDF data
2. Train/load embedding models
3. Configure LLM integration
4. Test GraphRAG queries in staging
5. Gradual rollout with A/B testing

### Phase 3: WASM (Week 5-6)

1. Build and optimize WASM binary
2. Deploy to CDN
3. Update browser clients
4. Enable offline support
5. Monitor browser performance

---

## Support & Resources

### Documentation

- [GraphRAG README](../ai/oxirs-graphrag/README.md)
- [DID/VC README](../security/oxirs-did/README.md)
- [WASM README](../platforms/oxirs-wasm/README.md)
- [Integration Guide](PHASE_C_INTEGRATION.md)

### Examples

- `cargo run -p oxirs-did --example simple_vc`
- `cargo run -p oxirs-graphrag --example research_assistant`
- Open `platforms/oxirs-wasm/examples/browser_demo.html`

### Community

- GitHub Issues: https://github.com/cool-japan/oxirs/issues
- Discord: https://discord.gg/oxirs (placeholder)
- Documentation: https://docs.rs/oxirs-graphrag, oxirs-did, oxirs-wasm

---

## Production Success Criteria

### Week 1-2 (DID/VC)

- [ ] 100% of new datasets have VCs
- [ ] VC verification success rate >99.9%
- [ ] Verification latency <10ms (p95)
- [ ] Zero security incidents

### Week 3-4 (GraphRAG)

- [ ] GraphRAG latency <500ms (p95)
- [ ] User satisfaction >85% (vs baseline)
- [ ] Answer accuracy >90% (human eval)
- [ ] Cache hit rate >80%

### Week 5-6 (WASM)

- [ ] WASM load time <200ms (p95)
- [ ] Browser support >95% (Chrome, Firefox, Safari, Edge)
- [ ] Offline functionality working
- [ ] Zero WASM runtime errors

---

## Summary

This deployment guide provides:

✅ **Production configurations** for all three Phase C features
✅ **Security hardening** with TLS, HSM, audit logging
✅ **Scalability** with horizontal scaling and caching
✅ **Monitoring** with Prometheus and Grafana
✅ **Disaster recovery** with backup/restore procedures
✅ **Compliance** with GDPR and SOC 2 considerations
✅ **Migration path** for incremental adoption

**Estimated deployment time**: 4-6 weeks for full production rollout
**Required expertise**: DevOps (Kubernetes), Security (cryptography), AI/ML (embeddings)
**Budget**: $2,000-5,000/month (AWS, moderate traffic)

For support, consult the documentation or open an issue on GitHub.
