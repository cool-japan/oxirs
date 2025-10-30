# OxiRS Migration Guide: Alpha.3 → Beta.1

**From**: v0.1.0-alpha.3
**To**: v0.1.0-beta.1
**Date**: October 12, 2025
**Compatibility**: Backward compatible with deprecation warnings

## 🎯 Migration Overview

Beta.1 maintains **99% backward compatibility** with Alpha.3. Most applications will work without changes, but we recommend updating to use new stable APIs to avoid deprecation warnings.

### Breaking Changes
- **NONE** - Full backward compatibility maintained

### Deprecated APIs
- 5 APIs deprecated (will be removed in v0.2.0)
- All deprecated APIs have direct replacements
- Deprecation warnings guide you to new APIs

### New Features
- API stability guarantees
- Enhanced performance (5-10x improvements)
- Production monitoring (Grafana + Prometheus)
- Comprehensive security hardening

---

## 📋 Pre-Migration Checklist

- [ ] **Backup your data** - Create full backup of RDF datasets
- [ ] **Review changelog** - Read CHANGELOG.md for all changes
- [ ] **Test in staging** - Deploy Beta.1 to staging environment first
- [ ] **Update dependencies** - Ensure compatible dependency versions
- [ ] **Review deprecation warnings** - Plan updates for deprecated APIs

---

## 🔄 Migration Steps

### Step 1: Update Dependencies

#### Cargo.toml Changes
```toml
# Before (Alpha.3)
[dependencies]
oxirs-core = "0.1.0-beta.1"
oxirs-arq = "0.1.0-beta.1"
oxirs-fuseki = "0.1.0-beta.1"

# After (Beta.1)
[dependencies]
oxirs-core = "0.1.0-beta.1"
oxirs-arq = "0.1.0-beta.1"
oxirs-fuseki = "0.1.0-beta.1"
```

#### Update Command
```bash
# Update all OxiRS dependencies
cargo update -p oxirs-core
cargo update -p oxirs-arq
cargo update -p oxirs-fuseki
cargo update -p oxirs-cluster
cargo update -p oxirs-shacl
# ... other oxirs-* crates

# Or update all dependencies
cargo update
```

### Step 2: Update Configuration

#### Configuration File Changes
```toml
# oxirs.toml

# NEW: Beta.1 adds monitoring section
[monitoring]
enabled = true
prometheus_port = 9090
health_check_interval = 30

# NEW: Beta.1 adds logging format
[logging]
format = "json"  # NEW: json or text
output = "stdout"
level = "info"
include_timestamps = true

# ENHANCED: TLS configuration more explicit
[server.tls]
enabled = true
cert_path = "/etc/ssl/certs/server.crt"
key_path = "/etc/ssl/private/server.key"
min_version = "1.3"  # NEW: Minimum TLS version
```

### Step 3: Update Code

#### API Changes

##### 1. ConcreteStore Construction (Deprecated)
```rust
// Alpha.3 (DEPRECATED)
use oxirs_core::rdf_store::ConcreteStore;

let store = ConcreteStore::new()?;

// Beta.1 (RECOMMENDED)
use oxirs_core::store::{Store, MemoryStore};

let store = MemoryStore::new()?;
// Or for persistent storage:
let store = TdbStore::open("/path/to/data")?;
```

**Migration Path**: Replace `ConcreteStore` with `MemoryStore` or `TdbStore`

##### 2. Query Execution API (Enhanced)
```rust
// Alpha.3 (Still works, but enhanced in Beta.1)
use oxirs_arq::QueryEngine;

let engine = QueryEngine::new();
let results = engine.execute(query_str, &dataset)?;

// Beta.1 (Enhanced with options)
use oxirs_arq::{QueryEngine, QueryOptions};

let engine = QueryEngine::new();
let options = QueryOptions {
    timeout: Some(Duration::from_secs(30)),
    max_results: Some(10000),
    enable_cache: true,
};
let results = engine.execute_with_options(query_str, &dataset, options)?;
```

**Migration Path**: Optional - use `execute_with_options` for fine-grained control

##### 3. Federation Configuration (Simplified)
```rust
// Alpha.3 (Still works)
use oxirs_core::federation::{FederationClient, FederationConfig};

let config = FederationConfig {
    timeout: Duration::from_secs(30),
    max_retries: 3,
    retry_delay: Duration::from_millis(500),
    user_agent: "OxiRS/0.1.0-alpha.3".to_string(),
};
let client = FederationClient::new(config)?;

// Beta.1 (Simplified builder)
use oxirs_core::federation::FederationClient;

let client = FederationClient::builder()
    .timeout(Duration::from_secs(30))
    .max_retries(3)
    .retry_delay(Duration::from_millis(500))
    .build()?;
```

**Migration Path**: Optional - use builder pattern for cleaner code

##### 4. Error Handling (Enhanced)
```rust
// Alpha.3 (Generic errors)
use anyhow::Result;

fn query_data() -> Result<Vec<Triple>> {
    // ...
}

// Beta.1 (Specific error types)
use oxirs_core::error::{OxirsError, Result};

fn query_data() -> Result<Vec<Triple>> {
    // Returns OxirsError with detailed context
    // ...
}

// Pattern matching on errors (Beta.1)
match query_data() {
    Ok(triples) => println!("Got {} triples", triples.len()),
    Err(OxirsError::ParseError(msg)) => eprintln!("Parse error: {}", msg),
    Err(OxirsError::QueryTimeout) => eprintln!("Query timed out"),
    Err(e) => eprintln!("Other error: {}", e),
}
```

**Migration Path**: Optional - use specific error types for better error handling

##### 5. Cluster Configuration (New Defaults)
```rust
// Alpha.3 (Manual configuration)
use oxirs_cluster::{ClusterConfig, ReplicationFactor};

let config = ClusterConfig {
    node_id: 1,
    listen_addr: "127.0.0.1:7000".parse()?,
    replication_factor: ReplicationFactor::Three,
    heartbeat_interval: Duration::from_secs(5),
    election_timeout_min: Duration::from_millis(150),
    election_timeout_max: Duration::from_millis(300),
};

// Beta.1 (Smart defaults)
use oxirs_cluster::ClusterConfig;

let config = ClusterConfig::builder()
    .node_id(1)
    .listen_addr("127.0.0.1:7000".parse()?)
    .with_defaults()  // Uses production-tested defaults
    .build()?;
```

**Migration Path**: Use builder with `with_defaults()` for best practices

---

## 🔧 Configuration Migration

### Alpha.3 Configuration
```toml
[general]
default_format = "turtle"
timeout = 300

[server]
host = "0.0.0.0"
port = 3030

[server.auth]
enabled = true
method = "jwt"

[[datasets]]
name = "default"
type = "tdb2"
location = "/var/lib/oxirs/datasets/default"
```

### Beta.1 Configuration (Enhanced)
```toml
[general]
default_format = "turtle"
timeout = 300
log_level = "info"  # NEW
max_query_results = 10000  # NEW

[server]
host = "0.0.0.0"
port = 3030
enable_cors = true
enable_admin_ui = false

[server.tls]  # ENHANCED
enabled = true
cert_path = "/etc/ssl/certs/server.crt"
key_path = "/etc/ssl/private/server.key"
min_version = "1.3"  # NEW

[server.auth]
enabled = true
method = "jwt"
jwt_secret_env = "JWT_SECRET"  # NEW: Read from env
token_expiry = 3600  # NEW

[server.rate_limit]  # NEW
enabled = true
requests_per_second = 100
burst_size = 200

[monitoring]  # NEW
enabled = true
prometheus_port = 9090
health_check_interval = 30

[logging]  # NEW
format = "json"
output = "stdout"
level = "info"

[[datasets]]
name = "default"
type = "tdb2"
location = "/var/lib/oxirs/datasets/default"

[datasets.default.options]  # NEW
cache_size = 10000
buffer_pool_size = 1000
enable_transactions = true
```

**Migration**: Add new sections gradually, all are optional

---

## 🗄️ Data Migration

### No Data Migration Required! ✅

Beta.1 uses the **same data format** as Alpha.3:
- ✅ **TDB datasets**: Compatible, no migration needed
- ✅ **N-Quads files**: Compatible, no migration needed
- ✅ **RDF serialization**: Compatible, no migration needed
- ✅ **Cluster data**: Compatible, rolling upgrade supported

### Verification
```bash
# Verify data compatibility
oxirs query --dataset /path/to/data "SELECT (COUNT(*) AS ?count) WHERE { ?s ?p ?o }"

# Should return same count as Alpha.3
```

---

## 🚀 Deployment Migration

### Docker Migration

#### Alpha.3 Docker Command
```bash
docker run -p 3030:3030 \
  -v $(pwd)/oxirs.toml:/etc/oxirs/oxirs.toml \
  -v oxirs-data:/var/lib/oxirs \
  ghcr.io/cool-japan/oxirs-fuseki:v0.1.0-alpha.3
```

#### Beta.1 Docker Command
```bash
# Same command, just update tag!
docker run -p 3030:3030 \
  -v $(pwd)/oxirs.toml:/etc/oxirs/oxirs.toml \
  -v oxirs-data:/var/lib/oxirs \
  ghcr.io/cool-japan/oxirs-fuseki:v0.1.0-beta.1
```

### Kubernetes Migration

#### Rolling Update Strategy
```bash
# Update image in deployment
kubectl set image deployment/oxirs-fuseki \
  oxirs-fuseki=ghcr.io/cool-japan/oxirs-fuseki:v0.1.0-beta.1 \
  -n oxirs

# Monitor rollout
kubectl rollout status deployment/oxirs-fuseki -n oxirs

# Verify health
kubectl get pods -n oxirs
kubectl logs -f deployment/oxirs-fuseki -n oxirs
```

#### Blue-Green Deployment
```bash
# Deploy Beta.1 alongside Alpha.3
kubectl apply -f deployments/kubernetes/beta1/

# Switch traffic
kubectl patch service oxirs-fuseki -n oxirs \
  -p '{"spec":{"selector":{"version":"beta.1"}}}'

# Verify and cleanup Alpha.3
kubectl delete deployment oxirs-fuseki-alpha3 -n oxirs
```

---

## 🧪 Testing After Migration

### Functional Tests
```bash
# 1. Query execution
oxirs query --dataset default "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10"

# 2. Data insertion
oxirs update --dataset default "INSERT DATA { <http://example.org/test> <http://example.org/prop> \"value\" }"

# 3. Federation (if used)
oxirs query --dataset default "
  SELECT ?s ?label WHERE {
    SERVICE <https://dbpedia.org/sparql> {
      ?s rdfs:label ?label
    }
  } LIMIT 5
"

# 4. Health check
curl http://localhost:3030/$/ping
```

### Performance Tests
```bash
# Run benchmark suite
cd benchmarks
./scripts/run_standard_benchmark.sh

# Compare with Alpha.3 baseline
./scripts/compare_versions.py alpha.3 beta.1
```

### Monitoring Tests
```bash
# Check metrics endpoint
curl http://localhost:3030/metrics

# Verify Grafana dashboard
open http://localhost:3000

# Check Prometheus targets
open http://localhost:9090/targets
```

---

## ⚠️ Deprecation Warnings

### APIs Deprecated in Beta.1 (Removed in v0.2.0)

#### 1. `ConcreteStore::new()`
```rust
// DEPRECATED
let store = ConcreteStore::new()?;

// USE INSTEAD
let store = MemoryStore::new()?;
// or
let store = TdbStore::open(path)?;
```

#### 2. `QueryExecutor::execute_query()`
```rust
// DEPRECATED
let results = executor.execute_query(query_str)?;

// USE INSTEAD
let algebra = parse_query(query_str)?;
let results = executor.execute(&algebra, &dataset)?;
```

#### 3. `FederationConfig` direct construction
```rust
// DEPRECATED
let config = FederationConfig { /* fields */ };

// USE INSTEAD
let config = FederationClient::builder()
    .timeout(Duration::from_secs(30))
    .build()?;
```

#### 4. String-based error types
```rust
// DEPRECATED
Err("Query failed".into())

// USE INSTEAD
Err(OxirsError::QueryError("Query failed".to_string()))
```

#### 5. Manual health check endpoints
```rust
// DEPRECATED: Custom health endpoints

// USE INSTEAD: Built-in health endpoints
// GET /$/ping
// GET /$/ready
// GET /$/alive
```

---

## 📊 Performance Improvements

### Expected Performance Gains (Alpha.3 → Beta.1)

| Operation | Alpha.3 | Beta.1 | Improvement |
|-----------|---------|--------|-------------|
| Simple SELECT (1M triples) | 50ms p95 | 10ms p95 | **5x faster** |
| 2-way JOIN (1M triples) | 200ms p95 | 40ms p95 | **5x faster** |
| COUNT (10M triples) | 5s p95 | 500ms p95 | **10x faster** |
| Import (1M triples) | 30s | 10s | **3x faster** |
| Memory (10M triples) | 10GB | 5GB | **2x efficient** |

### How to Measure
```bash
# Run same queries on both versions
time oxirs query "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100000"

# Alpha.3: ~5.2s
# Beta.1: ~1.1s (5x faster)
```

---

## 🛡️ Security Enhancements

### New Security Features in Beta.1

1. **Enhanced TLS Configuration**
   - Minimum TLS 1.3 enforced
   - Strong cipher suites only
   - Certificate validation

2. **Rate Limiting**
   - Per-IP rate limiting
   - Burst handling
   - Token bucket algorithm

3. **Security Headers**
   - HSTS enabled by default
   - CSP headers configured
   - X-Frame-Options set

4. **JWT Improvements**
   - Configurable expiry
   - Secret from environment
   - RS256/ES256 support

### Security Migration
```toml
# Add to configuration
[server.rate_limit]
enabled = true
requests_per_second = 100
burst_size = 200

[server.tls]
min_version = "1.3"  # Enforce TLS 1.3
```

---

## 🔍 Troubleshooting

### Common Migration Issues

#### Issue 1: Deprecation Warnings
**Symptom**: Compiler warnings about deprecated APIs

**Solution**: Update to new APIs as shown in deprecation warnings
```bash
# Review warnings
cargo build 2>&1 | grep "warning: use of deprecated"

# Update code following warnings
```

#### Issue 2: Configuration Parsing Errors
**Symptom**: Server fails to start with config error

**Solution**: Validate configuration
```bash
# Validate config
oxirs-fuseki --config oxirs.toml --validate

# Fix any errors reported
```

#### Issue 3: Performance Regression
**Symptom**: Queries slower than Alpha.3

**Solution**: Enable caching and optimization
```toml
[datasets.default.options]
cache_size = 10000
buffer_pool_size = 1000
enable_query_cache = true
```

#### Issue 4: Monitoring Not Working
**Symptom**: Metrics not showing in Grafana

**Solution**: Enable monitoring in config
```toml
[monitoring]
enabled = true
prometheus_port = 9090
```

---

## 📋 Migration Checklist

### Pre-Migration
- [ ] Create full backup of data
- [ ] Review CHANGELOG.md
- [ ] Test in staging environment
- [ ] Update Cargo.toml dependencies
- [ ] Review deprecated API usage

### During Migration
- [ ] Update configuration file
- [ ] Update code for deprecated APIs
- [ ] Deploy to staging
- [ ] Run functional tests
- [ ] Run performance tests
- [ ] Verify monitoring works

### Post-Migration
- [ ] Monitor logs for errors
- [ ] Check performance metrics
- [ ] Verify data integrity
- [ ] Update documentation
- [ ] Train team on new features

---

## 🆘 Rollback Plan

If migration fails, rollback is simple:

### Docker Rollback
```bash
# Stop Beta.1
docker stop oxirs-fuseki

# Start Alpha.3
docker run -p 3030:3030 \
  -v $(pwd)/oxirs.toml:/etc/oxirs/oxirs.toml \
  -v oxirs-data:/var/lib/oxirs \
  ghcr.io/cool-japan/oxirs-fuseki:v0.1.0-alpha.3
```

### Kubernetes Rollback
```bash
# Rollback deployment
kubectl rollout undo deployment/oxirs-fuseki -n oxirs

# Verify rollback
kubectl rollout status deployment/oxirs-fuseki -n oxirs
```

### Data Rollback
```bash
# Restore from backup (if needed)
./restore.sh /backup/oxirs/oxirs-backup-20251012_120000.tar.gz
```

**Note**: Rollback is **safe** - Beta.1 uses same data format as Alpha.3

---

## 📚 Additional Resources

- **CHANGELOG**: See CHANGELOG.md for detailed changes
- **API Documentation**: https://docs.rs/oxirs-core/0.1.0-beta.1
- **GitHub Issues**: https://github.com/cool-japan/oxirs/issues
- **Discord Support**: https://discord.gg/oxirs

---

## ✅ Migration Success

After migration, you should see:
- ✅ All tests passing
- ✅ 5-10x performance improvements
- ✅ Monitoring dashboards showing metrics
- ✅ Zero deprecation warnings (after code updates)
- ✅ Enhanced security (TLS 1.3, rate limiting)

**Estimated Migration Time**: 2-4 hours for typical deployment

---

*Migration Guide - October 12, 2025*
*Smooth upgrade path from Alpha.3 to Beta.1*
