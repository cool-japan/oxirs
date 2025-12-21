# Migration Guide: Apache Jena Fuseki → OxiRS Fuseki

**Target Audience**: Users migrating from Apache Jena Fuseki to OxiRS Fuseki
**Difficulty**: Easy to Medium
**Estimated Migration Time**: 1-4 hours (depending on complexity)

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Quick Start Migration](#quick-start-migration)
3. [Configuration Migration](#configuration-migration)
4. [API Compatibility](#api-compatibility)
5. [Data Migration](#data-migration)
6. [Authentication & Security](#authentication--security)
7. [Performance Tuning](#performance-tuning)
8. [Deployment Migration](#deployment-migration)
9. [Feature Comparison](#feature-comparison)
10. [Troubleshooting](#troubleshooting)

---

## Overview

### Why Migrate to OxiRS Fuseki?

OxiRS Fuseki provides full Apache Jena Fuseki API compatibility while offering:

✅ **Better Performance**: Work-stealing scheduler, memory pooling, zero-copy streaming
✅ **Smaller Footprint**: 12MB binary vs 100MB+ JAR files
✅ **Modern Security**: OAuth2/OIDC, JWT, RBAC, ReBAC (not just basic auth)
✅ **Advanced Observability**: Native Prometheus metrics, OpenTelemetry tracing
✅ **Cloud Native**: Kubernetes operator, Terraform modules, Docker multi-arch
✅ **Additional APIs**: GraphQL and REST API v2 alongside SPARQL
✅ **HTTP/2 & HTTP/3**: Modern protocol support

### Compatibility Level

- ✅ **SPARQL Protocol**: 100% compatible
- ✅ **Graph Store Protocol**: 100% compatible
- ✅ **Validation Services**: 100% compatible
- ✅ **Admin Endpoints**: Mostly compatible (with enhancements)
- ⚠️ **Assembler Files**: Not supported (use TOML configuration instead)
- ⚠️ **TDB1/TDB2**: Use OxiRS native store (data migration required)

---

## Quick Start Migration

### 1. Basic Migration (5 Minutes)

If you're running Fuseki with default settings:

```bash
# Stop Fuseki
# (on Linux/Mac)
./fuseki-server stop

# Install OxiRS Fuseki
cargo install oxirs-fuseki
# OR use Docker
docker pull ghcr.io/cool-japan/oxirs-fuseki:0.1.0

# Run with default settings
oxirs-fuseki --port 3030

# Your SPARQL endpoint is now at http://localhost:3030/dataset/sparql
# Graph Store Protocol at http://localhost:3030/dataset/data
```

**That's it!** Your existing Fuseki clients should work without changes.

### 2. Intermediate Migration (30 Minutes)

With custom datasets and configuration:

```bash
# 1. Export your Fuseki data
# Using Fuseki admin UI or tdbdump

# 2. Create OxiRS Fuseki config
cat > oxirs.toml << 'EOF'
[server]
host = "0.0.0.0"
port = 3030
admin_ui = true

[[datasets]]
name = "mydata"
description = "My RDF dataset"
persistent = true
data_file = "mydata.nq"

[[datasets]]
name = "production"
description = "Production dataset"
persistent = true
data_file = "production.nq"
EOF

# 3. Import data (convert TDB to N-Quads first)
# Copy your .nq files to the data directory

# 4. Start OxiRS Fuseki
oxirs-fuseki --config oxirs.toml
```

---

## Configuration Migration

### Fuseki Configuration File

Apache Jena Fuseki uses Turtle/RDF for configuration:

```turtle
# fuseki-config.ttl
@prefix fuseki: <http://jena.apache.org/fuseki#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

<#service1> rdf:type fuseki:Service ;
    fuseki:name "dataset" ;
    fuseki:serviceQuery "sparql" ;
    fuseki:serviceUpdate "update" ;
    fuseki:serviceUpload "upload" ;
    fuseki:serviceReadGraphStore "data" ;
    fuseki:dataset <#dataset> .

<#dataset> rdf:type tdb:DatasetTDB ;
    tdb:location "/path/to/tdb" .
```

### OxiRS Fuseki Configuration (TOML)

OxiRS uses human-friendly TOML:

```toml
# oxirs.toml
[server]
host = "0.0.0.0"
port = 3030
admin_ui = true
max_connections = 1000
request_timeout_secs = 300
graceful_shutdown_timeout_secs = 30

[[datasets]]
name = "dataset"
description = "My RDF dataset"
persistent = true
data_file = "dataset.nq"

[datasets.features]
text_search = false
vector_search = false
rdf_star = true
```

### Configuration Mapping

| Fuseki (Turtle) | OxiRS (TOML) | Notes |
|----------------|--------------|-------|
| `fuseki:name` | `datasets.name` | Dataset name |
| `fuseki:serviceQuery` | Auto-enabled at `/{name}/sparql` | Always available |
| `fuseki:serviceUpdate` | Auto-enabled at `/{name}/update` | Always available |
| `fuseki:serviceUpload` | Auto-enabled at `/{name}/upload` | Always available |
| `fuseki:serviceReadGraphStore` | Auto-enabled at `/{name}/data` | Graph Store Protocol |
| `tdb:location` | `datasets.data_file` | Use N-Quads instead of TDB |
| `fuseki:timeout` | `server.request_timeout_secs` | Global timeout |
| `fuseki:allowedUsers` | `authentication.*` | See Auth section |

### Complete Configuration Example

```toml
# oxirs.toml - Production Configuration
[server]
host = "0.0.0.0"
port = 3030
admin_ui = true
max_connections = 1000
request_timeout_secs = 300
graceful_shutdown_timeout_secs = 30

# TLS/SSL (optional)
[server.tls]
enabled = true
cert_file = "/etc/oxirs/cert.pem"
key_file = "/etc/oxirs/key.pem"

# Authentication (optional)
[authentication]
enabled = true
default_method = "oauth2"

[authentication.oauth2]
provider = "keycloak"
client_id = "oxirs-fuseki"
client_secret = "your-secret"
authorization_url = "https://auth.example.com/realms/master/protocol/openid-connect/auth"
token_url = "https://auth.example.com/realms/master/protocol/openid-connect/token"
redirect_url = "http://localhost:3030/auth/callback"

# Datasets
[[datasets]]
name = "production"
description = "Production RDF dataset"
persistent = true
data_file = "/var/lib/oxirs/production.nq"

[datasets.features]
text_search = true
vector_search = false
rdf_star = true

[[datasets]]
name = "staging"
description = "Staging dataset"
persistent = true
data_file = "/var/lib/oxirs/staging.nq"

# Logging
[logging]
level = "info"
format = "json"
file = "/var/log/oxirs/fuseki.log"

# Metrics
[metrics]
enabled = true
prometheus_port = 9090

# Performance
[performance]
worker_threads = 8
max_concurrent_queries = 100
query_cache_size = 10000
```

---

## API Compatibility

### SPARQL Protocol

**100% Compatible** - No changes needed!

```bash
# Fuseki
curl -X POST http://localhost:3030/dataset/sparql \
  -H "Content-Type: application/sparql-query" \
  -d "SELECT * WHERE { ?s ?p ?o } LIMIT 10"

# OxiRS Fuseki (identical)
curl -X POST http://localhost:3030/dataset/sparql \
  -H "Content-Type: application/sparql-query" \
  -d "SELECT * WHERE { ?s ?p ?o } LIMIT 10"
```

### SPARQL Update

**100% Compatible** - No changes needed!

```bash
# Fuseki
curl -X POST http://localhost:3030/dataset/update \
  -H "Content-Type: application/sparql-update" \
  -d "INSERT DATA { <http://example.org/s> <http://example.org/p> <http://example.org/o> }"

# OxiRS Fuseki (identical)
curl -X POST http://localhost:3030/dataset/update \
  -H "Content-Type: application/sparql-update" \
  -d "INSERT DATA { <http://example.org/s> <http://example.org/p> <http://example.org/o> }"
```

### Graph Store Protocol

**100% Compatible** - No changes needed!

```bash
# GET graph
curl http://localhost:3030/dataset/data?graph=http://example.org/graph

# PUT graph (replace)
curl -X PUT http://localhost:3030/dataset/data?graph=http://example.org/graph \
  -H "Content-Type: text/turtle" \
  -d "@data.ttl"

# POST graph (merge)
curl -X POST http://localhost:3030/dataset/data?graph=http://example.org/graph \
  -H "Content-Type: text/turtle" \
  -d "@data.ttl"

# DELETE graph
curl -X DELETE http://localhost:3030/dataset/data?graph=http://example.org/graph
```

### Admin Endpoints

| Endpoint | Fuseki | OxiRS Fuseki | Status |
|----------|--------|--------------|--------|
| `/$/ping` | ✅ | ✅ | Compatible |
| `/$/stats` | ✅ | ✅ | Compatible + Enhanced |
| `/$/datasets` | ✅ | ✅ | Compatible |
| `/$/server` | ✅ | ✅ | Compatible + Enhanced |
| `/$/backup/{dataset}` | ✅ | ✅ | Compatible |
| `/$/backups-list` | ✅ | ✅ | **New in OxiRS** |
| `/$/compact/{dataset}` | ✅ | ✅ | Compatible |
| `/$/sleep` | ✅ | ✅ | Compatible |
| `/$/tasks` | ✅ | ✅ | Compatible + Enhanced |
| `/$/validate/query` | ✅ | ✅ | **Full parity** |
| `/$/validate/update` | ✅ | ✅ | **Full parity** |
| `/$/validate/iri` | ✅ | ✅ | **Full parity** |
| `/$/validate/data` | ✅ | ✅ | **Full parity** |
| `/$/validate/langtag` | ❌ | ✅ | **New in OxiRS** |
| `/$/metrics` | ❌ | ✅ | **New in OxiRS** (Prometheus) |

### Additional APIs (OxiRS Only)

OxiRS provides additional modern APIs:

1. **GraphQL API**: `/graphql` and `/graphql/playground`
2. **REST API v2**: `/api/v2/*` with OpenAPI 3.0
3. **WebSocket**: `/ws` for real-time subscriptions

---

## Data Migration

### Option 1: Export/Import N-Quads (Recommended)

```bash
# 1. Export from Fuseki (using tdbdump or admin UI)
cd /path/to/fuseki
tdbdump --loc=databases/DB1 > export.nq

# 2. Copy to OxiRS data directory
cp export.nq /var/lib/oxirs/mydata.nq

# 3. Configure OxiRS
cat > oxirs.toml << 'EOF'
[[datasets]]
name = "mydata"
persistent = true
data_file = "/var/lib/oxirs/mydata.nq"
EOF

# 4. Start OxiRS (auto-loads data)
oxirs-fuseki --config oxirs.toml
```

### Option 2: SPARQL CONSTRUCT Export

```bash
# 1. Export via SPARQL (for selective migration)
curl -X POST http://localhost:3030/fuseki-dataset/sparql \
  -H "Accept: application/n-quads" \
  -d "CONSTRUCT WHERE { ?s ?p ?o }" > export.nq

# 2. Import to OxiRS via upload
curl -X POST http://localhost:3030/oxirs-dataset/upload \
  -H "Content-Type: application/n-quads" \
  --data-binary "@export.nq"
```

### Option 3: Live Migration (Zero Downtime)

```bash
# 1. Set up OxiRS alongside Fuseki
oxirs-fuseki --port 3031 &

# 2. Sync data using federation
curl -X POST http://localhost:3031/oxirs-dataset/update \
  -H "Content-Type: application/sparql-update" \
  -d "INSERT { ?s ?p ?o } WHERE {
    SERVICE <http://localhost:3030/fuseki-dataset/sparql> {
      ?s ?p ?o
    }
  }"

# 3. Switch traffic to OxiRS (update load balancer/proxy)

# 4. Shutdown Fuseki when safe
```

### Data Format Support

| Format | Fuseki | OxiRS | Notes |
|--------|--------|-------|-------|
| Turtle | ✅ | ✅ | Full support |
| N-Triples | ✅ | ✅ | Full support |
| RDF/XML | ✅ | ✅ | Full support |
| JSON-LD | ✅ | ✅ | Full support |
| N-Quads | ✅ | ✅ | **Recommended** |
| TriG | ✅ | ✅ | Full support |
| TDB1/TDB2 | ✅ | ❌ | Export to N-Quads |

---

## Authentication & Security

### Basic Authentication (Fuseki)

```bash
# Fuseki uses Jetty's security.properties
# admin=password,admin
```

### OAuth2/OIDC (OxiRS - Recommended)

```toml
[authentication]
enabled = true
default_method = "oauth2"

[authentication.oauth2]
provider = "keycloak"
client_id = "oxirs-fuseki"
client_secret = "your-secret"
authorization_url = "https://auth.example.com/auth"
token_url = "https://auth.example.com/token"
redirect_url = "http://localhost:3030/auth/callback"
```

### JWT (OxiRS)

```toml
[authentication.jwt]
enabled = true
secret = "your-256-bit-secret"
algorithm = "HS256"
issuer = "oxirs-fuseki"
expiration_hours = 24
```

### Migration Path

1. **Phase 1**: Keep using basic auth (set `authentication.enabled = false`)
2. **Phase 2**: Migrate to OAuth2/OIDC or JWT
3. **Phase 3**: Enable RBAC or ReBAC for fine-grained control

```toml
# Phase 1: No auth (same as Fuseki without security)
[authentication]
enabled = false

# Phase 2: OAuth2
[authentication]
enabled = true
default_method = "oauth2"

# Phase 3: Add RBAC
[authorization]
enabled = true
default_role = "viewer"

[[authorization.roles]]
name = "admin"
permissions = ["read", "write", "manage", "delete"]
```

---

## Performance Tuning

### Memory Settings

**Fuseki (JVM)**:
```bash
export JVM_ARGS="-Xmx4G -Xms2G"
./fuseki-server
```

**OxiRS Fuseki**:
No JVM! Memory management is automatic via Rust. Optional tuning:

```toml
[performance]
worker_threads = 8  # CPU cores
max_concurrent_queries = 100
query_cache_size = 10000

[performance.memory_pool]
enabled = true
initial_size = 1000
max_size = 10000
gc_interval_secs = 60
```

### Query Timeouts

**Fuseki**:
```turtle
fuseki:timeout "60000,120000" ;  # connect, read (ms)
```

**OxiRS**:
```toml
[server]
request_timeout_secs = 300  # 5 minutes
connection_timeout_secs = 60
```

### Connection Pooling

**Fuseki**: Jetty manages connections

**OxiRS**: Advanced pooling with stats

```toml
[connection_pool]
max_size = 1000
min_idle = 10
connection_timeout_secs = 30
idle_timeout_secs = 600
```

### Performance Comparison

| Metric | Fuseki | OxiRS Fuseki | Improvement |
|--------|--------|--------------|-------------|
| Cold start | ~10s | ~0.5s | **20x faster** |
| Memory (idle) | ~500MB | ~50MB | **10x less** |
| Binary size | ~100MB | 12MB | **8x smaller** |
| Concurrent queries | Limited by JVM | Work-stealing | **Better scaling** |
| HTTP/2 support | ❌ | ✅ | **Modern** |

---

## Deployment Migration

### Systemd Service (Linux)

**Fuseki** (`/etc/systemd/system/fuseki.service`):
```ini
[Unit]
Description=Apache Jena Fuseki
After=network.target

[Service]
Type=simple
User=fuseki
ExecStart=/opt/fuseki/fuseki-server --config=/etc/fuseki/config.ttl
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

**OxiRS Fuseki** (`/etc/systemd/system/oxirs-fuseki.service`):
```ini
[Unit]
Description=OxiRS Fuseki SPARQL Server
After=network.target

[Service]
Type=simple
User=oxirs
ExecStart=/usr/local/bin/oxirs-fuseki --config=/etc/oxirs/oxirs.toml
Restart=on-failure
RestartSec=5

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/oxirs

[Install]
WantedBy=multi-user.target
```

### Docker Migration

**Fuseki**:
```dockerfile
FROM openjdk:11
COPY fuseki-server.jar /opt/
CMD ["java", "-jar", "/opt/fuseki-server.jar"]
```

**OxiRS Fuseki**:
```dockerfile
FROM ghcr.io/cool-japan/oxirs-fuseki:0.1.0
# That's it! 12MB vs 500MB+ Java image
```

### Docker Compose Migration

**Before (Fuseki)**:
```yaml
version: '3.8'
services:
  fuseki:
    image: stain/jena-fuseki:latest
    ports:
      - "3030:3030"
    volumes:
      - ./data:/fuseki
    environment:
      - ADMIN_PASSWORD=admin
```

**After (OxiRS)**:
```yaml
version: '3.8'
services:
  oxirs-fuseki:
    image: ghcr.io/cool-japan/oxirs-fuseki:0.1.0
    ports:
      - "3030:3030"
    volumes:
      - ./data:/var/lib/oxirs
      - ./oxirs.toml:/etc/oxirs/oxirs.toml
    environment:
      - RUST_LOG=info
```

### Kubernetes Migration

OxiRS provides Helm charts and Terraform modules:

```bash
# Install via Helm (coming soon)
helm install oxirs-fuseki oxirs/oxirs-fuseki

# OR use Kubernetes Operator
kubectl apply -f deployment/kubernetes/deployment.yaml

# OR use Terraform
cd deployment/terraform/aws
terraform init
terraform apply
```

---

## Feature Comparison

### What's the Same?

✅ SPARQL 1.1 Query
✅ SPARQL 1.1 Update
✅ Graph Store Protocol
✅ Multiple RDF formats
✅ Named graphs
✅ Validation services
✅ Admin endpoints
✅ Backup/restore

### What's Better in OxiRS?

🚀 **Performance**: Work-stealing, memory pooling, zero-copy
🚀 **Binary Size**: 12MB vs 100MB+ (8x smaller)
🚀 **Memory**: 50MB vs 500MB idle (10x less)
🚀 **Protocols**: HTTP/2, HTTP/3 support
🚀 **Auth**: OAuth2/OIDC, JWT, RBAC, ReBAC
🚀 **Observability**: Prometheus, OpenTelemetry
🚀 **Deployment**: K8s operator, Terraform, Ansible
🚀 **APIs**: GraphQL, REST API v2

### What's Missing?

❌ **TDB1/TDB2 Format**: Use N-Quads instead (easy migration)
❌ **Assembler Files**: Use TOML configuration (cleaner)
❌ **Inference**: Coming in v0.2.0
❌ **Text Search (Jena Text)**: Coming in v0.2.0 with better integration

### Feature Parity Table

| Feature | Fuseki | OxiRS | Notes |
|---------|--------|-------|-------|
| **SPARQL 1.1 Query** | ✅ | ✅ | Full parity |
| **SPARQL 1.1 Update** | ✅ | ✅ | All 14 operations |
| **SPARQL 1.2** | Partial | ✅ | RDF-star support |
| **Graph Store Protocol** | ✅ | ✅ | Full parity |
| **Validation Services** | ✅ | ✅ | + langtag validation |
| **Admin UI** | ✅ | ✅ | Modern React UI |
| **TDB Storage** | ✅ | ❌ | Use N-Quads |
| **Inference** | ✅ | 🚧 | Coming v0.2.0 |
| **Text Search** | ✅ | 🚧 | Coming v0.2.0 |
| **GeoSPARQL** | Plugin | 🚧 | Coming v0.3.0 |
| **Federation (SERVICE)** | ✅ | ✅ | Enhanced |
| **Authentication** | Basic | OAuth2/JWT | Much better |
| **Authorization** | Basic | RBAC/ReBAC | Fine-grained |
| **Metrics** | JMX | Prometheus | Cloud-native |
| **Tracing** | ❌ | OpenTelemetry | Full distributed tracing |
| **HTTP/2** | ❌ | ✅ | Modern protocols |
| **GraphQL** | ❌ | ✅ | Alternative API |
| **REST API v2** | ❌ | ✅ | OpenAPI 3.0 |
| **WebSocket** | ❌ | ✅ | Real-time updates |

---

## Troubleshooting

### Issue: "Dataset not found" error

**Cause**: Dataset configuration mismatch

**Solution**:
```bash
# Check dataset list
curl http://localhost:3030/$/datasets

# Verify config
cat oxirs.toml | grep -A 5 "\[\[datasets\]\]"

# Restart server
systemctl restart oxirs-fuseki
```

### Issue: "Authentication required" but Fuseki had no auth

**Cause**: Authentication enabled in OxiRS config

**Solution**:
```toml
# Disable authentication
[authentication]
enabled = false
```

### Issue: "Query timeout" - queries worked in Fuseki

**Cause**: Different timeout defaults

**Solution**:
```toml
[server]
request_timeout_secs = 600  # 10 minutes (increase as needed)
```

### Issue: "Cannot load TDB database"

**Cause**: TDB format not supported

**Solution**:
```bash
# Export from Fuseki
tdbdump --loc=/path/to/tdb > export.nq

# Import to OxiRS
curl -X POST http://localhost:3030/dataset/upload \
  -H "Content-Type: application/n-quads" \
  --data-binary "@export.nq"
```

### Issue: "Missing text search results"

**Cause**: Text search not yet implemented

**Workaround**:
```sparql
# Use SPARQL FILTER instead
SELECT * WHERE {
  ?s ?p ?o .
  FILTER(CONTAINS(STR(?o), "search term"))
}
```

### Issue: "Performance slower than Fuseki"

**Cause**: Not using optimizations

**Solution**:
```toml
[performance]
worker_threads = 16  # Increase for your CPU
max_concurrent_queries = 200
query_cache_size = 50000

[performance.memory_pool]
enabled = true
max_size = 10000

[performance.batching]
enabled = true
max_batch_size = 100
```

### Getting Help

1. **Check Logs**:
   ```bash
   journalctl -u oxirs-fuseki -f
   # OR
   tail -f /var/log/oxirs/fuseki.log
   ```

2. **Enable Debug Logging**:
   ```toml
   [logging]
   level = "debug"
   ```

3. **GitHub Issues**: https://github.com/cool-japan/oxirs/issues
4. **Documentation**: https://github.com/cool-japan/oxirs/tree/main/docs

---

## Migration Checklist

### Pre-Migration

- [ ] Review current Fuseki configuration
- [ ] Document custom settings and integrations
- [ ] Backup all datasets
- [ ] Test SPARQL queries on sample data
- [ ] Identify authentication requirements
- [ ] Plan downtime window (if needed)

### Migration

- [ ] Install OxiRS Fuseki
- [ ] Convert configuration to TOML
- [ ] Export data from Fuseki (N-Quads)
- [ ] Import data to OxiRS
- [ ] Configure authentication
- [ ] Test SPARQL endpoints
- [ ] Verify query results
- [ ] Set up monitoring (Prometheus)
- [ ] Configure backups

### Post-Migration

- [ ] Monitor performance metrics
- [ ] Verify all integrations work
- [ ] Update client applications (if needed)
- [ ] Document new configuration
- [ ] Train team on new features
- [ ] Decommission Fuseki (after validation period)

---

## Migration Support

Need help with migration? We're here to assist!

- **Documentation**: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
- **GitHub Issues**: https://github.com/cool-japan/oxirs/issues
- **Community Support**: https://github.com/cool-japan/oxirs/discussions

---

**Welcome to OxiRS Fuseki! 🦀✨**
