# Getting Started with OxiRS Fuseki

**OxiRS Fuseki** is a high-performance, production-ready SPARQL 1.1/1.2 HTTP server with Apache Jena Fuseki compatibility, built in pure Rust.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/cool-japan/oxirs.git
cd oxirs

# Build oxirs-fuseki
cargo build --release -p oxirs-fuseki --all-features

# The binary will be at target/release/oxirs-fuseki
./target/release/oxirs-fuseki --version
```

### From Docker

```bash
# Pull the latest image
docker pull oxirs/fuseki:latest

# Run with default configuration
docker run -p 3030:3030 oxirs/fuseki:latest
```

### Binary Downloads

Download pre-built binaries from the [GitHub Releases](https://github.com/cool-japan/oxirs/releases) page:

- **Linux AMD64** (GNU libc)
- **Linux AMD64** (musl libc - fully static)
- **macOS Intel** (x86_64)
- **macOS Apple Silicon** (ARM64)

## Quick Start

### 1. Create a Configuration File

Create `oxirs.toml`:

```toml
[server]
host = "0.0.0.0"
port = 3030
enable_admin_ui = true

[datasets.default]
name = "default"
type = "Memory"
description = "Default in-memory dataset"

[security]
enable_auth = false  # Disable for quick start

[performance]
max_concurrent_queries = 100
query_timeout_secs = 300
```

### 2. Start the Server

```bash
./oxirs-fuseki --config oxirs.toml
```

You should see:

```
INFO  Starting OxiRS Fuseki v0.1.0
INFO  Binding to http://0.0.0.0:3030
INFO  Admin UI available at http://localhost:3030
INFO  Server started successfully
```

### 3. Test with a Query

Open your browser to `http://localhost:3030` for the Admin UI, or use curl:

```bash
# Query the dataset
curl -X POST http://localhost:3030/default/query \
  -H "Content-Type: application/sparql-query" \
  -d "SELECT * WHERE { ?s ?p ?o } LIMIT 10"
```

### 4. Add Some Data

```bash
# Insert triples
curl -X POST http://localhost:3030/default/update \
  -H "Content-Type: application/sparql-update" \
  -d "INSERT DATA {
    <http://example.org/person/Alice> <http://xmlns.com/foaf/0.1/name> \"Alice\" .
    <http://example.org/person/Alice> <http://xmlns.com/foaf/0.1/age> \"30\"^^<http://www.w3.org/2001/XMLSchema#integer> .
  }"
```

### 5. Query the Data

```bash
curl -X POST http://localhost:3030/default/query \
  -H "Content-Type: application/sparql-query" \
  -d "SELECT ?name ?age WHERE {
    ?person <http://xmlns.com/foaf/0.1/name> ?name .
    ?person <http://xmlns.com/foaf/0.1/age> ?age .
  }"
```

## Configuration

### Complete Configuration Example

```toml
# Server configuration
[server]
host = "0.0.0.0"
port = 3030
enable_admin_ui = true
enable_cors = true
graceful_shutdown_timeout_secs = 30

# TLS/SSL configuration (optional)
[server.tls]
enabled = false
cert_path = "/path/to/cert.pem"
key_path = "/path/to/key.pem"
# Optional: Client certificate authentication
client_auth = "Optional"  # None, Optional, Required

# Datasets
[datasets.default]
name = "default"
type = "Memory"  # or "Persistent"
description = "Default dataset"

[datasets.mydata]
name = "mydata"
type = "Persistent"
data_dir = "./data/mydata"
enable_inference = false

# Authentication & Authorization
[security]
enable_auth = true
jwt_secret = "your-secret-key-change-this"
jwt_expiration_hours = 24

# OAuth2/OIDC providers
[[security.oauth_providers]]
name = "google"
client_id = "your-client-id"
client_secret = "your-client-secret"
authorization_url = "https://accounts.google.com/o/oauth2/v2/auth"
token_url = "https://oauth2.googleapis.com/token"
scopes = ["openid", "email", "profile"]

# RBAC roles
[[security.roles]]
name = "admin"
permissions = ["read", "write", "admin"]

[[security.roles]]
name = "user"
permissions = ["read"]

# Performance tuning
[performance]
max_concurrent_queries = 100
query_timeout_secs = 300
enable_query_cache = true
cache_size_mb = 512
cache_ttl_secs = 3600

# Rate limiting
[performance.rate_limiting]
enabled = true
requests_per_minute = 1000
burst_size = 100

# Concurrency (v0.1.0 features)
[performance.concurrency]
enabled = true
worker_threads = 8
max_queue_size = 10000
priority_scheduling = true

# Memory management (v0.1.0 features)
[performance.memory]
enabled = true
pool_size = 1000
gc_interval_secs = 300
max_memory_mb = 4096

# Batch execution (v0.1.0 features)
[performance.batch_execution]
enabled = true
max_batch_size = 100
adaptive_sizing = true

# Streaming (v0.1.0 features)
[performance.streaming]
enabled = true
chunk_size = 65536
compression = "gzip"  # none, gzip, brotli
backpressure_threshold = 0.8

# Monitoring & Observability
[monitoring]
enable_metrics = true
metrics_port = 9090
enable_tracing = true
tracing_endpoint = "http://localhost:4317"
log_level = "info"  # trace, debug, info, warn, error

# Federation
[federation]
enabled = true
max_concurrent_requests = 10
timeout_secs = 60

[[federation.endpoints]]
name = "dbpedia"
url = "https://dbpedia.org/sparql"
enabled = true

# Clustering (experimental)
[clustering]
enabled = false
node_id = "node1"
bind_addr = "0.0.0.0:7070"
seeds = ["node2:7070", "node3:7070"]

# Backup & Recovery
[backup]
enabled = true
schedule = "0 2 * * *"  # 2 AM daily (cron format)
retention_days = 7
compression = true
destination = "./backups"

# Security Audit
[security_audit]
enabled = true
log_path = "./logs/security.log"
scan_interval_secs = 3600

# DDoS Protection
[ddos_protection]
enabled = true
max_requests_per_ip = 100
block_duration_secs = 3600
```

### Environment Variables

Configuration can also be set via environment variables:

```bash
export OXIRS_SERVER_HOST=0.0.0.0
export OXIRS_SERVER_PORT=3030
export OXIRS_SECURITY_JWT_SECRET=my-secret
export OXIRS_MONITORING_LOG_LEVEL=debug

./oxirs-fuseki
```

## Basic Usage

### SPARQL Query Endpoints

#### GET Request

```bash
curl "http://localhost:3030/default/query?query=SELECT%20*%20WHERE%20%7B%20%3Fs%20%3Fp%20%3Fo%20%7D%20LIMIT%2010"
```

#### POST Request (URL-encoded)

```bash
curl -X POST http://localhost:3030/default/query \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "query=SELECT * WHERE { ?s ?p ?o } LIMIT 10"
```

#### POST Request (Direct SPARQL)

```bash
curl -X POST http://localhost:3030/default/query \
  -H "Content-Type: application/sparql-query" \
  -d "SELECT * WHERE { ?s ?p ?o } LIMIT 10"
```

### SPARQL Update

```bash
curl -X POST http://localhost:3030/default/update \
  -H "Content-Type: application/sparql-update" \
  -d "
PREFIX dc: <http://purl.org/dc/elements/1.1/>
INSERT DATA {
  <http://example.org/book1> dc:title \"SPARQL Tutorial\" .
  <http://example.org/book1> dc:creator \"Alice\" .
}
"
```

### Graph Store Protocol

#### Get all triples in default graph

```bash
curl http://localhost:3030/default/data
```

#### Add triples to default graph

```bash
curl -X PUT http://localhost:3030/default/data \
  -H "Content-Type: text/turtle" \
  -d "@prefix ex: <http://example.org/> .
      ex:subject ex:predicate ex:object ."
```

#### Add triples to named graph

```bash
curl -X PUT "http://localhost:3030/default/data?graph=http://example.org/mygraph" \
  -H "Content-Type: text/turtle" \
  -d "@prefix ex: <http://example.org/> .
      ex:subject ex:predicate ex:object ."
```

### Upload RDF Files

```bash
curl -X POST http://localhost:3030/default/upload \
  -F "file=@data.ttl" \
  -F "format=turtle"
```

Supported formats:
- `turtle` (`.ttl`)
- `ntriples` (`.nt`)
- `rdfxml` (`.rdf`)
- `nquads` (`.nq`)
- `trig` (`.trig`)

### Authentication

#### Using API Keys

```bash
# Create an API key
curl -X POST http://localhost:3030/$/admin/api-keys \
  -H "Authorization: Bearer admin-token" \
  -d '{"name": "my-app", "permissions": ["read", "write"]}'

# Use the API key
curl -X POST http://localhost:3030/default/query \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/sparql-query" \
  -d "SELECT * WHERE { ?s ?p ?o } LIMIT 10"
```

#### Using JWT

```bash
# Login to get JWT token
curl -X POST http://localhost:3030/$/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "secret"}'

# Use JWT token
curl -X POST http://localhost:3030/default/query \
  -H "Authorization: Bearer eyJhbGc..." \
  -H "Content-Type: application/sparql-query" \
  -d "SELECT * WHERE { ?s ?p ?o } LIMIT 10"
```

## Advanced Features

### WebSocket Subscriptions

Connect to live query results:

```javascript
const ws = new WebSocket('ws://localhost:3030/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'subscribe',
    query: 'SELECT * WHERE { ?s ?p ?o }',
    dataset: 'default'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('New results:', data);
};
```

### GraphQL API

```bash
# Query via GraphQL
curl -X POST http://localhost:3030/graphql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "{ datasets { name tripleCount } }"
  }'
```

GraphQL Playground available at: `http://localhost:3030/graphql`

### REST API v2

OpenAPI 3.0 compliant REST API:

```bash
# List datasets
curl http://localhost:3030/api/v2/datasets

# Get dataset info
curl http://localhost:3030/api/v2/datasets/default

# Execute query
curl -X POST http://localhost:3030/api/v2/datasets/default/query \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT * WHERE { ?s ?p ?o } LIMIT 10"}'

# Get statistics
curl http://localhost:3030/api/v2/stats
```

Swagger UI available at: `http://localhost:3030/api/v2/docs`

### Federation (SERVICE queries)

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?person ?name ?friend
WHERE {
  ?person foaf:name ?name .

  # Query remote endpoint
  SERVICE <https://dbpedia.org/sparql> {
    ?person foaf:knows ?friend .
  }
}
```

### Real-time Notifications

Subscribe to dataset updates:

```javascript
const ws = new WebSocket('ws://localhost:3030/notifications');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'subscribe',
    filters: {
      event_types: ['dataset_updated', 'query_completed'],
      datasets: ['default']
    }
  }));
};

ws.onmessage = (event) => {
  const notification = JSON.parse(event.data);
  console.log('Notification:', notification);
};
```

## Deployment

### Docker

```bash
# Using docker-compose
cd deployment/docker
docker-compose up -d

# Access services
# - Fuseki: http://localhost:3030
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
```

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f deployment/kubernetes/

# Check status
kubectl get pods -n oxirs-fuseki
kubectl get svc -n oxirs-fuseki

# Access via port-forward
kubectl port-forward svc/oxirs-fuseki 3030:3030 -n oxirs-fuseki
```

### Terraform (AWS)

```bash
cd deployment/terraform/aws

# Initialize
terraform init

# Plan
terraform plan -out=tfplan

# Apply
terraform apply tfplan

# Outputs will show endpoints and connection info
```

### Ansible

```bash
cd deployment/ansible

# Install to production servers
ansible-playbook -i inventory/production site.yml

# Specific roles
ansible-playbook -i inventory/production site.yml --tags=oxirs-fuseki
```

## Troubleshooting

### Common Issues

#### Port Already in Use

```bash
# Change port in config
[server]
port = 3031
```

Or use environment variable:
```bash
OXIRS_SERVER_PORT=3031 ./oxirs-fuseki
```

#### Out of Memory

Increase memory limits:

```toml
[performance.memory]
max_memory_mb = 8192
gc_interval_secs = 180
```

#### Slow Queries

Enable query optimization:

```toml
[performance]
enable_query_cache = true
cache_size_mb = 1024

[performance.concurrency]
enabled = true
worker_threads = 16
```

#### Connection Timeouts

Adjust timeouts:

```toml
[performance]
query_timeout_secs = 600

[federation]
timeout_secs = 120
```

### Debugging

Enable debug logging:

```toml
[monitoring]
log_level = "debug"
```

Or via environment:
```bash
RUST_LOG=debug ./oxirs-fuseki
```

View performance metrics:
```bash
curl http://localhost:9090/metrics
```

Health check:
```bash
curl http://localhost:3030/$/ping
```

### Getting Help

- **Documentation**: https://docs.oxirs.org
- **Issues**: https://github.com/cool-japan/oxirs/issues
- **Discussions**: https://github.com/cool-japan/oxirs/discussions
- **Chat**: Join our Discord server

## Next Steps

- Read the [API Reference](API_REFERENCE.md)
- Explore [Examples](../examples/)
- Learn about [Performance Tuning](PERFORMANCE_TUNING.md)
- Set up [Monitoring](MONITORING.md)
- Configure [Security](SECURITY.md)

## License

OxiRS Fuseki is licensed under the Apache License 2.0 or MIT license.
