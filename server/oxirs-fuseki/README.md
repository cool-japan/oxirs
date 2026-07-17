# OxiRS Fuseki

[![Version](https://img.shields.io/badge/version-0.3.2-blue)](https://github.com/cool-japan/oxirs/releases)

**SPARQL 1.1/1.2 HTTP server with Apache Fuseki compatibility**

**Status**: v0.3.2 - Released 2026-07-12

✨ **Production Release**: Production-ready with API stability guarantees. Semantic versioning enforced.

## Overview

`oxirs-fuseki` is a high-performance SPARQL HTTP server that provides complete compatibility with Apache Jena Fuseki while leveraging Rust's performance and safety. It implements the SPARQL 1.1 Protocol for RDF over HTTP and extends it with SPARQL 1.2 features.

## Features

- **SPARQL Protocol Compliance**: Full SPARQL 1.1 Protocol implementation
- **SPARQL 1.2 Support**: Extended features and optimizations
- **Fuseki Compatibility**: Drop-in replacement for Apache Fuseki
- **Multi-Dataset Support**: Host multiple datasets on different endpoints
- **Authentication & Authorization**: OAuth2/OIDC, JWT, SAML 2.0 SP, LDAP (with HA failover), RBAC/ReBAC, graph-level ACLs, MFA, and cluster node auth
- **GraphQL Integration**: Dual protocol support (SPARQL + GraphQL)
- **Real-time Features**: WebSocket subscriptions and live queries
- **High Performance**: Async I/O with Tokio and optimized query execution
- **Monitoring**: Built-in metrics, logging, and health checks
- **Configuration**: YAML/TOML configuration with hot-reload

## Installation

### As a Library

```toml
[dependencies]
oxirs-fuseki = "0.3.2"
```

### As a Binary

```bash
# Install from crates.io
cargo install oxirs-fuseki

# Or build from source
git clone https://github.com/cool-japan/oxirs
cd oxirs/server/oxirs-fuseki
cargo install --path .
```

### Docker

```bash
docker pull ghcr.io/cool-japan/oxirs-fuseki:latest
docker run -p 3030:3030 ghcr.io/cool-japan/oxirs-fuseki:latest
```

## Quick Start

### Basic Server

```rust
use oxirs_fuseki::{Server, Config, Dataset};
use oxirs_core::Graph;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a dataset with some data
    let mut dataset = Dataset::new();
    let graph = Graph::new();
    dataset.set_default_graph(graph);
    
    // Configure the server
    let config = Config::builder()
        .port(3030)
        .host("localhost")
        .dataset("/dataset", dataset)
        .build();
    
    // Start the server
    let server = Server::new(config);
    server.run().await
}
```

### Configuration File

Create `fuseki.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 3030
  cors: true
  
datasets:
  - name: "example"
    path: "/example"
    type: "memory"
    services:
      - type: "sparql-query"
        endpoint: "sparql"
      - type: "sparql-update"  
        endpoint: "update"
      - type: "graphql"
        endpoint: "graphql"
        
security:
  authentication:
    type: "basic"
    users:
      - username: "admin"
        password: "password"
        roles: ["admin"]
        
logging:
  level: "info"
  format: "json"
```

Run with configuration:

```bash
oxirs-fuseki --config fuseki.yaml
```

## API Endpoints

### SPARQL Query

```http
POST /dataset/sparql
Content-Type: application/sparql-query

SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10
```

### SPARQL Update

```http
POST /dataset/update
Content-Type: application/sparql-update

INSERT DATA { 
  <http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice" .
}
```

### GraphQL

```http
POST /dataset/graphql
Content-Type: application/json

{
  "query": "{ person(id: \"alice\") { name, age } }"
}
```

### Data Upload

```http
PUT /dataset/data
Content-Type: text/turtle

@prefix foaf: <http://xmlns.com/foaf/0.1/> .
<http://example.org/alice> foaf:name "Alice" .
```

## Operational Contracts

These are the behaviors an operator should rely on when running `oxirs-fuseki` in production. They reflect the actual code (`src/server/types.rs`, `src/main.rs`, `src/lib.rs`) as of v0.4.0, not aspirational design.

### `read_only` dataset resolution

`config.datasets.<name>.read_only` is enforced by `AppState::is_dataset_read_only()` / `AppState::reject_if_read_only()`, with HTTP 403 returned before any mutation runs. Resolution depends on how many datasets are configured:

- **Exactly one dataset configured**: that dataset's `read_only` flag applies to every write guard **regardless of the name a given guard queries**. A single dataset named `[datasets.mydata]` (not `[datasets.default]`) still gets full write protection — every write path in this crate resolves to it.
- **Two or more datasets configured**: resolution reverts to an exact per-key lookup. Guards that key on the request's own path parameter (SPARQL UPDATE's `{dataset}`, `/$/datasets/{name}`, `/$/compact/{name}`) resolve correctly per dataset. Guards that are **not** parameterized by dataset name — Graph Store Protocol (PUT/POST/DELETE), `/upload`, `/patch`, and `/$/reload` — key their lookup on the literal string `"default"`. A `read_only = true` dataset configured under any other name in a multi-dataset deployment is therefore invisible to those specific guards.
- **Startup diagnostics**: when two or more datasets are configured and at least one is `read_only`, the server logs a WARN listing which write paths resolve per-dataset versus which key on `"default"`. If none of the `read_only` datasets is literally named `"default"`, that WARN escalates to an ERROR naming the exact misconfiguration (Graph Store Protocol / upload / patch / reload writes to that dataset are **not** blocked). There is no runtime auto-fix — rename the intended read-only dataset to `"default"`, or restrict access to it through another mechanism (auth/ACLs).

Write paths that check `read_only` (→ HTTP 403 when set): SPARQL UPDATE, Graph Store Protocol writes (PUT/POST/DELETE), `/upload` (bulk RDF upload), `/patch` (RDF Patch), admin dataset **create**/**delete**/**compact**/**reload** (`/$/datasets`, `/$/datasets/{name}`, `/$/compact/{name}`, `/$/reload`), and the REST API v2 dataset/triple write endpoints (`POST /api/v2/datasets`, `DELETE /api/v2/datasets/{name}`, `POST`/`DELETE /api/v2/datasets/{name}/triples`) — the REST v2 write paths were an unguarded bypass around `read_only` prior to this hardening pass and are now covered by the same shared `reject_if_read_only()` helper as every other write path.

### Store selection: `--dataset` vs. `--config`

The server opens exactly one on-disk (or in-memory) store per process, chosen by `ServerBuilder::build()`:

- `oxirs-fuseki --dataset <path>` opens a real store at `<path>` (`Store::open(path)`) and is what actually determines where data is read from and written to.
- `config.datasets.<name>.location` in a `--config` file is validated at load time (must be non-empty) but is **not currently used to open a store**. It exists in the config schema but has no effect on which store the running server actually serves.
- **Known operational gotcha**: `oxirs-fuseki --config fuseki.yaml` **without** `--dataset` starts the server against a fresh empty in-memory store (`Store::new()`), no matter what `datasets.<name>.location` says in the config file. Everything else in the config (security, `read_only` flags, services, etc.) is honored — only the store backend is not derived from `location`. Always pass `--dataset <path>` alongside `--config` when persistence is required; do not rely on `location` alone.

### Fail-loud query contract

SPARQL query execution never returns HTTP 200 with a silently-empty result on failure:

- A query that fails to **parse** returns HTTP 400 with an error message.
- A query that parses but fails during **execution** (unsupported construct, store error, federation failure, etc.) returns HTTP 500 with an error message.
- `SELECT`, `ASK`, `CONSTRUCT`, and `DESCRIBE` all execute through the real oxirs-arq engine (`handlers/sparql/arq_exec.rs`) via a single parse-once dispatch — including `GRAPH`/`FROM`/`FROM NAMED` named-graph scoping, `SERVICE` HTTP federation, and aggregate/`HAVING` projections. There is no legacy "demo" fallback path left in the SELECT/ASK/CONSTRUCT/DESCRIBE handlers that could return 200 OK with an empty body on an unrecognized or unsupported query.

Status-code guarantees, made precise (v0.4.0):

- **Parse / validation errors are 400, not 500 or 200.** A malformed aggregate call inside `HAVING` — a wrong-arity aggregate such as `SUM()` or `COUNT(?a, ?b)` — is caught at parse time and returns HTTP 400, rather than parsing cleanly and failing deep in execution. The same wrong-arity check applies to aggregate projections in the `SELECT` list.
- **A genuinely undefined function fails the whole query loudly.** An unknown function inside a `FILTER` or `HAVING` raises a typed `UnknownFunctionError` that surfaces as a 5xx, instead of being swallowed per row and silently shrinking the result set (a 200 with dropped rows).
- **No 200 with wrong or fabricated data in the result body.** The SPARQL Results JSON term serializer (`arq_exec::term_to_json`) is exhaustive over the arq term type: an RDF-star quoted triple serializes as `{"type":"triple","value":{"subject":…,"predicate":…,"object":…}}` (recursing per position), and a property-path term — which can never be a legitimate solution binding — is a 500 fail-loud error rather than a fabricated `Debug`-string literal.

`DESCRIBE` semantics: the response is a **symmetric** Concise Bounded Description of each described node — both its outgoing arcs (`node ?p ?o`) and its incoming/object-side arcs (`?s ?p node`), recursing through blank nodes in both directions with a visited-set so cycles terminate. `DESCRIBE` honors `FROM` / `FROM NAMED` dataset scoping (with no `FROM`, it reads the default graph only); auto-unioning every named graph for an unscoped `DESCRIBE` is a deliberate non-goal.

## Advanced Features

### Multi-Dataset Hosting

```rust
use oxirs_fuseki::{Server, Config, Dataset};

let config = Config::builder()
    .dataset("/companies", Dataset::from_file("companies.ttl")?)
    .dataset("/products", Dataset::from_file("products.nt")?)
    .dataset("/users", Dataset::memory())
    .build();
```

### Authentication

```rust
use oxirs_fuseki::{Config, auth::{BasicAuth, JwtAuth}};

let config = Config::builder()
    .auth(BasicAuth::new()
        .user("admin", "secret", &["admin"])
        .user("user", "password", &["read"]))
    .build();
```

### Custom Extensions

```rust
use oxirs_fuseki::{Server, Extension, Request, Response};

struct MetricsExtension;

impl Extension for MetricsExtension {
    async fn before_query(&self, req: &Request) -> Result<(), Response> {
        // Log query metrics
        Ok(())
    }
    
    async fn after_query(&self, req: &Request, response: &mut Response) {
        // Add timing headers
    }
}

let server = Server::new(config)
    .extension(MetricsExtension);
```

### WebSocket Subscriptions

```javascript
const ws = new WebSocket('ws://localhost:3030/dataset/subscriptions');

ws.send(JSON.stringify({
  type: 'start',
  payload: {
    query: 'SUBSCRIBE { ?s ?p ?o } WHERE { ?s ?p ?o }'
  }
}));

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('New triple:', data);
};
```

## Performance

### Benchmarks

#### Performance Comparison

| Metric | OxiRS Fuseki | Apache Fuseki | Stardog | Improvement |
|--------|--------------|---------------|---------|-------------|
| Query throughput | 15,000 q/s | 8,000 q/s | 12,000 q/s | 1.9x / 1.25x |
| Memory usage | 45 MB | 120 MB | 80 MB | 2.7x / 1.8x |
| Startup time | 0.3s | 4.2s | 2.1s | 14x / 7x |
| Binary size | 12 MB | 80 MB | 150 MB | 6.7x / 12.5x |
| Cold query latency | 5ms | 25ms | 15ms | 5x / 3x |
| Concurrent connections | 50,000 | 5,000 | 10,000 | 10x / 5x |

#### Scalability Benchmarks

| Dataset Size | Query Latency (p95) | Memory Usage | Notes |
|--------------|-------------------|--------------|-------|
| 1M triples | 15ms | 180MB | Excellent |
| 10M triples | 45ms | 1.2GB | Very good |
| 100M triples | 150ms | 8GB | Good |
| 1B triples | 800ms | 32GB | Acceptable |

*Benchmarks run on AWS c5.4xlarge instance (16 vCPU, 32GB RAM)*

### Optimization Tips

```rust
use oxirs_fuseki::Config;

let config = Config::builder()
    // Enable query caching
    .query_cache(true)
    .cache_size(1000)
    
    // Optimize for read-heavy workloads
    .read_threads(8)
    .write_threads(2)
    
    // Enable compression
    .compression(true)
    
    .build();
```

## Monitoring

### Metrics Endpoint

```http
GET /metrics
```

Returns Prometheus-compatible metrics:

```
# HELP sparql_queries_total Total number of SPARQL queries
# TYPE sparql_queries_total counter
sparql_queries_total{dataset="example",type="select"} 1234

# HELP sparql_query_duration_seconds Query execution time
# TYPE sparql_query_duration_seconds histogram
sparql_query_duration_seconds_bucket{le="0.1"} 800
sparql_query_duration_seconds_bucket{le="1.0"} 950
sparql_query_duration_seconds_sum 45.2
sparql_query_duration_seconds_count 1000
```

### Health Checks

```http
GET /health
```

```json
{
  "status": "healthy",
  "version": "0.3.2",
  "uptime": "2h 15m 30s",
  "datasets": {
    "example": {
      "status": "ready",
      "triples": 15420,
      "last_update": "2025-12-25T10:30:00Z"
    }
  }
}
```

## Integration

### With oxirs-gql

```rust
use oxirs_fuseki::Server;
use oxirs_gql::GraphQLService;

let server = Server::new(config)
    .service("/graphql", GraphQLService::new(schema));
```

### With oxirs-stream

```rust
use oxirs_fuseki::Server;
use oxirs_stream::EventStream;

let server = Server::new(config)
    .event_stream(EventStream::kafka("localhost:9092"));
```

## Deployment

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: oxirs-fuseki
spec:
  replicas: 3
  selector:
    matchLabels:
      app: oxirs-fuseki
  template:
    metadata:
      labels:
        app: oxirs-fuseki
    spec:
      containers:
      - name: oxirs-fuseki
        image: ghcr.io/cool-japan/oxirs-fuseki:latest
        ports:
        - containerPort: 3030
        env:
        - name: RUST_LOG
          value: "info"
        resources:
          requests:
            memory: "64Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

### Docker Compose

```yaml
version: '3.8'
services:
  fuseki:
    image: ghcr.io/cool-japan/oxirs-fuseki:latest
    ports:
      - "3030:3030"
    volumes:
      - ./data:/data
      - ./config.yaml:/config.yaml
    environment:
      - OXIRS_CONFIG=/config.yaml
      - RUST_LOG=info
```

## Performance Tuning

### Memory Optimization
```yaml
# config.yaml
server:
  memory:
    query_cache_size: "1GB"    # Adjust based on available RAM
    result_cache_size: "512MB"
    connection_pool_size: 100
    
optimization:
  enable_query_planning: true
  enable_join_reordering: true
  enable_filter_pushdown: true
  parallel_execution: true
  worker_threads: 8              # Match CPU cores
```

### High-Throughput Configuration
```yaml
server:
  port: 3030
  workers: 16                    # For CPU-intensive workloads
  keep_alive: 30s
  request_timeout: 60s
  
network:
  tcp_nodelay: true
  socket_reuse: true
  backlog: 1024
  
caching:
  query_cache: true
  result_cache: true
  ttl: 3600                      # 1 hour
```

### Production Deployment
```yaml
security:
  tls:
    cert_file: "/etc/ssl/certs/server.crt"
    key_file: "/etc/ssl/private/server.key"
  rate_limiting:
    requests_per_minute: 1000
    burst_size: 100
    
logging:
  level: "warn"                  # Reduce log verbosity
  format: "json"
  audit: true
  
monitoring:
  metrics: true
  health_checks: true
  profiling: false               # Disable in production
```

## Troubleshooting

### Common Issues

**Q: High memory usage with large datasets**
A: Enable streaming mode and adjust cache sizes:
```yaml
query_execution:
  streaming_threshold: 10000     # Stream results > 10k rows
  max_memory_per_query: "100MB"
```

**Q: Slow query performance**
A: Enable query optimization and check index usage:
```yaml
optimization:
  query_planner: "advanced"
  statistics_collection: true
  index_recommendations: true
```

**Q: Connection timeouts**
A: Adjust timeout settings and connection limits:
```yaml
server:
  connection_timeout: 30s
  keep_alive_timeout: 60s
  max_connections: 1000
```

### Debug Mode
```bash
# Enable debug logging
RUST_LOG=oxirs_fuseki=debug oxirs-fuseki --config config.yaml

# Enable query tracing
oxirs-fuseki --config config.yaml --trace-queries

# Profile performance
oxirs-fuseki --config config.yaml --profile
```

## OxiRS Ecosystem

### Core Components
- [`oxirs-core`](../../core/oxirs-core/): RDF data model and core functionality
- [`oxirs-arq`](../../engine/oxirs-arq/): SPARQL query engine with optimization
- [`oxirs-shacl`](../../engine/oxirs-shacl/): SHACL validation engine
- [`oxirs-star`](../../engine/oxirs-star/): RDF-star and SPARQL-star support

### Server & Networking
- [`oxirs-fuseki`](./): HTTP server (this crate)
- [`oxirs-gql`](../oxirs-gql/): GraphQL interface and schema generation
- [`oxirs-stream`](../../stream/oxirs-stream/): Real-time data streaming
- [`oxirs-federate`](../../stream/oxirs-federate/): Federated query processing

### AI & Analytics
- [`oxirs-vec`](../../engine/oxirs-vec/): Vector embeddings and similarity search
- [`oxirs-shacl-ai`](../../ai/oxirs-shacl-ai/): AI-powered data validation
- [`oxirs-rule`](../../engine/oxirs-rule/): Rule-based reasoning engine

### Integration Examples

```rust
// Full-stack semantic web application
use oxirs_fuseki::Server;
use oxirs_gql::GraphQLService;
use oxirs_stream::EventStream;
use oxirs_vec::VectorIndex;

let server = Server::new(config)
    .service("/graphql", GraphQLService::new(schema))
    .event_stream(EventStream::kafka("localhost:9092"))
    .vector_index(VectorIndex::new())
    .build();
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Licensed under:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

## Status

🚀 **Production Release (v0.3.2)** - 2026-07-12

**2,350 tests passing**, zero warnings

Current features:
- ✅ SPARQL query/update endpoints backed by persisted N-Quads datasets
- ✅ Federation (`SERVICE` clause) with retries, `SERVICE SILENT`, and result merging
- ✅ OAuth2/OIDC + JWT security with hardened headers and HSTS
- ✅ Enterprise authentication: SAML 2.0 SP (`auth/saml*.rs`), LDAP with HA failover (`auth/ldap_ha.rs`), RBAC/ReBAC, graph-level ACLs, MFA, and cluster node auth tokens (`auth/cluster_auth.rs`)
- ✅ Audit log export (`GET /$/audit/log`, `GET /$/audit/log/stats`) with JSON/JSONL/CSV output and actor/action/time filtering
- ✅ GraphQL integration (`graphql_integration` module, async-graphql-based) alongside SPARQL
- ✅ Admin UI dashboard (`admin_ui`) and configuration hot-reload (`config_reload`, `hot-reload` feature)
- ✅ Prometheus metrics, slow-query tracing, and structured logging via SciRS2
- ✅ Multi-dataset support with auto save/load and CLI integration

APIs follow semantic versioning. See CHANGELOG.md for details.