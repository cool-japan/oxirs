# OxiRS Tutorials

**Version**: v0.1.0-rc.2
**Date**: December 21, 2025
**Audience**: Developers, Data Engineers, Semantic Web Practitioners

---

## 🎯 Tutorial Index

| Tutorial | Level | Time | Description |
|----------|-------|------|-------------|
| [Quick Start](#quick-start) | Beginner | 5 min | Get OxiRS running in 5 minutes |
| [SPARQL Basics](#sparql-basics) | Beginner | 15 min | Essential SPARQL query patterns |
| [Data Management](#data-management) | Beginner | 20 min | Load, update, and manage RDF data |
| [Federation](#federation) | Intermediate | 25 min | Query multiple SPARQL endpoints |
| [Performance Tuning](#performance-tuning) | Intermediate | 30 min | Optimize query performance |
| [Production Deployment](#production-deployment) | Advanced | 45 min | Deploy to production with monitoring |
| [Security Hardening](#security-hardening) | Advanced | 30 min | Secure your SPARQL endpoint |
| [Cluster Setup](#cluster-setup) | Advanced | 60 min | Distributed RDF storage with Raft |

---

## 🚀 Quick Start

**Goal**: Get OxiRS running in 5 minutes

### Prerequisites
```bash
# Rust 1.75.0 or later
rustc --version

# (Optional) Docker for containerized deployment
docker --version
```

### Step 1: Installation

#### Option A: Install Pre-built Binary (Fastest)
```bash
# Download latest release
curl -L https://github.com/cool-japan/oxirs/releases/download/v0.1.0-rc.2/oxirs-linux-x86_64.tar.gz | tar xz

# Move to PATH
sudo mv oxirs-fuseki /usr/local/bin/

# Verify installation
oxirs-fuseki --version
# oxirs-fuseki 0.1.0-rc.2
```

#### Option B: Build from Source
```bash
# Clone repository
git clone https://github.com/cool-japan/oxirs.git
cd oxirs

# Build release binary
cargo build --release -p oxirs-fuseki

# Binary location
./target/release/oxirs-fuseki --version
```

#### Option C: Docker (Recommended for Production)
```bash
# Pull image
docker pull ghcr.io/cool-japan/oxirs-fuseki:v0.1.0-rc.2

# Run container
docker run -d \
  -p 3030:3030 \
  --name oxirs \
  ghcr.io/cool-japan/oxirs-fuseki:v0.1.0-rc.2

# Verify
curl http://localhost:3030/$/ping
# {"status": "ok"}
```

### Step 2: Create Configuration

```bash
# Create minimal config
cat > oxirs.toml <<EOF
[general]
default_format = "turtle"

[server]
host = "0.0.0.0"
port = 3030

[[datasets]]
name = "my_dataset"
type = "memory"
EOF
```

### Step 3: Start Server

```bash
# Start with config
oxirs-fuseki --config oxirs.toml

# Server starts
# INFO  oxirs_fuseki: Starting OxiRS Fuseki Server v0.1.0-rc.2
# INFO  oxirs_fuseki: Listening on http://0.0.0.0:3030
# INFO  oxirs_fuseki: Dataset 'my_dataset' ready
```

### Step 4: Load Sample Data

```bash
# Create sample RDF data (Turtle format)
cat > data.ttl <<EOF
@prefix ex: <http://example.org/> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .

ex:alice a foaf:Person ;
    foaf:name "Alice" ;
    foaf:age 30 ;
    foaf:knows ex:bob .

ex:bob a foaf:Person ;
    foaf:name "Bob" ;
    foaf:age 25 .
EOF

# Load data via HTTP
curl -X POST \
  -H "Content-Type: text/turtle" \
  --data-binary @data.ttl \
  http://localhost:3030/my_dataset/data
```

### Step 5: Run Your First Query

```bash
# Query via HTTP
curl -X POST \
  -H "Content-Type: application/sparql-query" \
  -d "SELECT ?name WHERE { ?person <http://xmlns.com/foaf/0.1/name> ?name }" \
  http://localhost:3030/my_dataset/query

# Response (JSON)
{
  "head": { "vars": ["name"] },
  "results": {
    "bindings": [
      { "name": { "type": "literal", "value": "Alice" } },
      { "name": { "type": "literal", "value": "Bob" } }
    ]
  }
}
```

**🎉 Congratulations!** You've successfully set up OxiRS, loaded RDF data, and executed your first SPARQL query.

---

## 📖 SPARQL Basics

**Goal**: Learn essential SPARQL query patterns

### 1. SELECT Queries (Retrieve Data)

#### Basic Pattern Matching
```sparql
# Find all people and their names
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?person ?name
WHERE {
  ?person a foaf:Person ;
          foaf:name ?name .
}
```

#### Filtering Results
```sparql
# Find people older than 25
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?name ?age
WHERE {
  ?person foaf:name ?name ;
          foaf:age ?age .
  FILTER (?age > 25)
}
```

#### OPTIONAL Patterns
```sparql
# Find people with optional email addresses
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?name ?email
WHERE {
  ?person foaf:name ?name .
  OPTIONAL { ?person foaf:mbox ?email }
}
```

### 2. CONSTRUCT Queries (Transform Data)

```sparql
# Transform FOAF to schema.org vocabulary
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX schema: <http://schema.org/>

CONSTRUCT {
  ?person a schema:Person ;
          schema:name ?name ;
          schema:age ?age .
}
WHERE {
  ?person a foaf:Person ;
          foaf:name ?name ;
          foaf:age ?age .
}
```

### 3. ASK Queries (Boolean Check)

```sparql
# Check if Alice exists
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

ASK {
  ?person foaf:name "Alice" .
}
# Returns: true or false
```

### 4. DESCRIBE Queries (Get All Data About Resource)

```sparql
# Get all triples about Alice
PREFIX ex: <http://example.org/>

DESCRIBE ex:alice
# Returns all triples with ex:alice as subject or object
```

### 5. Complex Patterns

#### Joins (Multiple Patterns)
```sparql
# Find friends of friends
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?person ?friendOfFriend ?fofName
WHERE {
  ?person foaf:name "Alice" ;
          foaf:knows ?friend .
  ?friend foaf:knows ?friendOfFriend .
  ?friendOfFriend foaf:name ?fofName .
}
```

#### Aggregation
```sparql
# Count people by age group
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?ageGroup (COUNT(?person) AS ?count)
WHERE {
  ?person a foaf:Person ;
          foaf:age ?age .
  BIND(IF(?age < 18, "minor", IF(?age < 65, "adult", "senior")) AS ?ageGroup)
}
GROUP BY ?ageGroup
ORDER BY DESC(?count)
```

#### Subqueries
```sparql
# Find people with more friends than average
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?person ?name ?friendCount
WHERE {
  {
    SELECT ?person (COUNT(?friend) AS ?friendCount)
    WHERE {
      ?person foaf:knows ?friend .
    }
    GROUP BY ?person
  }
  ?person foaf:name ?name .

  FILTER (?friendCount > (
    SELECT (AVG(?fc) AS ?avgFriends)
    WHERE {
      {
        SELECT (COUNT(?f) AS ?fc)
        WHERE { ?p foaf:knows ?f }
        GROUP BY ?p
      }
    }
  ))
}
```

### 6. Property Paths

```sparql
# Find all resources reachable via any path from Alice
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX ex: <http://example.org/>

SELECT ?connected
WHERE {
  ex:alice (foaf:knows+) ?connected .
  # + = one or more hops
  # * = zero or more hops
  # ? = zero or one hop
}
```

### Practice Exercises

**Exercise 1**: Find all people Alice knows directly or indirectly (friends of friends)

<details>
<summary>Solution</summary>

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX ex: <http://example.org/>

SELECT DISTINCT ?name
WHERE {
  ex:alice foaf:knows+ ?person .
  ?person foaf:name ?name .
}
```
</details>

**Exercise 2**: Find the oldest person in the dataset

<details>
<summary>Solution</summary>

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?name ?age
WHERE {
  ?person foaf:name ?name ;
          foaf:age ?age .
}
ORDER BY DESC(?age)
LIMIT 1
```
</details>

---

## 💾 Data Management

**Goal**: Learn to load, update, and manage RDF data

### Loading Data

#### 1. Load from File (CLI)
```bash
# Load Turtle file
oxirs load \
  --dataset my_dataset \
  --file data.ttl \
  --format turtle

# Load N-Triples file
oxirs load \
  --dataset my_dataset \
  --file large_dataset.nt \
  --format ntriples

# Load RDF/XML
oxirs load \
  --dataset my_dataset \
  --file legacy.rdf \
  --format rdfxml
```

#### 2. Load via HTTP (cURL)
```bash
# Upload Turtle data
curl -X POST \
  -H "Content-Type: text/turtle" \
  --data-binary @data.ttl \
  http://localhost:3030/my_dataset/data

# Upload N-Quads (with named graphs)
curl -X POST \
  -H "Content-Type: application/n-quads" \
  --data-binary @data.nq \
  http://localhost:3030/my_dataset/data

# Upload JSON-LD
curl -X POST \
  -H "Content-Type: application/ld+json" \
  --data-binary @data.jsonld \
  http://localhost:3030/my_dataset/data
```

#### 3. Bulk Import (Large Datasets)
```bash
# Import 100M triples efficiently
oxirs import \
  --dataset my_dataset \
  --file dbpedia_subset.nt.gz \
  --format ntriples \
  --compressed gzip \
  --batch-size 10000 \
  --num-threads 8

# Progress output
# Importing: 10M triples (10%)
# Importing: 20M triples (20%)
# ...
# Import complete: 100M triples in 5m32s (300K triples/sec)
```

### Updating Data

#### 1. INSERT DATA (Add Triples)
```sparql
PREFIX ex: <http://example.org/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

INSERT DATA {
  ex:charlie a foaf:Person ;
              foaf:name "Charlie" ;
              foaf:age 28 .
}
```

```bash
# Execute via HTTP
curl -X POST \
  -H "Content-Type: application/sparql-update" \
  -d "PREFIX ex: <http://example.org/>
      PREFIX foaf: <http://xmlns.com/foaf/0.1/>
      INSERT DATA {
        ex:charlie a foaf:Person ;
                    foaf:name \"Charlie\" ;
                    foaf:age 28 .
      }" \
  http://localhost:3030/my_dataset/update
```

#### 2. DELETE DATA (Remove Triples)
```sparql
PREFIX ex: <http://example.org/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

DELETE DATA {
  ex:bob foaf:age 25 .
}
```

#### 3. DELETE/INSERT (Update Existing Data)
```sparql
# Update Bob's age from 25 to 26
PREFIX ex: <http://example.org/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

DELETE { ex:bob foaf:age ?oldAge }
INSERT { ex:bob foaf:age 26 }
WHERE {
  ex:bob foaf:age ?oldAge .
}
```

#### 4. DELETE WHERE (Conditional Delete)
```sparql
# Delete all people younger than 18
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

DELETE WHERE {
  ?person a foaf:Person ;
          foaf:age ?age .
  FILTER (?age < 18)
}
```

### Managing Named Graphs

#### Create Graph
```sparql
# Create a new named graph
PREFIX ex: <http://example.org/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

INSERT DATA {
  GRAPH ex:friends_graph {
    ex:alice foaf:knows ex:bob .
    ex:bob foaf:knows ex:charlie .
  }
}
```

#### Query Specific Graph
```sparql
# Query only from friends_graph
PREFIX ex: <http://example.org/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?person1 ?person2
FROM <http://example.org/friends_graph>
WHERE {
  ?person1 foaf:knows ?person2 .
}
```

#### Query Across Graphs
```sparql
# Query from multiple named graphs
PREFIX ex: <http://example.org/>

SELECT ?s ?p ?o ?g
WHERE {
  GRAPH ?g {
    ?s ?p ?o .
  }
  FILTER (?g IN (ex:friends_graph, ex:family_graph))
}
```

### Exporting Data

```bash
# Export entire dataset to Turtle
oxirs export \
  --dataset my_dataset \
  --output export.ttl \
  --format turtle

# Export specific graph to N-Quads
oxirs export \
  --dataset my_dataset \
  --graph http://example.org/friends_graph \
  --output friends.nq \
  --format nquads

# Export compressed
oxirs export \
  --dataset my_dataset \
  --output export.nt.gz \
  --format ntriples \
  --compress gzip
```

### Dataset Statistics

```bash
# Get dataset statistics
curl http://localhost:3030/my_dataset/stats

# Response
{
  "dataset": "my_dataset",
  "triple_count": 1523456,
  "graph_count": 5,
  "subject_count": 523104,
  "predicate_count": 87,
  "object_count": 912345,
  "size_bytes": 52428800
}
```

---

## 🌐 Federation

**Goal**: Query multiple SPARQL endpoints simultaneously

### Basic Federation

```sparql
# Query local dataset AND remote DBpedia
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX dbo: <http://dbpedia.org/ontology/>

SELECT ?localPerson ?localName ?dbpediaInfo
WHERE {
  # Local data
  ?localPerson a foaf:Person ;
               foaf:name ?localName .

  # Federated query to DBpedia
  SERVICE <https://dbpedia.org/sparql> {
    ?dbpediaResource rdfs:label ?dbpediaInfo .
    FILTER (CONTAINS(LCASE(?dbpediaInfo), LCASE(?localName)))
    FILTER (LANG(?dbpediaInfo) = "en")
  }
}
LIMIT 10
```

### Multiple Endpoints

```sparql
# Query both DBpedia and Wikidata
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX dbo: <http://dbpedia.org/ontology/>

SELECT ?city ?population ?dbpediaInfo
WHERE {
  # Wikidata: Get cities with population
  SERVICE <https://query.wikidata.org/sparql> {
    ?city wdt:P31 wd:Q515 ;       # Instance of city
          wdt:P1082 ?population .  # Population
    FILTER (?population > 1000000)
  }

  # DBpedia: Get additional info
  SERVICE <https://dbpedia.org/sparql> {
    ?city dbo:abstract ?dbpediaInfo .
    FILTER (LANG(?dbpediaInfo) = "en")
  }
}
LIMIT 5
```

### Federation Configuration

```toml
# oxirs.toml
[federation]
enabled = true
timeout = 30          # Seconds
max_retries = 3
retry_delay = 500     # Milliseconds
cache_ttl = 3600      # Cache federated results for 1 hour

# Trusted endpoints (skip some security checks)
[[federation.trusted_endpoints]]
url = "https://dbpedia.org/sparql"

[[federation.trusted_endpoints]]
url = "https://query.wikidata.org/sparql"
```

### Federation Best Practices

1. **Use LIMIT** to avoid overwhelming remote endpoints
2. **Filter early** to reduce data transfer
3. **Cache results** for frequently accessed federated data
4. **Handle timeouts** gracefully with OPTIONAL

```sparql
# Good: Filter before federation
SERVICE <https://dbpedia.org/sparql> {
  ?resource a dbo:City .
  FILTER (?population > 1000000)  # Filter remote data
}

# Bad: Fetch all, filter locally
SERVICE <https://dbpedia.org/sparql> {
  ?resource a dbo:City ;
            dbo:population ?population .
}
FILTER (?population > 1000000)  # Filter after transfer
```

---

## ⚡ Performance Tuning

**Goal**: Optimize query performance for production workloads

### 1. Query Optimization

#### Use Selective Patterns First
```sparql
# Good: Most selective pattern first (fewer intermediate results)
SELECT ?person ?name ?age
WHERE {
  ?person foaf:name "Alice" .      # Selective (1 result)
  ?person foaf:age ?age .           # Filter on specific person
  ?person foaf:knows ?friend .      # Then find friends
}

# Bad: Least selective pattern first (many intermediate results)
SELECT ?person ?name ?age
WHERE {
  ?person foaf:knows ?friend .      # Broad (many results)
  ?person foaf:age ?age .           # Still many results
  ?person foaf:name "Alice" .       # Filter at end (wasteful)
}
```

#### Avoid OPTIONAL in Large Datasets
```sparql
# Good: Use FILTER NOT EXISTS instead of OPTIONAL + FILTER
SELECT ?person
WHERE {
  ?person a foaf:Person .
  FILTER NOT EXISTS { ?person foaf:email ?email }
}

# Bad: OPTIONAL generates many combinations
SELECT ?person
WHERE {
  ?person a foaf:Person .
  OPTIONAL { ?person foaf:email ?email }
  FILTER (!BOUND(?email))
}
```

### 2. Indexing and Storage

#### Enable Caching
```toml
# oxirs.toml
[datasets.my_dataset.options]
cache_size = 10000              # Cache 10K query results
buffer_pool_size = 1000         # Buffer pool for disk reads
enable_query_cache = true       # Cache compiled queries
```

#### Use Appropriate Store Type
```toml
# For < 10M triples: Memory store (fastest)
[[datasets]]
name = "small_dataset"
type = "memory"

# For 10M-1B triples: TDB store (disk-backed)
[[datasets]]
name = "large_dataset"
type = "tdb2"
location = "/var/lib/oxirs/data"
cache_size = 50000

# For > 1B triples: Cluster (distributed)
[[datasets]]
name = "huge_dataset"
type = "cluster"
cluster_nodes = ["node1:7000", "node2:7000", "node3:7000"]
replication_factor = 3
```

### 3. Query Profiling

```bash
# Enable query profiling
export OXIRS_PROFILE=1

# Run query
curl -X POST \
  -H "Content-Type: application/sparql-query" \
  -d @complex_query.rq \
  http://localhost:3030/my_dataset/query

# Check logs for profile
# INFO Query execution time: 523ms
#   - Parse: 12ms
#   - Optimize: 45ms
#   - Execute: 466ms
#     - Join (hash): 234ms
#     - Filter: 89ms
#     - Aggregate: 143ms
```

### 4. Batch Operations

```rust
// Bulk insert (Rust API)
use oxirs_core::store::{Store, TdbStore};

let mut store = TdbStore::open("/data")?;

// Good: Batch insert
store.begin_transaction()?;
for triple in &triples {
    store.insert_triple(triple)?;
}
store.commit()?;  // Single commit

// Bad: Individual inserts
for triple in &triples {
    store.insert_triple(triple)?;  // Commit each time (slow)
}
```

### 5. Connection Pooling

```toml
# oxirs.toml
[server.connection_pool]
max_connections = 100
min_idle = 10
connection_timeout = 30
idle_timeout = 600
```

### Performance Benchmarking

```bash
# Run built-in benchmarks
cd oxirs
cargo bench --bench sparql_queries

# Results
# test bench_simple_select ... bench:      8,234 ns/iter (+/- 421)
# test bench_join_2way     ... bench:     38,123 ns/iter (+/- 1,203)
# test bench_aggregation   ... bench:    142,567 ns/iter (+/- 5,432)
```

---

## 🏭 Production Deployment

**Goal**: Deploy OxiRS to production with monitoring and high availability

### Docker Deployment

#### 1. Build Production Image
```dockerfile
# Dockerfile (already provided at /deployments/docker/Dockerfile)
FROM rust:1.90-slim as builder
WORKDIR /app
COPY . .
RUN cargo build --release -p oxirs-fuseki

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 oxirs
COPY --from=builder /app/target/release/oxirs-fuseki /usr/local/bin/
USER oxirs
EXPOSE 3030
HEALTHCHECK --interval=30s CMD curl -f http://localhost:3030/$/ping || exit 1
CMD ["oxirs-fuseki", "--config", "/etc/oxirs/oxirs.toml"]
```

```bash
# Build image
docker build -t oxirs-fuseki:v0.1.0-rc.2 .

# Run production container
docker run -d \
  --name oxirs-prod \
  -p 3030:3030 \
  -v $(pwd)/oxirs.toml:/etc/oxirs/oxirs.toml:ro \
  -v oxirs-data:/var/lib/oxirs \
  --restart unless-stopped \
  --memory 16g \
  --cpus 8 \
  oxirs-fuseki:v0.1.0-rc.2
```

### Kubernetes Deployment

#### 2. Deploy to Kubernetes
See [DEPLOYMENT.md](DEPLOYMENT.md#kubernetes-deployment) for full manifests.

```bash
# Quick deployment
kubectl apply -f deployments/kubernetes/

# Verify deployment
kubectl get pods -n oxirs
# NAME                    READY   STATUS    RESTARTS   AGE
# oxirs-fuseki-0          1/1     Running   0          2m
# oxirs-fuseki-1          1/1     Running   0          2m
# oxirs-fuseki-2          1/1     Running   0          2m

# Check logs
kubectl logs -f oxirs-fuseki-0 -n oxirs

# Access via service
kubectl port-forward svc/oxirs-fuseki 3030:3030 -n oxirs
```

### Monitoring Setup

#### 3. Prometheus + Grafana

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'oxirs'
    static_configs:
      - targets: ['oxirs-fuseki:3030']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

```bash
# Deploy Prometheus
helm install prometheus prometheus-community/prometheus \
  --namespace monitoring \
  --values prometheus-values.yaml

# Deploy Grafana
helm install grafana grafana/grafana \
  --namespace monitoring \
  --set adminPassword=admin

# Import OxiRS dashboard
# Dashboard JSON: /monitoring/grafana/oxirs-dashboard.json
```

### High Availability Setup

#### 4. Multi-Node Cluster

```toml
# oxirs.toml (Node 1)
[cluster]
enabled = true
node_id = 1
listen_addr = "10.0.1.10:7000"
peers = ["10.0.1.11:7000", "10.0.1.12:7000"]
replication_factor = 3

# oxirs.toml (Node 2)
[cluster]
enabled = true
node_id = 2
listen_addr = "10.0.1.11:7000"
peers = ["10.0.1.10:7000", "10.0.1.12:7000"]
replication_factor = 3

# oxirs.toml (Node 3)
[cluster]
enabled = true
node_id = 3
listen_addr = "10.0.1.12:7000"
peers = ["10.0.1.10:7000", "10.0.1.11:7000"]
replication_factor = 3
```

### Backup and Recovery

```bash
# Automated backup script
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/oxirs"

# Export dataset
oxirs export \
  --dataset my_dataset \
  --output $BACKUP_DIR/backup_$TIMESTAMP.nq.gz \
  --format nquads \
  --compress gzip

# Upload to S3 (optional)
aws s3 cp $BACKUP_DIR/backup_$TIMESTAMP.nq.gz \
  s3://my-oxirs-backups/

# Keep only last 7 days
find $BACKUP_DIR -name "backup_*.nq.gz" -mtime +7 -delete
```

---

## 🔒 Security Hardening

**Goal**: Secure your SPARQL endpoint for production

### 1. Enable TLS/SSL

```toml
# oxirs.toml
[server.tls]
enabled = true
cert_path = "/etc/ssl/certs/oxirs.crt"
key_path = "/etc/ssl/private/oxirs.key"
min_version = "1.3"  # Enforce TLS 1.3
```

```bash
# Generate self-signed cert (for testing)
openssl req -x509 -newkey rsa:4096 \
  -keyout oxirs.key \
  -out oxirs.crt \
  -days 365 -nodes \
  -subj "/CN=oxirs.example.com"

# For production: Use Let's Encrypt
certbot certonly --standalone \
  -d oxirs.example.com \
  --email admin@example.com
```

### 2. Authentication (JWT)

```toml
# oxirs.toml
[server.auth]
enabled = true
method = "jwt"
jwt_secret_env = "JWT_SECRET"  # Read from environment
token_expiry = 3600            # 1 hour
```

```bash
# Generate JWT secret
export JWT_SECRET=$(openssl rand -base64 32)

# Create token (example using jwt.io)
{
  "sub": "user@example.com",
  "role": "admin",
  "exp": 1735689600
}

# Use token in requests
curl -X POST \
  -H "Authorization: Bearer eyJhbGc..." \
  -H "Content-Type: application/sparql-query" \
  -d "SELECT * WHERE { ?s ?p ?o } LIMIT 10" \
  https://oxirs.example.com:3030/my_dataset/query
```

### 3. Rate Limiting

```toml
# oxirs.toml
[server.rate_limit]
enabled = true
requests_per_second = 100   # Per IP
burst_size = 200
ban_duration = 300          # Ban for 5 minutes after abuse
```

### 4. Firewall Rules

```bash
# Allow only SPARQL endpoint (port 3030)
sudo ufw allow 3030/tcp

# Allow Prometheus metrics (internal only)
sudo ufw allow from 10.0.0.0/8 to any port 9090

# Deny all other traffic
sudo ufw default deny incoming
sudo ufw enable
```

### 5. Query Restrictions

```toml
# oxirs.toml
[query_restrictions]
max_query_time = 30              # Seconds
max_results = 10000              # Maximum result size
max_query_length = 10000         # Characters
disable_describe = false         # Allow DESCRIBE
disable_construct = false        # Allow CONSTRUCT
allowed_functions = ["*"]        # Allow all SPARQL functions
# Or restrict: allowed_functions = ["CONCAT", "REGEX", "FILTER"]
```

### Security Checklist

- [x] TLS 1.3 enabled
- [x] JWT authentication configured
- [x] Rate limiting enabled (100 req/s)
- [x] Firewall configured
- [x] Query timeouts set (30s)
- [x] Result limits enforced (10K max)
- [x] Security headers enabled (see [SECURITY_AUDIT.md](../SECURITY_AUDIT.md))
- [x] Secrets stored in environment variables (not config files)
- [x] Regular security audits scheduled

---

## 🌍 Cluster Setup

**Goal**: Deploy distributed RDF storage with Raft consensus

### Prerequisites

- 3+ servers (for quorum)
- Network connectivity between nodes
- Persistent storage on each node

### Step 1: Configure Each Node

**Node 1** (`10.0.1.10`):
```toml
# /etc/oxirs/oxirs.toml
[general]
node_name = "node1"

[server]
host = "0.0.0.0"
port = 3030

[cluster]
enabled = true
node_id = 1
listen_addr = "10.0.1.10:7000"
advertise_addr = "10.0.1.10:7000"

[[cluster.peers]]
node_id = 2
address = "10.0.1.11:7000"

[[cluster.peers]]
node_id = 3
address = "10.0.1.12:7000"

[cluster.raft]
heartbeat_interval = 1000      # ms
election_timeout_min = 2000    # ms
election_timeout_max = 4000    # ms

[cluster.replication]
factor = 3                     # Replicate to all 3 nodes
sync_writes = true             # Wait for quorum before ACK
```

**Node 2** (`10.0.1.11`) and **Node 3** (`10.0.1.12`): Similar config with adjusted `node_id` and `listen_addr`.

### Step 2: Start Cluster

```bash
# Node 1
ssh node1
oxirs-fuseki --config /etc/oxirs/oxirs.toml

# Node 2
ssh node2
oxirs-fuseki --config /etc/oxirs/oxirs.toml

# Node 3
ssh node3
oxirs-fuseki --config /etc/oxirs/oxirs.toml
```

### Step 3: Verify Cluster Health

```bash
# Check cluster status
curl http://10.0.1.10:3030/$/cluster/status

# Response
{
  "cluster_id": "oxirs-cluster-1",
  "leader": "node1",
  "nodes": [
    {"id": 1, "address": "10.0.1.10:7000", "state": "leader", "healthy": true},
    {"id": 2, "address": "10.0.1.11:7000", "state": "follower", "healthy": true},
    {"id": 3, "address": "10.0.1.12:7000", "state": "follower", "healthy": true}
  ],
  "quorum": true
}
```

### Step 4: Test Failover

```bash
# Simulate node failure (stop node1 - the leader)
ssh node1
sudo systemctl stop oxirs-fuseki

# Wait for election (~2-4 seconds)
sleep 5

# Check new leader (should be node2 or node3)
curl http://10.0.1.11:3030/$/cluster/status

# Response
{
  "leader": "node2",  # New leader elected
  "nodes": [
    {"id": 1, "state": "unavailable", "healthy": false},
    {"id": 2, "state": "leader", "healthy": true},
    {"id": 3, "state": "follower", "healthy": true}
  ],
  "quorum": true  # Still have quorum (2/3 nodes)
}

# Restart node1 (becomes follower)
ssh node1
sudo systemctl start oxirs-fuseki
```

### Step 5: Add New Node (Scale Out)

```bash
# Add node4 to existing cluster
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": 4,
    "address": "10.0.1.13:7000",
    "replication_factor": 3
  }' \
  http://10.0.1.10:3030/$/cluster/add-node

# Start node4 with cluster config
ssh node4
oxirs-fuseki --config /etc/oxirs/oxirs.toml
```

---

## 🎓 Advanced Topics

### Custom Functions in SPARQL

```rust
// Register custom function
use oxirs_arq::function::{CustomFunction, FunctionRegistry};

struct UppercaseFunction;

impl CustomFunction for UppercaseFunction {
    fn name(&self) -> &str { "http://example.org/uppercase" }

    fn execute(&self, args: &[Term]) -> Result<Term> {
        let text = args[0].as_str()?;
        Ok(Term::literal(text.to_uppercase()))
    }
}

// Register function
let registry = FunctionRegistry::global();
registry.register(Box::new(UppercaseFunction))?;
```

```sparql
# Use custom function in SPARQL
PREFIX ex: <http://example.org/>

SELECT ?upper
WHERE {
  ?person foaf:name ?name .
  BIND(ex:uppercase(?name) AS ?upper)
}
```

### Event Streaming Integration

```rust
// Stream RDF events to Kafka
use oxirs_stream::{StreamProcessor, KafkaConfig};

let kafka_config = KafkaConfig {
    brokers: vec!["localhost:9092".to_string()],
    topic: "rdf-updates".to_string(),
};

let mut processor = StreamProcessor::new(kafka_config)?;

// Stream all INSERT/DELETE operations
processor.stream_updates(|update| {
    println!("RDF Update: {:?}", update);
    // Publish to Kafka
})?;
```

---

## 📚 Additional Resources

### Documentation
- [Architecture Guide](ARCHITECTURE.md) - System design and internals
- [API Stability](API_STABILITY.md) - Version guarantees
- [Deployment Guide](DEPLOYMENT.md) - Production deployment
- [Migration Guide](MIGRATION_ALPHA3_BETA1.md) - Upgrade from Alpha.3

### Community
- GitHub: https://github.com/cool-japan/oxirs
- Discussions: https://github.com/cool-japan/oxirs/discussions
- Discord: https://discord.gg/oxirs

### Standards
- [SPARQL 1.1 Specification](https://www.w3.org/TR/sparql11-query/)
- [RDF 1.1 Concepts](https://www.w3.org/TR/rdf11-concepts/)
- [RDF-star Specification](https://www.w3.org/2021/12/rdf-star.html)

---

*Tutorials - October 12, 2025*
*Version: v0.1.0-rc.2*
*Happy SPARQL querying! 🚀*
