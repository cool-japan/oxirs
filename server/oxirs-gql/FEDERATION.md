# Apollo Federation v2 Support in OxiRS

Complete Apollo Federation v2 implementation with automatic RDF integration, intelligent query planning, and production-ready optimization.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage Guide](#usage-guide)
- [RDF Integration](#rdf-integration)
- [Query Optimization](#query-optimization)
- [Production Deployment](#production-deployment)
- [API Reference](#api-reference)

## Overview

OxiRS provides enterprise-grade Apollo Federation v2 support, enabling you to build distributed GraphQL architectures with intelligent query planning and automatic schema generation from RDF ontologies.

### Why Federation?

- **Distributed Architecture**: Split monolithic GraphQL APIs into focused microservices
- **Team Autonomy**: Different teams can own different parts of the schema
- **Incremental Migration**: Gradually migrate from monolith to microservices
- **Performance**: Intelligent caching and query optimization across services

## Features

### ✅ Core Federation Support

- **Apollo Federation v2 Compliance**: Full support for Federation v2 specification
- **Federation Directives**: @key, @external, @requires, @provides, @shareable, @override
- **Entity Resolution**: Automatic _entities query for cross-service lookups
- **Service Introspection**: _service query for schema composition

### ✅ Intelligent Query Planning

- **Cost-Based Optimization**: Choose optimal execution strategies
- **Parallel Execution**: Maximize parallelism across subgraphs
- **Query Deduplication**: Eliminate redundant subgraph queries
- **Request Batching**: Combine multiple requests to same subgraph

### ✅ Production-Ready Caching

- **TTL-Based Cache**: Configurable time-to-live for query results
- **LRU Eviction**: Intelligent cache eviction based on usage
- **Cache Statistics**: Monitor hit rates, evictions, and size
- **Per-Query TTL**: Fine-grained cache control

### ✅ RDF Native Integration

- **Automatic Schema Generation**: Generate Federation schemas from RDF ontologies
- **Entity Auto-Detection**: Automatically identify entities from RDF classes
- **Semantic Mapping**: Map RDF properties to GraphQL fields
- **Multiple Formats**: Support for Turtle, RDF/XML, JSON-LD, and more

## Quick Start

### 1. Define Federation Schemas

```rust
use oxirs_gql::apollo_federation::{EntityKey, FederationSchemaBuilder, FederationVersion};

// Users subgraph
let users_schema = FederationSchemaBuilder::new(FederationVersion::V2)
    .add_entity("User", EntityKey::new(vec!["id".to_string()]))
    .add_entity("User", EntityKey::new(vec!["email".to_string()]))
    .build()?;

// Products subgraph
let products_schema = FederationSchemaBuilder::new(FederationVersion::V2)
    .add_entity("Product", EntityKey::new(vec!["sku".to_string()]))
    .add_shareable("Product")
    .build()?;
```

### 2. Compose Subgraphs

```rust
use oxirs_gql::federation_composer::{Subgraph, SubgraphRegistry, SchemaComposer};

let mut registry = SubgraphRegistry::new();

registry.register(Subgraph::new(
    "users",
    "http://localhost:4001/graphql",
    users_schema.to_sdl(),
))?;

registry.register(Subgraph::new(
    "products",
    "http://localhost:4002/graphql",
    products_schema.to_sdl(),
))?;

let composer = SchemaComposer::new(registry);
let supergraph = composer.compose()?;
```

### 3. Plan and Optimize Queries

```rust
use oxirs_gql::federation_composer::QueryPlanner;
use oxirs_gql::federation_optimizer::{FederationOptimizer, FederationCache, OptimizationStrategy};
use std::sync::Arc;
use std::time::Duration;

// Create query planner
let planner = QueryPlanner::new(supergraph.clone());

// Generate execution plan
let query = "{ user(id: \"123\") { id name } }";
let plan = planner.plan(query)?;

// Optimize with caching
let cache = Arc::new(FederationCache::new(
    Duration::from_secs(300), // 5 min TTL
    100,                       // 100MB max
));

let optimizer = FederationOptimizer::new(
    supergraph,
    cache,
    OptimizationStrategy::Balanced,
);

let optimized_plan = optimizer.optimize(plan)?;
```

## Architecture

### Federation Components

```
┌─────────────────────────────────────────────────────────────┐
│                      Federation Gateway                      │
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │   Schema    │  │    Query     │  │    Federation     │  │
│  │  Composer   │─→│   Planner    │─→│    Optimizer      │  │
│  └─────────────┘  └──────────────┘  └────────────────────┘  │
│         │                │                     │              │
│         ↓                ↓                     ↓              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            Subgraph Registry                        │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────┬───────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ↓               ↓               ↓
    ┌──────────┐    ┌──────────┐   ┌──────────┐
    │  Users   │    │ Products │   │ Reviews  │
    │ Subgraph │    │ Subgraph │   │ Subgraph │
    └──────────┘    └──────────┘   └──────────┘
```

### Query Execution Flow

1. **Client Request**: GraphQL query arrives at gateway
2. **Query Planning**: Decompose query into subgraph operations
3. **Optimization**: Apply caching, batching, parallelization
4. **Execution**: Execute subgraph queries (parallel when possible)
5. **Result Merging**: Combine results from multiple subgraphs
6. **Response**: Return unified result to client

## Usage Guide

### Creating Federation Schemas

#### Basic Entity

```rust
let schema = FederationSchemaBuilder::new(FederationVersion::V2)
    .add_entity("User", EntityKey::new(vec!["id".to_string()]))
    .build()?;
```

**Generated SDL:**
```graphql
extend schema
  @link(url: "https://specs.apollo.dev/federation/v2.0",
        import: ["@key"])

type User @key(fields: "id") {
  id: ID!
}

extend type Query {
  _entities(representations: [_Any!]!): [_Entity]!
  _service: _Service!
}
```

#### Multiple Keys

```rust
let schema = FederationSchemaBuilder::new(FederationVersion::V2)
    .add_entity("User", EntityKey::new(vec!["id".to_string()]))
    .add_entity("User", EntityKey::new(vec!["email".to_string()]))
    .build()?;
```

**Generated SDL:**
```graphql
type User @key(fields: "id")
type User @key(fields: "email")
```

#### External Fields and Requirements

```rust
let schema = FederationSchemaBuilder::new(FederationVersion::V2)
    .add_entity("Review", EntityKey::new(vec!["id".to_string()]))
    .add_external_field("Review", "author")
    .add_requires(
        "Review",
        "authorName",
        vec!["author".to_string(), "email".to_string()],
    )
    .build()?;
```

#### Shareable Types

```rust
let schema = FederationSchemaBuilder::new(FederationVersion::V2)
    .add_entity("Product", EntityKey::new(vec!["sku".to_string()]))
    .add_shareable("Product")
    .build()?;
```

**Generated SDL:**
```graphql
type Product @key(fields: "sku")
  @shareable
```

### Subgraph Registry

#### Register Subgraphs

```rust
let mut registry = SubgraphRegistry::new();

registry.register(Subgraph::new(
    "users",
    "http://users-service:4001/graphql",
    users_sdl,
).with_health_endpoint("/health"))?;

registry.register(Subgraph::new(
    "products",
    "http://products-service:4002/graphql",
    products_sdl,
).with_metadata("version", "1.0.0"))?;
```

#### Query Registry

```rust
// Get specific subgraph
if let Some(subgraph) = registry.get("users") {
    println!("Users service: {}", subgraph.url);
}

// List all subgraphs
for subgraph in registry.list() {
    println!("Subgraph: {} at {}", subgraph.name, subgraph.url);
}

// Get count
println!("Total subgraphs: {}", registry.count());
```

#### Unregister Subgraphs

```rust
let removed = registry.unregister("old-service")?;
println!("Removed: {}", removed.name);
```

### Schema Composition

#### Compose Supergraph

```rust
let composer = SchemaComposer::new(registry);
let supergraph = composer.compose()?;

println!("Supergraph SDL:\n{}", supergraph.sdl);
println!("Entities: {:?}", supergraph.entities.keys());
println!("Field ownership: {:?}", supergraph.field_ownership);
```

#### Access Entity Information

```rust
for (type_name, entity) in &supergraph.entities {
    println!("Entity: {}", type_name);
    println!("  Keys: {:?}", entity.keys);
    println!("  Provided by: {:?}", entity.subgraphs);
    println!("  Fields: {:?}", entity.fields_by_subgraph);
}
```

### Query Planning

#### Generate Execution Plan

```rust
let planner = QueryPlanner::new(supergraph);

let query = r#"
    query GetUserAndProducts {
        user(id: "123") {
            id
            name
        }
        products(category: "electronics") {
            sku
            name
        }
    }
"#;

let plan = planner.plan(query)?;

println!("Estimated cost: {}", plan.estimated_cost);
println!("Subgraphs involved: {:?}", plan.subgraphs);
println!("Execution plan: {:#?}", plan.root);
```

#### Analyze Plan

```rust
match &plan.root {
    QueryPlanNode::Parallel { nodes } => {
        println!("Parallel execution of {} operations", nodes.len());
    }
    QueryPlanNode::Sequence { nodes } => {
        println!("Sequential execution required");
    }
    QueryPlanNode::Fetch { subgraph, query, .. } => {
        println!("Direct fetch from {}", subgraph);
    }
    _ => {}
}
```

## Query Optimization

### Optimization Strategies

#### MinLatency

Maximize parallelism to minimize response time:

```rust
let optimizer = FederationOptimizer::new(
    supergraph,
    cache,
    OptimizationStrategy::MinLatency,
);
```

- Converts sequential operations to parallel where safe
- Best for: Real-time applications, user-facing APIs
- Trade-off: May increase total resource usage

#### MinRequests

Minimize subgraph calls via batching and deduplication:

```rust
let optimizer = FederationOptimizer::new(
    supergraph,
    cache,
    OptimizationStrategy::MinRequests,
);
```

- Batches multiple queries to same subgraph
- Eliminates duplicate requests
- Best for: High-traffic scenarios, cost optimization
- Trade-off: Slightly higher latency

#### Balanced

Balance between latency and requests:

```rust
let optimizer = FederationOptimizer::new(
    supergraph,
    cache,
    OptimizationStrategy::Balanced,
);
```

- Applies both parallelization and deduplication
- Best for: General-purpose applications
- Trade-off: Moderate on both dimensions

#### MinCost

Maximize caching to minimize compute cost:

```rust
let optimizer = FederationOptimizer::new(
    supergraph,
    cache,
    OptimizationStrategy::MinCost,
);
```

- Aggressive cache usage
- Eliminates redundant computations
- Best for: Budget-conscious deployments
- Trade-off: Stale data risk (configurable via TTL)

### Cache Management

#### Configure Cache

```rust
let cache = FederationCache::new(
    Duration::from_secs(300), // 5 minute TTL
    100,                       // 100MB max size
);
```

#### Set and Get

```rust
// Store query result
let data = serde_json::json!({"user": {"id": "123"}});
cache.set("query_key".to_string(), data);

// Retrieve cached result
if let Some(cached) = cache.get("query_key") {
    println!("Cache hit!");
}
```

#### Custom TTL

```rust
// Short TTL for volatile data
cache.set_with_ttl(
    "real_time_data".to_string(),
    data,
    Duration::from_secs(10),
);

// Long TTL for stable data
cache.set_with_ttl(
    "static_content".to_string(),
    data,
    Duration::from_secs(3600),
);
```

#### Monitor Cache

```rust
let stats = cache.stats();

println!("Cache Statistics:");
println!("  Total Requests: {}", stats.total_requests);
println!("  Hits: {} ({:.2}%)", stats.hits, stats.hit_rate() * 100.0);
println!("  Misses: {}", stats.misses);
println!("  Evictions: {}", stats.evictions);
println!("  Size: {} bytes", cache.size_bytes());
```

#### Clear Cache

```rust
cache.clear();
```

## RDF Integration

### Generate Federation from RDF Ontology

```rust
use oxirs_gql::schema::{SchemaGenerator, SchemaGenerationConfig};
use oxirs_gql::apollo_federation::FederationVersion;

async fn generate_from_rdf() -> anyhow::Result<()> {
    // Configure Federation
    let mut config = SchemaGenerationConfig::default();
    config.enable_federation = true;
    config.federation_version = FederationVersion::V2;

    // Create generator
    let generator = SchemaGenerator::new().with_config(config);

    // Generate from HTTP ontology
    let schema = generator
        .generate_federation_sdl_from_ontology("http://xmlns.com/foaf/0.1/")
        .await?;

    println!("Federation SDL:\n{}", schema);
    Ok(())
}
```

### Configure Entity Classes

```rust
let mut config = SchemaGenerationConfig::default();
config.enable_federation = true;
config.federation_version = FederationVersion::V2;

// Specify which RDF classes should be entities
config.entity_classes.insert("http://xmlns.com/foaf/0.1/Person".to_string());
config.entity_classes.insert("http://schema.org/Product".to_string());

// Specify custom key fields
config.entity_keys.insert(
    "http://xmlns.com/foaf/0.1/Person".to_string(),
    vec!["id".to_string(), "email".to_string()],
);
```

### Generate from RDF Store

```rust
use oxirs_gql::RdfStore;

let store = RdfStore::new()?;
// ... load RDF data into store ...

let generator = SchemaGenerator::new().with_config(config);
let schema = generator.generate_federation_sdl_from_store(&store)?;
```

### Auto-Detection

By default, all RDF classes are treated as entities with `id` as the key:

```rust
let mut config = SchemaGenerationConfig::default();
config.enable_federation = true;
// entity_classes empty = auto-detect all classes

let generator = SchemaGenerator::new().with_config(config);
```

## Production Deployment

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: oxirs-federation-gateway
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: gateway
        image: oxirs/federation-gateway:latest
        env:
        - name: CACHE_SIZE_MB
          value: "1000"
        - name: CACHE_TTL_SECONDS
          value: "300"
        - name: OPTIMIZATION_STRATEGY
          value: "Balanced"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

### Environment Configuration

```bash
# Cache settings
export FEDERATION_CACHE_SIZE_MB=1000
export FEDERATION_CACHE_TTL_SECONDS=300

# Optimization
export FEDERATION_OPTIMIZATION_STRATEGY=Balanced

# Subgraph endpoints
export USERS_SERVICE_URL=http://users:4001/graphql
export PRODUCTS_SERVICE_URL=http://products:4002/graphql
export REVIEWS_SERVICE_URL=http://reviews:4003/graphql
```

### Monitoring

```rust
use prometheus::{IntCounter, Registry, Histogram};

let registry = Registry::new();

// Track cache hits/misses
let cache_hits = IntCounter::new("federation_cache_hits", "Cache hits")?;
let cache_misses = IntCounter::new("federation_cache_misses", "Cache misses")?;

// Track query latency
let query_latency = Histogram::new(
    "federation_query_latency_seconds",
    "Query latency in seconds",
)?;

registry.register(Box::new(cache_hits))?;
registry.register(Box::new(cache_misses))?;
registry.register(Box::new(query_latency))?;
```

### Health Checks

```rust
async fn health_check(registry: &SubgraphRegistry) -> bool {
    for subgraph in registry.list() {
        if let Some(health_endpoint) = &subgraph.health_endpoint {
            // Check subgraph health
            if !check_endpoint(health_endpoint).await {
                return false;
            }
        }
    }
    true
}
```

## API Reference

### FederationSchemaBuilder

```rust
pub struct FederationSchemaBuilder;

impl FederationSchemaBuilder {
    pub fn new(version: FederationVersion) -> Self;
    pub fn add_entity(self, type_name: impl Into<String>, key: EntityKey) -> Self;
    pub fn add_external_field(self, type_name: impl Into<String>, field_name: impl Into<String>) -> Self;
    pub fn add_requires(self, type_name: impl Into<String>, field_name: impl Into<String>, required_fields: Vec<String>) -> Self;
    pub fn add_provides(self, type_name: impl Into<String>, field_name: impl Into<String>, provided_fields: Vec<String>) -> Self;
    pub fn add_shareable(self, type_name: impl Into<String>) -> Self;
    pub fn add_override(self, field_name: impl Into<String>, from_subgraph: impl Into<String>) -> Self;
    pub fn build(self) -> Result<FederationSchema>;
}
```

### SubgraphRegistry

```rust
pub struct SubgraphRegistry;

impl SubgraphRegistry {
    pub fn new() -> Self;
    pub fn register(&mut self, subgraph: Subgraph) -> Result<()>;
    pub fn unregister(&mut self, name: &str) -> Result<Subgraph>;
    pub fn get(&self, name: &str) -> Option<&Subgraph>;
    pub fn list(&self) -> Vec<&Subgraph>;
    pub fn count(&self) -> usize;
}
```

### FederationOptimizer

```rust
pub struct FederationOptimizer;

impl FederationOptimizer {
    pub fn new(supergraph: Supergraph, cache: Arc<FederationCache>, strategy: OptimizationStrategy) -> Self;
    pub fn optimize(&self, plan: QueryPlan) -> Result<OptimizedQueryPlan>;
    pub fn cache_stats(&self) -> CacheStats;
}
```

### FederationCache

```rust
pub struct FederationCache;

impl FederationCache {
    pub fn new(default_ttl: Duration, max_size_mb: usize) -> Self;
    pub fn get(&self, query_key: &str) -> Option<serde_json::Value>;
    pub fn set(&self, query_key: String, data: serde_json::Value);
    pub fn set_with_ttl(&self, query_key: String, data: serde_json::Value, ttl: Duration);
    pub fn clear(&self);
    pub fn stats(&self) -> CacheStats;
    pub fn size_bytes(&self) -> usize;
}
```

## Examples

See `examples/federation_example.rs` for a complete working example:

```bash
cargo run --example federation_example -p oxirs-gql
```

## Testing

Run Federation tests:

```bash
# All Federation tests
cargo test -p oxirs-gql apollo_federation
cargo test -p oxirs-gql federation_composer
cargo test -p oxirs-gql federation_optimizer

# Specific test
cargo test -p oxirs-gql test_federation_schema_builder_v2
```

## Performance

Expected performance characteristics:

- **Cache Hit Latency**: < 1ms
- **Cache Miss + Query**: ~10-50ms (depending on subgraph)
- **Parallel Execution**: Up to N-way parallelism (N = number of subgraphs)
- **Request Batching**: 30-70% reduction in subgraph calls
- **Memory Usage**: Configurable (default 100MB cache)

## License

MIT OR Apache-2.0

## Contributing

Contributions welcome! See CONTRIBUTING.md for guidelines.
