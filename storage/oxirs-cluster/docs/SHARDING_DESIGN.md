# Sharding Design for OxiRS Cluster

## Overview

This document describes the sharding implementation for OxiRS cluster, providing horizontal scalability through intelligent data partitioning based on semantic relationships.

## Architecture

### Core Components

1. **Shard Router** (`shard.rs`)
   - Routes triples to appropriate shards based on configured strategy
   - Maintains shard metadata and statistics
   - Provides query routing for efficient distributed queries

2. **Shard Manager** (`shard_manager.rs`)
   - Manages shard lifecycle (creation, splitting, merging, migration)
   - Coordinates shard operations across nodes
   - Handles automatic rebalancing and optimization

3. **Query Router** (`shard_routing.rs`)
   - Creates optimized query plans for distributed execution
   - Manages query caching and statistics
   - Supports federated queries across multiple endpoints

## Sharding Strategies

### 1. Hash-Based Sharding
```rust
ShardingStrategy::Hash { num_shards: 16 }
```
- Distributes data evenly using hash function
- Best for uniform data distribution
- Simple and predictable

### 2. Subject-Based Sharding
```rust
ShardingStrategy::Subject { num_shards: 16 }
```
- All triples with same subject go to same shard
- Optimizes subject-based queries
- Good for entity-centric access patterns

### 3. Predicate-Based Sharding
```rust
ShardingStrategy::Predicate {
    predicate_groups: HashMap<String, ShardId>
}
```
- Groups related predicates together
- Optimizes predicate-based queries
- Good for property-centric access patterns

### 4. Namespace-Based Sharding
```rust
ShardingStrategy::Namespace {
    namespace_mapping: HashMap<String, ShardId>
}
```
- Routes based on IRI namespace
- Keeps related data together
- Good for multi-tenant scenarios

### 5. Semantic Clustering
```rust
ShardingStrategy::Semantic {
    concept_clusters: Vec<ConceptCluster>,
    similarity_threshold: f64
}
```
- Groups semantically related concepts
- Uses similarity calculation for routing
- Optimizes for domain-specific access patterns

### 6. Hybrid Strategy
```rust
ShardingStrategy::Hybrid {
    primary: Box<ShardingStrategy>,
    secondary: Box<ShardingStrategy>
}
```
- Combines multiple strategies
- Falls back to secondary if primary doesn't match
- Provides flexibility for complex requirements

## Key Features

### 1. Semantic-Aware Partitioning
- Groups related data based on semantic relationships
- Reduces cross-shard queries
- Improves query performance

### 2. Dynamic Shard Management
- Automatic shard splitting when size exceeds threshold
- Shard merging for underutilized shards
- Live migration without downtime

### 3. Intelligent Query Routing
- Cost-based query optimization
- Parallel query execution
- Result caching for repeated queries

### 4. Fault Tolerance
- Configurable replication factor per shard
- Automatic failover to replica nodes
- Consistent hashing for stable routing

## Configuration

### Shard Manager Configuration
```rust
ShardManagerConfig {
    replication_factor: 3,
    max_triples_per_shard: 10_000_000,
    min_triples_per_shard: 100_000,
    max_imbalance_ratio: 2.0,
    auto_manage: true,
    check_interval_secs: 60,
}
```

### Query Optimization Hints
```rust
QueryOptimizationHints {
    use_index: true,
    parallel_execution: true,
    limit: Some(1000),
    order_by: Some("?date"),
    enable_cache: true,
    timeout_ms: Some(5000),
}
```

## Usage Example

```rust
use oxirs_cluster::{
    shard::{ShardRouter, ShardingStrategy},
    shard_manager::{ShardManager, ShardManagerConfig},
    shard_routing::QueryRouter,
};

// Create namespace-based sharding
let mut namespace_mapping = HashMap::new();
namespace_mapping.insert("http://schema.org/", 0);
namespace_mapping.insert("http://example.org/", 1);

let strategy = ShardingStrategy::Namespace { namespace_mapping };
let router = Arc::new(ShardRouter::new(strategy));

// Initialize shard manager
let manager = ShardManager::new(
    node_id,
    router,
    ShardManagerConfig::default(),
    storage,
    network,
);

// Store triple - automatically routed to correct shard
manager.store_triple(triple).await?;

// Query with automatic shard routing
let results = manager.query_triples(
    Some("http://schema.org/Person"),
    None,
    None,
).await?;
```

## Performance Characteristics

### Routing Performance
- O(1) for hash-based routing
- O(1) for namespace lookup
- O(n) for semantic similarity calculation

### Query Performance
- Single-shard queries: No overhead
- Multi-shard queries: Parallel execution
- Federated queries: Network latency dependent

### Storage Efficiency
- Even distribution with hash sharding
- Locality optimization with semantic sharding
- Compression within shards

## Monitoring and Statistics

### Shard Statistics
```rust
ShardingStatistics {
    total_shards: 16,
    active_shards: 16,
    total_triples: 100_000_000,
    total_size: 10_737_418_240, // 10GB
    distribution: Vec<ShardDistribution>,
}
```

### Query Routing Statistics
```rust
RoutingStatistics {
    total_queries: 1_000_000,
    single_shard_queries: 800_000,
    multi_shard_queries: 200_000,
    avg_shards_per_query: 1.4,
    cache_hit_rate: 0.85,
    avg_latency_ms: 12.5,
}
```

## Future Enhancements

1. **Machine Learning Integration**
   - Learn optimal shard placement from query patterns
   - Predictive shard splitting/merging
   - Adaptive routing strategies

2. **Advanced Partitioning**
   - Time-based partitioning for temporal data
   - Geographic partitioning for spatial data
   - Multi-dimensional partitioning

3. **Cross-Shard Optimization**
   - Distributed join optimization
   - Shard-aware query planning
   - Result set streaming

4. **Global Secondary Indexes**
   - Cross-shard index structures
   - Bloom filters for existence queries
   - Materialized views across shards

## Testing

The sharding implementation includes comprehensive tests:
- Unit tests for routing algorithms
- Integration tests for shard operations
- Performance benchmarks
- Fault injection tests

Run tests with:
```bash
cargo test -p oxirs-cluster shard
```

## References

1. Amazon DynamoDB: Sharding and partitioning strategies
2. MongoDB: Sharding architecture
3. Apache Cassandra: Distributed data partitioning
4. Google Spanner: Globally distributed database