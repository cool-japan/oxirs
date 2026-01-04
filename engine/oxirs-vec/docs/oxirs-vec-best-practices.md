# OxiRS Vec - Best Practices Guide

**Version**: v0.1.0-rc.2
**Last Updated**: December 6, 2025

## Table of Contents

1. [Introduction](#introduction)
2. [Data Modeling](#data-modeling)
3. [Index Design](#index-design)
4. [Query Patterns](#query-patterns)
5. [Error Handling](#error-handling)
6. [Testing](#testing)
7. [Monitoring](#monitoring)
8. [Security](#security)
9. [Maintenance](#maintenance)
10. [Common Anti-Patterns](#common-anti-patterns)

---

## Introduction

This guide outlines production-proven best practices for building robust and efficient vector search applications with OxiRS Vec.

### Core Principles

1. **Plan for Scale**: Design for 10× your current requirements
2. **Measure Everything**: You can't optimize what you don't measure
3. **Test Thoroughly**: Test failure scenarios, not just happy paths
4. **Security First**: Authentication, authorization, and input validation
5. **Simplicity Wins**: Start simple, add complexity only when needed

---

## Data Modeling

### Vector Dimensionality

**Best Practice**: Choose dimensions based on your accuracy requirements

```rust
// ❌ DON'T: Use unnecessarily high dimensions
let vector = Vector::new(vec![0.5; 2048]); // Overkill for most tasks

// ✅ DO: Use appropriate dimensions
let vector = Vector::new(vec![0.5; 384]);  // Sentence transformers (good balance)
let vector = Vector::new(vec![0.5; 128]);  // Fast inference, lower memory
```

**Dimension Guidelines**:
- **128-256**: Fast inference, moderate accuracy
- **384-512**: Balanced (recommended)
- **768-1024**: High accuracy, slower
- **>1024**: Special use cases only (image embeddings, etc.)

### Vector Normalization

**Best Practice**: Always normalize vectors for cosine similarity

```rust
// ❌ DON'T: Use un-normalized vectors for cosine similarity
let query = Vector::new(vec![1.0, 2.0, 3.0]);
let results = index.search_knn(&query, 10)?;

// ✅ DO: Normalize vectors before insertion and search
let mut query = Vector::new(vec![1.0, 2.0, 3.0]);
query.normalize();
let results = index.search_knn(&query, 10)?;
```

### Metadata Management

**Best Practice**: Store searchable metadata separately

```rust
use std::collections::HashMap;

// ❌ DON'T: Embed metadata in vector
// (increases memory, can't filter efficiently)

// ✅ DO: Store metadata separately
struct VectorMetadata {
    id: String,
    category: String,
    timestamp: i64,
    tags: Vec<String>,
    custom_fields: HashMap<String, String>,
}

// Index vector
store.index_vector(metadata.id.clone(), vector)?;

// Store metadata in separate key-value store or database
metadata_store.insert(&metadata.id, metadata)?;
```

### Embedding Generation

**Best Practice**: Use consistent embedding models

```rust
use oxirs_vec::embeddings::{EmbeddingStrategy, EmbeddingManager};

// ✅ DO: Use the same model for indexing and querying
let strategy = EmbeddingStrategy::SentenceTransformer;
let mut manager = EmbeddingManager::new(strategy, 1000)?;

// Index documents
for doc in documents {
    let embedding = manager.get_embedding(&doc)?;
    store.index_vector(doc.id, embedding)?;
}

// Query with same model
let query_embedding = manager.get_embedding(&query)?;
let results = store.similarity_search_vector(&query_embedding, 10)?;
```

---

## Index Design

### Index Selection

**Best Practice**: Match index type to workload

```rust
use oxirs_vec::index::{IndexType, IndexConfig};

// Use Case 1: High-accuracy semantic search
let config = IndexConfig {
    index_type: IndexType::HNSW,
    dimensions: 384,
    // HNSW specific: high M and ef_construction
};

// Use Case 2: Memory-constrained environment
let config = IndexConfig {
    index_type: IndexType::ProductQuantization,
    dimensions: 384,
    // PQ: 16-32x memory reduction
};

// Use Case 3: Billion-scale datasets
let config = IndexConfig {
    index_type: IndexType::DiskANN,
    dimensions: 384,
    // DiskANN: disk-backed, minimal memory
};
```

### Index Building

**Best Practice**: Build indices offline, deploy as read-only

```rust
// ❌ DON'T: Build index in production serving path
fn serve_queries(store: &mut VectorStore) {
    // This blocks query serving!
    store.rebuild_index()?;
}

// ✅ DO: Build offline, load pre-built index
fn build_index_offline() -> anyhow::Result<()> {
    let mut store = VectorStore::new();

    // Build index (may take hours)
    for vector in vectors {
        store.index_vector(vector.id, vector.data)?;
    }

    // Save to disk
    store.save("/data/vectors.index")?;

    Ok(())
}

fn serve_queries() -> anyhow::Result<()> {
    // Load pre-built index (fast)
    let store = VectorStore::load("/data/vectors.index")?;

    // Serve queries
    Ok(())
}
```

### Index Updates

**Best Practice**: Use batching for index updates

```rust
use oxirs_vec::real_time_updates::{UpdateBatch, UpdateOperation};

// ❌ DON'T: Update index for every vector individually
for vector in new_vectors {
    store.index_vector(vector.id, vector.data)?;
    // Triggers index rebuild each time!
}

// ✅ DO: Batch updates
let mut batch = UpdateBatch::new();
for vector in new_vectors {
    batch.add(UpdateOperation::Insert {
        id: vector.id,
        data: vector.data,
    });
}

// Apply batch (single index rebuild)
store.apply_batch(batch)?;
```

---

## Query Patterns

### K-Nearest Neighbors

**Best Practice**: Choose k based on post-processing needs

```rust
// ❌ DON'T: Always use k=10
let results = store.similarity_search(query, 10)?;

// ✅ DO: Request more results for re-ranking
let k = 100; // Retrieve more candidates
let candidates = store.similarity_search(query, k)?;

// Re-rank with cross-encoder or filtering
let filtered = apply_filters(&candidates, &filters)?;
let reranked = rerank_with_cross_encoder(&filtered)?;
let top_10 = reranked.into_iter().take(10).collect();
```

### Query Expansion

**Best Practice**: Use hybrid search for better recall

```rust
use oxirs_vec::hybrid_search::{HybridSearchManager, HybridQuery};

// ❌ DON'T: Rely on semantic search alone
let results = store.similarity_search("rust programming", 10)?;

// ✅ DO: Combine keyword + semantic search
let hybrid_manager = HybridSearchManager::new()?;
let query = HybridQuery {
    text: "rust programming".to_string(),
    semantic_weight: 0.7,  // 70% semantic
    keyword_weight: 0.3,   // 30% keyword (BM25)
};

let results = hybrid_manager.search(&query, 10)?;
```

### Query Caching

**Best Practice**: Cache popular queries

```rust
use oxirs_vec::advanced_caching::{MultiLevelCache, CacheConfig};

// ✅ DO: Implement query result caching
let cache_config = CacheConfig {
    l1_capacity: 1000,   // Hot queries
    l2_capacity: 10000,  // Warm queries
    l3_capacity: 100000, // Cold queries
    ttl_seconds: 3600,   // 1 hour
};

let mut cache = MultiLevelCache::new(cache_config)?;

// Check cache before querying
if let Some(cached_results) = cache.get(&query_hash) {
    return Ok(cached_results);
}

// Query and cache result
let results = store.similarity_search(query, 10)?;
cache.put(query_hash, results.clone());
```

### Filtered Search

**Best Practice**: Apply pre-filtering when possible

```rust
use oxirs_vec::filtered_search::{FilterConfig, FilterCondition};

// ❌ DON'T: Filter after search (inefficient)
let all_results = store.similarity_search(query, 1000)?;
let filtered = all_results.into_iter()
    .filter(|r| matches_criteria(r))
    .take(10)
    .collect();

// ✅ DO: Pre-filter during search
let filter = FilterConfig {
    conditions: vec![
        FilterCondition::Equals("category".to_string(), "technology".to_string()),
        FilterCondition::GreaterThan("timestamp".to_string(), "2025-12-01".to_string()),
    ],
};

let results = store.filtered_search(query, 10, filter)?;
```

---

## Error Handling

### Graceful Degradation

**Best Practice**: Fallback to simpler strategies on errors

```rust
use anyhow::{Result, Context};

fn search_with_fallback(
    store: &VectorStore,
    query: &str,
    k: usize
) -> Result<Vec<(String, f32)>> {
    // Try GPU-accelerated search
    match store.gpu_search(query, k) {
        Ok(results) => Ok(results),
        Err(e) => {
            tracing::warn!("GPU search failed, falling back to CPU: {}", e);

            // Fallback to CPU search
            store.similarity_search(query, k)
                .context("Both GPU and CPU search failed")
        }
    }
}
```

### Input Validation

**Best Practice**: Validate all inputs

```rust
use oxirs_vec::validation::{VectorValidator, ValidationConfig};

fn validate_and_index(
    store: &mut VectorStore,
    id: String,
    vector: Vector
) -> Result<()> {
    // ✅ DO: Validate inputs
    let validator = VectorValidator::new(ValidationConfig {
        min_dimensions: 1,
        max_dimensions: 2048,
        allow_nan: false,
        allow_inf: false,
        max_magnitude: 1000.0,
    });

    validator.validate(&vector)
        .context("Invalid vector")?;

    // Validate ID
    if id.is_empty() || id.len() > 1000 {
        anyhow::bail!("Invalid ID length");
    }

    store.index_vector(id, vector)?;
    Ok(())
}
```

### Resource Limits

**Best Practice**: Enforce resource quotas

```rust
use oxirs_vec::multi_tenancy::{QuotaEnforcer, ResourceQuota};

fn create_tenant_quotas() -> Result<QuotaEnforcer> {
    let quota = ResourceQuota {
        max_vectors: 1_000_000,
        max_storage_bytes: 10_000_000_000, // 10 GB
        max_queries_per_second: 1000.0,
        max_index_size_bytes: 5_000_000_000, // 5 GB
    };

    QuotaEnforcer::new(quota)
}

fn check_quota_before_insert(
    enforcer: &QuotaEnforcer,
    tenant_id: &str,
    vector_count: usize
) -> Result<()> {
    if !enforcer.can_add_vectors(tenant_id, vector_count)? {
        anyhow::bail!("Quota exceeded for tenant {}", tenant_id);
    }
    Ok(())
}
```

---

## Testing

### Unit Testing

**Best Practice**: Test with realistic data

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_search() {
        // ✅ DO: Use realistic vector dimensions
        let dimensions = 384; // Sentence transformers

        // Create test vectors
        let query = Vector::new(vec![0.5; dimensions]);
        let doc1 = Vector::new(vec![0.6; dimensions]);
        let doc2 = Vector::new(vec![0.1; dimensions]);

        // Build index
        let mut store = VectorStore::new();
        store.index_vector("doc1".to_string(), doc1).unwrap();
        store.index_vector("doc2".to_string(), doc2).unwrap();

        // Search
        let results = store.similarity_search_vector(&query, 2).unwrap();

        // Verify
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "doc1"); // doc1 should be more similar
    }
}
```

### Integration Testing

**Best Practice**: Test end-to-end workflows

```rust
#[test]
fn test_full_indexing_and_search_workflow() {
    use tempfile::tempdir;

    // Create temporary directory
    let temp_dir = tempdir().unwrap();
    let index_path = temp_dir.path().join("test_index");

    // 1. Build index
    let mut store = VectorStore::new();
    for i in 0..1000 {
        let vector = Vector::new(vec![i as f32; 384]);
        store.index_vector(format!("vec_{}", i), vector).unwrap();
    }

    // 2. Save to disk
    store.save(&index_path).unwrap();

    // 3. Load from disk
    let loaded_store = VectorStore::load(&index_path).unwrap();

    // 4. Query
    let query = Vector::new(vec![500.0; 384]);
    let results = loaded_store.similarity_search_vector(&query, 10).unwrap();

    // 5. Verify
    assert_eq!(results.len(), 10);
    assert!(results[0].1 > 0.9); // High similarity expected
}
```

### Load Testing

**Best Practice**: Test under realistic load

```rust
use std::sync::Arc;
use std::thread;

#[test]
fn test_concurrent_queries() {
    let store = Arc::new(VectorStore::new());

    // Spawn multiple query threads
    let mut handles = vec![];
    for i in 0..10 {
        let store_clone = Arc::clone(&store);
        let handle = thread::spawn(move || {
            let query = Vector::new(vec![i as f32; 384]);
            for _ in 0..100 {
                store_clone.similarity_search_vector(&query, 10).unwrap();
            }
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
}
```

---

## Monitoring

### Key Metrics

**Best Practice**: Track business and technical metrics

```rust
use oxirs_vec::enhanced_performance_monitoring::{
    EnhancedPerformanceMonitor, MetricType
};

fn setup_comprehensive_monitoring() -> Result<EnhancedPerformanceMonitor> {
    let monitor = EnhancedPerformanceMonitor::new_default()?;

    // Business metrics
    monitor.track_metric("search_success_rate", MetricType::Gauge)?;
    monitor.track_metric("user_satisfaction", MetricType::Gauge)?;

    // Technical metrics
    monitor.track_metric("query_latency_ms", MetricType::Histogram)?;
    monitor.track_metric("index_size_bytes", MetricType::Gauge)?;
    monitor.track_metric("cache_hit_rate", MetricType::Gauge)?;

    // Quality metrics
    monitor.track_metric("recall_at_10", MetricType::Gauge)?;
    monitor.track_metric("precision_at_10", MetricType::Gauge)?;

    Ok(monitor)
}
```

### Alerting

**Best Practice**: Set up proactive alerts

```rust
use oxirs_vec::enhanced_performance_monitoring::{
    AlertManager, AlertThresholds, AlertSeverity
};

fn configure_alerts() -> Result<AlertManager> {
    let thresholds = AlertThresholds {
        max_query_latency_ms: 100.0,      // Critical
        max_memory_usage_percent: 85.0,   // Warning at 85%, critical at 95%
        min_recall: 0.90,                 // Warning if recall drops below 0.90
        max_error_rate: 0.01,             // Critical if >1% errors
    };

    let alert_manager = AlertManager::new(thresholds)?;

    // Configure alert destinations
    alert_manager.add_destination("email", "ops@example.com")?;
    alert_manager.add_destination("slack", "#alerts")?;

    Ok(alert_manager)
}
```

### Logging

**Best Practice**: Use structured logging

```rust
use tracing::{info, warn, error, instrument};

#[instrument(skip(store))]
fn search_with_logging(
    store: &VectorStore,
    query: &str,
    k: usize
) -> Result<Vec<(String, f32)>> {
    info!(query, k, "Starting search");

    let start = std::time::Instant::now();
    let results = store.similarity_search(query, k)?;
    let elapsed = start.elapsed();

    info!(
        query,
        k,
        num_results = results.len(),
        latency_ms = elapsed.as_millis(),
        "Search completed"
    );

    // Warn if latency is high
    if elapsed.as_millis() > 100 {
        warn!(
            query,
            latency_ms = elapsed.as_millis(),
            "Slow query detected"
        );
    }

    Ok(results)
}
```

---

## Security

### Authentication

**Best Practice**: Implement multi-layer authentication

```rust
use oxirs_vec::multi_tenancy::{AccessControl, Permission};

fn authenticate_request(
    acl: &AccessControl,
    user_id: &str,
    operation: Permission
) -> Result<()> {
    // Check if user has required permission
    if !acl.has_permission(user_id, operation)? {
        anyhow::bail!("Access denied for user {}", user_id);
    }

    Ok(())
}
```

### Rate Limiting

**Best Practice**: Implement per-tenant rate limiting

```rust
use oxirs_vec::multi_tenancy::RateLimiter;

fn rate_limit_request(
    limiter: &mut RateLimiter,
    tenant_id: &str
) -> Result<()> {
    if !limiter.allow_request(tenant_id)? {
        anyhow::bail!("Rate limit exceeded for tenant {}", tenant_id);
    }

    Ok(())
}
```

### Input Sanitization

**Best Practice**: Sanitize all user inputs

```rust
fn sanitize_query(query: &str) -> Result<String> {
    // Remove control characters
    let sanitized: String = query
        .chars()
        .filter(|c| !c.is_control())
        .collect();

    // Limit length
    if sanitized.len() > 10_000 {
        anyhow::bail!("Query too long");
    }

    Ok(sanitized)
}
```

---

## Maintenance

### Regular Maintenance Tasks

**Best Practice**: Automate maintenance

```rust
use oxirs_vec::compaction::CompactionManager;
use oxirs_vec::tiering::TieringManager;

async fn perform_maintenance(
    store: &mut VectorStore
) -> Result<()> {
    // 1. Compact index (reduce fragmentation)
    let compaction_manager = CompactionManager::new()?;
    compaction_manager.compact(store).await?;

    // 2. Optimize tiering (move cold data to cheaper storage)
    let tiering_manager = TieringManager::new_default()?;
    tiering_manager.optimize(store).await?;

    // 3. Update statistics
    store.update_statistics()?;

    // 4. Checkpoint WAL
    store.checkpoint_wal()?;

    Ok(())
}

// Schedule maintenance (e.g., daily at 2 AM)
// Run during low-traffic periods
```

### Index Rebuilding

**Best Practice**: Rebuild indices periodically

```rust
fn schedule_index_rebuild() -> Result<()> {
    // Rebuild indices weekly to:
    // 1. Incorporate accumulated updates
    // 2. Optimize index structure
    // 3. Reclaim deleted space

    // Steps:
    // 1. Build new index (blue)
    // 2. Verify new index quality
    // 3. Switch traffic to new index (blue-green deployment)
    // 4. Delete old index (green)

    Ok(())
}
```

### Backup Strategy

**Best Practice**: Implement automated backups

```rust
use oxirs_vec::persistence::BackupManager;

async fn automated_backup() -> Result<()> {
    let backup_manager = BackupManager::new()?;

    // Daily incremental backups
    backup_manager.create_incremental_backup("/backups/daily").await?;

    // Weekly full backups
    if is_sunday() {
        backup_manager.create_full_backup("/backups/weekly").await?;
    }

    // Retain backups for 30 days
    backup_manager.prune_old_backups(30).await?;

    Ok(())
}

fn is_sunday() -> bool {
    use chrono::Weekday;
    chrono::Local::now().weekday() == Weekday::Sun
}
```

---

## Common Anti-Patterns

### Anti-Pattern 1: Over-Indexing

```rust
// ❌ DON'T: Create too many specialized indices
let hnsw_index = HnswIndex::new(config)?;
let ivf_index = IvfIndex::new(config)?;
let lsh_index = LshIndex::new(config)?;
let pq_index = PQIndex::new(config)?;
// Too many indices = high memory usage

// ✅ DO: Use one or two complementary indices
let primary_index = HnswIndex::new(config)?; // High recall
let fallback_index = IvfIndex::new(config)?; // Fast search
```

### Anti-Pattern 2: Ignoring Errors

```rust
// ❌ DON'T: Ignore errors
let results = store.similarity_search(query, 10).unwrap();

// ✅ DO: Handle errors gracefully
let results = match store.similarity_search(query, 10) {
    Ok(r) => r,
    Err(e) => {
        error!("Search failed: {}", e);
        // Return cached results or empty vec
        vec![]
    }
};
```

### Anti-Pattern 3: Not Normalizing Vectors

```rust
// ❌ DON'T: Skip normalization for cosine similarity
let vector = Vector::new(vec![1.0, 2.0, 3.0]);
index.insert("doc1", vector)?;

// ✅ DO: Always normalize
let mut vector = Vector::new(vec![1.0, 2.0, 3.0]);
vector.normalize();
index.insert("doc1", vector)?;
```

### Anti-Pattern 4: Premature Optimization

```rust
// ❌ DON'T: Optimize before measuring
let complex_config = /* super optimized config */;
let index = HnswIndex::new(complex_config)?;

// ✅ DO: Start simple, measure, then optimize
let simple_config = HnswConfig::default();
let index = HnswIndex::new(simple_config)?;

// Measure performance
let metrics = benchmark(&index)?;

// Optimize based on measurements
if metrics.recall < 0.95 {
    // Increase accuracy
} else if metrics.latency_ms > 100 {
    // Optimize for speed
}
```

### Anti-Pattern 5: Not Testing Failure Scenarios

```rust
// ❌ DON'T: Only test happy paths
#[test]
fn test_search() {
    let results = store.similarity_search("query", 10).unwrap();
    assert!(!results.is_empty());
}

// ✅ DO: Test failure scenarios
#[test]
fn test_search_with_empty_index() {
    let store = VectorStore::new();
    let results = store.similarity_search("query", 10).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_search_with_invalid_query() {
    let result = store.similarity_search("", 10);
    assert!(result.is_err());
}

#[test]
fn test_search_during_index_rebuild() {
    // Test concurrent access during maintenance
}
```

---

## Quick Reference

### DO

- ✅ Normalize vectors for cosine similarity
- ✅ Batch index updates
- ✅ Validate all inputs
- ✅ Use caching for popular queries
- ✅ Monitor query latency and recall
- ✅ Test failure scenarios
- ✅ Implement rate limiting
- ✅ Use structured logging
- ✅ Automate backups
- ✅ Choose appropriate index type

### DON'T

- ❌ Skip error handling
- ❌ Use too many indices
- ❌ Ignore monitoring
- ❌ Build indices in serving path
- ❌ Over-optimize prematurely
- ❌ Use unsafe defaults
- ❌ Skip input validation
- ❌ Ignore resource limits
- ❌ Mix embedding models
- ❌ Skip testing

---

## Checklist

### Pre-Launch Checklist

- [ ] Vectors are normalized
- [ ] Appropriate index selected
- [ ] Caching configured
- [ ] Error handling implemented
- [ ] Rate limiting enabled
- [ ] Monitoring set up
- [ ] Alerts configured
- [ ] Logging structured
- [ ] Load tested
- [ ] Backups automated
- [ ] Security reviewed
- [ ] Documentation complete

---

## Next Steps

1. **Review Code**: Check your implementation against this guide
2. **Add Tests**: Ensure comprehensive test coverage
3. **Set Up Monitoring**: Implement metrics and alerts
4. **Load Test**: Verify performance under load
5. **Security Audit**: Review authentication and authorization

---

**Document Version**: 1.0
**OxiRS Vec Version**: v0.1.0-rc.2
**Last Updated**: December 6, 2025
