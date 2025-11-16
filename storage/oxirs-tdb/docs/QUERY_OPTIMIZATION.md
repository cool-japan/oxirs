# Query Optimization Guide for OxiRS TDB

## Overview

OxiRS TDB provides advanced query optimization capabilities through cost-based query planning. The query optimizer automatically selects the most efficient execution plan for triple pattern queries using statistics and heuristics.

## Quick Start

### Basic Usage (Automatic Optimization)

The query optimizer works automatically behind the scenes:

```rust
use oxirs_tdb::TdbStore;

let store = TdbStore::open("/path/to/data")?;

// Simple query - optimizer automatically selects best index
let results = store.query_triples(
    Some(&Term::iri("http://example.org/alice")),
    None,
    None,
)?;
```

### Advanced Usage with Hints

For fine-grained control, use query hints:

```rust
use oxirs_tdb::{TdbStore, QueryHints, IndexType};

let store = TdbStore::open("/path/to/data")?;

// Create hints to guide optimization
let hints = QueryHints::new()
    .with_index(IndexType::POS)  // Prefer POS index
    .with_limit(100)              // Limit results
    .with_offset(20)              // Skip first 20
    .with_stats(true);            // Collect execution statistics

// Execute with hints
let (results, stats) = store.query_triples_with_hints(
    None,
    Some(&Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")),
    None,
    &hints,
)?;

// Examine statistics
println!("Execution time: {}μs", stats.execution_time_us);
println!("Results found: {}", stats.results_found);
println!("Index used: {:?}", stats.index_used);
```

## Index Selection

OxiRS TDB maintains three indexes for optimal query performance:

### SPO Index (Subject-Predicate-Object)
**Best for**: Queries with subject bound
```rust
// Efficient - uses SPO index
store.query_triples(
    Some(&Term::iri("http://example.org/alice")),  // Subject bound
    None,
    None,
)?;
```

### POS Index (Predicate-Object-Subject)
**Best for**: Queries with predicate bound (especially when subject is not)
```rust
// Efficient - uses POS index
store.query_triples(
    None,
    Some(&Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")),  // Predicate bound
    None,
)?;
```

### OSP Index (Object-Subject-Predicate)
**Best for**: Queries with object bound (especially when subject and predicate are not)
```rust
// Efficient - uses OSP index
store.query_triples(
    None,
    None,
    Some(&Term::literal("Alice")),  // Object bound
)?;
```

## Optimization Levels

The query optimizer operates at three levels:

### Level 0: Disabled
- No optimization, uses simple heuristics
- Fastest planning, potentially slower execution
- Use for: Simple queries or when statistics are unavailable

```rust
// Configured at optimizer creation (internal use)
let optimizer = QueryOptimizer::new(statistics)
    .with_optimization_level(0);
```

### Level 1: Basic (Default for Small Datasets)
- Pattern-based optimization with simple statistics
- Balanced planning and execution speed
- Use for: Small to medium datasets (<1M triples)

### Level 2: Advanced (Default for Large Datasets)
- Cost-based optimization with detailed statistics
- Slower planning, optimal execution
- Plan caching for repeated queries
- Use for: Large datasets (>1M triples)

## Query Patterns and Performance

### Pattern: All Bound (Exact Match)
```rust
// Best case - O(log n) lookup
store.query_triples(
    Some(&subject),
    Some(&predicate),
    Some(&object),
)?;
```
**Performance**: O(log n) - Exact B+Tree lookup
**Estimated results**: 1 (or 0 if not found)

### Pattern: Two Bound
```rust
// Good - O(log n + k) range scan
store.query_triples(
    Some(&subject),
    Some(&predicate),
    None,  // Wildcard
)?;
```
**Performance**: O(log n + k) where k = number of results
**Estimated results**: ~1% of total triples

### Pattern: One Bound
```rust
// Moderate - larger range scan
store.query_triples(
    Some(&subject),
    None,
    None,
)?;
```
**Performance**: O(log n + k) where k can be large
**Estimated results**: ~10% of total triples

### Pattern: None Bound (Full Scan)
```rust
// Slowest - full index scan
store.query_triples(None, None, None)?;
```
**Performance**: O(n) - Full scan
**Estimated results**: All triples
**Recommendation**: Avoid in production, use hints with limit

## Statistics and Cost Estimation

The optimizer uses statistics to estimate query costs:

### Cardinality Statistics
- **Subject cardinality**: Number of distinct subjects
- **Predicate cardinality**: Number of distinct predicates
- **Object cardinality**: Number of distinct objects

### Cost Model

```
Total Cost = I/O Cost + CPU Cost

I/O Cost = log(total_triples) + log(estimated_results)
CPU Cost = estimated_results × comparison_factor
```

### Updating Statistics

Statistics are automatically updated:
- After every 1,000 insertions/deletions
- Can be manually triggered for critical operations

```rust
// Check if statistics need update
if store.statistics().needs_update() {
    // Statistics will be automatically updated on next operation
}
```

## Best Practices

### 1. Use Appropriate Indexes

Always bind the most selective term first:

```rust
// GOOD: Predicate is very selective (rdf:type)
store.query_triples(
    None,
    Some(&rdf_type),  // Very selective
    None,
)?;

// LESS OPTIMAL: Subject may be less selective
store.query_triples(
    Some(&some_subject),
    None,
    None,
)?;
```

### 2. Use Limits for Unbounded Queries

```rust
// GOOD: Limited results for exploration
let hints = QueryHints::new().with_limit(100);
let (results, _) = store.query_triples_with_hints(
    None, None, None,
    &hints,
)?;

// BAD: May return millions of triples
let results = store.query_triples(None, None, None)?;
```

### 3. Enable Query Caching

For repeated queries, caching provides significant speedup:

```rust
// Caching enabled by default
let config = TdbConfig::new("/path/to/data")
    .with_query_cache(true);

let store = TdbStore::open_with_config(config)?;

// First execution: ~100ms
let results1 = store.query_triples(Some(&s), Some(&p), None)?;

// Second execution: ~1ms (cached)
let results2 = store.query_triples(Some(&s), Some(&p), None)?;
```

### 4. Use Estimated Result Size

For memory-intensive operations, provide size hints:

```rust
let hints = QueryHints::new()
    .with_estimated_size(1000);  // Expect ~1000 results

// Optimizer can pre-allocate memory
let (results, _) = store.query_triples_with_hints(
    Some(&subject), None, None,
    &hints,
)?;
```

### 5. Monitor Query Performance

Enable statistics collection for analysis:

```rust
let hints = QueryHints::new().with_stats(true);

let (results, stats) = store.query_triples_with_hints(
    Some(&s), None, None,
    &hints,
)?;

// Analyze performance
if stats.execution_time_us > 10_000 {
    eprintln!("Slow query detected: {}μs", stats.execution_time_us);
    eprintln!("Index used: {:?}", stats.index_used);
    eprintln!("Results: {}", stats.results_found);
}
```

## Common Query Patterns

### Finding All Properties of a Resource

```rust
// Efficient with SPO index
let properties = store.query_triples(
    Some(&resource_iri),  // Subject bound
    None,                 // Any predicate
    None,                 // Any object
)?;
```

### Finding All Resources of a Type

```rust
// Efficient with POS index
let rdf_type = Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
let person_class = Term::iri("http://xmlns.com/foaf/0.1/Person");

let people = store.query_triples(
    None,                 // Any subject
    Some(&rdf_type),      // rdf:type predicate
    Some(&person_class),  // foaf:Person object
)?;
```

### Finding Resources with Specific Literal Values

```rust
// Efficient with OSP index
let name_value = Term::literal("Alice");

let resources = store.query_triples(
    None,              // Any subject
    None,              // Any predicate
    Some(&name_value), // Literal "Alice"
)?;
```

### Pagination

```rust
// Page 1 (results 0-99)
let page1_hints = QueryHints::new()
    .with_limit(100)
    .with_offset(0);

let (page1, _) = store.query_triples_with_hints(
    Some(&subject), None, None,
    &page1_hints,
)?;

// Page 2 (results 100-199)
let page2_hints = QueryHints::new()
    .with_limit(100)
    .with_offset(100);

let (page2, _) = store.query_triples_with_hints(
    Some(&subject), None, None,
    &page2_hints,
)?;
```

## Performance Tuning

### Buffer Pool Size

Larger buffer pools improve cache hit rates:

```rust
let config = TdbConfig::new("/path/to/data")
    .with_buffer_pool_size(10000);  // 10,000 pages (~40MB)

let store = TdbStore::open_with_config(config)?;
```

### Bloom Filters

Enable bloom filters for existence checks:

```rust
let config = TdbConfig::new("/path/to/data")
    .with_bloom_filters(true);

let store = TdbStore::open_with_config(config)?;

// Bloom filter eliminates 99% of non-existent lookups
if store.contains(&subject, &predicate, &object)? {
    // Triple exists
}
```

### Statistics Collection

Adjust statistics update frequency:

```rust
use oxirs_tdb::statistics::StatisticsConfig;

let stats_config = StatisticsConfig {
    enabled: true,
    update_threshold: 5000,  // Update every 5,000 modifications
    use_sampling: true,
    sample_rate: 0.1,        // 10% sampling for large datasets
};
```

## Troubleshooting

### Slow Queries

**Problem**: Query takes too long to execute

**Diagnosis**:
```rust
let hints = QueryHints::new().with_stats(true);
let (results, stats) = store.query_triples_with_hints(s, p, o, &hints)?;

println!("Pages accessed: {}", stats.pages_accessed);
println!("Index entries scanned: {}", stats.index_entries_scanned);
println!("Results found: {}", stats.results_found);
```

**Solutions**:
1. Ensure the most selective term is bound
2. Use appropriate index hints
3. Enable query result caching
4. Increase buffer pool size
5. Add limits to unbounded queries

### High Memory Usage

**Problem**: Query consumes too much memory

**Solutions**:
1. Use pagination with limits
2. Process results in batches
3. Reduce buffer pool size if system-wide
4. Use streaming APIs (future feature)

### Cache Thrashing

**Problem**: Cache hit rate is low

**Diagnosis**:
```rust
let cache_stats = store.query_cache_stats();
println!("Hit rate: {:.2}%", cache_stats.hit_rate * 100.0);
```

**Solutions**:
1. Increase cache size in configuration
2. Analyze query patterns for optimization opportunities
3. Consider pre-warming cache for common queries

## Advanced Topics

### Custom Cost Models

For specialized workloads, you can implement custom cost estimation (future enhancement).

### Query Plan Inspection

```rust
// Enable detailed statistics
let hints = QueryHints::new()
    .with_stats(true)
    .collect_stats(true);

let (results, stats) = store.query_triples_with_hints(s, p, o, &hints)?;

// Inspect execution plan
println!("Query Plan:");
println!("  Index: {:?}", stats.index_used);
println!("  Bloom filter: {}", stats.bloom_filter_used);
println!("  Cached: {}", stats.cached_result);
println!("  Execution: {}μs", stats.execution_time_us);
```

### Batch Query Optimization

For multiple related queries, consider batch processing (future enhancement).

## See Also

- [TDB Storage Architecture](./ARCHITECTURE.md)
- [Index Design](./INDEXES.md)
- [Statistics Collection](./STATISTICS.md)
- [Performance Benchmarks](./BENCHMARKS.md)
