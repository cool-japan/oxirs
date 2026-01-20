# OxiRS-Star Troubleshooting Guide

[![Documentation](https://docs.rs/oxirs-star/badge.svg)](https://docs.rs/oxirs-star)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

**Comprehensive troubleshooting guide for OxiRS-Star RDF-star implementation with solutions for common issues and performance optimization tips.**

## Table of Contents

- [Quick Diagnostics](#quick-diagnostics)
- [Installation Issues](#installation-issues)
- [Parsing Problems](#parsing-problems)
- [Query Issues](#query-issues)
- [Performance Problems](#performance-problems)
- [Memory Issues](#memory-issues)
- [Concurrency Issues](#concurrency-issues)
- [Configuration Problems](#configuration-problems)
- [Integration Issues](#integration-issues)
- [Error Messages Reference](#error-messages-reference)
- [Debugging Tools](#debugging-tools)
- [Best Practices](#best-practices)

## Quick Diagnostics

### Health Check Script

```rust
use oxirs_star::{StarStore, StarTriple, StarTerm, StarQueryEngine};

/// Quick health check for OxiRS-Star installation
fn run_health_check() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 OxiRS-Star Health Check");
    
    // 1. Basic functionality test
    println!("├─ Testing basic functionality...");
    let mut store = StarStore::new();
    let triple = StarTriple::new(
        StarTerm::iri("http://example.org/test")?,
        StarTerm::iri("http://example.org/prop")?,
        StarTerm::literal("test value"),
    )?;
    store.insert(&triple)?;
    assert_eq!(store.size(), 1);
    println!("│  ✅ Basic operations working");
    
    // 2. RDF-star functionality test
    println!("├─ Testing RDF-star functionality...");
    let quoted_triple = StarTriple::new(
        StarTerm::iri("http://example.org/subject")?,
        StarTerm::iri("http://example.org/predicate")?,
        StarTerm::iri("http://example.org/object")?,
    )?;
    let meta_triple = StarTriple::new(
        StarTerm::quoted_triple(quoted_triple),
        StarTerm::iri("http://example.org/confidence")?,
        StarTerm::typed_literal("0.9", "http://www.w3.org/2001/XMLSchema#decimal")?,
    )?;
    store.insert(&meta_triple)?;
    println!("│  ✅ RDF-star operations working");
    
    // 3. Parser test
    println!("├─ Testing parser functionality...");
    let turtle_star = r#"
        @prefix ex: <http://example.org/> .
        <<ex:alice ex:knows ex:bob>> ex:confidence 0.95 .
    "#;
    let parsed_triples = store.parse_turtle_star(turtle_star)?;
    assert!(!parsed_triples.is_empty());
    println!("│  ✅ Parser functionality working");
    
    // 4. Query engine test
    println!("├─ Testing query engine...");
    let engine = StarQueryEngine::new(&store);
    let query = "SELECT * WHERE { ?s ?p ?o }";
    let results = engine.execute(query)?;
    assert!(!results.is_empty());
    println!("│  ✅ Query engine working");
    
    // 5. Memory usage check
    println!("└─ Checking memory usage...");
    let memory_info = store.get_memory_usage()?;
    println!("   📊 Memory usage: {} KB", memory_info.total_kb);
    if memory_info.total_kb > 100_000 {
        println!("   ⚠️  High memory usage detected");
    } else {
        println!("   ✅ Memory usage normal");
    }
    
    println!("\n🎉 Health check completed successfully!");
    Ok(())
}
```

### System Information

```rust
use oxirs_star::diagnostics::SystemInfo;

fn print_system_info() -> Result<(), Box<dyn std::error::Error>> {
    let info = SystemInfo::collect()?;
    
    println!("📋 System Information");
    println!("├─ OxiRS-Star version: {}", info.oxirs_star_version);
    println!("├─ Rust version: {}", info.rust_version);
    println!("├─ Target: {}", info.target_triple);
    println!("├─ Features enabled: {:?}", info.enabled_features);
    println!("├─ Available memory: {} GB", info.available_memory_gb);
    println!("├─ CPU cores: {}", info.cpu_cores);
    println!("└─ Optimal thread count: {}", info.recommended_threads);
    
    Ok(())
}
```

## Installation Issues

### Compilation Errors

**Problem**: Compilation fails with dependency conflicts
```bash
error: failed to resolve dependencies
```

**Solution**:
```bash
# Clear cache and rebuild
cargo clean
rm Cargo.lock
cargo build --release

# If still failing, check Rust version
rustc --version  # Should be 1.70+
rustup update

# Check for conflicting features
cargo check --no-default-features
cargo check --all-features
```

**Problem**: Missing system dependencies
```bash
error: could not find system library 'xyz'
```

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install build-essential pkg-config libssl-dev

# macOS
brew install pkg-config openssl

# Update environment if needed
export PKG_CONFIG_PATH="/usr/local/opt/openssl/lib/pkgconfig"
```

### Feature Configuration Issues

**Problem**: RDF-star features not available

**Solution**:
```toml
# Cargo.toml
[dependencies]
oxirs-star = { version = "0.1", features = ["full"] }

# Or specific features
oxirs-star = { 
    version = "0.1", 
    features = ["turtle-star", "sparql-star", "reification"] 
}
```

**Verification**:
```rust
#[cfg(feature = "turtle-star")]
fn test_turtle_star_available() {
    // This will compile only if turtle-star feature is enabled
    use oxirs_star::parser::TurtleStarParser;
    let parser = TurtleStarParser::new();
}
```

## Parsing Problems

### Malformed RDF-star Syntax

**Problem**: Parse errors with quoted triples
```
ParseError: Expected '>>' but found '>'
```

**Debugging**:
```rust
use oxirs_star::parser::{TurtleStarParser, ParseOptions};

let malformed_data = r#"
    @prefix ex: <http://example.org/> .
    <<ex:alice ex:knows ex:bob> ex:confidence 0.9 .  # Missing closing >>
"#;

// Enable detailed error reporting
let options = ParseOptions {
    strict_mode: false,
    report_line_numbers: true,
    collect_detailed_errors: true,
    ..Default::default()
};

let parser = TurtleStarParser::with_options(options);
match parser.parse(malformed_data) {
    Ok(triples) => println!("Parsed {} triples", triples.len()),
    Err(e) => {
        println!("Parse error at line {}: {}", e.line_number, e.message);
        println!("Context: {}", e.context);
        
        // Suggest correction
        if e.message.contains("Expected '>>'") {
            println!("💡 Tip: Check that all quoted triples are properly closed with '>>'");
        }
    }
}
```

**Solution**: Enable error recovery parsing
```rust
let recovery_parser = TurtleStarParser::with_error_recovery();
let (triples, errors) = recovery_parser.parse_with_errors(malformed_data)?;

println!("Successfully parsed {} triples", triples.len());
println!("Encountered {} recoverable errors", errors.len());

for error in errors {
    println!("⚠️  Recovered from error at line {}: {}", error.line, error.message);
}
```

### Large File Parsing Issues

**Problem**: Out of memory when parsing large files
```
thread 'main' panicked at 'allocation failed'
```

**Solution**: Use streaming parser
```rust
use oxirs_star::parser::streaming::StreamingTurtleStarParser;

let streaming_parser = StreamingTurtleStarParser::new()
    .with_buffer_size(64 * 1024) // 64KB buffer
    .with_memory_limit(512 * 1024 * 1024); // 512MB limit

let file = std::fs::File::open("large_file.ttls")?;
for result in streaming_parser.parse_file(file)? {
    match result {
        Ok(triple) => {
            // Process one triple at a time
            process_triple_immediately(triple)?;
        },
        Err(e) => {
            eprintln!("Parse error: {}", e);
            // Continue processing or break based on strategy
        }
    }
}
```

### Encoding Issues

**Problem**: Unicode characters not parsed correctly

**Solution**:
```rust
use oxirs_star::parser::ParseOptions;

let options = ParseOptions {
    strict_unicode: false,
    normalize_unicode: true,
    encoding: Some("UTF-8".to_string()),
    ..Default::default()
};

// Handle files with BOM
let content = std::fs::read("file_with_bom.ttls")?;
let content_str = if content.starts_with(&[0xEF, 0xBB, 0xBF]) {
    // Remove UTF-8 BOM
    std::str::from_utf8(&content[3..])?
} else {
    std::str::from_utf8(&content)?
};

let triples = parser.parse(content_str)?;
```

## Query Issues

### SPARQL-star Syntax Problems

**Problem**: Query fails with syntax error
```
SyntaxError: Unexpected token '<<' at position 45
```

**Debugging**:
```rust
use oxirs_star::query::{StarQueryEngine, QueryAnalyzer};

let problematic_query = r#"
    SELECT ?stmt ?conf WHERE {
        ?stmt <<ex:confidence>> ?conf .  # Wrong syntax
    }
"#;

let analyzer = QueryAnalyzer::new();
let analysis = analyzer.analyze(problematic_query)?;

if !analysis.is_valid {
    println!("❌ Query syntax errors:");
    for error in analysis.syntax_errors {
        println!("  Line {}: {}", error.line, error.message);
        println!("  Suggestion: {}", error.suggestion);
    }
}

// Correct syntax
let correct_query = r#"
    SELECT ?stmt ?conf WHERE {
        ?stmt ex:confidence ?conf .
        ?stmt { ?s ex:prop ?o }  # Correct RDF-star syntax
    }
"#;
```

**Common SPARQL-star patterns**:
```sparql
# ✅ Correct patterns
SELECT ?qt ?conf WHERE {
    ?qt ex:confidence ?conf .
    ?qt { ?s ex:knows ?o }
}

# ✅ Nested quoted triples
SELECT ?meta WHERE {
    ?meta ex:describes <<?person ex:believes <<?s ex:prop ?o>>>> .
}

# ❌ Common mistakes
SELECT ?qt WHERE {
    <<qt>> ex:confidence ?conf .    # Wrong: variables in quoted triples
}

SELECT ?qt WHERE {
    ?qt <<ex:confidence ?conf>> .   # Wrong: quoted triple as predicate
}
```

### Query Performance Issues

**Problem**: Queries taking too long to execute

**Diagnosis**:
```rust
use oxirs_star::query::{StarQueryEngine, QueryProfiler};

let profiler = QueryProfiler::new();
let engine = StarQueryEngine::with_profiler(&store, profiler);

let slow_query = "SELECT * WHERE { ?s ?p ?o }";
let start = std::time::Instant::now();
let results = engine.execute(slow_query)?;
let duration = start.elapsed();

println!("Query executed in {:?}", duration);

let profile = engine.get_query_profile()?;
println!("Query plan: {:#?}", profile.execution_plan);
println!("Index usage: {:?}", profile.index_usage);
println!("Bottlenecks: {:?}", profile.bottlenecks);

if duration > std::time::Duration::from_secs(5) {
    println!("🐌 Slow query detected!");
    
    // Optimization suggestions
    if profile.bottlenecks.contains(&"sequential_scan") {
        println!("💡 Consider adding indices for frequently queried patterns");
    }
    
    if profile.bottlenecks.contains(&"cartesian_product") {
        println!("💡 Add more selective filters to reduce cartesian products");
    }
}
```

**Solutions**:

1. **Add indices**:
```rust
use oxirs_star::indexing::{IndexManager, IndexType};

let mut index_manager = IndexManager::new();
index_manager.create_index(
    "subject_predicate_index",
    IndexType::BTree,
    &["subject", "predicate"]
)?;
```

2. **Optimize query structure**:
```sparql
# ❌ Inefficient
SELECT ?person ?name ?age WHERE {
    ?person foaf:name ?name .
    ?person foaf:age ?age .
    FILTER(?age > 18)
    FILTER(contains(?name, "Smith"))
}

# ✅ More efficient (filters earlier)
SELECT ?person ?name ?age WHERE {
    ?person foaf:age ?age .
    FILTER(?age > 18)
    ?person foaf:name ?name .
    FILTER(contains(?name, "Smith"))
}
```

3. **Use query hints**:
```sparql
SELECT ?s ?p ?o WHERE {
    ?s ?p ?o .
    FILTER(?s = ex:specificResource)
} 
OPTION (INDEX "subject_index")
```

### Federation Issues

**Problem**: Federated queries timing out or failing

**Debugging**:
```rust
use oxirs_star::query::federation::{FederatedQueryEngine, EndpointStatus};

let federated_engine = FederatedQueryEngine::new();

// Check endpoint health
for endpoint_name in federated_engine.list_endpoints() {
    match federated_engine.check_endpoint_status(&endpoint_name).await {
        Ok(EndpointStatus::Healthy) => {
            println!("✅ Endpoint {} is healthy", endpoint_name);
        },
        Ok(EndpointStatus::Slow(duration)) => {
            println!("🐌 Endpoint {} is slow: {:?}", endpoint_name, duration);
        },
        Ok(EndpointStatus::Unreachable) => {
            println!("❌ Endpoint {} is unreachable", endpoint_name);
        },
        Err(e) => {
            println!("⚠️  Error checking endpoint {}: {}", endpoint_name, e);
        }
    }
}

// Configure timeouts and retries
let federated_config = FederationConfig {
    default_timeout: std::time::Duration::from_secs(30),
    max_retries: 3,
    circuit_breaker_threshold: 5,
    enable_fallback: true,
};

let engine = FederatedQueryEngine::with_config(federated_config);
```

## Performance Problems

### Memory Usage Issues

**Problem**: High memory consumption

**Diagnosis**:
```rust
use oxirs_star::diagnostics::{MemoryProfiler, MemoryReport};

let profiler = MemoryProfiler::new();
let baseline = profiler.capture_baseline()?;

// Perform operations
let mut store = StarStore::new();
for i in 0..1_000_000 {
    let triple = create_large_triple(i)?;
    store.insert(&triple)?;
}

let after_insert = profiler.capture_snapshot()?;
let report = MemoryReport::diff(&baseline, &after_insert);

println!("Memory usage analysis:");
println!("├─ Total allocated: {} MB", report.total_allocated_mb);
println!("├─ Peak usage: {} MB", report.peak_usage_mb);
println!("├─ Current usage: {} MB", report.current_usage_mb);
println!("├─ Largest allocation: {} KB", report.largest_allocation_kb);
println!("└─ Fragmentation: {:.1}%", report.fragmentation_percent);

if report.current_usage_mb > 1000 {
    println!("⚠️  High memory usage detected!");
    
    // Memory optimization suggestions
    println!("💡 Optimization suggestions:");
    if report.fragmentation_percent > 20.0 {
        println!("  - Consider using memory-mapped storage");
        println!("  - Enable compression");
    }
    
    if report.largest_allocation_kb > 10_000 {
        println!("  - Use streaming for large operations");
        println!("  - Implement batch processing");
    }
}
```

**Solutions**:

1. **Enable memory-mapped storage**:
```rust
use oxirs_star::storage::{MemoryMappedStore, StorageOptions};

let options = StorageOptions {
    enable_mmap: true,
    mmap_file_path: "/tmp/rdf_star_store.db".to_string(),
    mmap_size: 2 * 1024 * 1024 * 1024, // 2GB
    enable_compression: true,
};

let mmap_store = MemoryMappedStore::with_options(options)?;
```

2. **Use streaming operations**:
```rust
use oxirs_star::streaming::StreamingStore;

let streaming_store = StreamingStore::new()
    .with_memory_limit(512 * 1024 * 1024) // 512MB limit
    .with_disk_cache("/tmp/rdf_star_cache");

// Large operations use disk when memory limit reached
for chunk in large_dataset.chunks(10000) {
    streaming_store.insert_batch(chunk)?;
}
```

3. **Configure garbage collection**:
```rust
use oxirs_star::memory::{GarbageCollector, GcConfig};

let gc_config = GcConfig {
    trigger_threshold_mb: 1000,
    target_memory_mb: 500,
    aggressive_mode: false,
};

let gc = GarbageCollector::with_config(gc_config);
gc.schedule_periodic_collection(std::time::Duration::from_secs(60))?;
```

### Query Performance Optimization

**Problem**: Slow query execution

**Profiling**:
```rust
use oxirs_star::query::performance::{QueryProfiler, OptimizationHints};

let profiler = QueryProfiler::detailed();
let engine = StarQueryEngine::with_profiler(&store, profiler);

let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o . ?s ex:type ?type }";
let results = engine.execute(query)?;

let profile = engine.get_detailed_profile()?;
println!("Query performance analysis:");
println!("├─ Execution time: {:?}", profile.total_time);
println!("├─ Planning time: {:?}", profile.planning_time);
println!("├─ Index lookups: {}", profile.index_lookups);
println!("├─ Sequential scans: {}", profile.sequential_scans);
println!("├─ Join operations: {}", profile.join_operations);
println!("└─ Memory peak: {} MB", profile.peak_memory_mb);

// Get optimization hints
let hints = OptimizationHints::analyze(&profile, &store)?;
for hint in hints {
    println!("💡 {}", hint.description);
    println!("   Expected improvement: {}", hint.expected_improvement);
}
```

**Optimization strategies**:

1. **Index optimization**:
```rust
use oxirs_star::indexing::{IndexOptimizer, IndexStrategy};

let optimizer = IndexOptimizer::new(&store);
let recommendations = optimizer.analyze_query_patterns()?;

for rec in recommendations {
    println!("Index recommendation: {}", rec.description);
    println!("  Estimated speedup: {}x", rec.estimated_speedup);
    println!("  Memory cost: {} MB", rec.memory_cost_mb);
    
    if rec.estimated_speedup > 2.0 && rec.memory_cost_mb < 100 {
        // Apply beneficial index
        optimizer.create_recommended_index(&rec)?;
    }
}
```

2. **Query rewriting**:
```rust
use oxirs_star::query::optimization::{QueryRewriter, RewriteRules};

let rewriter = QueryRewriter::new()
    .with_rule(RewriteRules::PushdownFilters)
    .with_rule(RewriteRules::EliminateSubqueries)
    .with_rule(RewriteRules::OptimizeJoinOrder);

let original_query = "SELECT ?s ?name WHERE { ?s foaf:name ?name . ?s foaf:age ?age . FILTER(?age > 18) }";
let optimized_query = rewriter.rewrite(original_query)?;

println!("Original: {}", original_query);
println!("Optimized: {}", optimized_query);

// Use optimized query
let results = engine.execute(&optimized_query)?;
```

## Memory Issues

### Memory Leaks

**Detection**:
```rust
use oxirs_star::diagnostics::LeakDetector;

let leak_detector = LeakDetector::new();
leak_detector.start_monitoring()?;

// Perform operations that might leak
for i in 0..1000 {
    let store = StarStore::new();
    // ... operations ...
    // store should be dropped here
}

let leak_report = leak_detector.generate_report()?;
if leak_report.potential_leaks > 0 {
    println!("⚠️  Potential memory leaks detected:");
    for leak in leak_report.leak_sources {
        println!("  {} allocations not freed: {}", leak.count, leak.location);
    }
}
```

**Solutions**:
```rust
// Ensure proper cleanup
{
    let mut store = StarStore::new();
    // ... use store ...
    
    // Explicit cleanup if needed
    store.clear()?;
    store.compact()?;
} // store is dropped here

// Use RAII patterns
struct RdfStarProcessor {
    store: StarStore,
}

impl Drop for RdfStarProcessor {
    fn drop(&mut self) {
        // Cleanup resources
        let _ = self.store.flush();
        let _ = self.store.close();
    }
}
```

### Stack Overflow Issues

**Problem**: Stack overflow with deeply nested quoted triples

**Solution**:
```rust
use oxirs_star::config::RecursionConfig;

let config = RecursionConfig {
    max_nesting_depth: 100, // Adjust based on needs
    use_heap_for_deep_nesting: true,
    stack_size_mb: 8, // Increase stack size if needed
};

oxirs_star::configure_recursion(config)?;

// Alternative: iterative processing
use oxirs_star::processing::IterativeProcessor;

let processor = IterativeProcessor::new();
let result = processor.process_deeply_nested_structure(complex_rdf_star_data)?;
```

## Concurrency Issues

### Race Conditions

**Problem**: Data corruption with concurrent access

**Detection**:
```rust
use oxirs_star::concurrent::{ConcurrentStore, ConsistencyChecker};

let store = ConcurrentStore::new_with_consistency_checking();
let checker = ConsistencyChecker::new(&store);

// Simulate concurrent access
let handles: Vec<_> = (0..10).map(|i| {
    let store_clone = store.clone();
    std::thread::spawn(move || {
        for j in 0..100 {
            let triple = create_test_triple(i, j);
            store_clone.insert(&triple)?;
        }
        Ok::<_, StarError>(())
    })
}).collect();

// Wait for all threads
for handle in handles {
    handle.join().unwrap()?;
}

// Check consistency
let consistency_report = checker.check_full_consistency()?;
if !consistency_report.is_consistent {
    println!("❌ Data inconsistency detected:");
    for issue in consistency_report.issues {
        println!("  {}", issue.description);
    }
}
```

**Solutions**:
```rust
use oxirs_star::concurrent::{LockFreeStore, TransactionalStore};

// Option 1: Lock-free data structures
let lock_free_store = LockFreeStore::new();

// Option 2: Transactional approach
let tx_store = TransactionalStore::new();
let tx = tx_store.begin_transaction()?;
tx.insert(&triple1)?;
tx.insert(&triple2)?;
tx.commit()?; // Atomic commit

// Option 3: Actor-based model
use oxirs_star::concurrent::ActorStore;

let actor_store = ActorStore::new();
let handle = actor_store.spawn_worker()?;

// Send operations to actor
handle.send_insert(triple).await?;
handle.send_query(query).await?;
```

### Deadlock Prevention

**Problem**: Deadlocks with multiple locks

**Solution**:
```rust
use oxirs_star::concurrent::{LockOrdering, DeadlockDetector};

// Establish lock ordering
let lock_ordering = LockOrdering::new()
    .add_resource("store", 1)
    .add_resource("index", 2)
    .add_resource("cache", 3);

// Use ordered locking
let _store_lock = lock_ordering.acquire_lock("store")?;
let _index_lock = lock_ordering.acquire_lock("index")?;
// Locks are acquired in predefined order, preventing deadlocks

// Deadlock detection
let detector = DeadlockDetector::new();
detector.monitor_locks()?;

if let Some(deadlock) = detector.check_for_deadlock()? {
    println!("⚠️  Deadlock detected between threads: {:?}", deadlock.thread_ids);
    
    // Recovery strategies
    detector.break_deadlock(&deadlock)?;
}
```

## Configuration Problems

### Invalid Configuration

**Problem**: Application fails to start with configuration errors

**Validation**:
```rust
use oxirs_star::config::{StarConfig, ConfigValidator};

let config = StarConfig::from_file("config.toml")?;
let validator = ConfigValidator::new();

let validation_result = validator.validate(&config)?;
if !validation_result.is_valid {
    println!("❌ Configuration errors:");
    for error in validation_result.errors {
        println!("  [{}] {}: {}", error.severity, error.path, error.message);
        if let Some(suggestion) = error.suggestion {
            println!("    💡 Suggestion: {}", suggestion);
        }
    }
    
    return Err("Invalid configuration".into());
}

// Apply validated configuration
oxirs_star::init_with_config(config)?;
```

**Common configuration issues**:

1. **Memory limits too low**:
```toml
# ❌ Too restrictive
[memory]
limit_mb = 64  # Too low for most workloads

# ✅ More reasonable
[memory]
limit_mb = 1024  # 1GB
```

2. **Invalid file paths**:
```toml
# ❌ Non-existent path
[storage]
data_dir = "/non/existent/path"

# ✅ Valid path with fallback
[storage]
data_dir = "/var/lib/oxirs"
fallback_dir = "/tmp/oxirs"
create_if_missing = true
```

3. **Conflicting feature flags**:
```toml
# ❌ Conflicting settings
[features]
enable_strict_mode = true
enable_error_recovery = true  # Conflicts with strict mode

# ✅ Consistent settings
[features]
enable_strict_mode = false
enable_error_recovery = true
```

## Integration Issues

### OxiRS Core Integration

**Problem**: Type conversion issues between oxirs-star and oxirs-core

**Solution**:
```rust
use oxirs_star::integration::{CoreBridge, ConversionOptions};

let bridge = CoreBridge::new();
let conversion_options = ConversionOptions {
    preserve_quoted_triples: true,
    fallback_to_reification: true,
    validate_conversions: true,
};

// Convert from oxirs-core types
let core_triple = oxirs_core::Triple::new(/* ... */);
let star_triple = bridge.from_core_triple(core_triple, &conversion_options)?;

// Convert to oxirs-core types
let core_compatible = bridge.to_core_compatible(&star_triple, &conversion_options)?;

// Batch conversion
let core_triples = vec![/* ... */];
let star_triples = bridge.batch_from_core(core_triples, &conversion_options)?;
```

### SPARQL Engine Integration

**Problem**: SPARQL queries not recognizing RDF-star syntax

**Solution**:
```rust
use oxirs_star::integration::sparql::{SparqlStarExtension, ExtensionConfig};

let extension_config = ExtensionConfig {
    enable_quoted_triple_patterns: true,
    enable_star_functions: true,
    enable_annotation_syntax: true,
    backward_compatibility: true,
};

let sparql_extension = SparqlStarExtension::with_config(extension_config);

// Register with existing SPARQL engine
let mut sparql_engine = existing_sparql_engine;
sparql_engine.register_extension(sparql_extension)?;

// Now SPARQL-star queries work
let star_query = r#"
    SELECT ?qt ?conf WHERE {
        ?qt ex:confidence ?conf .
        ?qt { ?s ex:knows ?o }
    }
"#;

let results = sparql_engine.execute(star_query)?;
```

## Error Messages Reference

### Common Error Patterns

| Error Pattern | Meaning | Solution |
|---------------|---------|----------|
| `ParseError: Expected '>>'` | Unclosed quoted triple | Add missing `>>` |
| `ValidationError: Invalid IRI` | Malformed IRI in RDF-star data | Fix IRI syntax |
| `QueryError: Unknown function 'TRIPLE'` | SPARQL-star functions not enabled | Enable star functions |
| `StorageError: Index corruption` | Storage index is corrupted | Rebuild indices |
| `MemoryError: Allocation failed` | Out of memory | Use streaming or increase memory |
| `ConcurrencyError: Lock timeout` | Deadlock or contention | Review locking strategy |
| `ConfigError: Invalid value` | Configuration parameter invalid | Check configuration docs |
| `IntegrationError: Type mismatch` | Incompatible types between modules | Use conversion utilities |

### Detailed Error Analysis

```rust
use oxirs_star::error::{StarError, ErrorAnalyzer, ErrorContext};

fn analyze_error(error: &StarError) {
    let analyzer = ErrorAnalyzer::new();
    let analysis = analyzer.analyze(error);
    
    println!("🔍 Error Analysis:");
    println!("├─ Type: {:?}", analysis.error_type);
    println!("├─ Severity: {:?}", analysis.severity);
    println!("├─ Root cause: {}", analysis.root_cause);
    println!("├─ Affected component: {}", analysis.component);
    
    if !analysis.suggestions.is_empty() {
        println!("├─ Suggestions:");
        for (i, suggestion) in analysis.suggestions.iter().enumerate() {
            println!("│  {}. {}", i + 1, suggestion);
        }
    }
    
    if let Some(recovery) = analysis.recovery_strategy {
        println!("└─ Recovery: {}", recovery.description);
    }
}

// Error recovery
fn handle_error_with_recovery(error: StarError) -> Result<(), StarError> {
    match error.kind {
        StarErrorKind::ParseError => {
            // Attempt parsing with error recovery
            eprintln!("⚠️  Parse error, attempting recovery...");
            // recovery logic
            Ok(())
        },
        StarErrorKind::ValidationError => {
            // Continue with warnings
            eprintln!("⚠️  Validation failed, continuing with warnings");
            Ok(())
        },
        _ => Err(error) // Re-raise unrecoverable errors
    }
}
```

## Debugging Tools

### Debug Logging

```rust
use oxirs_star::logging::{DebugLogger, LogLevel, LogConfig};

let log_config = LogConfig {
    level: LogLevel::Debug,
    components: vec![
        "parser".to_string(),
        "query".to_string(),
        "storage".to_string(),
    ],
    output_file: Some("/tmp/oxirs_star_debug.log".to_string()),
    structured_output: true,
};

let debug_logger = DebugLogger::with_config(log_config);
debug_logger.enable()?;

// Logs are now generated for debugging
// tail -f /tmp/oxirs_star_debug.log
```

### Interactive Debugging

```rust
use oxirs_star::debug::{DebugRepl, DebugCommands};

let mut debug_repl = DebugRepl::new(&store);
debug_repl.add_commands(DebugCommands::standard());

// Start interactive debugging session
debug_repl.run()?;

// Available commands:
// > inspect <triple_id>
// > query <sparql>  
// > memory
// > indices
// > validate
// > profile <operation>
// > export <format>
// > help
```

### Performance Profiling

```rust
use oxirs_star::profiling::{PerformanceProfiler, ProfileConfig};

let profile_config = ProfileConfig {
    sample_rate: 1000, // Sample every 1000 operations
    collect_stack_traces: true,
    profile_memory: true,
    profile_cpu: true,
    output_format: "flamegraph".to_string(),
};

let profiler = PerformanceProfiler::with_config(profile_config);
profiler.start()?;

// Perform operations to profile
perform_workload();

profiler.stop()?;
let profile_data = profiler.export_profile()?;

// Generate performance report
profiler.generate_flamegraph(&profile_data, "performance.svg")?;
println!("📊 Performance profile saved to performance.svg");
```

## Best Practices

### Error Handling

```rust
use anyhow::{Context, Result};
use oxirs_star::{StarError, StarErrorKind};

// ✅ Good error handling
fn robust_rdf_processing(data: &str) -> Result<Vec<StarTriple>> {
    let parser = TurtleStarParser::new();
    
    parser.parse(data)
        .with_context(|| "Failed to parse RDF-star data")
        .or_else(|e| {
            // Attempt recovery for certain error types
            if matches!(e.downcast_ref::<StarError>(), Some(StarError { kind: StarErrorKind::ParseError, .. })) {
                eprintln!("⚠️  Parse error, attempting recovery");
                let recovery_parser = TurtleStarParser::with_error_recovery();
                let (triples, errors) = recovery_parser.parse_with_errors(data)
                    .context("Recovery parsing also failed")?;
                
                if !errors.is_empty() {
                    eprintln!("⚠️  Recovered with {} errors", errors.len());
                }
                
                Ok(triples)
            } else {
                Err(e)
            }
        })
}
```

### Resource Management

```rust
use oxirs_star::resources::{ResourcePool, PoolConfig};

// ✅ Use resource pools for expensive objects
let pool_config = PoolConfig {
    min_size: 2,
    max_size: 10,
    max_idle_time: std::time::Duration::from_secs(300),
};

let query_engine_pool = ResourcePool::<StarQueryEngine>::with_config(pool_config);

// Get engine from pool
let engine = query_engine_pool.acquire().await?;
let results = engine.execute(query)?;
// Engine is automatically returned to pool when dropped

// ✅ Proper cleanup
struct RdfStarApplication {
    store: StarStore,
    engine: StarQueryEngine,
}

impl Drop for RdfStarApplication {
    fn drop(&mut self) {
        // Ensure proper cleanup
        let _ = self.store.flush();
        let _ = self.store.close();
        
        // Clear caches
        let _ = self.engine.clear_cache();
    }
}
```

### Performance Optimization

```rust
// ✅ Efficient batch processing
fn process_large_dataset(triples: &[StarTriple]) -> Result<()> {
    const BATCH_SIZE: usize = 1000;
    
    let mut store = StarStore::new();
    
    for batch in triples.chunks(BATCH_SIZE) {
        // Process in batches to avoid memory issues
        store.insert_batch(batch)?;
        
        // Periodic memory management
        if store.size() % 50000 == 0 {
            store.compact()?;
        }
    }
    
    Ok(())
}

// ✅ Efficient querying
fn efficient_querying(store: &StarStore) -> Result<Vec<StarTriple>> {
    // Use specific patterns instead of broad queries
    let specific_pattern = StarPattern::new(
        Some(StarTerm::iri("http://example.org/specific_subject")?),
        None,
        None,
    );
    
    // Better than: StarPattern::any()
    store.match_pattern(&specific_pattern)
}
```

### Testing Strategies

```rust
use oxirs_star::testing::{TestStoreBuilder, TestDataGenerator};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quoted_triple_handling() -> Result<()> {
        // Use test utilities
        let test_store = TestStoreBuilder::new()
            .with_quoted_triples(100)
            .with_metadata_triples(50)
            .build()?;
        
        let test_data = TestDataGenerator::new()
            .generate_quoted_triples(10)
            .with_confidence_scores()
            .with_nested_depth(3);
        
        // Test with generated data
        for triple in test_data {
            test_store.insert(&triple)?;
        }
        
        assert_eq!(test_store.size(), 10);
        Ok(())
    }
    
    #[test]
    fn test_error_recovery() -> Result<()> {
        let malformed_data = r#"
            <<ex:alice ex:knows ex:bob> ex:confidence 0.9 .
        "#;
        
        // Should handle gracefully
        let result = robust_rdf_processing(malformed_data);
        assert!(result.is_ok()); // Should recover
        
        Ok(())
    }
}
```

This comprehensive troubleshooting guide covers the most common issues users encounter with OxiRS-Star and provides practical solutions for diagnosing and resolving problems. For additional help, consult the [API Reference](API_REFERENCE.md) and [Performance Tuning Guide](PERFORMANCE.md).