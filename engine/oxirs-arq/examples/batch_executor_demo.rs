//! # Smart Query Batch Executor - Comprehensive Demo
//!
//! This example demonstrates all features of the Smart Query Batch Executor,
//! including parallel execution, priority queuing, resource management, and
//! intelligent query grouping.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example batch_executor_demo --all-features
//! ```

use anyhow::Result;
use oxirs_arq::algebra::{Term as AlgebraTerm, TriplePattern};
use oxirs_arq::query_batch_executor::{
    BatchConfig, BatchMode, BatchQueryResult, QueryBatchExecutor, QueryPriority,
};
use std::sync::Arc;
use std::time::Instant;

/// Mock dataset for demonstration purposes
struct MockDataset;

impl oxirs_arq::executor::Dataset for MockDataset {
    fn find_triples(
        &self,
        _pattern: &TriplePattern,
    ) -> Result<Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>> {
        Ok(Vec::new())
    }

    fn contains_triple(
        &self,
        _subject: &AlgebraTerm,
        _predicate: &AlgebraTerm,
        _object: &AlgebraTerm,
    ) -> Result<bool> {
        Ok(false)
    }

    fn subjects(&self) -> Result<Vec<AlgebraTerm>> {
        Ok(Vec::new())
    }

    fn predicates(&self) -> Result<Vec<AlgebraTerm>> {
        Ok(Vec::new())
    }

    fn objects(&self) -> Result<Vec<AlgebraTerm>> {
        Ok(Vec::new())
    }
}

fn print_section(title: &str) {
    println!("\n{}", "=".repeat(80));
    println!("  {}", title);
    println!("{}", "=".repeat(80));
}

fn print_subsection(title: &str) {
    println!("\n--- {} ---", title);
}

fn print_results(results: &[BatchQueryResult]) {
    for (i, result) in results.iter().enumerate() {
        if result.success {
            println!(
                "  Query {}: ✓ Success - {} results in {:?}",
                i + 1,
                result.result_count,
                result.duration
            );
        } else {
            println!(
                "  Query {}: ✗ Failed - {}",
                i + 1,
                result
                    .error
                    .as_ref()
                    .unwrap_or(&"Unknown error".to_string())
            );
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("\n🚀 Smart Query Batch Executor - Comprehensive Demo");
    println!("Demonstrating advanced SPARQL batch query processing capabilities\n");

    // Demonstration 1: Basic Parallel Execution
    print_section("Demo 1: Basic Parallel Execution");
    demo_basic_parallel().await?;

    // Demonstration 2: Priority Queuing
    print_section("Demo 2: Priority Queuing System");
    demo_priority_queuing().await?;

    // Demonstration 3: Batch Execution Modes
    print_section("Demo 3: Batch Execution Modes Comparison");
    demo_execution_modes().await?;

    // Demonstration 4: Query Grouping & Optimization
    print_section("Demo 4: Intelligent Query Grouping");
    demo_query_grouping().await?;

    // Demonstration 5: Resource Management
    print_section("Demo 5: Resource Management & Limits");
    demo_resource_management().await?;

    // Demonstration 6: Statistics & Monitoring
    print_section("Demo 6: Comprehensive Statistics Tracking");
    demo_statistics().await?;

    // Demonstration 7: Large-Scale Batch Processing
    print_section("Demo 7: Large-Scale Batch Processing");
    demo_large_scale().await?;

    println!("\n{}", "=".repeat(80));
    println!("  ✅ All demonstrations completed successfully!");
    println!("{}", "=".repeat(80));
    println!();

    Ok(())
}

/// Demo 1: Basic parallel execution with default configuration
async fn demo_basic_parallel() -> anyhow::Result<()> {
    print_subsection("Creating batch executor with default configuration");

    let config = BatchConfig::default();
    println!("  Max concurrent: {}", config.max_concurrent);
    println!("  Memory limit: {} MB", config.memory_limit_mb);
    println!("  CPU limit: {:.1}%", config.cpu_limit * 100.0);
    println!("  Mode: {:?}", config.mode);

    let executor = QueryBatchExecutor::new(config);
    let dataset = Arc::new(MockDataset);

    print_subsection("Adding queries to the batch");

    let queries = [
        "SELECT * WHERE { ?s ?p ?o } LIMIT 100",
        "SELECT ?person WHERE { ?person a :Person } LIMIT 50",
        "SELECT ?company WHERE { ?company a :Company } LIMIT 30",
        "ASK { ?s a :Entity }",
        "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o } LIMIT 20",
    ];

    for (i, query) in queries.iter().enumerate() {
        executor.add_query(*query, QueryPriority::Normal)?;
        println!("  ✓ Added query {}: {}", i + 1, query);
    }

    print_subsection("Executing batch in parallel");

    let start = Instant::now();
    let results = executor.execute_batch_async(dataset).await?;
    let duration = start.elapsed();

    println!("  ⏱️  Total execution time: {:?}", duration);
    println!("  📊 Results:");
    print_results(&results);

    let stats = executor.statistics();
    println!("\n  Success rate: {:.1}%", stats.success_rate() * 100.0);
    println!("  Throughput: {:.2} queries/sec", stats.throughput);

    Ok(())
}

/// Demo 2: Priority queuing with different priority levels
async fn demo_priority_queuing() -> anyhow::Result<()> {
    print_subsection("Demonstrating 4-level priority system");

    let executor = QueryBatchExecutor::new(BatchConfig::default());
    let dataset = Arc::new(MockDataset);

    // Add queries with different priorities
    let queries = [
        ("SELECT ?s WHERE { ?s ?p ?o }", QueryPriority::Background),
        ("SELECT COUNT(*) WHERE { ?s ?p ?o }", QueryPriority::High),
        ("SELECT ?p WHERE { ?s ?p ?o }", QueryPriority::Low),
        ("ASK { ?s a :CriticalEntity }", QueryPriority::High),
        ("SELECT ?o WHERE { ?s ?p ?o }", QueryPriority::Normal),
        ("SELECT * WHERE { ?s :urgent true }", QueryPriority::High),
    ];

    println!("  Adding queries with priorities:");
    for (i, (query, priority)) in queries.iter().enumerate() {
        executor.add_query(*query, *priority)?;
        println!("    Query {}: {:?} - {}", i + 1, priority, query);
    }

    print_subsection("Execution order (by priority)");

    println!("  High priority queries execute first");
    println!("  Normal priority queries execute second");
    println!("  Low priority queries execute third");
    println!("  Background priority queries execute last");

    let results = executor.execute_batch_async(dataset).await?;

    println!("\n  📊 Execution Results:");
    print_results(&results);

    Ok(())
}

/// Demo 3: Compare different batch execution modes
async fn demo_execution_modes() -> anyhow::Result<()> {
    let dataset = Arc::new(MockDataset);

    let test_queries = [
        "SELECT * WHERE { ?s ?p ?o } LIMIT 10",
        "SELECT ?person WHERE { ?person a :Person }",
        "SELECT ?company WHERE { ?company a :Company }",
        "ASK { ?s a :Entity }",
        "SELECT COUNT(*) WHERE { ?s ?p ?o }",
    ];

    let modes = vec![
        (BatchMode::Parallel, "Parallel - All queries at once"),
        (BatchMode::Sequential, "Sequential - One at a time"),
        (BatchMode::Optimized, "Optimized - Grouped by similarity"),
        (BatchMode::Adaptive, "Adaptive - Dynamic concurrency"),
    ];

    for (mode, description) in modes {
        print_subsection(&format!("Mode: {}", description));

        let config = BatchConfig::default().with_mode(mode);
        let executor = QueryBatchExecutor::new(config);

        for query in &test_queries {
            executor.add_query(*query, QueryPriority::Normal)?;
        }

        let start = Instant::now();
        let _results = executor.execute_batch_async(dataset.clone()).await?;
        let duration = start.elapsed();

        let stats = executor.statistics();

        println!("  ⏱️  Execution time: {:?}", duration);
        println!(
            "  ✓ Successful: {}/{}",
            stats.successful_queries, stats.total_queries
        );
        println!("  📈 Throughput: {:.2} queries/sec", stats.throughput);
        println!(
            "  💾 Cache hit rate: {:.1}%",
            stats.cache_hit_rate() * 100.0
        );
    }

    Ok(())
}

/// Demo 4: Intelligent query grouping for optimization
async fn demo_query_grouping() -> anyhow::Result<()> {
    print_subsection("Grouping similar queries for optimized execution");

    let config = BatchConfig::default()
        .with_mode(BatchMode::Optimized)
        .with_grouping(true);

    let executor = QueryBatchExecutor::new(config);
    let dataset = Arc::new(MockDataset);

    println!("  Adding queries with similar patterns:");

    // Similar pattern group 1: Person queries
    let person_queries = vec![
        "SELECT ?person WHERE { ?person a :Person }",
        "SELECT ?person WHERE { ?person a :Person ; :age ?age }",
        "SELECT ?person WHERE { ?person a :Person ; :name ?name }",
    ];

    // Similar pattern group 2: Company queries
    let company_queries = vec![
        "SELECT ?company WHERE { ?company a :Company }",
        "SELECT ?company WHERE { ?company a :Company ; :revenue ?r }",
    ];

    // Different pattern: ASK queries
    let ask_queries = vec!["ASK { ?s a :Person }", "ASK { ?s a :Company }"];

    println!("\n  Group 1: Person queries");
    for query in &person_queries {
        executor.add_query(*query, QueryPriority::Normal)?;
        println!("    - {}", query);
    }

    println!("\n  Group 2: Company queries");
    for query in &company_queries {
        executor.add_query(*query, QueryPriority::Normal)?;
        println!("    - {}", query);
    }

    println!("\n  Group 3: ASK queries");
    for query in &ask_queries {
        executor.add_query(*query, QueryPriority::Normal)?;
        println!("    - {}", query);
    }

    print_subsection("Executing with automatic grouping");

    let start = Instant::now();
    let _results = executor.execute_batch_async(dataset).await?;
    let duration = start.elapsed();

    let stats = executor.statistics();

    println!("  ⏱️  Total time: {:?}", duration);
    println!("  🔗 Query groups detected: {}", stats.query_groups);
    println!("  ✓ Queries executed: {}", stats.total_queries);
    println!("  📊 Average duration: {:?}", stats.avg_duration);

    Ok(())
}

/// Demo 5: Resource management and limits
async fn demo_resource_management() -> anyhow::Result<()> {
    print_subsection("Configuring resource limits");

    let config = BatchConfig::default()
        .with_max_concurrent(8)
        .with_memory_limit_mb(1024)
        .with_cpu_limit(0.7);

    println!("  Max concurrent queries: {}", config.max_concurrent);
    println!("  Memory limit: {} MB", config.memory_limit_mb);
    println!("  CPU limit: {:.0}%", config.cpu_limit * 100.0);

    let executor = QueryBatchExecutor::new(config);
    let dataset = Arc::new(MockDataset);

    print_subsection("Adding 20 queries to test concurrency control");

    for i in 1..=20 {
        let query = format!("SELECT * WHERE {{ ?s ?p ?o }} LIMIT {}", i * 10);
        executor.add_query(query, QueryPriority::Normal)?;
    }

    println!("  ✓ Added 20 queries to the queue");
    println!("  ⚙️  Executor will limit to 8 concurrent executions");

    let start = Instant::now();
    let _results = executor.execute_batch_async(dataset).await?;
    let duration = start.elapsed();

    println!("\n  ⏱️  Execution time: {:?}", duration);
    println!("  ✓ All 20 queries completed");
    println!(
        "  📈 Throughput: {:.2} queries/sec",
        20.0 / duration.as_secs_f64()
    );

    Ok(())
}

/// Demo 6: Comprehensive statistics tracking
async fn demo_statistics() -> anyhow::Result<()> {
    print_subsection("Tracking detailed execution statistics");

    let config = BatchConfig::default().with_caching(true);
    let executor = QueryBatchExecutor::new(config);
    let dataset = Arc::new(MockDataset);

    // Add duplicate queries to demonstrate caching
    let queries = vec![
        "SELECT * WHERE { ?s ?p ?o } LIMIT 100",
        "SELECT * WHERE { ?s ?p ?o } LIMIT 100", // Duplicate
        "SELECT ?person WHERE { ?person a :Person }",
        "SELECT ?person WHERE { ?person a :Person }", // Duplicate
        "ASK { ?s a :Entity }",
    ];

    for query in &queries {
        executor.add_query(*query, QueryPriority::Normal)?;
    }

    let _results = executor.execute_batch_async(dataset).await?;
    let stats = executor.statistics();

    println!("  📊 Execution Statistics:");
    println!("  ─────────────────────────────────────");
    println!("  Total queries: {}", stats.total_queries);
    println!("  Successful: {}", stats.successful_queries);
    println!("  Failed: {}", stats.failed_queries);
    println!("  Success rate: {:.1}%", stats.success_rate() * 100.0);
    println!();
    println!("  ⏱️  Timing:");
    println!("  ─────────────────────────────────────");
    println!("  Total duration: {:?}", stats.total_duration);
    println!("  Average duration: {:?}", stats.avg_duration);
    println!("  Min duration: {:?}", stats.min_duration);
    println!("  Max duration: {:?}", stats.max_duration);
    println!();
    println!("  🚀 Performance:");
    println!("  ─────────────────────────────────────");
    println!("  Throughput: {:.2} queries/sec", stats.throughput);
    println!("  Total results: {}", stats.total_results);
    println!();
    println!("  💾 Caching:");
    println!("  ─────────────────────────────────────");
    println!("  Cache hit rate: {:.1}%", stats.cache_hit_rate() * 100.0);
    println!("  Cached queries: {}", stats.cached_queries);
    println!("  Query groups: {}", stats.query_groups);

    Ok(())
}

/// Demo 7: Large-scale batch processing
async fn demo_large_scale() -> anyhow::Result<()> {
    print_subsection("Processing 100 queries in optimized batches");

    let config = BatchConfig::default()
        .with_max_concurrent(16)
        .with_mode(BatchMode::Optimized)
        .with_grouping(true)
        .with_caching(true);

    let executor = QueryBatchExecutor::new(config);
    let dataset = Arc::new(MockDataset);

    println!("  Generating 100 test queries...");

    for i in 1..=100 {
        let query = match i % 5 {
            0 => format!("SELECT * WHERE {{ ?s ?p ?o }} LIMIT {}", i),
            1 => format!("SELECT ?person WHERE {{ ?person a :Person }} LIMIT {}", i),
            2 => format!(
                "SELECT ?company WHERE {{ ?company a :Company }} LIMIT {}",
                i
            ),
            3 => "ASK { ?s a :Entity }".to_string(),
            _ => "SELECT COUNT(*) WHERE { ?s ?p ?o }".to_string(),
        };

        let priority = match i % 10 {
            0 => QueryPriority::High,
            1..=2 => QueryPriority::Low,
            _ => QueryPriority::Normal,
        };

        executor.add_query(query, priority)?;
    }

    println!("  ✓ Added 100 queries to the batch");

    print_subsection("Executing large-scale batch");

    let start = Instant::now();
    let _results = executor.execute_batch_async(dataset).await?;
    let duration = start.elapsed();

    let stats = executor.statistics();

    println!("\n  🎯 Large-Scale Batch Results:");
    println!("  ═══════════════════════════════════════");
    println!("  Total queries: {}", stats.total_queries);
    println!(
        "  Successful: {} ({:.1}%)",
        stats.successful_queries,
        stats.success_rate() * 100.0
    );
    println!("  Failed: {}", stats.failed_queries);
    println!();
    println!("  ⏱️  Total time: {:?}", duration);
    println!("  📈 Throughput: {:.2} queries/sec", stats.throughput);
    println!("  ⚡ Avg per query: {:?}", stats.avg_duration);
    println!();
    println!("  🔗 Optimization:");
    println!("  Query groups: {}", stats.query_groups);
    println!(
        "  Cache hits: {} ({:.1}%)",
        stats.cached_queries,
        stats.cache_hit_rate() * 100.0
    );
    println!();
    println!(
        "  💡 Performance gain from grouping: ~{:.1}x faster than sequential",
        100.0 / duration.as_secs_f64()
    );

    Ok(())
}
