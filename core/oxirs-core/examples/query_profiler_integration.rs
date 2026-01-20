//! Query Profiler Integration Example
//!
//! This example demonstrates how to integrate the query profiler into your
//! SPARQL query execution pipeline for production monitoring and optimization.
//!
//! Run with: cargo run --example query_profiler_integration --all-features

use oxirs_core::model::{GraphName, NamedNode, Object, Predicate, Quad, Subject};
use oxirs_core::query::query_profiler::{ProfilerConfig, QueryProfiler};
use oxirs_core::rdf_store::RdfStore;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Query Profiler Integration Example\n");

    // Step 1: Create a profiler with custom configuration
    let profiler_config = ProfilerConfig {
        enable_detailed: true,
        track_memory: true,
        profile_patterns: true,
        profile_joins: true,
        profile_indexes: true,
        max_history: 100,
        slow_query_threshold_ms: 100,
        sample_rate: 1.0, // Profile all queries
    };

    let profiler = Arc::new(QueryProfiler::new(profiler_config));
    println!("✅ Created profiler with custom configuration");

    // Step 2: Create a sample RDF store and populate with data
    let mut store = RdfStore::new()?;
    populate_sample_data(&mut store)?;
    println!("✅ Populated store with sample data\n");

    // Step 3: Execute queries with profiling
    println!("📊 Executing queries with profiling...\n");

    // Query 1: Fast query (should be efficient)
    execute_profiled_query(
        &profiler,
        "SELECT * WHERE { ?s <http://example.org/type> <http://example.org/Person> }",
        "fast_query",
        || {
            // Simulate query execution
            std::thread::sleep(std::time::Duration::from_millis(50));
            (1000, 10) // (triples_matched, results_count)
        },
    );

    // Query 2: Slow query with many joins
    execute_profiled_query(
        &profiler,
        "SELECT * WHERE { ?s ?p1 ?o1 . ?o1 ?p2 ?o2 . ?o2 ?p3 ?o3 }",
        "complex_join_query",
        || {
            // Simulate expensive join operations
            std::thread::sleep(std::time::Duration::from_millis(150));
            (50000, 50) // Many triples matched, few results
        },
    );

    // Query 3: Query with low selectivity
    execute_profiled_query(
        &profiler,
        "SELECT * WHERE { ?s ?p ?o }",
        "full_scan_query",
        || {
            std::thread::sleep(std::time::Duration::from_millis(200));
            (100000, 100) // Very poor selectivity
        },
    );

    // Step 4: Display profiling statistics
    println!("\n📈 Profiling Statistics:\n");
    let stats = profiler.get_statistics();

    println!("Total queries profiled: {}", stats.total_queries);
    println!(
        "Average execution time: {:.2}ms",
        stats.avg_execution_time_ms
    );
    println!(
        "Median execution time: {:.2}ms",
        stats.median_execution_time_ms
    );
    println!("95th percentile: {:.2}ms", stats.p95_execution_time_ms);
    println!("99th percentile: {:.2}ms", stats.p99_execution_time_ms);
    println!("Min execution time: {}ms", stats.min_execution_time_ms);
    println!("Max execution time: {}ms", stats.max_execution_time_ms);
    println!("Total triples matched: {}", stats.total_triples_matched);
    println!(
        "Average triples per query: {:.0}",
        stats.avg_triples_per_query
    );
    println!(
        "Overall cache hit rate: {:.1}%",
        stats.overall_cache_hit_rate * 100.0
    );
    println!("Slow query count: {}", stats.slow_query_count);

    // Step 5: Show slow queries with optimization hints
    println!("\n🐌 Slow Queries (with optimization hints):\n");
    let slow_queries = profiler.get_slow_queries(10);

    for (i, query) in slow_queries.iter().enumerate() {
        println!("{}. Query: {}", i + 1, query.query_text);
        println!("   Type: {}", query.query_type);
        println!("   Execution time: {}ms", query.statistics.total_time_ms);
        println!("   Triples matched: {}", query.statistics.triples_matched);
        println!("   Results: {}", query.statistics.results_count);

        if !query.optimization_hints.is_empty() {
            println!("   Optimization hints:");
            for hint in &query.optimization_hints {
                println!("     {}", hint);
            }
        }
        println!();
    }

    // Step 6: Export profiling data as JSON
    println!("💾 Exporting profiling data to JSON...");
    match profiler.export_json() {
        Ok(json) => {
            std::fs::write("/tmp/query_profiling_stats.json", &json)?;
            println!("✅ Profiling data exported to /tmp/query_profiling_stats.json");
            println!("\nFirst 200 characters of JSON:");
            println!("{}", &json[..json.len().min(200)]);
            if json.len() > 200 {
                println!("...");
            }
        }
        Err(e) => eprintln!("❌ Failed to export JSON: {}", e),
    }

    // Step 7: Demonstrate concurrent profiling
    println!("\n🔄 Testing concurrent profiling...");
    test_concurrent_profiling(profiler.clone());

    println!("\n✅ Example completed successfully!");

    Ok(())
}

fn populate_sample_data(store: &mut RdfStore) -> Result<(), Box<dyn std::error::Error>> {
    // Add some sample quads
    for i in 0..100 {
        let subject =
            Subject::NamedNode(NamedNode::new(format!("http://example.org/person{}", i))?);
        let predicate = Predicate::NamedNode(NamedNode::new("http://example.org/type")?);
        let object = Object::NamedNode(NamedNode::new("http://example.org/Person")?);
        let graph = GraphName::DefaultGraph;

        let quad = Quad::new(subject, predicate, object, graph);
        store.insert(&quad)?;
    }

    Ok(())
}

fn execute_profiled_query<F>(
    profiler: &QueryProfiler,
    query_text: &str,
    query_type: &str,
    execute_fn: F,
) where
    F: FnOnce() -> (u64, u64),
{
    // Start profiling session
    let mut session = profiler.start_session(query_text);

    // Simulate query phases
    session.start_phase("parse");
    std::thread::sleep(std::time::Duration::from_millis(5));
    session.end_phase("parse");

    session.start_phase("planning");
    std::thread::sleep(std::time::Duration::from_millis(10));
    session.end_phase("planning");

    session.start_phase("execution");

    // Record some pattern matches and index accesses
    session.record_pattern("SPO".to_string());
    session.record_index_access("SPO_index".to_string());

    if query_text.contains("?p1") {
        // Complex query with joins
        for _ in 0..15 {
            session.record_join();
        }
        session.record_pattern("OPS".to_string());
        session.record_index_access("OPS_index".to_string());
    }

    // Execute the query
    let (triples_matched, results_count) = execute_fn();
    session.record_triples_matched(triples_matched);
    session.record_results(results_count);

    // Simulate some cache hits
    for i in 0..10 {
        session.record_cache_access(i % 3 == 0);
    }

    session.end_phase("execution");

    // Finish profiling and record
    let stats = session.finish();
    let total_time = stats.total_time_ms;
    profiler.record_query(query_text.to_string(), stats, query_type.to_string());

    println!(
        "✅ Executed: {} ({} ms, {} triples, {} results)",
        query_type, total_time, triples_matched, results_count
    );
}

fn test_concurrent_profiling(profiler: Arc<QueryProfiler>) {
    use std::thread;

    let mut handles = vec![];

    // Spawn multiple threads executing queries concurrently
    for i in 0..5 {
        let profiler_clone = profiler.clone();
        let handle = thread::spawn(move || {
            for j in 0..3 {
                let query = format!(
                    "SELECT * WHERE {{ ?s <http://example.org/prop{}> ?o }}",
                    i * 3 + j
                );
                let mut session = profiler_clone.start_session(&query);

                session.start_phase("execution");
                thread::sleep(std::time::Duration::from_millis(20));
                session.record_triples_matched(100 + j * 50);
                session.record_results(10 + j * 5);
                session.end_phase("execution");

                let stats = session.finish();
                profiler_clone.record_query(query, stats, "SELECT".to_string());
            }
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    println!("✅ Concurrent profiling completed - 15 queries executed across 5 threads");
}
