//! Comprehensive example showcasing SPARQL query profiling
//!
//! This example demonstrates:
//! - Query performance profiling
//! - Phase-based timing tracking
//! - Performance metrics collection
//! - Historical analysis
//!
//! Run with: cargo run --example query_profiling

use oxirs_arq::{QueryPhase, QueryProfiler};
use std::thread;
use std::time::Duration;

fn main() {
    println!("🚀 OxiRS ARQ - SPARQL Query Profiling Demo\n");
    println!("This example demonstrates comprehensive query performance profiling:\n");

    // Example 1: Basic profiling
    demo_basic_profiling();

    // Example 2: Phase-based profiling
    demo_phase_profiling();

    // Example 3: Historical analysis
    demo_historical_analysis();

    // Example 4: Performance comparison
    demo_performance_comparison();

    println!("\n✅ All profiling demonstrations complete!");
}

fn demo_basic_profiling() {
    println!("═══════════════════════════════════════════════════════════");
    println!("1️⃣  Basic Query Profiling");
    println!("═══════════════════════════════════════════════════════════");

    let mut profiler = QueryProfiler::new();

    // Start profiling a query
    profiler.start_query("SELECT * WHERE { ?s ?p ?o } LIMIT 100".to_string());

    // Simulate query execution
    thread::sleep(Duration::from_millis(50));

    // Record metrics
    profiler.record_triples(1000);
    profiler.record_results(100);
    profiler.record_join();
    profiler.record_memory(5_000_000); // 5 MB
    profiler.record_cache_hit_rate(0.85);

    // End profiling
    let stats = profiler.end_query().unwrap();

    println!("✓ Query profiled successfully:");
    println!("{}\n", stats.report());
}

fn demo_phase_profiling() {
    println!("═══════════════════════════════════════════════════════════");
    println!("2️⃣  Phase-Based Profiling");
    println!("═══════════════════════════════════════════════════════════");

    let mut profiler = QueryProfiler::new();

    profiler.start_query("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }".to_string());

    // Parsing phase
    profiler.start_phase(QueryPhase::Parsing);
    thread::sleep(Duration::from_millis(10));
    profiler.end_phase(QueryPhase::Parsing);
    println!("✓ Parsing phase: 10ms");

    // Planning phase
    profiler.start_phase(QueryPhase::Planning);
    thread::sleep(Duration::from_millis(30));
    profiler.end_phase(QueryPhase::Planning);
    println!("✓ Planning phase: 30ms");

    // Execution phase
    profiler.start_phase(QueryPhase::Execution);
    thread::sleep(Duration::from_millis(100));
    profiler.record_triples(5000);
    profiler.record_joins(3);
    profiler.end_phase(QueryPhase::Execution);
    println!("✓ Execution phase: 100ms");

    // Materialization phase
    profiler.start_phase(QueryPhase::Materialization);
    thread::sleep(Duration::from_millis(20));
    profiler.record_results(500);
    profiler.end_phase(QueryPhase::Materialization);
    println!("✓ Materialization phase: 20ms");

    let stats = profiler.end_query().unwrap();

    println!("\n📊 Phase Breakdown:");
    println!(
        "  - Parsing:         {:>6.1}ms ({:.1}%)",
        stats.phase_duration(QueryPhase::Parsing).as_secs_f64() * 1000.0,
        (stats.phase_duration(QueryPhase::Parsing).as_secs_f64()
            / stats.total_duration.as_secs_f64())
            * 100.0
    );
    println!(
        "  - Planning:        {:>6.1}ms ({:.1}%)",
        stats.phase_duration(QueryPhase::Planning).as_secs_f64() * 1000.0,
        (stats.phase_duration(QueryPhase::Planning).as_secs_f64()
            / stats.total_duration.as_secs_f64())
            * 100.0
    );
    println!(
        "  - Execution:       {:>6.1}ms ({:.1}%)",
        stats.phase_duration(QueryPhase::Execution).as_secs_f64() * 1000.0,
        (stats.phase_duration(QueryPhase::Execution).as_secs_f64()
            / stats.total_duration.as_secs_f64())
            * 100.0
    );
    println!(
        "  - Materialization: {:>6.1}ms ({:.1}%)\n",
        stats
            .phase_duration(QueryPhase::Materialization)
            .as_secs_f64()
            * 1000.0,
        (stats
            .phase_duration(QueryPhase::Materialization)
            .as_secs_f64()
            / stats.total_duration.as_secs_f64())
            * 100.0
    );
}

fn demo_historical_analysis() {
    println!("═══════════════════════════════════════════════════════════");
    println!("3️⃣  Historical Query Analysis");
    println!("═══════════════════════════════════════════════════════════");

    let mut profiler = QueryProfiler::new();

    // Simulate multiple queries
    let queries = [
        ("SELECT ?s WHERE { ?s rdf:type foaf:Person }", 100, 50),
        ("SELECT ?name WHERE { ?s foaf:name ?name }", 200, 75),
        ("ASK { ?s foaf:knows ?o }", 50, 1),
        (
            "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o } LIMIT 10",
            500,
            10,
        ),
        ("DESCRIBE <http://example.org/alice>", 150, 20),
    ];

    for (i, (query, triples, results)) in queries.iter().enumerate() {
        profiler.start_query(query.to_string());
        thread::sleep(Duration::from_millis(10 + (i as u64 * 5)));
        profiler.record_triples(*triples);
        profiler.record_results(*results);
        profiler.record_cache_hit_rate(0.7 + (i as f64 * 0.05));
        profiler.end_query();
    }

    println!("✓ Profiled {} queries", profiler.history().len());
    println!("\n{}\n", profiler.summary_report());
}

fn demo_performance_comparison() {
    println!("═══════════════════════════════════════════════════════════");
    println!("4️⃣  Performance Comparison");
    println!("═══════════════════════════════════════════════════════════");

    let mut profiler_cached = QueryProfiler::new();
    let mut profiler_uncached = QueryProfiler::new();

    // Simulate cached query
    profiler_cached.start_query("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100".to_string());
    thread::sleep(Duration::from_millis(20));
    profiler_cached.record_triples(1000);
    profiler_cached.record_results(100);
    profiler_cached.record_cache_hit_rate(0.95); // High cache hit
    let cached_stats = profiler_cached.end_query().unwrap();

    // Simulate uncached query
    profiler_uncached.start_query("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100".to_string());
    thread::sleep(Duration::from_millis(150));
    profiler_uncached.record_triples(1000);
    profiler_uncached.record_results(100);
    profiler_uncached.record_cache_hit_rate(0.10); // Low cache hit
    let uncached_stats = profiler_uncached.end_query().unwrap();

    println!("📊 Cached vs Uncached Query:");
    println!("\nWith Cache (95% hit rate):");
    println!(
        "  - Duration: {:.1}ms",
        cached_stats.total_duration.as_secs_f64() * 1000.0
    );
    println!("  - Throughput: {:.0} results/s", cached_stats.throughput());

    println!("\nWithout Cache (10% hit rate):");
    println!(
        "  - Duration: {:.1}ms",
        uncached_stats.total_duration.as_secs_f64() * 1000.0
    );
    println!(
        "  - Throughput: {:.0} results/s",
        uncached_stats.throughput()
    );

    let speedup =
        uncached_stats.total_duration.as_secs_f64() / cached_stats.total_duration.as_secs_f64();
    println!("\n💡 Cache Speedup: {:.1}x faster\n", speedup);
}

// Helper trait for recording multiple joins
trait ProfilerExt {
    fn record_joins(&mut self, count: usize);
}

impl ProfilerExt for QueryProfiler {
    fn record_joins(&mut self, count: usize) {
        for _ in 0..count {
            self.record_join();
        }
    }
}
