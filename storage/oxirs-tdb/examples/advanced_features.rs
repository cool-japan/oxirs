//! Advanced features demonstration for oxirs-tdb
//!
//! This example demonstrates the advanced features added in v0.1.0-rc.1:
//! - Query result caching with LRU eviction
//! - Statistics collection for cost-based optimization
//! - Query monitoring with timeout enforcement
//! - Slow query logging
//! - Advanced diagnostics system
//!
//! Run with: cargo run --example advanced_features

use oxirs_tdb::{diagnostics::DiagnosticLevel, TdbConfig, TdbStore};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiRS TDB Advanced Features Demo ===\n");

    // Create a temporary directory for the database
    let temp_dir = std::env::temp_dir().join("oxirs_advanced_demo");
    std::fs::create_dir_all(&temp_dir)?;

    // 1. Create a TDB store with all advanced features enabled
    println!("1. Creating TDB store with advanced features...");
    let config = TdbConfig::new(&temp_dir)
        .with_buffer_pool_size(1000)
        .with_query_cache(true) // Enable query caching
        .with_statistics(true) // Enable statistics collection
        .with_query_monitoring(true); // Enable query monitoring

    let mut store = TdbStore::open_with_config(config)?;
    println!("   ✓ Store created with caching, statistics, and monitoring\n");

    // 2. Insert sample data
    println!("2. Inserting sample triples...");
    let triples = vec![
        (
            "http://example.org/alice",
            "http://example.org/knows",
            "http://example.org/bob",
        ),
        (
            "http://example.org/alice",
            "http://example.org/knows",
            "http://example.org/charlie",
        ),
        (
            "http://example.org/bob",
            "http://example.org/likes",
            "http://example.org/pizza",
        ),
        (
            "http://example.org/charlie",
            "http://example.org/worksAt",
            "http://example.org/acme",
        ),
        (
            "http://example.org/alice",
            "http://example.org/age",
            "http://example.org/30",
        ),
    ];

    for (s, p, o) in &triples {
        store.insert(s, p, o)?;
    }
    println!("   ✓ Inserted {} triples\n", triples.len());

    // 3. Demonstrate query caching
    println!("3. Demonstrating query result caching...");

    // First query - cache miss
    let term = oxirs_tdb::dictionary::Term::Iri("http://example.org/alice".to_string());
    let results1 = store.query_triples(Some(&term), None, None)?;
    println!("   First query (cache miss): {} results", results1.len());

    // Get cache stats after first query
    let cache_stats = store.query_cache_stats();
    println!(
        "   Cache stats: {} misses, {} hits",
        cache_stats
            .misses
            .load(std::sync::atomic::Ordering::Relaxed),
        cache_stats.hits.load(std::sync::atomic::Ordering::Relaxed)
    );

    // Second query - cache hit
    let results2 = store.query_triples(Some(&term), None, None)?;
    println!("   Second query (cache hit): {} results", results2.len());

    let cache_stats = store.query_cache_stats();
    println!(
        "   Cache stats: {} misses, {} hits",
        cache_stats
            .misses
            .load(std::sync::atomic::Ordering::Relaxed),
        cache_stats.hits.load(std::sync::atomic::Ordering::Relaxed)
    );
    println!(
        "   Cache hit rate: {:.2}%\n",
        cache_stats.hit_rate() * 100.0
    );

    // 4. Demonstrate statistics collection
    println!("4. Demonstrating statistics collection...");
    let stats = store.triple_statistics();
    println!("   Total triples: {}", stats.total_triples());
    println!("   Distinct subjects: {}", stats.distinct_subjects());
    println!("   Distinct predicates: {}", stats.distinct_predicates());
    println!("   Distinct objects: {}", stats.distinct_objects());

    // Export statistics snapshot
    let snapshot = store.export_statistics();
    println!("\n   Statistics Snapshot:");
    println!("   - Total triples: {}", snapshot.total_triples);
    println!("   - Distinct subjects: {}", snapshot.distinct_subjects);
    println!("   - Distinct predicates: {}", snapshot.distinct_predicates);
    println!("   - Distinct objects: {}", snapshot.distinct_objects);
    println!();

    // 5. Demonstrate query monitoring
    println!("5. Demonstrating query monitoring...");

    // Run several queries to generate monitoring data
    for i in 0..10 {
        if i % 3 == 0 {
            // Some slower queries by querying all triples
            let _ = store.query_triples(None, None, None)?;
        } else {
            // Faster queries with specific patterns
            let _ = store.query_triples(Some(&term), None, None)?;
        }
    }

    let monitor_stats = store.query_monitor_stats();
    println!(
        "   Total queries executed: {}",
        monitor_stats
            .total_queries
            .load(std::sync::atomic::Ordering::Relaxed)
    );
    println!(
        "   Average execution time: {} μs",
        monitor_stats.avg_execution_time_us()
    );
    println!(
        "   Slow queries detected: {}",
        monitor_stats
            .slow_queries
            .load(std::sync::atomic::Ordering::Relaxed)
    );

    // Show slow query history
    let slow_queries = store.slow_query_history();
    if !slow_queries.is_empty() {
        println!("\n   Slow Query History:");
        for (i, sq) in slow_queries.iter().take(3).enumerate() {
            println!("   {}. Pattern: {}", i + 1, sq.pattern.describe());
            println!("      Time: {:?}", sq.execution_time);
            println!("      Results: {}", sq.result_count);
        }
    }
    println!();

    // 6. Demonstrate diagnostics system
    println!("6. Running comprehensive diagnostics...");

    // Run quick diagnostics
    let quick_report = store.run_diagnostics(DiagnosticLevel::Quick);
    println!("   Quick Diagnostics:");
    println!("   - Health Status: {:?}", quick_report.health_status);
    println!("   - Duration: {:?}", quick_report.duration);
    println!("   - Checks run: {}", quick_report.results.len());

    // Show diagnostic results by severity
    let warnings = quick_report.warnings();
    let _errors = quick_report.errors();
    let _criticals = quick_report.critical_issues();

    println!("\n   Diagnostic Summary:");
    println!("   - Info: {}", quick_report.summary.info_count);
    println!("   - Warnings: {}", quick_report.summary.warning_count);
    println!("   - Errors: {}", quick_report.summary.error_count);
    println!("   - Critical: {}", quick_report.summary.critical_count);

    if !warnings.is_empty() {
        println!("\n   Warnings:");
        for warning in warnings.iter().take(3) {
            println!("   - {}: {}", warning.name, warning.description);
            if let Some(ref recommendation) = warning.recommendation {
                println!("     Recommendation: {}", recommendation);
            }
        }
    }

    // Run standard diagnostics for more detailed analysis
    let standard_report = store.run_diagnostics(DiagnosticLevel::Standard);
    println!("\n   Standard Diagnostics:");
    println!("   - Health Status: {:?}", standard_report.health_status);
    println!("   - Checks run: {}", standard_report.results.len());
    println!();

    // 7. Demonstrate enhanced statistics
    println!("7. Enhanced store statistics...");
    let enhanced_stats = store.enhanced_stats();

    println!("   Buffer Pool:");
    println!(
        "      Hit rate: {:.2}%",
        enhanced_stats.buffer_pool.hit_rate() * 100.0
    );
    println!(
        "      Total fetches: {}",
        enhanced_stats
            .buffer_pool
            .total_fetches
            .load(std::sync::atomic::Ordering::Relaxed)
    );

    println!("\n   Storage:");
    println!(
        "      Total size: {} bytes",
        enhanced_stats.storage.total_size_bytes
    );
    println!(
        "      Page size: {} bytes",
        enhanced_stats.storage.page_size
    );
    println!(
        "      Pages allocated: {}",
        enhanced_stats.storage.pages_allocated
    );
    println!(
        "      Efficiency: {:.2}%",
        enhanced_stats.storage.efficiency() * 100.0
    );
    println!(
        "      Fragmentation: {:.2}%",
        enhanced_stats.storage.fragmentation()
    );

    println!("\n   Index:");
    println!("      SPO entries: {}", enhanced_stats.index.spo_entries);
    println!("      POS entries: {}", enhanced_stats.index.pos_entries);
    println!("      OSP entries: {}", enhanced_stats.index.osp_entries);
    println!(
        "      Total entries: {}",
        enhanced_stats.index.total_entries()
    );
    println!();

    // 8. Show active queries (should be empty at this point)
    println!("8. Active queries:");
    let active = store.active_queries();
    println!("   Currently active queries: {}\n", active.len());

    // 9. Cleanup and summary
    println!("9. Cleanup...");
    store.clear_query_cache();
    store.clear_slow_query_history();
    println!("   ✓ Query cache cleared");
    println!("   ✓ Slow query history cleared");

    // Clean up temp directory
    std::fs::remove_dir_all(&temp_dir).ok();
    println!("   ✓ Temporary files removed\n");

    println!("=== Demo Complete ===");
    println!("\nKey Features Demonstrated:");
    println!("  ✓ Query Result Caching (LRU with TTL)");
    println!("  ✓ Statistics Collection (for cost-based optimization)");
    println!("  ✓ Query Monitoring (timeout enforcement & slow query detection)");
    println!("  ✓ Advanced Diagnostics (multi-level health checks)");
    println!("  ✓ Enhanced Metrics (buffer pool, storage, indexes)");

    Ok(())
}
