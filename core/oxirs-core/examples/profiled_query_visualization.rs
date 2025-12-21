//! Profiled Query Visualization Example
//!
//! This example demonstrates the integration between the query profiler and
//! query plan visualizer, showing how to automatically generate visual query
//! plans with real execution statistics.

use oxirs_core::query::profiled_plan_builder::ProfiledPlanBuilder;
use oxirs_core::query::query_profiler::QueryStatistics;
use std::collections::HashMap;

fn main() {
    println!("=== OxiRS Profiled Query Visualization Demo ===\n");

    // Example 1: Basic profiling with visualization
    println!("Example 1: Basic Query Profiling with Visualization");
    println!("----------------------------------------------------");
    let stats1 = create_simple_query_stats();
    demonstrate_basic_profiling(&stats1);
    println!("\n");

    // Example 2: Complex query with performance analysis
    println!("Example 2: Complex Query Performance Analysis");
    println!("---------------------------------------------");
    let stats2 = create_complex_query_stats();
    demonstrate_performance_analysis(&stats2);
    println!("\n");

    // Example 3: Slow query detection and optimization
    println!("Example 3: Slow Query Detection");
    println!("--------------------------------");
    let stats3 = create_slow_query_stats();
    demonstrate_slow_query_analysis(&stats3);
    println!("\n");

    // Example 4: Query execution comparison
    println!("Example 4: Query Execution Comparison");
    println!("-------------------------------------");
    let baseline = create_baseline_stats();
    let optimized = create_optimized_stats();
    demonstrate_execution_comparison(&baseline, &optimized);
    println!("\n");

    // Example 5: Regression detection
    println!("Example 5: Performance Regression Detection");
    println!("-------------------------------------------");
    let regressed = create_regressed_stats();
    demonstrate_regression_detection(&baseline, &regressed);
    println!("\n");

    // Example 6: Complete profiling report
    println!("Example 6: Complete Profiling Report");
    println!("------------------------------------");
    demonstrate_complete_report(&stats2);
}

fn create_simple_query_stats() -> QueryStatistics {
    let mut pattern_matches = HashMap::new();
    pattern_matches.insert("?person rdf:type foaf:Person".to_string(), 100);

    let mut index_accesses = HashMap::new();
    index_accesses.insert("SPO".to_string(), 1);

    QueryStatistics {
        total_time_ms: 50,
        parse_time_ms: 5,
        planning_time_ms: 10,
        execution_time_ms: 35,
        triples_matched: 100,
        results_count: 10,
        peak_memory_bytes: 512 * 1024, // 512KB
        cache_hit_rate: 0.8,
        pattern_matches,
        index_accesses,
        ..Default::default()
    }
}

fn create_complex_query_stats() -> QueryStatistics {
    let mut pattern_matches = HashMap::new();
    pattern_matches.insert("?person rdf:type foaf:Person".to_string(), 2000);
    pattern_matches.insert("?person foaf:name ?name".to_string(), 1800);
    pattern_matches.insert("?person foaf:knows ?friend".to_string(), 1200);

    let mut index_accesses = HashMap::new();
    index_accesses.insert("SPO".to_string(), 2);
    index_accesses.insert("POS".to_string(), 1);

    QueryStatistics {
        total_time_ms: 250,
        parse_time_ms: 15,
        planning_time_ms: 35,
        execution_time_ms: 200,
        triples_matched: 5000,
        results_count: 150,
        peak_memory_bytes: 2 * 1024 * 1024, // 2MB
        join_operations: 3,
        cache_hit_rate: 0.65,
        pattern_matches,
        index_accesses,
        ..Default::default()
    }
}

fn create_slow_query_stats() -> QueryStatistics {
    let mut pattern_matches = HashMap::new();
    pattern_matches.insert("?s ?p ?o".to_string(), 30000); // Very broad!
    pattern_matches.insert("?article schema:author ?author".to_string(), 20000);

    // No index usage!
    let index_accesses = HashMap::new();

    QueryStatistics {
        total_time_ms: 5000, // 5 seconds!
        parse_time_ms: 10,
        planning_time_ms: 40,
        execution_time_ms: 4950,
        triples_matched: 50000,
        results_count: 500,
        peak_memory_bytes: 10 * 1024 * 1024, // 10MB
        join_operations: 5,
        cache_hit_rate: 0.15, // Poor cache usage
        pattern_matches,
        index_accesses,
        ..Default::default()
    }
}

fn create_baseline_stats() -> QueryStatistics {
    QueryStatistics {
        total_time_ms: 150,
        parse_time_ms: 10,
        planning_time_ms: 20,
        execution_time_ms: 120,
        triples_matched: 1000,
        results_count: 50,
        peak_memory_bytes: 1024 * 1024,
        join_operations: 2,
        cache_hit_rate: 0.5,
        ..Default::default()
    }
}

fn create_optimized_stats() -> QueryStatistics {
    let mut stats = create_baseline_stats();
    stats.total_time_ms = 80; // 47% faster!
    stats.execution_time_ms = 50;
    stats.cache_hit_rate = 0.85; // Better cache usage
    stats.peak_memory_bytes = 768 * 1024; // 25% less memory

    let mut index_accesses = HashMap::new();
    index_accesses.insert("SPO".to_string(), 2);
    stats.index_accesses = index_accesses;

    stats
}

fn create_regressed_stats() -> QueryStatistics {
    let mut stats = create_baseline_stats();
    stats.total_time_ms = 350; // 133% slower!
    stats.execution_time_ms = 320;
    stats.cache_hit_rate = 0.2; // Worse cache usage
    stats.peak_memory_bytes = 2 * 1024 * 1024; // 2x memory

    stats
}

fn demonstrate_basic_profiling(stats: &QueryStatistics) {
    let builder = ProfiledPlanBuilder::new();
    let query = "SELECT ?person ?name WHERE { ?person a foaf:Person . ?person foaf:name ?name }";

    println!("Query: {}\n", query);

    let plan = builder.build_from_stats(stats, query);
    let visualizer = oxirs_core::query::query_plan_visualizer::QueryPlanVisualizer::new();
    let tree = visualizer.visualize_as_tree(&plan);

    println!("{}", tree);

    println!("\nExecution Summary:");
    println!("  Total Time:    {}ms", stats.total_time_ms);
    println!("  Parse:         {}ms", stats.parse_time_ms);
    println!("  Planning:      {}ms", stats.planning_time_ms);
    println!("  Execution:     {}ms", stats.execution_time_ms);
    println!("  Results:       {}", stats.results_count);
    println!("  Cache Hit:     {:.1}%", stats.cache_hit_rate * 100.0);
}

fn demonstrate_performance_analysis(stats: &QueryStatistics) {
    let builder = ProfiledPlanBuilder::new();
    let query = "SELECT ?person ?name ?friend WHERE { ?person a foaf:Person . ?person foaf:name ?name . ?person foaf:knows ?friend }";

    println!("Query: {}\n", query);

    let analysis = builder.analyze_performance(stats);

    println!("Performance Analysis:");
    println!("  Overall Grade: {:?}", analysis.overall_grade);
    println!("  Is Slow:       {}", analysis.is_slow);
    println!("  Cache:         {:?}", analysis.cache_effectiveness);

    if !analysis.slow_phases.is_empty() {
        println!("\n  Slow Phases:");
        for phase in &analysis.slow_phases {
            println!("    - {}", phase);
        }
    }

    if !analysis.inefficient_patterns.is_empty() {
        println!("\n  Inefficient Patterns:");
        for pattern in &analysis.inefficient_patterns {
            println!("    - {}", pattern);
        }
    }

    if !analysis.index_recommendations.is_empty() {
        println!("\n  Index Recommendations:");
        for rec in &analysis.index_recommendations {
            println!("    - {}", rec);
        }
    }
}

fn demonstrate_slow_query_analysis(stats: &QueryStatistics) {
    let builder = ProfiledPlanBuilder::new();
    let query = "SELECT * WHERE { ?s ?p ?o . ?article schema:author ?author }";

    println!("Query: {}\n", query);

    let analysis = builder.analyze_performance(stats);

    println!("🔴 SLOW QUERY DETECTED!");
    println!("\nPerformance Analysis:");
    println!("  Overall Grade: {:?}", analysis.overall_grade);
    println!("  Total Time:    {}ms (SLOW!)", stats.total_time_ms);
    println!(
        "  Cache Hit:     {:.1}% (Poor)",
        stats.cache_hit_rate * 100.0
    );

    println!("\n⚠️  Issues Detected:");
    if !analysis.slow_phases.is_empty() {
        println!("\n  Slow Phases:");
        for phase in &analysis.slow_phases {
            println!("    🐌 {}", phase);
        }
    }

    if !analysis.inefficient_patterns.is_empty() {
        println!("\n  Inefficient Patterns:");
        for pattern in &analysis.inefficient_patterns {
            println!("    📊 {}", pattern);
        }
    }

    if !analysis.index_recommendations.is_empty() {
        println!("\n  Critical Index Recommendations:");
        for rec in &analysis.index_recommendations {
            println!("    🔍 {}", rec);
        }
    }

    println!("\n💡 Recommendations:");
    println!("   1. Add indexes for frequently matched patterns");
    println!("   2. Increase query selectivity (avoid ?s ?p ?o)");
    println!("   3. Consider result set limits");
    println!("   4. Review cache configuration");
}

fn demonstrate_execution_comparison(baseline: &QueryStatistics, optimized: &QueryStatistics) {
    let builder = ProfiledPlanBuilder::new();
    let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o . FILTER(?s = <http://example.org/alice>) }";

    println!("Query: {}\n", query);

    let comparison = builder.compare_executions(baseline, optimized);

    println!("Baseline vs Optimized Execution:\n");
    println!("{}", comparison);

    println!("\n✅ Performance Improvement:");
    println!("   Time improved by {:.1}%", -comparison.time_diff_pct);
    println!("   Memory reduced by {:.1}%", -comparison.memory_diff_pct);
    println!(
        "   Cache effectiveness increased by {:.1}%",
        comparison.cache_hit_diff * 100.0
    );
}

fn demonstrate_regression_detection(baseline: &QueryStatistics, regressed: &QueryStatistics) {
    let builder = ProfiledPlanBuilder::new();

    let comparison = builder.compare_executions(baseline, regressed);

    println!("Baseline vs Current Execution:\n");
    println!("{}", comparison);

    println!("\n🔴 PERFORMANCE REGRESSION DETECTED!");
    println!("   Status: {:?}", comparison.improvement);
    println!("   Time regressed by {:.1}%", comparison.time_diff_pct);
    println!("   Memory increased by {:.1}%", comparison.memory_diff_pct);
    println!(
        "   Cache effectiveness decreased by {:.1}%",
        comparison.cache_hit_diff * 100.0
    );

    println!("\n⚠️  Action Required:");
    println!("   - Review recent changes to query execution");
    println!("   - Check for index degradation");
    println!("   - Verify cache configuration");
    println!("   - Consider rolling back recent changes");
}

fn demonstrate_complete_report(stats: &QueryStatistics) {
    let builder = ProfiledPlanBuilder::new();
    let query = "SELECT ?person ?name ?friend WHERE { ?person a foaf:Person . ?person foaf:name ?name . ?person foaf:knows ?friend }";

    let report = builder.generate_report(stats, query);

    println!("\n");
    report.print();
}
