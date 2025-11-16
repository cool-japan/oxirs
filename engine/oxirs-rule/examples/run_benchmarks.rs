//! Benchmark Runner for Integration Performance Analysis
//!
//! Executes the comprehensive integration benchmark suite and provides
//! detailed performance analysis including:
//! - Operations per second for all integration features
//! - Average execution time per operation
//! - Performance comparison across different modes
//! - Identification of performance bottlenecks
//! - Scaling characteristics for distributed reasoning

use anyhow::Result;
use oxirs_rule::integration_benchmarks::IntegrationBenchmarkSuite;

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  OxiRS Rule Engine - Integration Performance Benchmarks");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Create and run benchmark suite
    let mut suite = IntegrationBenchmarkSuite::new();

    println!("Running comprehensive benchmark suite...\n");
    let results = suite.run_all_benchmarks()?.to_vec();

    // Print summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Benchmark Summary");
    println!("═══════════════════════════════════════════════════════════════\n");
    suite.print_summary();

    // Analyze SPARQL performance
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  SPARQL Integration Analysis");
    println!("═══════════════════════════════════════════════════════════════\n");
    analyze_sparql_performance(&results);

    // Analyze SHACL performance
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  SHACL Integration Analysis");
    println!("═══════════════════════════════════════════════════════════════\n");
    analyze_shacl_performance(&results);

    // Analyze distributed performance
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Distributed Reasoning Analysis");
    println!("═══════════════════════════════════════════════════════════════\n");
    analyze_distributed_performance(&results);

    // Analyze composition performance
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Composition System Analysis");
    println!("═══════════════════════════════════════════════════════════════\n");
    analyze_composition_performance(&results);

    // Identify optimization opportunities
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Optimization Recommendations");
    println!("═══════════════════════════════════════════════════════════════\n");
    provide_optimization_recommendations(&results);

    // Overall statistics
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Overall Statistics");
    println!("═══════════════════════════════════════════════════════════════\n");
    print_overall_statistics(&results);

    Ok(())
}

/// Analyze SPARQL integration performance across different modes
fn analyze_sparql_performance(results: &[oxirs_rule::integration_benchmarks::BenchmarkResult]) {
    let sparql_results: Vec<_> = results
        .iter()
        .filter(|r| r.name.starts_with("SPARQL"))
        .collect();

    if sparql_results.is_empty() {
        println!("  No SPARQL benchmarks found.");
        return;
    }

    println!("  Query Mode Performance Comparison:");
    println!("  ┌─────────────────────────────────┬──────────────┬──────────────┐");
    println!("  │ Mode                            │ Ops/sec      │ Avg Time (μs)│");
    println!("  ├─────────────────────────────────┼──────────────┼──────────────┤");

    for result in &sparql_results {
        let mode_name = result.name.replace("SPARQL - ", "");
        println!(
            "  │ {:<31} │ {:>12.2} │ {:>12.2} │",
            mode_name, result.ops_per_sec, result.avg_time_us
        );
    }
    println!("  └─────────────────────────────────┴──────────────┴──────────────┘");

    // Find fastest and slowest modes
    if let Some(fastest) = sparql_results
        .iter()
        .max_by(|a, b| a.ops_per_sec.partial_cmp(&b.ops_per_sec).unwrap())
    {
        println!(
            "\n  ⚡ Fastest: {} ({:.2} ops/sec)",
            fastest.name.replace("SPARQL - ", ""),
            fastest.ops_per_sec
        );
    }

    if let Some(slowest) = sparql_results
        .iter()
        .min_by(|a, b| a.ops_per_sec.partial_cmp(&b.ops_per_sec).unwrap())
    {
        println!(
            "  🐌 Slowest: {} ({:.2} ops/sec)",
            slowest.name.replace("SPARQL - ", ""),
            slowest.ops_per_sec
        );
    }
}

/// Analyze SHACL validation performance across different modes
fn analyze_shacl_performance(results: &[oxirs_rule::integration_benchmarks::BenchmarkResult]) {
    let shacl_results: Vec<_> = results
        .iter()
        .filter(|r| r.name.starts_with("SHACL"))
        .collect();

    if shacl_results.is_empty() {
        println!("  No SHACL benchmarks found.");
        return;
    }

    println!("  Validation Mode Performance Comparison:");
    println!("  ┌─────────────────────────────────┬──────────────┬──────────────┐");
    println!("  │ Mode                            │ Ops/sec      │ Avg Time (μs)│");
    println!("  ├─────────────────────────────────┼──────────────┼──────────────┤");

    for result in &shacl_results {
        let mode_name = result.name.replace("SHACL - ", "");
        println!(
            "  │ {:<31} │ {:>12.2} │ {:>12.2} │",
            mode_name, result.ops_per_sec, result.avg_time_us
        );
    }
    println!("  └─────────────────────────────────┴──────────────┴──────────────┘");

    // Calculate overhead of reasoning vs direct validation
    let direct = shacl_results.iter().find(|r| r.name.contains("Direct"));
    let full = shacl_results.iter().find(|r| r.name.contains("Full"));

    if let (Some(direct), Some(full)) = (direct, full) {
        let overhead_pct = ((full.avg_time_us - direct.avg_time_us) / direct.avg_time_us) * 100.0;
        println!(
            "\n  📊 Reasoning Overhead: {:.1}% slower than direct validation",
            overhead_pct
        );
    }
}

/// Analyze distributed reasoning performance and scaling
fn analyze_distributed_performance(
    results: &[oxirs_rule::integration_benchmarks::BenchmarkResult],
) {
    let distributed_results: Vec<_> = results
        .iter()
        .filter(|r| r.name.starts_with("Distributed"))
        .collect();

    if distributed_results.is_empty() {
        println!("  No distributed benchmarks found.");
        return;
    }

    println!("  Partitioning Strategy Performance:");
    println!("  ┌─────────────────────────────────┬──────────────┬──────────────┐");
    println!("  │ Strategy                        │ Ops/sec      │ Avg Time (μs)│");
    println!("  ├─────────────────────────────────┼──────────────┼──────────────┤");

    for result in distributed_results
        .iter()
        .filter(|r| !r.name.contains("Scaling"))
    {
        let strategy_name = result.name.replace("Distributed - ", "");
        println!(
            "  │ {:<31} │ {:>12.2} │ {:>12.2} │",
            strategy_name, result.ops_per_sec, result.avg_time_us
        );
    }
    println!("  └─────────────────────────────────┴──────────────┴──────────────┘");

    // Analyze scaling characteristics
    let scaling_results: Vec<_> = distributed_results
        .iter()
        .filter(|r| r.name.contains("Scaling"))
        .collect();

    if !scaling_results.is_empty() {
        println!("\n  Horizontal Scaling Analysis:");
        println!("  ┌───────────┬──────────────┬──────────────┬──────────────┐");
        println!("  │ Nodes     │ Ops/sec      │ Avg Time (μs)│ Speedup      │");
        println!("  ├───────────┼──────────────┼──────────────┼──────────────┤");

        let baseline = scaling_results
            .first()
            .map(|r| r.ops_per_sec)
            .unwrap_or(1.0);

        for (idx, result) in scaling_results.iter().enumerate() {
            let nodes = 1 << idx; // 1, 2, 4, 8 nodes
            let speedup = result.ops_per_sec / baseline;
            println!(
                "  │ {:>9} │ {:>12.2} │ {:>12.2} │ {:>12.2}x│",
                nodes, result.ops_per_sec, result.avg_time_us, speedup
            );
        }
        println!("  └───────────┴──────────────┴──────────────┴──────────────┘");

        // Calculate scaling efficiency
        if scaling_results.len() >= 2 {
            let first = &scaling_results[0];
            let last = &scaling_results[scaling_results.len() - 1];
            let theoretical_max = (scaling_results.len() as f64) * first.ops_per_sec;
            let actual = last.ops_per_sec;
            let efficiency = (actual / theoretical_max) * 100.0;
            println!(
                "\n  📈 Scaling Efficiency: {:.1}% of theoretical maximum",
                efficiency
            );
        }
    }
}

/// Analyze composition system performance
fn analyze_composition_performance(
    results: &[oxirs_rule::integration_benchmarks::BenchmarkResult],
) {
    let composition_results: Vec<_> = results
        .iter()
        .filter(|r| r.name.starts_with("Composition"))
        .collect();

    if composition_results.is_empty() {
        println!("  No composition benchmarks found.");
        return;
    }

    println!("  Composition Operations Performance:");
    println!("  ┌─────────────────────────────────┬──────────────┬──────────────┐");
    println!("  │ Operation                       │ Ops/sec      │ Avg Time (μs)│");
    println!("  ├─────────────────────────────────┼──────────────┼──────────────┤");

    for result in &composition_results {
        let op_name = result.name.replace("Composition - ", "");
        println!(
            "  │ {:<31} │ {:>12.2} │ {:>12.2} │",
            op_name, result.ops_per_sec, result.avg_time_us
        );
    }
    println!("  └─────────────────────────────────┴──────────────┴──────────────┘");

    // Calculate average overhead per operation
    let avg_time: f64 = composition_results
        .iter()
        .map(|r| r.avg_time_us)
        .sum::<f64>()
        / composition_results.len() as f64;
    println!("\n  ⏱️  Average Operation Time: {:.2} μs", avg_time);
}

/// Provide optimization recommendations based on benchmark results
fn provide_optimization_recommendations(
    results: &[oxirs_rule::integration_benchmarks::BenchmarkResult],
) {
    let mut recommendations = Vec::new();

    // Find slowest operations
    let mut sorted_results = results.to_vec();
    sorted_results.sort_by(|a, b| b.avg_time_us.partial_cmp(&a.avg_time_us).unwrap());

    println!("  Top 5 Performance Bottlenecks:");
    println!("  ┌────┬──────────────────────────────────┬──────────────┐");
    println!("  │ #  │ Operation                        │ Avg Time (μs)│");
    println!("  ├────┼──────────────────────────────────┼──────────────┤");

    for (idx, result) in sorted_results.iter().take(5).enumerate() {
        let short_name = if result.name.len() > 32 {
            format!("{}...", &result.name[..29])
        } else {
            result.name.clone()
        };
        println!(
            "  │ {:>2} │ {:<32} │ {:>12.2} │",
            idx + 1,
            short_name,
            result.avg_time_us
        );

        // Generate specific recommendations
        if (result.name.contains("Backward") || result.name.contains("Hybrid"))
            && !recommendations.contains(&"query_optimization")
        {
            recommendations.push("query_optimization");
        }
        if (result.name.contains("Post-Reasoning") || result.name.contains("Full"))
            && !recommendations.contains(&"validation_caching")
        {
            recommendations.push("validation_caching");
        }
        if result.name.contains("Distributed") && !recommendations.contains(&"network_optimization")
        {
            recommendations.push("network_optimization");
        }
    }
    println!("  └────┴──────────────────────────────────┴──────────────┘");

    println!("\n  Recommended Optimizations:");
    if recommendations.contains(&"query_optimization") {
        println!("  • Implement query plan caching for backward/hybrid reasoning");
        println!("  • Add join order optimization for complex patterns");
    }
    if recommendations.contains(&"validation_caching") {
        println!("  • Implement incremental validation for SHACL constraints");
        println!("  • Add constraint result caching with invalidation strategy");
    }
    if recommendations.contains(&"network_optimization") {
        println!("  • Implement batch processing for distributed operations");
        println!("  • Add compression for inter-node communication");
    }

    // Check for operations that could benefit from SIMD
    let avg_time: f64 = results.iter().map(|r| r.avg_time_us).sum::<f64>() / results.len() as f64;
    if avg_time > 100.0 {
        println!("  • Consider SIMD vectorization for graph operations (via scirs2-core)");
        println!("  • Evaluate GPU acceleration for embeddings (via scirs2-core::gpu)");
    }
}

/// Print overall statistics across all benchmarks
fn print_overall_statistics(results: &[oxirs_rule::integration_benchmarks::BenchmarkResult]) {
    let total_operations: usize = results.iter().map(|r| r.operations).sum();
    let total_time: std::time::Duration = results.iter().map(|r| r.duration).sum();
    let avg_ops_per_sec: f64 =
        results.iter().map(|r| r.ops_per_sec).sum::<f64>() / results.len() as f64;
    let avg_time_us: f64 =
        results.iter().map(|r| r.avg_time_us).sum::<f64>() / results.len() as f64;

    println!("  Total Benchmarks:      {}", results.len());
    println!("  Total Operations:      {}", total_operations);
    println!("  Total Time:            {:.2}s", total_time.as_secs_f64());
    println!("  Average Throughput:    {:.2} ops/sec", avg_ops_per_sec);
    println!("  Average Latency:       {:.2} μs/op", avg_time_us);

    // Find extreme values
    if let Some(fastest) = results
        .iter()
        .max_by(|a, b| a.ops_per_sec.partial_cmp(&b.ops_per_sec).unwrap())
    {
        println!(
            "\n  ⚡ Fastest Operation:    {} ({:.2} ops/sec)",
            fastest.name, fastest.ops_per_sec
        );
    }

    if let Some(slowest) = results
        .iter()
        .min_by(|a, b| a.ops_per_sec.partial_cmp(&b.ops_per_sec).unwrap())
    {
        println!(
            "  🐌 Slowest Operation:   {} ({:.2} ops/sec)",
            slowest.name, slowest.ops_per_sec
        );
    }

    // Performance rating
    let rating = if avg_ops_per_sec > 10000.0 {
        "Excellent ⭐⭐⭐⭐⭐"
    } else if avg_ops_per_sec > 5000.0 {
        "Good ⭐⭐⭐⭐"
    } else if avg_ops_per_sec > 1000.0 {
        "Fair ⭐⭐⭐"
    } else {
        "Needs Improvement ⭐⭐"
    };

    println!("\n  Overall Performance:   {}", rating);
}
