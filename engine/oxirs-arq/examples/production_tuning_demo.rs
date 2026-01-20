//! Production Tuning Demonstration
//!
//! This example shows how to use different optimizer configurations
//! for various production workload profiles.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example production_tuning_demo --features star
//! ```

use oxirs_arq::optimizer::{ProductionOptimizerConfig, WorkloadProfile};

fn main() -> anyhow::Result<()> {
    println!("=== Production Optimizer Tuning Guide ===\n");

    println!("This guide demonstrates workload-specific optimizer configurations");
    println!("for achieving optimal performance in different deployment scenarios.\n");

    // Part 1: High Throughput OLTP
    println!("📊 Profile 1: High-Throughput OLTP");
    println!("════════════════════════════════════\n");
    demo_profile(WorkloadProfile::HighThroughput);

    // Part 2: Analytical Queries OLAP
    println!("\n📈 Profile 2: Analytical Queries (OLAP)");
    println!("════════════════════════════════════════\n");
    demo_profile(WorkloadProfile::AnalyticalQueries);

    // Part 3: Mixed Workload
    println!("\n⚖️  Profile 3: Mixed Workload");
    println!("════════════════════════════════\n");
    demo_profile(WorkloadProfile::Mixed);

    // Part 4: Low Memory
    println!("\n💾 Profile 4: Low Memory (Edge/Container)");
    println!("═══════════════════════════════════════\n");
    demo_profile(WorkloadProfile::LowMemory);

    // Part 5: Low CPU
    println!("\n🔋 Profile 5: Low CPU (Resource Constrained)");
    println!("═════════════════════════════════════════════\n");
    demo_profile(WorkloadProfile::LowCpu);

    // Part 6: Maximum Performance
    println!("\n🚀 Profile 6: Maximum Performance");
    println!("══════════════════════════════════\n");
    demo_profile(WorkloadProfile::MaxPerformance);

    // Summary and Recommendations
    println!("\n✨ Production Deployment Recommendations");
    println!("════════════════════════════════════════\n");
    print_recommendations();

    println!("\n✅ Demo complete! Choose the profile that matches your workload.\n");

    Ok(())
}

fn demo_profile(profile: WorkloadProfile) {
    let config = ProductionOptimizerConfig::for_workload(profile);

    println!("Workload Profile: {:?}", profile);
    println!("\nOptimizer Settings:");
    println!(
        "  • Join Reordering: {}",
        config.base_config.join_reordering
    );
    println!(
        "  • Filter Pushdown: {}",
        config.base_config.filter_pushdown
    );
    println!(
        "  • Cost-Based Optimization: {}",
        config.base_config.cost_based
    );
    println!(
        "  • Max Optimization Passes: {}",
        config.base_config.max_passes
    );
    println!("  • Estimation Method: {:?}", config.estimation_method);

    println!("\nCaching Configuration:");
    println!(
        "  • Plan Cache Size: {} queries",
        config.max_plan_cache_size
    );
    println!(
        "  • Result Cache: {}",
        if config.enable_result_cache {
            "Enabled"
        } else {
            "Disabled"
        }
    );
    if config.enable_result_cache {
        println!(
            "  • Result Cache TTL: {} seconds",
            config.result_cache_ttl_secs
        );
    }

    println!("\nAdaptive Learning:");
    println!("  • Adaptive Learning: {}", config.adaptive_learning);
    if config.adaptive_learning {
        println!(
            "  • ML Training Threshold: {} samples",
            config.ml_training_threshold
        );
        println!(
            "  • Stats Update Frequency: every {} queries",
            config.stats_update_frequency
        );
    }

    let resources = config.estimate_resource_requirements();
    println!("\nResource Requirements:");
    println!("  • Memory: ~{}MB", resources.memory_mb);
    println!("  • CPU Cores: {} recommended", resources.cpu_cores);
    println!("  • Cache Memory: ~{}MB", resources.cache_memory_mb);
    println!(
        "  • Max Concurrent Queries: {}",
        resources.max_concurrent_queries
    );

    // Validate configuration
    let warnings = config.validate();
    if !warnings.is_empty() {
        println!("\n⚠️  Configuration Warnings:");
        for warning in warnings {
            println!("  • {}", warning);
        }
    } else {
        println!("\n✓ Configuration validated - no warnings");
    }

    // Print use cases
    println!("\n🎯 Best For:");
    match profile {
        WorkloadProfile::HighThroughput => {
            println!("  • Simple queries (2-5 triple patterns)");
            println!("  • High query rate (>1000 QPS)");
            println!("  • Low latency requirements (<10ms p95)");
            println!("  • Repeated query patterns");
            println!("  • E-commerce, real-time APIs, web applications");
        }
        WorkloadProfile::AnalyticalQueries => {
            println!("  • Complex queries (10-100 triple patterns)");
            println!("  • Low query rate (<10 QPS)");
            println!("  • Large result sets (>10K rows)");
            println!("  • Data warehousing, business intelligence");
            println!("  • Ad-hoc analytical queries");
        }
        WorkloadProfile::Mixed => {
            println!("  • Combination of simple and complex queries");
            println!("  • Moderate query rate (10-1000 QPS)");
            println!("  • Variable result sizes");
            println!("  • General-purpose SPARQL endpoints");
            println!("  • Most production deployments");
        }
        WorkloadProfile::LowMemory => {
            println!("  • Limited RAM (<2GB available)");
            println!("  • Containerized deployments (Docker, Kubernetes)");
            println!("  • Edge computing devices");
            println!("  • Embedded systems");
            println!("  • Development/testing environments");
        }
        WorkloadProfile::LowCpu => {
            println!("  • CPU-constrained environments");
            println!("  • Shared hosting");
            println!("  • Mobile/IoT devices");
            println!("  • Minimizing CPU usage");
            println!("  • Battery-powered devices");
        }
        WorkloadProfile::MaxPerformance => {
            println!("  • Dedicated servers (16+ cores, 32GB+ RAM)");
            println!("  • Mission-critical queries");
            println!("  • Maximum optimization needed");
            println!("  • Premium hosting environments");
            println!("  • Research/academic workloads");
        }
    }

    println!("\n💡 Configuration Tips:");
    match profile {
        WorkloadProfile::HighThroughput => {
            println!("  • Use aggressive plan caching for repeated queries");
            println!("  • Minimize optimization overhead with fewer passes");
            println!("  • Enable result caching for frequently-run queries");
            println!("  • Consider read replicas for scaling beyond 1K QPS");
        }
        WorkloadProfile::AnalyticalQueries => {
            println!("  • Enable ML-based cardinality estimation");
            println!("  • Use adaptive learning to improve over time");
            println!("  • Allocate more resources for complex optimizations");
            println!("  • Monitor query execution for bottleneck identification");
        }
        WorkloadProfile::Mixed => {
            println!("  • Balance between optimization depth and speed");
            println!("  • Use histogram-based estimation for good accuracy");
            println!("  • Enable adaptive learning for workload adaptation");
            println!("  • Monitor workload patterns and adjust if skewed");
        }
        WorkloadProfile::LowMemory => {
            println!("  • Disable result caching if memory is critical");
            println!("  • Use HyperLogLog sketches (only 16KB per predicate)");
            println!("  • Minimize plan cache size (100 plans = ~100KB)");
            println!("  • Consider streaming results instead of materialization");
        }
        WorkloadProfile::LowCpu => {
            println!("  • Rely on plan and result caching to avoid re-computation");
            println!("  • Use simple heuristics instead of cost-based optimization");
            println!("  • Limit optimization passes to 2-3");
            println!("  • Enable filter pushdown for early pruning");
        }
        WorkloadProfile::MaxPerformance => {
            println!("  • Enable all optimizations (30 passes)");
            println!("  • Use large caches (50K plans, 2-hour TTL)");
            println!("  • Train ML models aggressively (50 sample threshold)");
            println!("  • Monitor and tune based on actual query patterns");
        }
    }
}

fn print_recommendations() {
    println!("🎯 Quick Selection Guide:");
    println!("   ┌──────────────────────┬─────────────────────────────────┐");
    println!("   │ Your Scenario        │ Recommended Profile             │");
    println!("   ├──────────────────────┼─────────────────────────────────┤");
    println!("   │ REST API (<10ms)     │ HighThroughput                  │");
    println!("   │ Business Intelligence│ AnalyticalQueries               │");
    println!("   │ General SPARQL       │ Mixed (default)                 │");
    println!("   │ Docker/K8s (<2GB)    │ LowMemory                       │");
    println!("   │ Edge Device          │ LowCpu or LowMemory             │");
    println!("   │ Dedicated Server     │ MaxPerformance                  │");
    println!("   └──────────────────────┴─────────────────────────────────┘");

    println!("\n📈 Performance Expectations:");
    println!("   • HighThroughput: 1000+ QPS, <10ms p95 latency");
    println!("   • AnalyticalQueries: 5-10 QPS, optimized for complex queries");
    println!("   • Mixed: 100-500 QPS, balanced performance");
    println!("   • LowMemory: 50-100 QPS with <100MB overhead");
    println!("   • LowCpu: 20-50 QPS, minimal CPU usage");
    println!("   • MaxPerformance: 500+ QPS with advanced optimization");

    println!("\n🔧 Advanced Tuning:");
    println!("   1. Start with recommended profile");
    println!("   2. Monitor actual query patterns and performance");
    println!("   3. Adjust cache sizes based on hit rates");
    println!("   4. Enable adaptive learning for workload-specific optimization");
    println!("   5. Use ML estimation after collecting 100+ execution samples");

    println!("\n📊 Monitoring Recommendations:");
    println!("   • Track cache hit rates (target: >80% for repeated queries)");
    println!("   • Monitor optimization overhead (should be <10% of execution time)");
    println!("   • Measure p95/p99 latencies for SLA compliance");
    println!("   • Watch memory usage trends for cache sizing");
    println!("   • Profile slow queries for optimization opportunities");

    println!("\n⚡ Quick Wins:");
    println!("   ✓ Enable plan caching for 2-5x speedup on repeated queries");
    println!("   ✓ Use result caching for 10-100x speedup on identical queries");
    println!("   ✓ Enable filter pushdown for 3-10x reduction in intermediate results");
    println!("   ✓ Train ML model for 20-50% cardinality estimation improvement");
    println!("   ✓ Use cost-based join ordering for 2-10x speedup on complex joins");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_profiles() {
        let profiles = vec![
            WorkloadProfile::HighThroughput,
            WorkloadProfile::AnalyticalQueries,
            WorkloadProfile::Mixed,
            WorkloadProfile::LowMemory,
            WorkloadProfile::LowCpu,
            WorkloadProfile::MaxPerformance,
        ];

        for profile in profiles {
            println!("Testing profile: {:?}", profile);
            demo_profile(profile);
        }
    }
}
