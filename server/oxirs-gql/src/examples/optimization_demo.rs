//! Advanced Query Optimization Demo
//!
//! This example demonstrates the hybrid quantum-ML optimization capabilities
//! of OxiRS GraphQL with performance benchmarking and strategy selection.

use anyhow::Result;
use oxirs_gql::{
    hybrid_optimizer::{
        HybridQueryOptimizer, HybridOptimizerConfig, OptimizationStrategy,
        PerformanceTracker, OptimizationResult,
    },
    benchmarking::{BenchmarkSuite, BenchmarkConfig, OptimizationBenchmark},
    GraphQLConfig, GraphQLServer, RdfStore,
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("Starting OxiRS GraphQL Advanced Optimization Demo");

    // Create RDF store with sample data
    let store = Arc::new(RdfStore::new()?);
    load_optimization_test_data(&store).await?;

    // Configure hybrid optimization
    let optimizer_config = HybridOptimizerConfig {
        optimization_strategy: OptimizationStrategy::Adaptive,
        adaptive_strategy_selection: true,
        parallel_optimization: true,
        quantum_config: Default::default(),
        ml_config: Default::default(),
        ensemble_voting_enabled: true,
        performance_learning_enabled: true,
        cache_optimization_results: true,
        max_optimization_time: Duration::from_millis(500),
        confidence_threshold: 0.7,
    };

    // Create performance tracker
    let performance_tracker = Arc::new(PerformanceTracker::new());

    // Create hybrid optimizer
    let optimizer = HybridQueryOptimizer::new(optimizer_config.clone(), performance_tracker.clone());

    info!("Running optimization demonstrations...");

    // Demo 1: Simple query optimization
    demo_simple_optimization(&optimizer).await?;

    // Demo 2: Complex query optimization
    demo_complex_optimization(&optimizer).await?;

    // Demo 3: Comparative strategy benchmarking
    demo_strategy_comparison(&optimizer).await?;

    // Demo 4: Adaptive optimization learning
    demo_adaptive_learning(&optimizer).await?;

    // Demo 5: Performance benchmarking suite
    demo_comprehensive_benchmarking(&store, optimizer_config).await?;

    // Display final performance statistics
    display_performance_summary(&performance_tracker).await?;

    info!("Optimization demo completed successfully!");
    Ok(())
}

/// Load test data optimized for demonstrating optimization strategies
async fn load_optimization_test_data(store: &Arc<RdfStore>) -> Result<()> {
    info!("Loading optimization test dataset...");
    
    let mut store_mut = RdfStore::new()?;

    // Create a complex dataset with multiple entity types and relationships
    // This will provide various optimization scenarios

    // Add companies
    for i in 1..=100 {
        store_mut.insert_triple(
            &format!("http://example.org/company/{}", i),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://example.org/Company",
        )?;
        
        store_mut.insert_triple(
            &format!("http://example.org/company/{}", i),
            "http://example.org/name",
            &format!("\"Company {}\"", i),
        )?;
        
        store_mut.insert_triple(
            &format!("http://example.org/company/{}", i),
            "http://example.org/industry",
            &format!("\"Industry {}\"", i % 10),
        )?;
    }

    // Add employees with complex relationships
    for i in 1..=1000 {
        let company_id = (i % 100) + 1;
        store_mut.insert_triple(
            &format!("http://example.org/employee/{}", i),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://example.org/Employee",
        )?;
        
        store_mut.insert_triple(
            &format!("http://example.org/employee/{}", i),
            "http://example.org/worksFor",
            &format!("http://example.org/company/{}", company_id),
        )?;
        
        store_mut.insert_triple(
            &format!("http://example.org/employee/{}", i),
            "http://example.org/salary",
            &format!("\"{}\"", 30000 + (i * 100) % 100000),
        )?;
    }

    // Add projects with dependencies
    for i in 1..=500 {
        let company_id = (i % 100) + 1;
        store_mut.insert_triple(
            &format!("http://example.org/project/{}", i),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://example.org/Project",
        )?;
        
        store_mut.insert_triple(
            &format!("http://example.org/project/{}", i),
            "http://example.org/ownedBy",
            &format!("http://example.org/company/{}", company_id),
        )?;
        
        // Add project dependencies (creates complex query patterns)
        if i > 1 {
            store_mut.insert_triple(
                &format!("http://example.org/project/{}", i),
                "http://example.org/dependsOn",
                &format!("http://example.org/project/{}", i - 1),
            )?;
        }
    }

    info!("Test dataset loaded: 100 companies, 1000 employees, 500 projects");
    Ok(())
}

/// Demonstrate simple query optimization
async fn demo_simple_optimization(optimizer: &HybridQueryOptimizer) -> Result<()> {
    info!("=== Demo 1: Simple Query Optimization ===");

    let simple_query = r#"
    query SimpleCompanyQuery {
        companies(limit: 10) {
            id
            name
            industry
        }
    }
    "#;

    let document = juniper::parse_query(simple_query)?;
    let start_time = Instant::now();
    
    let result = optimizer.optimize_query(&document).await?;
    let optimization_time = start_time.elapsed();

    info!("Simple query optimization results:");
    info!("  Strategy used: {:?}", result.final_strategy);
    info!("  Confidence score: {:.3}", result.confidence_score);
    info!("  Optimization time: {:?}", optimization_time);
    info!("  Expected performance improvement: {:.1}%", result.performance_improvement_estimate * 100.0);

    Ok(())
}

/// Demonstrate complex query optimization
async fn demo_complex_optimization(optimizer: &HybridQueryOptimizer) -> Result<()> {
    info!("=== Demo 2: Complex Query Optimization ===");

    let complex_query = r#"
    query ComplexAnalyticsQuery {
        companies {
            id
            name
            industry
            employees {
                id
                name
                salary
                projects {
                    id
                    name
                    dependencies {
                        id
                        name
                    }
                }
            }
            totalSalaryBudget: employees(aggregation: SUM, field: salary)
            averageSalary: employees(aggregation: AVG, field: salary)
            projectCount: projects(aggregation: COUNT)
        }
    }
    "#;

    let document = juniper::parse_query(complex_query)?;
    let start_time = Instant::now();
    
    let result = optimizer.optimize_query(&document).await?;
    let optimization_time = start_time.elapsed();

    info!("Complex query optimization results:");
    info!("  Strategy used: {:?}", result.final_strategy);
    info!("  Confidence score: {:.3}", result.confidence_score);
    info!("  Optimization time: {:?}", optimization_time);
    info!("  Expected performance improvement: {:.1}%", result.performance_improvement_estimate * 100.0);
    
    if let Some(ref details) = result.optimization_details {
        info!("  Optimization details: {}", details);
    }

    Ok(())
}

/// Demonstrate strategy comparison
async fn demo_strategy_comparison(optimizer: &HybridQueryOptimizer) -> Result<()> {
    info!("=== Demo 3: Strategy Comparison ===");

    let test_query = r#"
    query StrategyTestQuery {
        companies(limit: 50) {
            id
            name
            employees(limit: 10) {
                id
                name
                projects(limit: 5) {
                    id
                    name
                }
            }
        }
    }
    "#;

    let document = juniper::parse_query(test_query)?;
    let strategies = vec![
        OptimizationStrategy::QuantumOnly,
        OptimizationStrategy::MLOnly,
        OptimizationStrategy::Hybrid,
        OptimizationStrategy::Adaptive,
    ];

    for strategy in strategies {
        info!("Testing strategy: {:?}", strategy);
        
        // Create temporary config for this strategy
        let mut config = optimizer.config.clone();
        config.optimization_strategy = strategy.clone();
        config.adaptive_strategy_selection = false; // Force specific strategy
        
        let temp_optimizer = HybridQueryOptimizer::new(config, optimizer.performance_tracker.clone());
        
        let start_time = Instant::now();
        let result = temp_optimizer.optimize_query(&document).await?;
        let time_taken = start_time.elapsed();
        
        info!("  Time: {:?}, Confidence: {:.3}, Improvement: {:.1}%", 
              time_taken, result.confidence_score, result.performance_improvement_estimate * 100.0);
    }

    Ok(())
}

/// Demonstrate adaptive learning
async fn demo_adaptive_learning(optimizer: &HybridQueryOptimizer) -> Result<()> {
    info!("=== Demo 4: Adaptive Learning ===");

    let queries = vec![
        "query Q1 { companies(limit: 5) { id name } }",
        "query Q2 { companies { id employees(limit: 3) { id } } }",
        "query Q3 { companies(industry: \"Tech\") { id name industry } }",
        "query Q4 { companies { id projects { id name } } }",
        "query Q5 { companies { id employees { salary } } }",
    ];

    info!("Running {} queries to demonstrate adaptive learning...", queries.len());

    for (i, query) in queries.iter().enumerate() {
        let document = juniper::parse_query(query)?;
        let result = optimizer.optimize_query(&document).await?;
        
        info!("Query {}: Strategy: {:?}, Confidence: {:.3}", 
              i + 1, result.final_strategy, result.confidence_score);
    }

    // Show learning progress
    if let Some(stats) = optimizer.get_learning_stats().await? {
        info!("Learning statistics:");
        info!("  Total optimizations: {}", stats.total_optimizations);
        info!("  Strategy distribution:");
        for (strategy, count) in stats.strategy_usage {
            info!("    {:?}: {} times", strategy, count);
        }
        info!("  Average confidence improvement: {:.3}", stats.confidence_improvement);
    }

    Ok(())
}

/// Demonstrate comprehensive benchmarking
async fn demo_comprehensive_benchmarking(
    store: &Arc<RdfStore>, 
    optimizer_config: HybridOptimizerConfig
) -> Result<()> {
    info!("=== Demo 5: Comprehensive Benchmarking Suite ===");

    let benchmark_config = BenchmarkConfig {
        warmup_iterations: 5,
        benchmark_iterations: 20,
        timeout_duration: Duration::from_secs(30),
        strategies_to_test: vec![
            OptimizationStrategy::QuantumOnly,
            OptimizationStrategy::MLOnly,
            OptimizationStrategy::Hybrid,
        ],
        query_complexities: vec!["simple", "medium", "complex"].into_iter().map(String::from).collect(),
        enable_detailed_metrics: true,
        enable_memory_profiling: true,
    };

    let mut benchmark_suite = BenchmarkSuite::new(benchmark_config);

    // Add optimization benchmark
    let optimization_benchmark = OptimizationBenchmark::new(
        store.clone(),
        optimizer_config,
    );

    benchmark_suite.add_benchmark(Box::new(optimization_benchmark));

    info!("Running comprehensive benchmark suite...");
    let results = benchmark_suite.run_all_benchmarks().await?;

    // Display benchmark results
    info!("Benchmark Results Summary:");
    for result in &results.benchmark_results {
        info!("  {}: {:.2}ms avg, {:.1}% improvement", 
              result.name, result.average_time_ms, result.performance_improvement_percent);
    }

    info!("Best performing strategy: {:?}", results.best_strategy);
    info!("Overall performance improvement: {:.1}%", results.overall_improvement_percent);

    // Generate detailed report
    let report_path = "/tmp/optimization_benchmark_report.json";
    benchmark_suite.generate_report(&results, report_path).await?;
    info!("Detailed report saved to: {}", report_path);

    Ok(())
}

/// Display performance summary
async fn display_performance_summary(tracker: &PerformanceTracker) -> Result<()> {
    info!("=== Performance Summary ===");

    let summary = tracker.get_summary().await;
    
    info!("Total optimizations performed: {}", summary.total_optimizations);
    info!("Average optimization time: {:.2}ms", summary.average_optimization_time_ms);
    info!("Total time saved: {:.2}ms", summary.total_time_saved_ms);
    info!("Average confidence score: {:.3}", summary.average_confidence_score);
    
    info!("Strategy performance:");
    for (strategy, metrics) in summary.strategy_metrics {
        info!("  {:?}:", strategy);
        info!("    Usage count: {}", metrics.usage_count);
        info!("    Average time: {:.2}ms", metrics.average_time_ms);
        info!("    Success rate: {:.1}%", metrics.success_rate * 100.0);
        info!("    Average improvement: {:.1}%", metrics.average_improvement * 100.0);
    }

    Ok(())
}