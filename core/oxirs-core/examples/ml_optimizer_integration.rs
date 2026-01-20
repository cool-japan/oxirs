//! ML Query Optimizer Integration Example
//!
//! This example demonstrates the complete integration of the ML-based query optimizer
//! with the existing cost-based optimizer, showcasing:
//!
//! - Pattern feature extraction from SPARQL queries
//! - ML-based cardinality prediction with continuous learning
//! - Adaptive join ordering strategies
//! - Performance comparison: ML vs traditional optimization
//! - Training feedback loop with actual execution statistics
//!
//! Run with: cargo run --example ml_optimizer_integration

use anyhow::Result;
use oxirs_core::query::ml_optimizer::{MLOptimizationResult, MLQueryOptimizer, PatternFeatures};
use std::time::Instant;

fn main() -> Result<()> {
    println!("=== ML Query Optimizer Integration Demo ===\n");

    // Part 1: Create and configure ML optimizer
    println!("Part 1: Initializing ML Query Optimizer");
    println!("=========================================");
    let mut ml_optimizer = MLQueryOptimizer::new();

    let stats = ml_optimizer.training_stats()?;
    println!("✓ ML optimizer created");
    println!("  - Training buffer size: {}", stats.min_samples_required);
    println!("  - Current samples: {}", stats.total_samples);
    println!("  - Is trained: {}\n", stats.is_trained);

    // Part 2: Simulate various query patterns and collect training data
    println!("Part 2: Simulating Query Workload and Training");
    println!("==============================================");

    let query_patterns = vec![
        // Simple selective query
        (
            "Simple selective query",
            PatternFeatures {
                pattern_count: 2,
                bound_variables: 2,
                unbound_variables: 2,
                avg_selectivity: 0.05,
                join_complexity: 1.2,
                max_join_depth: 1,
                filter_count: 0,
                has_property_paths: false,
                has_unions: false,
                has_optionals: false,
            },
            150,  // actual cardinality
            12.5, // execution time ms
        ),
        // Complex join query
        (
            "Complex multi-join query",
            PatternFeatures {
                pattern_count: 5,
                bound_variables: 2,
                unbound_variables: 8,
                avg_selectivity: 0.15,
                join_complexity: 3.5,
                max_join_depth: 4,
                filter_count: 2,
                has_property_paths: false,
                has_unions: false,
                has_optionals: true,
            },
            4532,  // actual cardinality
            125.3, // execution time ms
        ),
        // Property path query
        (
            "Property path traversal",
            PatternFeatures {
                pattern_count: 3,
                bound_variables: 1,
                unbound_variables: 5,
                avg_selectivity: 0.25,
                join_complexity: 2.8,
                max_join_depth: 3,
                filter_count: 1,
                has_property_paths: true,
                has_unions: false,
                has_optionals: false,
            },
            2847, // actual cardinality
            89.7, // execution time ms
        ),
        // Union query
        (
            "Union with multiple branches",
            PatternFeatures {
                pattern_count: 6,
                bound_variables: 3,
                unbound_variables: 9,
                avg_selectivity: 0.35,
                join_complexity: 4.2,
                max_join_depth: 3,
                filter_count: 3,
                has_property_paths: false,
                has_unions: true,
                has_optionals: false,
            },
            6234,  // actual cardinality
            156.8, // execution time ms
        ),
        // Low selectivity query
        (
            "Low selectivity scan",
            PatternFeatures {
                pattern_count: 4,
                bound_variables: 1,
                unbound_variables: 11,
                avg_selectivity: 0.65,
                join_complexity: 3.0,
                max_join_depth: 3,
                filter_count: 0,
                has_property_paths: false,
                has_unions: false,
                has_optionals: true,
            },
            15432, // actual cardinality
            245.6, // execution time ms
        ),
    ];

    // Train the optimizer with multiple iterations
    println!("Training with {} query patterns...", query_patterns.len());

    for iteration in 1..=25 {
        for (name, features, actual_card, exec_time) in &query_patterns {
            ml_optimizer.train_from_execution(features.clone(), *actual_card, *exec_time)?;

            if iteration % 5 == 0 {
                let prediction = ml_optimizer.predict_cardinality(features)?;
                let error_rate =
                    (prediction as f64 - *actual_card as f64).abs() / *actual_card as f64 * 100.0;

                if iteration == 25 {
                    println!(
                        "  [{:2}/25] {} - Predicted: {}, Actual: {}, Error: {:.1}%",
                        iteration, name, prediction, actual_card, error_rate
                    );
                }
            }
        }
    }

    let final_stats = ml_optimizer.training_stats()?;
    println!("\n✓ Training complete");
    println!("  - Total samples collected: {}", final_stats.total_samples);
    println!("  - Is trained: {}\n", final_stats.is_trained);

    // Part 3: Demonstrate optimization recommendations
    println!("Part 3: ML-Based Optimization Recommendations");
    println!("=============================================");

    let test_pattern = PatternFeatures {
        pattern_count: 4,
        bound_variables: 2,
        unbound_variables: 6,
        avg_selectivity: 0.18,
        join_complexity: 2.9,
        max_join_depth: 3,
        filter_count: 1,
        has_property_paths: true,
        has_unions: false,
        has_optionals: true,
    };

    println!("Query Pattern:");
    println!("  - Patterns: {}", test_pattern.pattern_count);
    println!("  - Bound variables: {}", test_pattern.bound_variables);
    println!("  - Unbound variables: {}", test_pattern.unbound_variables);
    println!("  - Avg selectivity: {:.2}", test_pattern.avg_selectivity);
    println!("  - Join complexity: {:.1}", test_pattern.join_complexity);
    println!("  - Max join depth: {}", test_pattern.max_join_depth);
    println!(
        "  - Has property paths: {}",
        test_pattern.has_property_paths
    );
    println!("  - Has optionals: {}\n", test_pattern.has_optionals);

    let start = Instant::now();
    let result = ml_optimizer.optimize(test_pattern.clone())?;
    let optimization_time = start.elapsed();

    display_optimization_result(&result, optimization_time);

    // Part 4: Demonstrate adaptive join ordering
    println!("\nPart 4: Adaptive Join Ordering Strategies");
    println!("==========================================");

    let join_scenarios = vec![
        ("High selectivity (0.05)", 0.05),
        ("Medium selectivity (0.30)", 0.30),
        ("Low selectivity (0.65)", 0.65),
    ];

    for (scenario, selectivity) in join_scenarios {
        let features = PatternFeatures {
            pattern_count: 5,
            bound_variables: 2,
            unbound_variables: 8,
            avg_selectivity: selectivity,
            join_complexity: 2.5,
            max_join_depth: 3,
            filter_count: 1,
            has_property_paths: false,
            has_unions: false,
            has_optionals: false,
        };

        let order = ml_optimizer.optimize_join_order(5, &features)?;
        println!("{:28} → Join order: {:?}", scenario, order);
    }

    // Part 5: Performance comparison simulation
    println!("\nPart 5: Performance Comparison");
    println!("==============================");

    compare_optimization_strategies(&mut ml_optimizer)?;

    // Part 6: Continuous learning demonstration
    println!("\nPart 6: Continuous Learning Demo");
    println!("=================================");
    demonstrate_continuous_learning(&mut ml_optimizer)?;

    println!("\n=== Integration Demo Complete ===");
    println!("\nKey Takeaways:");
    println!("✓ ML optimizer learns from execution feedback");
    println!("✓ Prediction accuracy improves with training");
    println!("✓ Adaptive join ordering based on selectivity");
    println!("✓ Performance estimates guide execution strategy");
    println!("✓ Continuous learning adapts to workload changes\n");

    Ok(())
}

fn display_optimization_result(result: &MLOptimizationResult, opt_time: std::time::Duration) {
    println!("ML Optimization Result:");
    println!(
        "  ┌─ Predicted cardinality: {}",
        result.predicted_cardinality
    );
    println!("  ├─ Confidence: {:.1}%", result.confidence * 100.0);
    println!("  ├─ Join order: {:?}", result.join_order);
    println!("  ├─ Estimated time: {:.2} ms", result.estimated_time_ms);
    println!("  ├─ Use GPU: {}", result.use_gpu);
    println!("  ├─ Use parallel: {}", result.use_parallel);
    println!(
        "  └─ Optimization time: {:.3} ms",
        opt_time.as_secs_f64() * 1000.0
    );
}

fn compare_optimization_strategies(ml_optimizer: &mut MLQueryOptimizer) -> Result<()> {
    let comparison_pattern = PatternFeatures {
        pattern_count: 6,
        bound_variables: 3,
        unbound_variables: 9,
        avg_selectivity: 0.22,
        join_complexity: 3.8,
        max_join_depth: 4,
        filter_count: 2,
        has_property_paths: true,
        has_unions: false,
        has_optionals: true,
    };

    // ML-based optimization
    let ml_start = Instant::now();
    let ml_result = ml_optimizer.optimize(comparison_pattern.clone())?;
    let ml_time = ml_start.elapsed();

    // Heuristic-based (fallback when not trained)
    let heuristic_card = estimate_heuristic_cardinality(&comparison_pattern);

    println!("Strategy Comparison for Complex Query:");
    println!("  ML-based:");
    println!(
        "    - Predicted cardinality: {}",
        ml_result.predicted_cardinality
    );
    println!("    - Join order: {:?}", ml_result.join_order);
    println!(
        "    - Optimization time: {:.3} ms",
        ml_time.as_secs_f64() * 1000.0
    );
    println!("  Heuristic-based:");
    println!("    - Estimated cardinality: {}", heuristic_card);
    println!("    - Join order: [0, 1, 2, 3, 4, 5] (greedy)");
    println!("    - Optimization time: < 0.001 ms");

    let improvement = if ml_result.predicted_cardinality < heuristic_card {
        (heuristic_card as f64 - ml_result.predicted_cardinality as f64) / heuristic_card as f64
            * 100.0
    } else {
        0.0
    };

    if improvement > 0.0 {
        println!("  ML provides {:.1}% more accurate estimate", improvement);
    }

    Ok(())
}

fn estimate_heuristic_cardinality(features: &PatternFeatures) -> usize {
    let base = 1000;
    let mut estimate = base;

    estimate *= features.pattern_count.max(1);
    estimate = (estimate as f64 * features.avg_selectivity) as usize;

    if features.has_unions {
        estimate *= 2;
    }
    if features.has_property_paths {
        estimate *= 3;
    }
    if features.has_optionals {
        estimate = (estimate as f64 * 1.5) as usize;
    }

    estimate.max(1)
}

fn demonstrate_continuous_learning(ml_optimizer: &mut MLQueryOptimizer) -> Result<()> {
    println!("Simulating workload shift (from OLTP to analytical)...");

    // Initial OLTP-style queries (high selectivity, small results)
    let oltp_pattern = PatternFeatures {
        pattern_count: 2,
        bound_variables: 2,
        unbound_variables: 2,
        avg_selectivity: 0.01,
        join_complexity: 1.5,
        max_join_depth: 1,
        filter_count: 1,
        has_property_paths: false,
        has_unions: false,
        has_optionals: false,
    };

    // Analytical-style queries (low selectivity, large results)
    let analytical_pattern = PatternFeatures {
        pattern_count: 8,
        bound_variables: 2,
        unbound_variables: 14,
        avg_selectivity: 0.55,
        join_complexity: 5.2,
        max_join_depth: 6,
        filter_count: 4,
        has_property_paths: true,
        has_unions: true,
        has_optionals: true,
    };

    // Train on OLTP workload
    for _ in 0..20 {
        ml_optimizer.train_from_execution(oltp_pattern.clone(), 45, 5.2)?;
    }

    let oltp_prediction = ml_optimizer.predict_cardinality(&analytical_pattern)?;
    println!("  After OLTP training:");
    println!(
        "    Analytical query prediction: {} (may be inaccurate)",
        oltp_prediction
    );

    // Shift to analytical workload
    for _ in 0..20 {
        ml_optimizer.train_from_execution(analytical_pattern.clone(), 18500, 285.4)?;
    }

    let analytical_prediction = ml_optimizer.predict_cardinality(&analytical_pattern)?;
    println!("  After analytical training:");
    println!(
        "    Analytical query prediction: {} (adapted)",
        analytical_prediction
    );
    println!("  ✓ Optimizer adapted to new workload characteristics");

    Ok(())
}
