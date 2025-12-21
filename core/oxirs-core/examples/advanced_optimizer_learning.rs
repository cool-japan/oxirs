//! Advanced Optimizer Learning - Histogram-Based Cardinality Estimation
//!
//! This example demonstrates the self-learning capabilities of OxiRS's
//! cost-based query optimizer with advanced statistics collection.
//!
//! # Features Demonstrated
//! - Histogram-based cardinality estimation (median of 100 observations per term)
//! - Adaptive join selectivity learning (1000 observations with similarity matching)
//! - Execution history tracking (1000 recent query executions)
//! - Continuous improvement through feedback loops
//! - Performance comparison: initial estimates vs learned estimates

use oxirs_core::model::{NamedNode, Variable};
use oxirs_core::query::algebra::{AlgebraTriplePattern, GraphPattern, TermPattern};
use oxirs_core::query::cost_based_optimizer::CostBasedOptimizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiRS Advanced Optimizer Learning Demo ===\n");

    // Part 1: Histogram-Based Cardinality Estimation
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  Part 1: Histogram-Based Cardinality Estimation     ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    demonstrate_histogram_cardinality()?;

    // Part 2: Adaptive Join Selectivity Learning
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║  Part 2: Adaptive Join Selectivity Learning         ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    demonstrate_join_selectivity_learning()?;

    // Part 3: Execution History Tracking
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║  Part 3: Execution History Tracking                 ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    demonstrate_execution_history()?;

    // Part 4: Complete Learning Cycle
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║  Part 4: Complete Learning Cycle                    ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    demonstrate_learning_cycle()?;

    println!("\n✅ Advanced optimizer learning demonstration complete!\n");
    println!("💡 Key Takeaways:");
    println!("   - Histogram-based estimation uses median (robust to outliers)");
    println!("   - Join selectivity adapts based on similar historical queries");
    println!("   - Execution history enables pattern-specific optimization");
    println!("   - Optimizer continuously improves with more query executions");

    Ok(())
}

fn demonstrate_histogram_cardinality() -> Result<(), Box<dyn std::error::Error>> {
    let optimizer = CostBasedOptimizer::new();

    // Create a pattern that we'll execute multiple times
    let foaf_name = NamedNode::new("http://xmlns.com/foaf/0.1/name").unwrap();
    let pattern = AlgebraTriplePattern {
        subject: TermPattern::Variable(Variable::new("person").unwrap()),
        predicate: TermPattern::NamedNode(foaf_name),
        object: TermPattern::Variable(Variable::new("name").unwrap()),
    };

    println!("Query Pattern: ?person foaf:name ?name\n");

    // Initial optimization (no historical data)
    let graph_pattern = GraphPattern::Bgp(vec![pattern.clone()]);
    let initial_plan = optimizer.optimize_pattern(&graph_pattern)?;
    println!(
        "Initial Estimate (no history): {} results",
        initial_plan.estimated_cardinality
    );

    // Simulate 20 query executions with varying cardinalities
    let actual_cardinalities = vec![
        950, 980, 1000, 1020, 950,  // Cluster around 1000
        2000, // Outlier
        990, 1010, 970, 1030, // More around 1000
        3000, // Another outlier
        1000, 980, 1020, 950, 990, 1010, 970, 1000, 1020,
    ];

    println!(
        "\nRecording {} executions with varying cardinalities...",
        actual_cardinalities.len()
    );
    for (i, &actual_card) in actual_cardinalities.iter().enumerate() {
        optimizer.update_stats_with_time(&graph_pattern, actual_card, 10 + i as u64);

        if (i + 1) % 5 == 0 {
            print!("  After {} executions: ", i + 1);
            let plan = optimizer.optimize_pattern(&graph_pattern)?;
            println!("estimated cardinality = {}", plan.estimated_cardinality);
        }
    }

    // Final optimization after learning
    let learned_plan = optimizer.optimize_pattern(&graph_pattern)?;
    println!(
        "\nFinal Estimate (histogram-based): {} results",
        learned_plan.estimated_cardinality
    );

    // Calculate median for comparison
    let mut sorted_cards = actual_cardinalities.clone();
    sorted_cards.sort_unstable();
    let median = sorted_cards[sorted_cards.len() / 2];

    println!("  Actual median: {} results", median);
    println!(
        "  Mean (affected by outliers): {:.0} results",
        actual_cardinalities.iter().sum::<usize>() as f64 / actual_cardinalities.len() as f64
    );

    println!("\n💡 Histogram uses median → robust to outliers (2000, 3000)");

    // Show advanced statistics
    let stats = optimizer.advanced_stats();
    println!("\nAdvanced Statistics:");
    println!("  Queries analyzed: {}", stats.queries_analyzed);
    println!(
        "  Predicate histogram size: {}",
        stats.predicate_histogram_size
    );
    println!("  Execution history size: {}", stats.history_size);

    Ok(())
}

fn demonstrate_join_selectivity_learning() -> Result<(), Box<dyn std::error::Error>> {
    let optimizer = CostBasedOptimizer::new();

    // Create two patterns for a join
    let foaf_name = NamedNode::new("http://xmlns.com/foaf/0.1/name").unwrap();
    let foaf_mbox = NamedNode::new("http://xmlns.com/foaf/0.1/mbox").unwrap();

    let pattern1 = GraphPattern::Bgp(vec![AlgebraTriplePattern {
        subject: TermPattern::Variable(Variable::new("person").unwrap()),
        predicate: TermPattern::NamedNode(foaf_name),
        object: TermPattern::Variable(Variable::new("name").unwrap()),
    }]);

    let pattern2 = GraphPattern::Bgp(vec![AlgebraTriplePattern {
        subject: TermPattern::Variable(Variable::new("person").unwrap()),
        predicate: TermPattern::NamedNode(foaf_mbox),
        object: TermPattern::Variable(Variable::new("email").unwrap()),
    }]);

    println!("Join Query:");
    println!("  ?person foaf:name ?name .");
    println!("  ?person foaf:mbox ?email .\n");

    // Initial join estimation (default selectivity = 0.1)
    let join_pattern = GraphPattern::Join(Box::new(pattern1.clone()), Box::new(pattern2.clone()));
    let initial_plan = optimizer.optimize_pattern(&join_pattern)?;
    println!("Initial Join Estimate:");
    println!(
        "  Estimated cardinality: {}",
        initial_plan.estimated_cardinality
    );

    // Record multiple join executions with realistic cardinalities
    println!("\nRecording 15 join executions...");
    let join_results = vec![
        (1000, 1000, 850),  // High selectivity
        (1200, 1100, 1000), // High selectivity
        (950, 980, 800),    // High selectivity
        (1500, 1400, 1200), // High selectivity
        (800, 850, 650),    // High selectivity
        (1100, 1050, 900),  // High selectivity
        (900, 920, 750),    // High selectivity
        (1300, 1250, 1050), // High selectivity
        (1000, 1020, 820),  // High selectivity
        (1150, 1100, 950),  // High selectivity
        (980, 1000, 800),   // High selectivity
        (1080, 1050, 880),  // High selectivity
        (920, 950, 750),    // High selectivity
        (1200, 1150, 980),  // High selectivity
        (1050, 1000, 850),  // High selectivity
    ];

    for (i, &(left_card, right_card, result_card)) in join_results.iter().enumerate() {
        optimizer.record_join_execution(&pattern1, &pattern2, left_card, right_card, result_card);

        if (i + 1) % 5 == 0 {
            print!("  After {} joins: ", i + 1);
            let plan = optimizer.optimize_pattern(&join_pattern)?;
            let actual_selectivity = result_card as f64 / (left_card as f64 * right_card as f64);
            println!(
                "estimated = {}, actual selectivity = {:.6}",
                plan.estimated_cardinality, actual_selectivity
            );
        }
    }

    // Final estimate after learning
    let learned_plan = optimizer.optimize_pattern(&join_pattern)?;
    println!("\nFinal Join Estimate:");
    println!(
        "  Estimated cardinality: {}",
        learned_plan.estimated_cardinality
    );

    // Calculate average actual selectivity
    let avg_selectivity: f64 = join_results
        .iter()
        .map(|&(l, r, res)| res as f64 / (l as f64 * r as f64))
        .sum::<f64>()
        / join_results.len() as f64;

    println!("  Average actual selectivity: {:.6}", avg_selectivity);
    println!("  Default selectivity (before learning): 0.100000");

    println!("\n💡 Join selectivity adapts based on similar historical joins");

    // Show join statistics
    let stats = optimizer.advanced_stats();
    println!("\nAdvanced Statistics:");
    println!("  Join samples recorded: {}", stats.join_samples);

    Ok(())
}

fn demonstrate_execution_history() -> Result<(), Box<dyn std::error::Error>> {
    let optimizer = CostBasedOptimizer::new();

    // Create a pattern
    let foaf_knows = NamedNode::new("http://xmlns.com/foaf/0.1/knows").unwrap();
    let pattern = AlgebraTriplePattern {
        subject: TermPattern::Variable(Variable::new("person").unwrap()),
        predicate: TermPattern::NamedNode(foaf_knows),
        object: TermPattern::Variable(Variable::new("friend").unwrap()),
    };

    println!("Query Pattern: ?person foaf:knows ?friend\n");

    // Execute the pattern 10 times with different cardinalities and times
    println!("Recording 10 executions with timing data...");
    let executions = vec![
        (500, 15), // 500 results in 15ms
        (520, 16), // 520 results in 16ms
        (480, 14), // 480 results in 14ms
        (510, 15), // 510 results in 15ms
        (490, 15), // 490 results in 15ms
        (530, 17), // 530 results in 17ms
        (475, 14), // 475 results in 14ms
        (505, 16), // 505 results in 16ms
        (495, 15), // 495 results in 15ms
        (515, 16), // 515 results in 16ms
    ];

    for &(cardinality, time_ms) in &executions {
        let graph_pattern = GraphPattern::Bgp(vec![pattern.clone()]);
        optimizer.update_stats_with_time(&graph_pattern, cardinality, time_ms);
    }

    println!("✓ Recorded {} executions\n", executions.len());

    // Retrieve execution history
    let history = optimizer.get_pattern_history(&pattern);
    println!("Execution History (showing all {} records):", history.len());

    for (i, execution) in history.iter().enumerate() {
        println!(
            "  {}: {} results in {}ms",
            i + 1,
            execution.cardinality,
            execution.execution_time_ms
        );
    }

    // Calculate statistics from history
    let total_results: usize = history.iter().map(|e| e.cardinality).sum();
    let total_time: u64 = history.iter().map(|e| e.execution_time_ms).sum();
    let avg_results = total_results as f64 / history.len() as f64;
    let avg_time = total_time as f64 / history.len() as f64;

    println!("\nHistory Statistics:");
    println!("  Average cardinality: {:.1} results", avg_results);
    println!("  Average execution time: {:.1}ms", avg_time);
    println!("  Results per millisecond: {:.1}", avg_results / avg_time);

    println!("\n💡 Execution history enables pattern-specific performance analysis");

    Ok(())
}

fn demonstrate_learning_cycle() -> Result<(), Box<dyn std::error::Error>> {
    let optimizer = CostBasedOptimizer::new();

    println!("Complete Learning Cycle: Training → Optimization → Execution → Feedback\n");

    // Create a realistic query with multiple patterns
    let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
    let foaf_person = NamedNode::new("http://xmlns.com/foaf/0.1/Person").unwrap();
    let foaf_name = NamedNode::new("http://xmlns.com/foaf/0.1/name").unwrap();
    let foaf_mbox = NamedNode::new("http://xmlns.com/foaf/0.1/mbox").unwrap();

    let pattern1 = AlgebraTriplePattern {
        subject: TermPattern::Variable(Variable::new("person").unwrap()),
        predicate: TermPattern::NamedNode(rdf_type),
        object: TermPattern::NamedNode(foaf_person),
    };

    let pattern2 = AlgebraTriplePattern {
        subject: TermPattern::Variable(Variable::new("person").unwrap()),
        predicate: TermPattern::NamedNode(foaf_name),
        object: TermPattern::Variable(Variable::new("name").unwrap()),
    };

    let pattern3 = AlgebraTriplePattern {
        subject: TermPattern::Variable(Variable::new("person").unwrap()),
        predicate: TermPattern::NamedNode(foaf_mbox),
        object: TermPattern::Variable(Variable::new("email").unwrap()),
    };

    let complex_pattern = GraphPattern::Bgp(vec![pattern1, pattern2, pattern3]);

    println!("Complex Query:");
    println!("  ?person rdf:type foaf:Person .");
    println!("  ?person foaf:name ?name .");
    println!("  ?person foaf:mbox ?email .\n");

    // Phase 1: Initial optimization (cold start)
    println!("Phase 1: Cold Start (no historical data)");
    let cold_plan = optimizer.optimize_pattern(&complex_pattern)?;
    println!(
        "  Estimated cardinality: {}",
        cold_plan.estimated_cardinality
    );
    println!("  Estimated cost: {:.2}", cold_plan.estimated_cost);

    // Phase 2: Training phase (10 executions)
    println!("\nPhase 2: Training (10 query executions)");
    let training_results = [
        (800, 25),
        (820, 26),
        (790, 24),
        (810, 25),
        (805, 25),
        (815, 26),
        (795, 24),
        (825, 27),
        (800, 25),
        (810, 25),
    ];

    for (i, &(cardinality, time_ms)) in training_results.iter().enumerate() {
        optimizer.update_stats_with_time(&complex_pattern, cardinality, time_ms);
        print!(".");
        if (i + 1) % 10 == 0 {
            println!(" {} executions recorded", i + 1);
        }
    }

    // Phase 3: Warm optimization (with learned statistics)
    println!("\nPhase 3: Warm Start (with learned statistics)");
    let warm_plan = optimizer.optimize_pattern(&complex_pattern)?;
    println!(
        "  Estimated cardinality: {}",
        warm_plan.estimated_cardinality
    );
    println!("  Estimated cost: {:.2}", warm_plan.estimated_cost);

    // Calculate actual averages
    let avg_cardinality: f64 = training_results.iter().map(|&(c, _)| c as f64).sum::<f64>()
        / training_results.len() as f64;
    let avg_time: f64 = training_results.iter().map(|&(_, t)| t as f64).sum::<f64>()
        / training_results.len() as f64;

    println!("\nActual Execution Statistics:");
    println!("  Average cardinality: {:.1}", avg_cardinality);
    println!("  Average time: {:.1}ms", avg_time);

    // Comparison
    println!("\nLearning Impact:");
    println!(
        "  Cold estimate error: {:.1}%",
        ((cold_plan.estimated_cardinality as f64 - avg_cardinality).abs() / avg_cardinality
            * 100.0)
    );
    println!(
        "  Warm estimate error: {:.1}%",
        ((warm_plan.estimated_cardinality as f64 - avg_cardinality).abs() / avg_cardinality
            * 100.0)
    );

    // Show comprehensive statistics
    let stats = optimizer.advanced_stats();
    println!("\nFinal Optimizer State:");
    println!("  Total queries analyzed: {}", stats.queries_analyzed);
    println!("  Subject histogram size: {}", stats.subject_histogram_size);
    println!(
        "  Predicate histogram size: {}",
        stats.predicate_histogram_size
    );
    println!("  Object histogram size: {}", stats.object_histogram_size);
    println!("  Join samples: {}", stats.join_samples);
    println!("  Execution history size: {}", stats.history_size);

    println!("\n💡 Optimizer learns from every query execution!");
    println!("   → Better estimates → Better plans → Faster queries");

    Ok(())
}
