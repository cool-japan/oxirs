//! Comprehensive Integration: Optimizer → Profiler → Visualizer
//!
//! This example demonstrates the complete query optimization, profiling, and
//! visualization pipeline in OxiRS. It shows how these systems work together
//! to provide deep insights into SPARQL query execution.
//!
//! # Features Demonstrated
//! - Cost-based query optimization with cardinality estimation
//! - Real-time query profiling with execution statistics
//! - Visual query plan generation with optimization hints
//! - Performance analysis and regression detection
//! - Complete optimization feedback loop

use oxirs_core::model::{NamedNode, Variable};
use oxirs_core::query::algebra::{AlgebraTriplePattern, GraphPattern, TermPattern};
use oxirs_core::query::cost_based_optimizer::{CostBasedOptimizer, CostConfiguration};
use oxirs_core::query::profiled_plan_builder::ProfiledPlanBuilder;
use oxirs_core::query::query_plan_visualizer::QueryPlanVisualizer;
use oxirs_core::query::query_profiler::{ProfilerConfig, QueryProfiler, QueryStatistics};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiRS Optimizer → Profiler → Visualizer Integration ===\n");

    // Part 1: Cost-Based Optimization
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  Part 1: Cost-Based Query Optimization              ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    demonstrate_optimization()?;

    // Part 2: Query Profiling
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║  Part 2: Query Profiling with Statistics            ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    let stats = demonstrate_profiling()?;

    // Part 3: Query Plan Visualization
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║  Part 3: Query Plan Visualization                   ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    demonstrate_visualization(&stats)?;

    // Part 4: Complete Integration
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║  Part 4: Complete Integration with Feedback Loop    ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    demonstrate_complete_integration()?;

    // Part 5: Performance Regression Detection
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║  Part 5: Performance Regression Detection           ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    demonstrate_regression_detection()?;

    println!("\n✅ Integration demonstration complete!");
    println!("\n💡 Key Takeaways:");
    println!("   - Cost-based optimization reduces execution time by reordering joins");
    println!("   - Profiling provides real execution statistics for validation");
    println!("   - Visualization makes complex query plans easy to understand");
    println!("   - Integration enables continuous query performance improvement");

    Ok(())
}

fn demonstrate_optimization() -> Result<(), Box<dyn std::error::Error>> {
    let optimizer = CostBasedOptimizer::with_config(CostConfiguration::default());

    // Create a complex query pattern with multiple joins
    let pattern = create_complex_query_pattern();

    println!("Original Query Pattern:");
    println!("  Find all people with their names and email addresses");
    println!("  Pattern: ?person rdf:type foaf:Person");
    println!("           ?person foaf:name ?name");
    println!("           ?person foaf:mbox ?email\n");

    // Optimize the pattern
    let optimized_plan = optimizer.optimize_pattern(&pattern)?;

    println!("Optimization Results:");
    println!(
        "  ✓ Estimated cardinality: {}",
        optimized_plan.estimated_cardinality
    );
    println!(
        "  ✓ Estimated cost:        {:.2}",
        optimized_plan.estimated_cost
    );
    println!("  ✓ Use index:             {}", optimized_plan.use_index);
    println!(
        "  ✓ Parallel execution:    {}",
        optimized_plan.parallel_execution
    );
    println!(
        "  ✓ Optimizations applied: {} strategies",
        optimized_plan.optimizations.len()
    );

    // Generate visual representation
    let visual_plan = optimizer.to_visual_plan(&pattern, &optimized_plan);
    let visualizer = QueryPlanVisualizer::new();

    println!("\nOptimized Execution Plan:");
    println!("{}", visualizer.visualize_as_tree(&visual_plan));

    Ok(())
}

fn demonstrate_profiling() -> Result<QueryStatistics, Box<dyn std::error::Error>> {
    let profiler_config = ProfilerConfig::default();
    let profiler = QueryProfiler::new(profiler_config);

    let query = "SELECT ?person ?name ?email WHERE { \
                 ?person rdf:type foaf:Person . \
                 ?person foaf:name ?name . \
                 ?person foaf:mbox ?email \
                 }";

    // Start profiling session
    let session = profiler.start_session(query);

    // Simulate query execution phases
    println!("Profiling Query Execution:");
    println!("  Phase 1: Parsing query... (10ms)");
    std::thread::sleep(std::time::Duration::from_millis(10));

    println!("  Phase 2: Planning optimization... (20ms)");
    std::thread::sleep(std::time::Duration::from_millis(20));

    println!("  Phase 3: Executing query... (120ms)");
    std::thread::sleep(std::time::Duration::from_millis(120));

    // Create realistic statistics
    let mut pattern_matches = HashMap::new();
    pattern_matches.insert("?person rdf:type foaf:Person".to_string(), 1000);
    pattern_matches.insert("?person foaf:name ?name".to_string(), 950);
    pattern_matches.insert("?person foaf:mbox ?email".to_string(), 800);

    let mut index_accesses = HashMap::new();
    index_accesses.insert("SPO".to_string(), 3);

    let stats = QueryStatistics {
        total_time_ms: 150,
        parse_time_ms: 10,
        planning_time_ms: 20,
        execution_time_ms: 120,
        triples_matched: 2750,
        results_count: 800,
        peak_memory_bytes: 2 * 1024 * 1024, // 2MB
        join_operations: 2,
        cache_hit_rate: 0.65,
        pattern_matches,
        index_accesses,
        ..Default::default()
    };

    // Finish session
    drop(session);

    println!("\n✓ Profiling Complete");
    println!("  Total time:     {}ms", stats.total_time_ms);
    println!("  Triples matched: {}", stats.triples_matched);
    println!("  Results:        {}", stats.results_count);
    println!("  Cache hit rate: {:.1}%", stats.cache_hit_rate * 100.0);
    println!("  Peak memory:    {}KB", stats.peak_memory_bytes / 1024);

    Ok(stats)
}

fn demonstrate_visualization(stats: &QueryStatistics) -> Result<(), Box<dyn std::error::Error>> {
    let builder = ProfiledPlanBuilder::new();
    let query = "SELECT ?person ?name ?email WHERE { ... }";

    // Generate profiling report with visualization
    let report = builder.generate_report(stats, query);

    println!("Query Plan Visualization:");
    println!("{}", report.tree_visualization);

    println!("\n{}", report.summary);

    if !report.optimization_hints.is_empty() {
        println!("\nOptimization Recommendations:");
        for (i, hint) in report.optimization_hints.iter().enumerate() {
            println!("  {}. {:?}: {}", i + 1, hint.severity, hint.message);
            println!("     → {}", hint.suggestion);
        }
    }

    // Performance analysis
    let analysis = builder.analyze_performance(stats);
    println!("\nPerformance Analysis:");
    println!("  Overall Grade: {:?}", analysis.overall_grade);
    println!("  Cache Effectiveness: {:?}", analysis.cache_effectiveness);

    if !analysis.slow_phases.is_empty() {
        println!("  ⚠️  Slow Phases:");
        for phase in &analysis.slow_phases {
            println!("      - {}", phase);
        }
    }

    Ok(())
}

fn demonstrate_complete_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("Complete Integration Workflow:");
    println!("  1️⃣  Parse SPARQL query");
    println!("  2️⃣  Optimize using cost-based optimizer");
    println!("  3️⃣  Profile execution with real statistics");
    println!("  4️⃣  Generate visual plan with actual vs estimated");
    println!("  5️⃣  Analyze performance and generate hints");
    println!("  6️⃣  Feed actual statistics back to optimizer\n");

    // Create optimizer with learning capability
    let optimizer = CostBasedOptimizer::new();
    let pattern = create_complex_query_pattern();

    // Step 1-2: Optimize
    let initial_plan = optimizer.optimize_pattern(&pattern)?;
    println!("Initial Optimization:");
    println!(
        "  Estimated cardinality: {}",
        initial_plan.estimated_cardinality
    );

    // Step 3: Execute and profile (simulated)
    let actual_cardinality = 750; // Actual result from execution

    // Step 6: Feedback loop - update optimizer statistics
    optimizer.update_stats(&pattern, actual_cardinality);

    // Re-optimize with learned statistics
    let learned_plan = optimizer.optimize_pattern(&pattern)?;
    println!("\nRe-Optimization with Learned Statistics:");
    println!(
        "  Updated cardinality estimate: {}",
        learned_plan.estimated_cardinality
    );

    if let Some(learned_card) = optimizer.get_learned_cardinality(&pattern) {
        println!("  Learned average cardinality: {:.1}", learned_card);
        println!("  ✓ Optimizer is now adapting based on real execution data!");
    }

    Ok(())
}

fn demonstrate_regression_detection() -> Result<(), Box<dyn std::error::Error>> {
    let builder = ProfiledPlanBuilder::new();

    // Baseline execution (optimized)
    let baseline = QueryStatistics {
        total_time_ms: 150,
        execution_time_ms: 120,
        peak_memory_bytes: 2 * 1024 * 1024,
        results_count: 800,
        cache_hit_rate: 0.65,
        ..Default::default()
    };

    // Current execution (regression)
    let current = QueryStatistics {
        total_time_ms: 350, // 133% slower!
        execution_time_ms: 320,
        peak_memory_bytes: 5 * 1024 * 1024, // 2.5x more memory
        results_count: 800,
        cache_hit_rate: 0.25, // Worse cache hit rate
        ..Default::default()
    };

    println!("Performance Regression Detection:");
    let comparison = builder.compare_executions(&baseline, &current);
    println!("{}", comparison);

    match comparison.improvement {
        oxirs_core::query::profiled_plan_builder::ImprovementLevel::Critical => {
            println!("🔴 CRITICAL REGRESSION DETECTED!");
            println!("   Action required: investigate query plan changes");
        }
        oxirs_core::query::profiled_plan_builder::ImprovementLevel::Regression => {
            println!("⚠️  Performance regression detected");
        }
        oxirs_core::query::profiled_plan_builder::ImprovementLevel::Significant => {
            println!("✅ Significant performance improvement!");
        }
        _ => {
            println!("ℹ️  No significant performance change");
        }
    }

    // Analyze both versions
    println!("\nBaseline Analysis:");
    let baseline_analysis = builder.analyze_performance(&baseline);
    println!("  Grade: {:?}", baseline_analysis.overall_grade);

    println!("\nCurrent Analysis:");
    let current_analysis = builder.analyze_performance(&current);
    println!("  Grade: {:?}", current_analysis.overall_grade);

    if current_analysis.overall_grade < baseline_analysis.overall_grade {
        println!("\n⚠️  Performance degradation confirmed by grade analysis");
    }

    Ok(())
}

// Helper functions

fn create_complex_query_pattern() -> GraphPattern {
    // Create triple patterns
    let person_var = Variable::new("person").unwrap();
    let name_var = Variable::new("name").unwrap();
    let email_var = Variable::new("email").unwrap();

    let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
    let foaf_person = NamedNode::new("http://xmlns.com/foaf/0.1/Person").unwrap();
    let foaf_name = NamedNode::new("http://xmlns.com/foaf/0.1/name").unwrap();
    let foaf_mbox = NamedNode::new("http://xmlns.com/foaf/0.1/mbox").unwrap();

    let pattern1 = AlgebraTriplePattern {
        subject: TermPattern::Variable(person_var.clone()),
        predicate: TermPattern::NamedNode(rdf_type),
        object: TermPattern::NamedNode(foaf_person),
    };

    let pattern2 = AlgebraTriplePattern {
        subject: TermPattern::Variable(person_var.clone()),
        predicate: TermPattern::NamedNode(foaf_name),
        object: TermPattern::Variable(name_var),
    };

    let pattern3 = AlgebraTriplePattern {
        subject: TermPattern::Variable(person_var),
        predicate: TermPattern::NamedNode(foaf_mbox),
        object: TermPattern::Variable(email_var),
    };

    // Create BGP with multiple patterns
    GraphPattern::Bgp(vec![pattern1, pattern2, pattern3])
}

// PerformanceGrade now derives PartialOrd and Ord in the source module
