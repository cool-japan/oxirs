//! Query Plan Visualizer Demonstration
//!
//! This example demonstrates the query plan visualization and debugging features
//! for SPARQL queries. It shows how to create query plans, visualize them as
//! ASCII trees, generate summaries, and get optimization hints.

use oxirs_core::query::query_plan_visualizer::{HintSeverity, QueryPlanNode, QueryPlanVisualizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiRS Query Plan Visualizer Demo ===\n");

    // Example 1: Simple triple pattern query
    println!("Example 1: Simple Triple Pattern Query");
    println!("---------------------------------------");
    let simple_plan = create_simple_query_plan();
    visualize_plan(&simple_plan, "Simple Query");
    println!();

    // Example 2: Complex join query with optimization opportunities
    println!("\nExample 2: Complex Join Query with Optimization Opportunities");
    println!("--------------------------------------------------------------");
    let complex_plan = create_complex_query_plan();
    visualize_plan(&complex_plan, "Complex Join Query");
    println!();

    // Example 3: Query with cardinality misestimation
    println!("\nExample 3: Query with Cardinality Misestimation");
    println!("-----------------------------------------------");
    let misestimated_plan = create_misestimated_query_plan();
    visualize_with_optimizations(&misestimated_plan, "Misestimated Query");
    println!();

    // Example 4: Slow query with performance issues
    println!("\nExample 4: Slow Query Analysis");
    println!("-------------------------------");
    let slow_plan = create_slow_query_plan();
    analyze_performance(&slow_plan, "Slow Query");
    println!();

    // Example 5: JSON export for external tools
    println!("\nExample 5: JSON Export");
    println!("----------------------");
    export_to_json(&complex_plan)?;
    println!();

    // Example 6: Multiple visualization options
    println!("\nExample 6: Customized Visualization");
    println!("------------------------------------");
    demonstrate_visualization_options(&complex_plan);
    println!();

    Ok(())
}

fn create_simple_query_plan() -> QueryPlanNode {
    QueryPlanNode::new("TriplePattern", "?person rdf:type foaf:Person")
        .with_estimated_cardinality(1000)
        .with_actual_cardinality(1050)
        .with_estimated_cost(10.0)
        .with_execution_time(500)
        .with_index("SPO")
        .with_metadata("selectivity", "0.1")
}

fn create_complex_query_plan() -> QueryPlanNode {
    let mut root = QueryPlanNode::new("HashJoin", "on ?person")
        .with_estimated_cardinality(100)
        .with_actual_cardinality(95)
        .with_estimated_cost(250.0)
        .with_execution_time(8000);

    // First child: Person type pattern
    let child1 = QueryPlanNode::new("TriplePattern", "?person rdf:type foaf:Person")
        .with_estimated_cardinality(1000)
        .with_actual_cardinality(1000)
        .with_estimated_cost(10.0)
        .with_execution_time(1500)
        .with_index("SPO")
        .with_metadata("pattern_type", "simple");

    // Second child: Nested join for person details
    let mut child2 = QueryPlanNode::new("HashJoin", "on ?person (nested)")
        .with_estimated_cardinality(100)
        .with_actual_cardinality(95)
        .with_estimated_cost(150.0)
        .with_execution_time(5000);

    let grandchild1 = QueryPlanNode::new("TriplePattern", "?person foaf:name ?name")
        .with_estimated_cardinality(950)
        .with_actual_cardinality(95)
        .with_estimated_cost(10.0)
        .with_execution_time(800)
        .with_index("SPO");

    let grandchild2 = QueryPlanNode::new("TriplePattern", "?person foaf:mbox ?email")
        .with_estimated_cardinality(100)
        .with_actual_cardinality(95)
        .with_estimated_cost(10.0)
        .with_execution_time(600)
        .with_index("SPO");

    child2.add_child(grandchild1);
    child2.add_child(grandchild2);

    // Filter operation
    let mut child3 = QueryPlanNode::new("Filter", "FILTER(contains(?name, \"John\"))")
        .with_estimated_cardinality(10)
        .with_actual_cardinality(8)
        .with_estimated_cost(50.0)
        .with_execution_time(1000);

    let filter_input = child2.clone();
    child3.add_child(filter_input);

    root.add_child(child1);
    root.add_child(child3);
    root
}

fn create_misestimated_query_plan() -> QueryPlanNode {
    let mut root = QueryPlanNode::new("HashJoin", "on ?article")
        .with_estimated_cardinality(100)
        .with_actual_cardinality(50000) // Massive underestimate!
        .with_estimated_cost(200.0)
        .with_execution_time(150000); // 150ms - slow due to underestimation

    let child1 = QueryPlanNode::new("TriplePattern", "?article rdf:type schema:Article")
        .with_estimated_cardinality(1000)
        .with_actual_cardinality(100000) // 100x off!
        .with_estimated_cost(10.0)
        .with_execution_time(50000)
        .with_index("SPO");

    let child2 = QueryPlanNode::new("TriplePattern", "?article schema:author ?author")
        .with_estimated_cardinality(500)
        .with_actual_cardinality(75000) // 150x off!
        .with_estimated_cost(10.0)
        .with_execution_time(80000);
    // Note: No index used here - potential optimization opportunity

    root.add_child(child1);
    root.add_child(child2);
    root
}

fn create_slow_query_plan() -> QueryPlanNode {
    let mut root = QueryPlanNode::new("Union", "union of two branches")
        .with_estimated_cardinality(1000)
        .with_actual_cardinality(950)
        .with_estimated_cost(500.0)
        .with_execution_time(250000); // 250ms - very slow!

    let mut left_branch = QueryPlanNode::new("HashJoin", "on ?paper")
        .with_estimated_cardinality(500)
        .with_actual_cardinality(475)
        .with_estimated_cost(200.0)
        .with_execution_time(120000); // 120ms

    let left_child1 = QueryPlanNode::new("TriplePattern", "?paper rdf:type bib:Paper")
        .with_estimated_cardinality(5000)
        .with_actual_cardinality(5000)
        .with_estimated_cost(15.0)
        .with_execution_time(3000)
        .with_index("SPO");

    let left_child2 = QueryPlanNode::new("TriplePattern", "?paper bib:year ?year")
        .with_estimated_cardinality(4000)
        .with_actual_cardinality(4500)
        .with_estimated_cost(15.0)
        .with_execution_time(110000); // Very slow - no index!
                                      // Missing index on large dataset

    left_branch.add_child(left_child1);
    left_branch.add_child(left_child2);

    let mut right_branch = QueryPlanNode::new("HashJoin", "on ?book")
        .with_estimated_cardinality(500)
        .with_actual_cardinality(475)
        .with_estimated_cost(200.0)
        .with_execution_time(125000); // 125ms

    let right_child1 = QueryPlanNode::new("TriplePattern", "?book rdf:type bib:Book")
        .with_estimated_cardinality(3000)
        .with_actual_cardinality(3000)
        .with_estimated_cost(15.0)
        .with_execution_time(2000)
        .with_index("SPO");

    let right_child2 = QueryPlanNode::new("TriplePattern", "?book bib:isbn ?isbn")
        .with_estimated_cardinality(2500)
        .with_actual_cardinality(2500)
        .with_estimated_cost(15.0)
        .with_execution_time(120000); // Very slow - no index!

    right_branch.add_child(right_child1);
    right_branch.add_child(right_child2);

    root.add_child(left_branch);
    root.add_child(right_branch);
    root
}

fn visualize_plan(plan: &QueryPlanNode, title: &str) {
    println!("Query: {}", title);
    println!();

    let visualizer = QueryPlanVisualizer::new();
    let tree = visualizer.visualize_as_tree(plan);
    println!("{}", tree);

    let summary = visualizer.generate_summary(plan);
    println!("{}", summary);
}

fn visualize_with_optimizations(plan: &QueryPlanNode, title: &str) {
    println!("Query: {}", title);
    println!();

    let visualizer = QueryPlanVisualizer::new();
    let tree = visualizer.visualize_as_tree(plan);
    println!("{}", tree);

    let summary = visualizer.generate_summary(plan);
    println!("{}", summary);

    // Get optimization hints
    let hints = visualizer.suggest_optimizations(plan);
    if !hints.is_empty() {
        println!("\n🔍 Optimization Hints:");
        println!("---------------------");
        for hint in hints {
            let icon = match hint.severity {
                HintSeverity::Info => "ℹ️",
                HintSeverity::Warning => "⚠️",
                HintSeverity::Critical => "🔴",
            };
            println!("{} {}", icon, hint);
        }
    }
}

fn analyze_performance(plan: &QueryPlanNode, title: &str) {
    println!("Query: {}", title);
    println!();

    let visualizer = QueryPlanVisualizer::new();

    // Show full tree
    let tree = visualizer.visualize_as_tree(plan);
    println!("{}", tree);

    // Performance summary
    let summary = visualizer.generate_summary(plan);
    println!("{}", summary);

    // Detailed optimization analysis
    let hints = visualizer.suggest_optimizations(plan);
    println!("\n🔍 Performance Analysis:");
    println!("-----------------------");
    println!("Total optimization hints: {}", hints.len());

    let critical_hints: Vec<_> = hints
        .iter()
        .filter(|h| matches!(h.severity, HintSeverity::Critical))
        .collect();
    let warning_hints: Vec<_> = hints
        .iter()
        .filter(|h| matches!(h.severity, HintSeverity::Warning))
        .collect();
    let info_hints: Vec<_> = hints
        .iter()
        .filter(|h| matches!(h.severity, HintSeverity::Info))
        .collect();

    println!("  Critical issues: {}", critical_hints.len());
    println!("  Warnings:        {}", warning_hints.len());
    println!("  Info:            {}", info_hints.len());

    if !critical_hints.is_empty() {
        println!("\n🔴 Critical Issues:");
        for hint in critical_hints {
            println!("  - {}", hint.message);
            println!("    → {}", hint.suggestion);
        }
    }

    if !warning_hints.is_empty() {
        println!("\n⚠️  Warnings:");
        for hint in warning_hints {
            println!("  - {}", hint.message);
            println!("    → {}", hint.suggestion);
        }
    }

    if !info_hints.is_empty() {
        println!("\nℹ️  Suggestions:");
        for hint in info_hints {
            println!("  - {}", hint.message);
            println!("    → {}", hint.suggestion);
        }
    }
}

fn export_to_json(plan: &QueryPlanNode) -> Result<(), Box<dyn std::error::Error>> {
    let visualizer = QueryPlanVisualizer::new();
    let json = visualizer.export_as_json(plan)?;

    println!("JSON Export Preview:");
    // Show first 500 characters
    let preview = if json.len() > 500 {
        format!(
            "{}...\n(truncated, total {} bytes)",
            &json[..500],
            json.len()
        )
    } else {
        json.clone()
    };
    println!("{}", preview);

    println!("\nFull JSON can be written to a file for external analysis tools.");
    println!("Example: Grafana, Kibana, or custom dashboards");

    Ok(())
}

fn demonstrate_visualization_options(plan: &QueryPlanNode) {
    // Show different visualization configurations

    println!("1. Full detail (default):");
    let full_visualizer = QueryPlanVisualizer::new();
    println!("{}", full_visualizer.visualize_as_tree(plan));

    println!("\n2. Without costs:");
    let no_costs = QueryPlanVisualizer::new().with_costs(false);
    println!("{}", no_costs.visualize_as_tree(plan));

    println!("\n3. Without indexes:");
    let no_indexes = QueryPlanVisualizer::new().with_indexes(false);
    println!("{}", no_indexes.visualize_as_tree(plan));

    println!("\n4. Cardinality only:");
    let card_only = QueryPlanVisualizer::new()
        .with_costs(false)
        .with_indexes(false)
        .with_stats(false);
    println!("{}", card_only.visualize_as_tree(plan));
}
