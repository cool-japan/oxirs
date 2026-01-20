//! Query EXPLAIN/ANALYZE command
//!
//! Provides query optimization insights, execution plans, and performance analysis

use anyhow::{Context, Result};
use oxirs_arq::query::parse_query;
use std::fmt::Write as FmtWrite;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Query analysis mode
#[derive(Debug, Clone, Copy)]
pub enum AnalysisMode {
    /// Show query plan only (no execution)
    Explain,
    /// Execute query and show performance metrics
    Analyze,
    /// Show both plan and execution metrics
    Full,
}

/// Query execution statistics
#[derive(Debug)]
pub struct QueryStats {
    pub parse_time_ms: f64,
    pub optimization_time_ms: f64,
    pub execution_time_ms: f64,
    pub total_time_ms: f64,
    pub result_count: usize,
    pub intermediate_results: usize,
    pub memory_used_mb: f64,
}

/// Explain query execution plan
pub async fn explain_query(
    dataset: String,
    query: String,
    is_file: bool,
    mode: AnalysisMode,
) -> Result<()> {
    explain_query_with_options(dataset, query, is_file, mode, None).await
}

/// Explain query execution plan with optional graphical output
pub async fn explain_query_with_options(
    dataset: String,
    query: String,
    is_file: bool,
    mode: AnalysisMode,
    graphviz_output: Option<PathBuf>,
) -> Result<()> {
    let total_start = Instant::now();

    // Read query from file if needed
    let query_str = if is_file {
        std::fs::read_to_string(&query)
            .with_context(|| format!("Failed to read query file: {}", query))?
    } else {
        query
    };

    println!("🔍 Query Analysis Mode: {:?}\n", mode);
    println!("📊 Query:\n{}\n", query_str);
    println!("{}\n", "=".repeat(80));

    // Parse query
    let parse_start = Instant::now();
    let parsed_query = parse_query(&query_str).with_context(|| "Failed to parse SPARQL query")?;
    let parse_time = parse_start.elapsed();

    println!("✅ Query Type: {:?}", parsed_query.query_type);
    println!(
        "⏱️  Parse Time: {:.3}ms\n",
        parse_time.as_secs_f64() * 1000.0
    );

    // Show query structure
    println!("📋 Query Structure:");
    println!("  Variables: {:?}", parsed_query.select_variables);
    println!("  Distinct: {}", parsed_query.distinct);
    println!("  Order By: {} clauses", parsed_query.order_by.len());
    println!("  Group By: {} clauses", parsed_query.group_by.len());
    if let Some(limit) = parsed_query.limit {
        println!("  Limit: {}", limit);
    }
    if let Some(offset) = parsed_query.offset {
        println!("  Offset: {}", offset);
    }
    println!();

    // Analyze algebra
    println!("🧮 Query Algebra:");
    println!("{:#?}\n", parsed_query.where_clause);

    // Estimate query complexity
    let complexity = estimate_complexity(&parsed_query.where_clause);
    println!("📈 Estimated Complexity:");
    println!("  Complexity Score: {}", complexity.score);
    println!("  Join Operations: {}", complexity.joins);
    println!("  Filters: {}", complexity.filters);
    println!("  Optional Patterns: {}", complexity.optionals);
    println!("  Unions: {}", complexity.unions);
    println!();

    // Show optimization opportunities
    show_optimization_hints(&parsed_query, &complexity);

    // Generate graphical query plan if requested
    if let Some(ref output_path) = graphviz_output {
        println!("{}\n", "=".repeat(80));
        println!("📊 Generating Graphical Query Plan\n");

        let dot_content = generate_query_plan_dot(&parsed_query, &complexity)?;
        std::fs::write(output_path, &dot_content)?;

        println!(
            "✅ Query plan visualization saved to: {}",
            output_path.display()
        );
        println!(
            "   Render with: dot -Tpng {} -o {}.png",
            output_path.display(),
            output_path.with_extension("").display()
        );
        println!("   Or view interactively: xdot {}\n", output_path.display());
    }

    // Execute query if requested
    if matches!(mode, AnalysisMode::Analyze | AnalysisMode::Full) {
        println!("{}\n", "=".repeat(80));
        println!("⚡ Query Execution Analysis\n");

        let _dataset_path = Path::new(&dataset);

        // Note: Full query execution with timing will be implemented in Beta.1
        // For now, we provide the query plan and optimization analysis
        println!("💡 Query execution with detailed timing will be available in Beta.1");
        println!("   For now, use 'oxirs query' to execute queries");
        println!("   This command focuses on query optimization analysis");
        println!();

        let parse_and_analysis_time = total_start.elapsed();
        println!(
            "⏱️  Analysis Time: {:.3}ms",
            parse_and_analysis_time.as_secs_f64() * 1000.0
        );
        println!();
    }

    Ok(())
}

/// Complexity metrics
#[derive(Debug, Default)]
struct ComplexityMetrics {
    score: usize,
    joins: usize,
    filters: usize,
    optionals: usize,
    unions: usize,
}

/// Estimate query complexity
fn estimate_complexity(algebra: &oxirs_arq::Algebra) -> ComplexityMetrics {
    use oxirs_arq::Algebra;

    let mut metrics = ComplexityMetrics::default();

    match algebra {
        Algebra::Bgp(patterns) => {
            metrics.score = patterns.len();
        }
        Algebra::Join { left, right } => {
            metrics.joins += 1;
            metrics.score += 2;
            let left_metrics = estimate_complexity(left);
            let right_metrics = estimate_complexity(right);
            metrics.score += left_metrics.score + right_metrics.score;
            metrics.joins += left_metrics.joins + right_metrics.joins;
            metrics.filters += left_metrics.filters + right_metrics.filters;
            metrics.optionals += left_metrics.optionals + right_metrics.optionals;
            metrics.unions += left_metrics.unions + right_metrics.unions;
        }
        Algebra::LeftJoin { left, right, .. } => {
            metrics.optionals += 1;
            metrics.score += 3;
            let left_metrics = estimate_complexity(left);
            let right_metrics = estimate_complexity(right);
            metrics.score += left_metrics.score + right_metrics.score;
            metrics.joins += left_metrics.joins + right_metrics.joins;
            metrics.filters += left_metrics.filters + right_metrics.filters;
            metrics.optionals += left_metrics.optionals + right_metrics.optionals;
            metrics.unions += left_metrics.unions + right_metrics.unions;
        }
        Algebra::Union { left, right } => {
            metrics.unions += 1;
            metrics.score += 2;
            let left_metrics = estimate_complexity(left);
            let right_metrics = estimate_complexity(right);
            metrics.score += left_metrics.score + right_metrics.score;
            metrics.joins += left_metrics.joins + right_metrics.joins;
            metrics.filters += left_metrics.filters + right_metrics.filters;
            metrics.optionals += left_metrics.optionals + right_metrics.optionals;
            metrics.unions += left_metrics.unions + right_metrics.unions;
        }
        Algebra::Filter { pattern, .. } => {
            metrics.filters += 1;
            metrics.score += 1;
            let inner = estimate_complexity(pattern);
            metrics.score += inner.score;
            metrics.joins += inner.joins;
            metrics.filters += inner.filters;
            metrics.optionals += inner.optionals;
            metrics.unions += inner.unions;
        }
        _ => {
            metrics.score = 1;
        }
    }

    metrics
}

/// Show optimization hints
fn show_optimization_hints(query: &oxirs_arq::query::Query, complexity: &ComplexityMetrics) {
    println!("💡 Optimization Hints:");

    let mut hints = Vec::new();

    // Check for missing LIMIT
    if query.limit.is_none() && complexity.score > 5 {
        hints.push("Consider adding LIMIT clause to reduce result size");
    }

    // Check for multiple filters
    if complexity.filters > 3 {
        hints.push("Multiple FILTER clauses detected - consider combining with && operator");
    }

    // Check for OPTIONAL without filter
    if complexity.optionals > 0 {
        hints.push("OPTIONAL patterns can be expensive - ensure they are necessary");
    }

    // Check for UNION
    if complexity.unions > 2 {
        hints.push("Multiple UNION operations - consider rewriting as a single BGP if possible");
    }

    // Check for complex joins
    if complexity.joins > 4 {
        hints.push("Many JOIN operations - ensure join order is optimal");
        hints.push("Consider adding more specific triple patterns to reduce intermediate results");
    }

    // Check for DISTINCT
    if query.distinct && complexity.score > 10 {
        hints.push("DISTINCT on complex query - this may be expensive");
    }

    // Check for GROUP BY without aggregate
    if !query.group_by.is_empty() {
        hints.push("GROUP BY detected - ensure aggregates are used efficiently");
    }

    if hints.is_empty() {
        println!("  ✅ Query appears well-optimized");
    } else {
        for (i, hint) in hints.iter().enumerate() {
            println!("  {}. {}", i + 1, hint);
        }
    }
    println!();
}

/// Generate Graphviz DOT format for query plan visualization
fn generate_query_plan_dot(
    query: &oxirs_arq::query::Query,
    complexity: &ComplexityMetrics,
) -> Result<String> {
    let mut dot = String::new();
    writeln!(dot, "digraph QueryPlan {{")?;
    writeln!(dot, "  rankdir=BT;")?;
    writeln!(
        dot,
        "  node [shape=box, style=\"rounded,filled\", fontname=\"Arial\"];"
    )?;
    writeln!(dot, "  edge [fontname=\"Arial\", fontsize=10];")?;
    writeln!(dot)?;

    // Add title with complexity metrics
    writeln!(dot, "  labelloc=\"t\";")?;
    writeln!(
        dot,
        "  label=<Query Plan<BR/><FONT POINT-SIZE=\"10\">Complexity: {} | Joins: {} | Filters: {} | Optionals: {}</FONT>>;",
        complexity.score, complexity.joins, complexity.filters, complexity.optionals
    )?;
    writeln!(dot)?;

    // Node counter for unique IDs
    let mut node_id = 0;

    // Generate root node
    let root_id = node_id;
    node_id += 1;

    let query_type_str = format!("{:?}", query.query_type);
    writeln!(
        dot,
        "  n{} [label=\"{}\", fillcolor=\"#90EE90\"];",
        root_id, query_type_str
    )?;

    // Add modifiers
    if query.distinct {
        let distinct_id = node_id;
        node_id += 1;
        writeln!(
            dot,
            "  n{} [label=\"DISTINCT\", fillcolor=\"#FFD700\"];",
            distinct_id
        )?;
        writeln!(dot, "  n{} -> n{};", distinct_id, root_id)?;
    }

    if query.limit.is_some() || query.offset.is_some() {
        let limit_id = node_id;
        node_id += 1;
        let label = match (query.limit, query.offset) {
            (Some(l), Some(o)) => format!("LIMIT {} OFFSET {}", l, o),
            (Some(l), None) => format!("LIMIT {}", l),
            (None, Some(o)) => format!("OFFSET {}", o),
            _ => String::new(),
        };
        writeln!(
            dot,
            "  n{} [label=\"{}\", fillcolor=\"#FFD700\"];",
            limit_id, label
        )?;
        writeln!(dot, "  n{} -> n{};", limit_id, root_id)?;
    }

    // Add ORDER BY if present
    if !query.order_by.is_empty() {
        let order_id = node_id;
        node_id += 1;
        writeln!(
            dot,
            "  n{} [label=\"ORDER BY ({})\", fillcolor=\"#FFD700\"];",
            order_id,
            query.order_by.len()
        )?;
        writeln!(dot, "  n{} -> n{};", order_id, root_id)?;
    }

    // Add GROUP BY if present
    if !query.group_by.is_empty() {
        let group_id = node_id;
        node_id += 1;
        writeln!(
            dot,
            "  n{} [label=\"GROUP BY ({})\", fillcolor=\"#FFD700\"];",
            group_id,
            query.group_by.len()
        )?;
        writeln!(dot, "  n{} -> n{};", group_id, root_id)?;
    }

    // Generate algebra tree
    let algebra_root = generate_algebra_nodes(&query.where_clause, &mut node_id, &mut dot)?;

    // Connect algebra to query root
    writeln!(dot, "  n{} -> n{};", algebra_root, root_id)?;

    writeln!(dot)?;
    writeln!(dot, "}}")?;

    Ok(dot)
}

/// Generate nodes for algebra tree (recursive)
fn generate_algebra_nodes(
    algebra: &oxirs_arq::Algebra,
    node_id: &mut usize,
    dot: &mut String,
) -> Result<usize> {
    use oxirs_arq::Algebra;

    let current_id = *node_id;
    *node_id += 1;

    match algebra {
        Algebra::Bgp(patterns) => {
            writeln!(
                dot,
                "  n{} [label=\"BGP\\n({} patterns)\", fillcolor=\"#87CEEB\"];",
                current_id,
                patterns.len()
            )?;

            // Add pattern details
            for (i, pattern) in patterns.iter().enumerate().take(3) {
                let pattern_id = *node_id;
                *node_id += 1;
                let label = format!("Pattern {}: {:?}", i + 1, pattern);
                let truncated = if label.len() > 50 {
                    format!("{}...", &label[..50])
                } else {
                    label
                };
                writeln!(
                    dot,
                    "  n{} [label=\"{}\", fillcolor=\"#E0FFFF\", shape=ellipse];",
                    pattern_id,
                    escape_dot_label(&truncated)
                )?;
                writeln!(dot, "  n{} -> n{};", current_id, pattern_id)?;
            }

            if patterns.len() > 3 {
                let more_id = *node_id;
                *node_id += 1;
                writeln!(
                    dot,
                    "  n{} [label=\"... {} more\", fillcolor=\"#E0FFFF\", shape=ellipse];",
                    more_id,
                    patterns.len() - 3
                )?;
                writeln!(dot, "  n{} -> n{};", current_id, more_id)?;
            }
        }
        Algebra::Join { left, right } => {
            writeln!(
                dot,
                "  n{} [label=\"JOIN\", fillcolor=\"#FFA07A\"];",
                current_id
            )?;

            let left_id = generate_algebra_nodes(left, node_id, dot)?;
            let right_id = generate_algebra_nodes(right, node_id, dot)?;

            writeln!(dot, "  n{} -> n{} [label=\"left\"];", current_id, left_id)?;
            writeln!(dot, "  n{} -> n{} [label=\"right\"];", current_id, right_id)?;
        }
        Algebra::LeftJoin { left, right, .. } => {
            writeln!(
                dot,
                "  n{} [label=\"LEFT JOIN\\n(OPTIONAL)\", fillcolor=\"#FFB6C1\"];",
                current_id
            )?;

            let left_id = generate_algebra_nodes(left, node_id, dot)?;
            let right_id = generate_algebra_nodes(right, node_id, dot)?;

            writeln!(dot, "  n{} -> n{} [label=\"left\"];", current_id, left_id)?;
            writeln!(
                dot,
                "  n{} -> n{} [label=\"optional\"];",
                current_id, right_id
            )?;
        }
        Algebra::Union { left, right } => {
            writeln!(
                dot,
                "  n{} [label=\"UNION\", fillcolor=\"#DDA0DD\"];",
                current_id
            )?;

            let left_id = generate_algebra_nodes(left, node_id, dot)?;
            let right_id = generate_algebra_nodes(right, node_id, dot)?;

            writeln!(dot, "  n{} -> n{} [label=\"left\"];", current_id, left_id)?;
            writeln!(dot, "  n{} -> n{} [label=\"right\"];", current_id, right_id)?;
        }
        Algebra::Filter { pattern, .. } => {
            writeln!(
                dot,
                "  n{} [label=\"FILTER\", fillcolor=\"#F0E68C\"];",
                current_id
            )?;

            let inner_id = generate_algebra_nodes(pattern, node_id, dot)?;
            writeln!(dot, "  n{} -> n{};", current_id, inner_id)?;
        }
        Algebra::Graph { graph, pattern } => {
            let graph_str = format!("{:?}", graph);
            let truncated = if graph_str.len() > 30 {
                format!("{}...", &graph_str[..30])
            } else {
                graph_str
            };

            writeln!(
                dot,
                "  n{} [label=\"GRAPH\\n{}\", fillcolor=\"#98FB98\"];",
                current_id,
                escape_dot_label(&truncated)
            )?;

            let inner_id = generate_algebra_nodes(pattern, node_id, dot)?;
            writeln!(dot, "  n{} -> n{};", current_id, inner_id)?;
        }
        Algebra::Extend { pattern, .. } => {
            writeln!(
                dot,
                "  n{} [label=\"EXTEND\", fillcolor=\"#FFDAB9\"];",
                current_id
            )?;

            let inner_id = generate_algebra_nodes(pattern, node_id, dot)?;
            writeln!(dot, "  n{} -> n{};", current_id, inner_id)?;
        }
        Algebra::Minus { left, right } => {
            writeln!(
                dot,
                "  n{} [label=\"MINUS\", fillcolor=\"#F08080\"];",
                current_id
            )?;

            let left_id = generate_algebra_nodes(left, node_id, dot)?;
            let right_id = generate_algebra_nodes(right, node_id, dot)?;

            writeln!(dot, "  n{} -> n{} [label=\"left\"];", current_id, left_id)?;
            writeln!(dot, "  n{} -> n{} [label=\"minus\"];", current_id, right_id)?;
        }
        Algebra::Service {
            endpoint, pattern, ..
        } => {
            let endpoint_str = format!("{:?}", endpoint);
            let truncated = if endpoint_str.len() > 30 {
                format!("{}...", &endpoint_str[..30])
            } else {
                endpoint_str
            };

            writeln!(
                dot,
                "  n{} [label=\"SERVICE\\n{}\", fillcolor=\"#AFEEEE\"];",
                current_id,
                escape_dot_label(&truncated)
            )?;

            let inner_id = generate_algebra_nodes(pattern, node_id, dot)?;
            writeln!(dot, "  n{} -> n{};", current_id, inner_id)?;
        }
        _ => {
            writeln!(
                dot,
                "  n{} [label=\"{:?}\", fillcolor=\"#D3D3D3\"];",
                current_id,
                format!("{:?}", algebra)
                    .split('(')
                    .next()
                    .unwrap_or("Unknown")
            )?;
        }
    }

    Ok(current_id)
}

/// Escape special characters for DOT label
fn escape_dot_label(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_arq::query::parse_query;

    #[test]
    fn test_generate_query_plan_dot_basic() {
        let query_str = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
        let query = parse_query(query_str).expect("Failed to parse query");
        let complexity = estimate_complexity(&query.where_clause);

        let dot = generate_query_plan_dot(&query, &complexity).expect("Failed to generate DOT");

        // Print for debugging
        println!("Generated DOT:\n{}", dot);

        // Verify basic structure
        assert!(
            dot.contains("digraph QueryPlan"),
            "Should contain digraph QueryPlan"
        );
        assert!(dot.contains("rankdir=BT"), "Should contain rankdir=BT");
        assert!(dot.contains("BGP"), "Should contain BGP");
        // Query type is Debug format, could be "Select" not "SELECT"
        assert!(
            dot.contains("Select") || dot.contains("SELECT"),
            "Should contain Select/SELECT"
        );
    }

    #[test]
    fn test_generate_query_plan_dot_with_join() {
        let query_str =
            "SELECT ?s WHERE { ?s <http://example.org/p1> ?o1 . ?s <http://example.org/p2> ?o2 }";
        let query = parse_query(query_str).expect("Failed to parse query");
        let complexity = estimate_complexity(&query.where_clause);

        let dot = generate_query_plan_dot(&query, &complexity).expect("Failed to generate DOT");

        // Should contain complexity metrics
        assert!(
            dot.contains("Complexity:"),
            "Should contain complexity metrics"
        );
        assert!(
            dot.contains("Select") || dot.contains("SELECT"),
            "Should contain Select"
        );
        assert!(dot.contains("BGP"), "Should contain BGP");
    }

    #[test]
    fn test_generate_query_plan_dot_with_filter() {
        let query_str = "SELECT ?s WHERE { ?s ?p ?o FILTER(?o > 10) }";
        let query = parse_query(query_str).expect("Failed to parse query");
        let complexity = estimate_complexity(&query.where_clause);

        let dot = generate_query_plan_dot(&query, &complexity).expect("Failed to generate DOT");

        // Should mention filter in complexity metrics
        assert!(
            dot.contains("Filters:"),
            "Should mention Filters in metrics"
        );
        assert!(dot.contains("digraph QueryPlan"), "Should contain digraph");
        // Check for FILTER node
        assert!(
            dot.contains("FILTER") || complexity.filters > 0,
            "Should have filter node or complexity"
        );
    }

    #[test]
    fn test_generate_query_plan_dot_with_optional() {
        let query_str =
            "SELECT ?s WHERE { ?s ?p ?o OPTIONAL { ?s <http://example.org/name> ?name } }";
        let query = parse_query(query_str).expect("Failed to parse query");
        let complexity = estimate_complexity(&query.where_clause);

        let dot = generate_query_plan_dot(&query, &complexity).expect("Failed to generate DOT");

        // Should mention optionals in complexity metrics
        assert!(
            dot.contains("Optionals:"),
            "Should mention Optionals in metrics"
        );
        assert!(dot.contains("digraph QueryPlan"), "Should contain digraph");
        // Check that optional pattern is detected
        assert!(complexity.optionals > 0, "Should detect optional patterns");
    }

    #[test]
    fn test_generate_query_plan_dot_with_modifiers() {
        // Test with DISTINCT which is supported by the parser
        let query_str = "SELECT DISTINCT ?s WHERE { ?s ?p ?o }";
        let query = parse_query(query_str).expect("Failed to parse query");
        let complexity = estimate_complexity(&query.where_clause);

        let dot = generate_query_plan_dot(&query, &complexity).expect("Failed to generate DOT");

        // Should contain DISTINCT modifier
        assert!(dot.contains("DISTINCT"), "Should contain DISTINCT node");
        assert!(dot.contains("Select"), "Should contain Select node");
        assert!(dot.contains("BGP"), "Should contain BGP node");
        assert!(
            dot.contains("digraph QueryPlan"),
            "Should be valid DOT format"
        );
    }

    #[test]
    fn test_escape_dot_label() {
        assert_eq!(escape_dot_label("test"), "test");
        assert_eq!(escape_dot_label("test & test"), "test &amp; test");
        assert_eq!(escape_dot_label("test < test"), "test &lt; test");
        assert_eq!(escape_dot_label("test > test"), "test &gt; test");
        assert_eq!(escape_dot_label("test \"quote\""), "test &quot;quote&quot;");
        assert_eq!(escape_dot_label("test\nline"), "test\\nline");
    }

    #[test]
    fn test_complexity_estimation_bgp() {
        let query_str = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
        let query = parse_query(query_str).expect("Failed to parse query");
        let complexity = estimate_complexity(&query.where_clause);

        assert_eq!(complexity.score, 1);
        assert_eq!(complexity.joins, 0);
        assert_eq!(complexity.filters, 0);
        assert_eq!(complexity.optionals, 0);
    }

    #[test]
    fn test_complexity_estimation_with_filter() {
        let query_str = "SELECT ?s WHERE { ?s ?p ?o FILTER(?o > 10) }";
        let query = parse_query(query_str).expect("Failed to parse query");
        let complexity = estimate_complexity(&query.where_clause);

        assert!(complexity.filters > 0, "Should detect filter");
        assert!(
            complexity.score > 1,
            "Filtered queries should have higher complexity"
        );
    }
}
