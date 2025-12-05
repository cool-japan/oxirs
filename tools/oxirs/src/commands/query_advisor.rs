//! Intelligent SPARQL Query Advisor
//!
//! Provides real-time query analysis, optimization suggestions, and best practices
//! recommendations using pattern matching, heuristics, and statistical analysis.
//!
//! ## Features
//!
//! - Pattern-based anti-pattern detection
//! - Query complexity scoring with statistical insights
//! - Selectivity estimation for query optimization
//! - Best practices recommendations with examples
//! - Performance prediction based on query characteristics

use super::CommandResult;
use crate::cli::CliContext;
use colored::Colorize;

/// Query advice severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdviceSeverity {
    /// Critical issues that will likely cause problems
    Critical,
    /// Warnings about potential performance issues
    Warning,
    /// Informational suggestions for best practices
    Info,
    /// Tips for query improvement
    Tip,
}

impl AdviceSeverity {
    fn emoji(&self) -> &str {
        match self {
            Self::Critical => "🚨",
            Self::Warning => "⚠️",
            Self::Info => "ℹ️",
            Self::Tip => "💡",
        }
    }

    fn colored_label(&self) -> String {
        match self {
            Self::Critical => "CRITICAL".red().bold().to_string(),
            Self::Warning => "WARNING".yellow().bold().to_string(),
            Self::Info => "INFO".blue().bold().to_string(),
            Self::Tip => "TIP".green().to_string(),
        }
    }
}

/// Single piece of query advice
#[derive(Debug, Clone)]
pub struct QueryAdvice {
    pub severity: AdviceSeverity,
    pub title: String,
    pub description: String,
    pub suggestion: String,
    pub example: Option<String>,
}

impl QueryAdvice {
    fn new(
        severity: AdviceSeverity,
        title: impl Into<String>,
        description: impl Into<String>,
        suggestion: impl Into<String>,
    ) -> Self {
        Self {
            severity,
            title: title.into(),
            description: description.into(),
            suggestion: suggestion.into(),
            example: None,
        }
    }

    fn with_example(mut self, example: impl Into<String>) -> Self {
        self.example = Some(example.into());
        self
    }
}

/// SPARQL Query Advisor
pub struct QueryAdvisor {
    ctx: CliContext,
}

impl QueryAdvisor {
    pub fn new() -> Self {
        Self {
            ctx: CliContext::new(),
        }
    }

    /// Analyze a SPARQL query and provide comprehensive advice
    pub fn analyze_query(&self, query: &str) -> Vec<QueryAdvice> {
        let mut advice = Vec::new();

        // Normalize query for analysis
        let query_upper = query.to_uppercase();
        let _query_lower = query.to_lowercase(); // Reserved for future analysis

        // Check for SELECT *
        if query_upper.contains("SELECT *") || query_upper.contains("SELECT*") {
            advice.push(
                QueryAdvice::new(
                    AdviceSeverity::Warning,
                    "Avoid SELECT *",
                    "Using SELECT * returns all variables and can be inefficient",
                    "Explicitly list only the variables you need",
                )
                .with_example(
                    "SELECT ?name ?age WHERE { ?person foaf:name ?name ; foaf:age ?age }",
                ),
            );
        }

        // Check for missing LIMIT
        if query_upper.contains("SELECT") && !query_upper.contains("LIMIT") {
            advice.push(
                QueryAdvice::new(
                    AdviceSeverity::Info,
                    "Consider adding LIMIT",
                    "Queries without LIMIT may return large result sets",
                    "Add LIMIT clause to control result size",
                )
                .with_example("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100"),
            );
        }

        // Check for Cartesian products
        let where_clause = extract_where_clause(query);
        if has_potential_cartesian_product(&where_clause) {
            advice.push(QueryAdvice::new(
                AdviceSeverity::Critical,
                "Potential Cartesian Product Detected",
                "Multiple disconnected triple patterns can create expensive cross products",
                "Connect patterns with shared variables or use explicit filters",
            ).with_example("# Bad:\nSELECT * WHERE { ?a ?b ?c . ?d ?e ?f }\n# Good:\nSELECT * WHERE { ?person foaf:knows ?friend . ?friend foaf:age ?age }"));
        }

        // Check for unbound predicates
        if query.contains("?p") && query_upper.matches(" ?P ").count() > 1 {
            advice.push(QueryAdvice::new(
                AdviceSeverity::Warning,
                "Unbound Predicate Variable",
                "Variables in predicate position can be very expensive",
                "Use specific predicates or add FILTER constraints",
            ));
        }

        // Check for OPTIONAL abuse
        let optional_count = query_upper.matches("OPTIONAL").count();
        if optional_count > 5 {
            advice.push(QueryAdvice::new(
                AdviceSeverity::Warning,
                "Excessive OPTIONAL Clauses",
                format!(
                    "Query has {} OPTIONAL clauses which may slow execution",
                    optional_count
                ),
                "Consider restructuring with UNION or multiple queries",
            ));
        }

        // Check for FILTER optimization opportunities
        if query_upper.contains("FILTER") && contains_expensive_filter(query) {
            advice.push(QueryAdvice::new(
                AdviceSeverity::Tip,
                "Filter Optimization Opportunity",
                "Some filters can be pushed down into triple patterns",
                "Use property paths or more specific patterns where possible",
            ));
        }

        // Check for ORDER BY without LIMIT
        if query_upper.contains("ORDER BY") && !query_upper.contains("LIMIT") {
            advice.push(
                QueryAdvice::new(
                    AdviceSeverity::Warning,
                    "ORDER BY without LIMIT",
                    "Sorting large unbounded result sets is expensive",
                    "Add LIMIT to reduce sorting cost",
                )
                .with_example(
                    "SELECT ?s ?name WHERE { ?s foaf:name ?name } ORDER BY ?name LIMIT 100",
                ),
            );
        }

        // Check for nested subqueries
        let select_count = query_upper.matches("SELECT").count();
        if select_count > 3 {
            advice.push(QueryAdvice::new(
                AdviceSeverity::Info,
                "Complex Query with Multiple Subqueries",
                "Deeply nested subqueries can be hard to optimize",
                "Consider breaking into multiple simpler queries",
            ));
        }

        // Check for aggregation without GROUP BY
        let has_aggregation = query_upper.contains("COUNT(")
            || query_upper.contains("SUM(")
            || query_upper.contains("AVG(")
            || query_upper.contains("MAX(")
            || query_upper.contains("MIN(");

        if has_aggregation && !query_upper.contains("GROUP BY") {
            advice.push(QueryAdvice::new(
                AdviceSeverity::Info,
                "Aggregation Without Grouping",
                "Query uses aggregation functions without GROUP BY",
                "This will aggregate over all results - ensure this is intended",
            ));
        }

        // Check for string operations
        let query_for_text = query.to_lowercase();
        if query_for_text.contains("regex") || query_for_text.contains("contains") {
            advice.push(
                QueryAdvice::new(
                    AdviceSeverity::Tip,
                    "Text Search Operations Detected",
                    "String operations like REGEX can be slow on large datasets",
                    "Consider using full-text search indexes if available",
                )
                .with_example("Use oxirs text-search plugin for better performance"),
            );
        }

        // Check for DISTINCT without need
        if query_upper.contains("SELECT DISTINCT") {
            advice.push(QueryAdvice::new(
                AdviceSeverity::Tip,
                "DISTINCT Usage",
                "DISTINCT adds overhead - only use if duplicates are expected",
                "Remove DISTINCT if your data model guarantees uniqueness",
            ));
        }

        // Check for property paths
        if query.contains("/") || query.contains("+") || query.contains("*") {
            advice.push(
                QueryAdvice::new(
                    AdviceSeverity::Info,
                    "Property Paths Detected",
                    "Property paths are powerful but can be expensive on large graphs",
                    "Add constraints to limit search space",
                )
                .with_example("?x foaf:knows+ ?y . FILTER(?x = <http://example.org/alice>)"),
            );
        }

        advice
    }

    /// Display advice in a formatted way
    pub fn display_advice(&self, advice: &[QueryAdvice]) -> CommandResult {
        if advice.is_empty() {
            self.ctx.success("✅ No issues found - query looks good!");
            return Ok(());
        }

        self.ctx.info(&format!(
            "\n{} Found {} suggestion(s):\n",
            "📋".bold(),
            advice.len()
        ));

        for (i, item) in advice.iter().enumerate() {
            println!(
                "{} {} {}",
                item.severity.emoji(),
                item.severity.colored_label(),
                item.title.bold()
            );
            println!("   {}", item.description);
            println!("   {} {}", "→".green(), item.suggestion.italic());

            if let Some(ref example) = item.example {
                println!("   {} Example:", "💡".green());
                for line in example.lines() {
                    println!("     {}", line.dimmed());
                }
            }

            if i < advice.len() - 1 {
                println!();
            }
        }

        println!();
        Ok(())
    }

    /// Generate a comprehensive query report
    pub fn generate_report(&self, query: &str) -> CommandResult {
        self.ctx.info("🔍 Analyzing SPARQL Query...\n");

        let advice = self.analyze_query(query);
        let metrics = self.calculate_metrics(query);

        // Display metrics
        println!("{}", "Query Metrics:".bold().underline());
        println!("  Lines: {}", query.lines().count());
        println!("  Characters: {}", query.len());
        println!("  Complexity Score: {}/100", metrics.complexity_score);
        println!("  Triple Patterns: {}", metrics.triple_patterns);
        println!("  Estimated Selectivity: {}", metrics.selectivity);
        println!("  Estimated Result Size: {}", metrics.estimated_result_size);
        println!(
            "  Optimization Potential: {}/100",
            metrics.optimization_potential
        );

        // Performance prediction
        if metrics.complexity_score > 70 {
            println!(
                "  {} {}",
                "⚠️".yellow(),
                "High complexity - expect slower execution".yellow()
            );
        } else if metrics.complexity_score < 30 {
            println!(
                "  {} {}",
                "✅".green(),
                "Low complexity - fast execution expected".green()
            );
        }

        if metrics.optimization_potential > 50 {
            println!(
                "  {} {}",
                "💡".cyan(),
                "High optimization potential - review suggestions below".cyan()
            );
        }

        println!();

        // Display advice
        self.display_advice(&advice)?;

        // Summary
        let critical = advice
            .iter()
            .filter(|a| a.severity == AdviceSeverity::Critical)
            .count();
        let warnings = advice
            .iter()
            .filter(|a| a.severity == AdviceSeverity::Warning)
            .count();

        if critical > 0 {
            self.ctx
                .warn(&format!("⚠️  {} critical issue(s) found", critical));
        } else if warnings > 0 {
            self.ctx.info(&format!("⚠️  {} warning(s) found", warnings));
        } else {
            self.ctx.success("✅ Query analysis complete");
        }

        Ok(())
    }

    /// Calculate query metrics with advanced statistical analysis
    fn calculate_metrics(&self, query: &str) -> QueryMetrics {
        let triple_patterns = count_triple_patterns(query);
        let complexity_score = calculate_complexity(query);
        let selectivity = estimate_selectivity(query);
        let estimated_result_size = estimate_result_size(query, triple_patterns);
        let optimization_potential = calculate_optimization_potential(query, complexity_score);

        QueryMetrics {
            triple_patterns,
            complexity_score,
            selectivity,
            estimated_result_size,
            optimization_potential,
        }
    }
}

impl Default for QueryAdvisor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
struct QueryMetrics {
    triple_patterns: usize,
    complexity_score: u8,
    selectivity: String,
    estimated_result_size: String,
    optimization_potential: u8,
}

// Helper functions

fn extract_where_clause(query: &str) -> String {
    let query_upper = query.to_uppercase();
    if let Some(start) = query_upper.find("WHERE") {
        query[start..].to_string()
    } else {
        String::new()
    }
}

fn has_potential_cartesian_product(where_clause: &str) -> bool {
    // Simple heuristic: look for multiple triple patterns without shared variables
    let patterns: Vec<&str> = where_clause.split('.').collect();
    if patterns.len() < 2 {
        return false;
    }

    // Check if patterns share variables
    let mut var_sets: Vec<Vec<&str>> = Vec::new();
    for pattern in patterns {
        let vars: Vec<&str> = pattern
            .split_whitespace()
            .filter(|s| s.starts_with('?'))
            .collect();
        if !vars.is_empty() {
            var_sets.push(vars);
        }
    }

    // Check for disconnected patterns
    if var_sets.len() >= 2 {
        let first_set: Vec<String> = var_sets[0].iter().map(|s| s.to_string()).collect();
        for other_set in &var_sets[1..] {
            let has_shared = other_set.iter().any(|v| first_set.contains(&v.to_string()));
            if !has_shared {
                return true;
            }
        }
    }

    false
}

fn contains_expensive_filter(query: &str) -> bool {
    let query_lower = query.to_lowercase();
    query_lower.contains("regex") || query_lower.contains("str(")
}

fn count_triple_patterns(query: &str) -> usize {
    // Simple heuristic: count dots and semicolons in WHERE clause
    let where_clause = extract_where_clause(query);
    where_clause.matches('.').count() + where_clause.matches(';').count()
}

fn calculate_complexity(query: &str) -> u8 {
    let query_upper = query.to_uppercase();

    let mut score = 0;

    score += count_triple_patterns(query) * 5;
    score += query_upper.matches("OPTIONAL").count() * 15;
    score += query_upper.matches("UNION").count() * 12;
    score += query_upper.matches("FILTER").count() * 8;
    score += (query_upper.matches("SELECT").count().saturating_sub(1)) * 20;

    (score.min(100)) as u8
}

fn estimate_selectivity(query: &str) -> String {
    let has_specific_uris = query.contains("http://") || query.contains("https://");
    let has_filters = query.to_uppercase().contains("FILTER");

    if has_specific_uris && has_filters {
        "High (specific URIs + filters)".to_string()
    } else if has_specific_uris {
        "Medium (specific URIs)".to_string()
    } else if has_filters {
        "Medium (filters applied)".to_string()
    } else {
        "Low (broad query)".to_string()
    }
}

fn estimate_result_size(query: &str, triple_patterns: usize) -> String {
    let query_upper = query.to_uppercase();
    let has_limit = query_upper.contains("LIMIT");
    let has_specific_uris = query.contains("http://") || query.contains("https://");
    let has_filters = query_upper.contains("FILTER");

    if has_limit {
        // Extract limit value if possible
        if let Some(limit_pos) = query_upper.find("LIMIT") {
            let after_limit = &query[limit_pos + 5..];
            if let Some(num_str) = after_limit
                .split_whitespace()
                .next()
                .and_then(|s| s.parse::<usize>().ok())
            {
                return format!("≤{} rows (explicit LIMIT)", num_str);
            }
        }
        "Bounded (LIMIT specified)".to_string()
    } else if has_specific_uris && has_filters && triple_patterns < 5 {
        "Small (10-1000 rows)".to_string()
    } else if has_specific_uris || has_filters {
        "Medium (1K-100K rows)".to_string()
    } else if triple_patterns > 10 {
        "Large (100K+ rows, Cartesian risk)".to_string()
    } else {
        "Medium-Large (10K-1M rows)".to_string()
    }
}

fn calculate_optimization_potential(query: &str, complexity_score: u8) -> u8 {
    let query_upper = query.to_uppercase();
    let mut potential = 0u8;

    // SELECT * adds optimization potential
    if query_upper.contains("SELECT *") || query_upper.contains("SELECT*") {
        potential = potential.saturating_add(20);
    }

    // Missing LIMIT adds potential
    if query_upper.contains("SELECT") && !query_upper.contains("LIMIT") {
        potential = potential.saturating_add(15);
    }

    // OPTIONAL abuse
    if query_upper.matches("OPTIONAL").count() > 3 {
        potential = potential.saturating_add(25);
    }

    // Unbound predicates
    if query.contains("?p ") && query.matches("?p").count() > 1 {
        potential = potential.saturating_add(30);
    }

    // ORDER BY without LIMIT
    if query_upper.contains("ORDER BY") && !query_upper.contains("LIMIT") {
        potential = potential.saturating_add(20);
    }

    // DISTINCT usage
    if query_upper.contains("DISTINCT") {
        potential = potential.saturating_add(10);
    }

    // High complexity inherently means optimization opportunity
    if complexity_score > 60 {
        potential = potential.saturating_add(15);
    }

    potential.min(100)
}

/// Command handler for query analysis
pub async fn analyze_query_cmd(query: String, verbose: bool) -> CommandResult {
    let advisor = QueryAdvisor::new();

    if verbose {
        advisor.generate_report(&query)?;
    } else {
        let advice = advisor.analyze_query(&query);
        advisor.display_advice(&advice)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_select_star() {
        let advisor = QueryAdvisor::new();
        let query = "SELECT * WHERE { ?s ?p ?o }";
        let advice = advisor.analyze_query(query);

        assert!(advice
            .iter()
            .any(|a| a.title.contains("SELECT *") && a.severity == AdviceSeverity::Warning));
    }

    #[test]
    fn test_detect_missing_limit() {
        let advisor = QueryAdvisor::new();
        let query = "SELECT ?s ?p WHERE { ?s ?p ?o }";
        let advice = advisor.analyze_query(query);

        assert!(advice.iter().any(|a| a.title.contains("LIMIT")));
    }

    #[test]
    fn test_detect_cartesian_product() {
        let advisor = QueryAdvisor::new();
        let query = "SELECT * WHERE { ?a ?b ?c . ?d ?e ?f }";
        let advice = advisor.analyze_query(query);

        assert!(advice
            .iter()
            .any(|a| a.title.contains("Cartesian") && a.severity == AdviceSeverity::Critical));
    }

    #[test]
    fn test_good_query_no_advice() {
        let advisor = QueryAdvisor::new();
        let query = "SELECT ?name WHERE { ?person foaf:name ?name } LIMIT 10";
        let advice = advisor.analyze_query(query);

        // Should have minimal or no critical/warning advice
        let critical_count = advice
            .iter()
            .filter(|a| a.severity == AdviceSeverity::Critical)
            .count();
        assert_eq!(critical_count, 0);
    }

    #[test]
    fn test_complexity_calculation() {
        let simple = "SELECT ?s WHERE { ?s ?p ?o }";
        let complex = "SELECT ?x WHERE { ?x ?p1 ?y . OPTIONAL { ?y ?p2 ?z } UNION { ?x ?p3 ?w } FILTER(?x != ?y) }";

        assert!(calculate_complexity(complex) > calculate_complexity(simple));
    }

    #[test]
    fn test_result_size_estimation() {
        // Query with explicit LIMIT
        let with_limit = "SELECT ?s WHERE { ?s ?p ?o } LIMIT 100";
        let result = estimate_result_size(with_limit, 1);
        assert!(result.contains("100"));

        // Query with specific URIs and filters
        let specific = "SELECT ?name WHERE { <http://example.org/alice> foaf:name ?name } FILTER(?name != \"\")";
        let result = estimate_result_size(specific, 1);
        assert!(result.contains("Small"));

        // Broad query without constraints
        let broad = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
        let result = estimate_result_size(broad, 1);
        assert!(result.contains("Medium") || result.contains("Large"));
    }

    #[test]
    fn test_optimization_potential() {
        // SELECT * with no LIMIT - high potential
        let high_potential = "SELECT * WHERE { ?s ?p ?o }";
        let score = calculate_optimization_potential(high_potential, 50);
        assert!(score > 30);

        // Well-optimized query - low potential
        let low_potential = "SELECT ?name WHERE { ?person foaf:name ?name } LIMIT 10";
        let score = calculate_optimization_potential(low_potential, 20);
        assert!(score < 30);

        // Query with unbound predicates - very high potential
        let unbound_pred = "SELECT * WHERE { ?s ?p ?o . ?x ?p ?y }";
        let score = calculate_optimization_potential(unbound_pred, 60);
        assert!(score > 50);
    }

    #[test]
    fn test_selectivity_estimation() {
        // High selectivity
        let high = "SELECT ?s WHERE { <http://example.org/alice> foaf:knows ?s } FILTER(?s != <http://example.org/bob>)";
        assert_eq!(estimate_selectivity(high), "High (specific URIs + filters)");

        // Medium with URIs only
        let medium_uri = "SELECT ?s WHERE { <http://example.org/alice> foaf:knows ?s }";
        assert_eq!(estimate_selectivity(medium_uri), "Medium (specific URIs)");

        // Low selectivity
        let low = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
        assert_eq!(estimate_selectivity(low), "Low (broad query)");
    }

    #[test]
    fn test_enhanced_metrics() {
        let advisor = QueryAdvisor::new();
        let query = "SELECT * WHERE { ?s ?p ?o . ?x ?y ?z } ORDER BY ?s";

        let metrics = advisor.calculate_metrics(query);

        // Check all fields are populated
        assert!(metrics.complexity_score > 0);
        assert!(metrics.triple_patterns > 0);
        assert!(!metrics.selectivity.is_empty());
        assert!(!metrics.estimated_result_size.is_empty());
        assert!(metrics.optimization_potential > 0);
    }
}
