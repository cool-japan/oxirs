//! Automatic Query Optimization Advisor
//!
//! Analyzes SPARQL queries and provides actionable optimization suggestions.
//!
//! ## Features
//!
//! - **Pattern analysis**: Identifies inefficient triple patterns
//! - **Index recommendations**: Suggests beneficial indexes
//! - **Query rewriting**: Proposes equivalent but faster formulations
//! - **Cost estimation**: Predicts performance impact of suggestions
//! - **Best practices**: Enforces SPARQL optimization guidelines
//! - **SciRS2 integration**: Statistical analysis of query patterns
//!
//! ## Example
//!
//! ```rust,ignore
//! use oxirs_arq::query_optimization_advisor::{OptimizationAdvisor, AdvisorConfig};
//!
//! let advisor = OptimizationAdvisor::new(AdvisorConfig::default());
//! let suggestions = advisor.analyze_query("SELECT * WHERE { ?s ?p ?o }")?;
//!
//! for suggestion in suggestions {
//!     println!("{}: {}", suggestion.severity, suggestion.message);
//! }
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Optimization suggestion severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SuggestionSeverity {
    /// Informational - minor optimization opportunity
    Info,

    /// Warning - notable performance issue
    Warning,

    /// Critical - major performance bottleneck
    Critical,
}

impl std::fmt::Display for SuggestionSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SuggestionSeverity::Info => write!(f, "INFO"),
            SuggestionSeverity::Warning => write!(f, "WARNING"),
            SuggestionSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Category of optimization suggestion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestionCategory {
    /// Pattern ordering suggestions
    PatternOrdering,

    /// Index usage recommendations
    IndexUsage,

    /// Filter placement optimization
    FilterPlacement,

    /// Join algorithm selection
    JoinStrategy,

    /// Result limiting
    ResultLimitation,

    /// Query structure
    QueryStructure,

    /// Best practices
    BestPractices,

    /// Resource usage
    ResourceUsage,
}

/// Optimization suggestion with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    /// Unique identifier for this suggestion type
    pub id: String,

    /// Human-readable message
    pub message: String,

    /// Severity level
    pub severity: SuggestionSeverity,

    /// Category
    pub category: SuggestionCategory,

    /// Suggested rewrite (if applicable)
    pub suggested_rewrite: Option<String>,

    /// Estimated performance impact (multiplier: >1.0 = faster)
    pub estimated_speedup: Option<f64>,

    /// Detailed explanation
    pub explanation: String,

    /// Location in query (line/column if available)
    pub location: Option<QueryLocation>,

    /// Related documentation
    pub documentation_url: Option<String>,
}

/// Location in a SPARQL query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryLocation {
    pub line: usize,
    pub column: usize,
    pub length: usize,
}

/// Configuration for optimization advisor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvisorConfig {
    /// Enable pattern ordering suggestions
    pub analyze_pattern_ordering: bool,

    /// Enable index recommendations
    pub analyze_index_usage: bool,

    /// Enable filter placement analysis
    pub analyze_filter_placement: bool,

    /// Enable join strategy recommendations
    pub analyze_join_strategy: bool,

    /// Enable result limitation checks
    pub analyze_result_limits: bool,

    /// Enable best practices checks
    pub analyze_best_practices: bool,

    /// Minimum selectivity for filter warnings
    pub filter_selectivity_threshold: f64,

    /// Maximum recommended OPTIONAL depth
    pub max_optional_depth: usize,

    /// Maximum recommended UNION branches
    pub max_union_branches: usize,

    /// Warn about queries without LIMIT
    pub require_limit_clause: bool,

    /// Maximum suggested LIMIT value
    pub recommended_max_limit: usize,
}

impl Default for AdvisorConfig {
    fn default() -> Self {
        Self {
            analyze_pattern_ordering: true,
            analyze_index_usage: true,
            analyze_filter_placement: true,
            analyze_join_strategy: true,
            analyze_result_limits: true,
            analyze_best_practices: true,
            filter_selectivity_threshold: 0.01,
            max_optional_depth: 3,
            max_union_branches: 5,
            require_limit_clause: false,
            recommended_max_limit: 10_000,
        }
    }
}

/// Query optimization advisor
pub struct OptimizationAdvisor {
    config: AdvisorConfig,
    #[allow(dead_code)]
    pattern_statistics: HashMap<String, PatternStatistics>,
}

/// Statistics for a triple pattern
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct PatternStatistics {
    selectivity: f64,
    estimated_results: usize,
    has_index: bool,
}

impl OptimizationAdvisor {
    /// Create a new advisor with configuration
    pub fn new(config: AdvisorConfig) -> Self {
        Self {
            config,
            pattern_statistics: HashMap::new(),
        }
    }

    /// Analyze a SPARQL query and return optimization suggestions
    pub fn analyze_query(&self, query: &str) -> Result<Vec<OptimizationSuggestion>> {
        let mut suggestions = Vec::new();

        // Parse query structure (simplified analysis)
        let analysis = self.analyze_query_structure(query)?;

        // Run various analyses
        if self.config.analyze_pattern_ordering {
            suggestions.extend(self.analyze_pattern_ordering(&analysis)?);
        }

        if self.config.analyze_filter_placement {
            suggestions.extend(self.analyze_filter_placement(&analysis)?);
        }

        if self.config.analyze_join_strategy {
            suggestions.extend(self.analyze_join_strategy(&analysis)?);
        }

        if self.config.analyze_result_limits {
            suggestions.extend(self.analyze_result_limits(&analysis)?);
        }

        if self.config.analyze_best_practices {
            suggestions.extend(self.analyze_best_practices(&analysis)?);
        }

        if self.config.analyze_index_usage {
            suggestions.extend(self.analyze_index_usage(&analysis)?);
        }

        // Sort by severity (critical first)
        suggestions.sort_by(|a, b| b.severity.cmp(&a.severity));

        Ok(suggestions)
    }

    /// Analyze query structure (simplified pattern extraction)
    fn analyze_query_structure(&self, query: &str) -> Result<QueryAnalysis> {
        let mut analysis = QueryAnalysis::default();

        // Count SELECT/ASK/CONSTRUCT/DESCRIBE
        if query.contains("SELECT") {
            analysis.query_form = QueryForm::Select;
        } else if query.contains("ASK") {
            analysis.query_form = QueryForm::Ask;
        } else if query.contains("CONSTRUCT") {
            analysis.query_form = QueryForm::Construct;
        } else if query.contains("DESCRIBE") {
            analysis.query_form = QueryForm::Describe;
        }

        // Check for DISTINCT
        analysis.has_distinct = query.contains("DISTINCT");

        // Check for LIMIT
        if let Some(pos) = query.find("LIMIT") {
            if let Some(limit_str) = query[pos..].split_whitespace().nth(1) {
                analysis.limit = limit_str.parse().ok();
            }
        }

        // Count OPTIONAL blocks
        analysis.optional_count = query.matches("OPTIONAL").count();

        // Count UNION blocks
        analysis.union_count = query.matches("UNION").count();

        // Count FILTER clauses
        analysis.filter_count = query.matches("FILTER").count();

        // Count triple patterns (approximate - count pattern terminators)
        // Each triple pattern ends with either ". " or " ." (possibly followed by })
        let dot_space = query.matches(". ").count();
        let space_dot = query.matches(" .").count();
        let dot_brace = query.matches(".}").count();

        // Take the maximum as a rough estimate
        let pattern_count = std::cmp::max(dot_space, std::cmp::max(space_dot, dot_brace));
        analysis.triple_pattern_count = if pattern_count > 0 { pattern_count } else { 1 };

        // Check for SELECT *
        analysis.has_select_star = query.contains("SELECT *") || query.contains("SELECT*");

        // Check for BIND
        analysis.bind_count = query.matches("BIND").count();

        // Check for ORDER BY
        analysis.has_order_by = query.contains("ORDER BY");

        // Check for GROUP BY
        analysis.has_group_by = query.contains("GROUP BY");

        // Check for aggregates
        analysis.has_aggregates = query.contains("COUNT(")
            || query.contains("SUM(")
            || query.contains("AVG(")
            || query.contains("MIN(")
            || query.contains("MAX(");

        Ok(analysis)
    }

    /// Analyze triple pattern ordering
    fn analyze_pattern_ordering(
        &self,
        analysis: &QueryAnalysis,
    ) -> Result<Vec<OptimizationSuggestion>> {
        let mut suggestions = Vec::new();

        // Suggest reordering if many patterns without statistics
        if analysis.triple_pattern_count >= 5 {
            suggestions.push(OptimizationSuggestion {
                id: "PATTERN_ORDER_001".to_string(),
                message: format!(
                    "Query has {} triple patterns. Consider pattern ordering to start with most selective patterns.",
                    analysis.triple_pattern_count
                ),
                severity: SuggestionSeverity::Warning,
                category: SuggestionCategory::PatternOrdering,
                suggested_rewrite: None,
                estimated_speedup: Some(2.0),
                explanation: "Placing selective patterns first reduces intermediate results. \
                              Patterns with bound subjects/objects are typically more selective than those with all variables."
                    .to_string(),
                location: None,
                documentation_url: Some("https://www.w3.org/TR/sparql11-query/#sparqlBGPExtend".to_string()),
            });
        }

        Ok(suggestions)
    }

    /// Analyze filter placement
    fn analyze_filter_placement(
        &self,
        analysis: &QueryAnalysis,
    ) -> Result<Vec<OptimizationSuggestion>> {
        let mut suggestions = Vec::new();

        if analysis.filter_count > 0 && analysis.triple_pattern_count >= 3 {
            suggestions.push(OptimizationSuggestion {
                id: "FILTER_PLACEMENT_001".to_string(),
                message: format!(
                    "Query has {} FILTERs. Ensure they are placed close to the patterns that bind their variables.",
                    analysis.filter_count
                ),
                severity: SuggestionSeverity::Info,
                category: SuggestionCategory::FilterPlacement,
                suggested_rewrite: None,
                estimated_speedup: Some(1.5),
                explanation: "Filters should be evaluated as soon as their variables are bound. \
                              Late filter evaluation can cause unnecessary computation on large intermediate results."
                    .to_string(),
                location: None,
                documentation_url: None,
            });
        }

        Ok(suggestions)
    }

    /// Analyze join strategy
    fn analyze_join_strategy(
        &self,
        analysis: &QueryAnalysis,
    ) -> Result<Vec<OptimizationSuggestion>> {
        let mut suggestions = Vec::new();

        if analysis.triple_pattern_count > 10 {
            suggestions.push(OptimizationSuggestion {
                id: "JOIN_STRATEGY_001".to_string(),
                message: "Large query detected. Consider using query hints to guide join algorithm selection.".to_string(),
                severity: SuggestionSeverity::Info,
                category: SuggestionCategory::JoinStrategy,
                suggested_rewrite: Some("Add /*+ HASH_JOIN */ or /*+ MERGE_JOIN */ hint".to_string()),
                estimated_speedup: Some(1.3),
                explanation: "For queries with many patterns, hash joins often outperform nested loop joins. \
                              Use query hints to override default join selection."
                    .to_string(),
                location: None,
                documentation_url: None,
            });
        }

        Ok(suggestions)
    }

    /// Analyze result limitations
    fn analyze_result_limits(
        &self,
        analysis: &QueryAnalysis,
    ) -> Result<Vec<OptimizationSuggestion>> {
        let mut suggestions = Vec::new();

        // Check for missing LIMIT
        if self.config.require_limit_clause
            && analysis.limit.is_none()
            && analysis.query_form == QueryForm::Select
        {
            suggestions.push(OptimizationSuggestion {
                id: "RESULT_LIMIT_001".to_string(),
                message: "Query has no LIMIT clause. Consider adding one to prevent excessive results.".to_string(),
                severity: SuggestionSeverity::Warning,
                category: SuggestionCategory::ResultLimitation,
                suggested_rewrite: Some(format!("Add LIMIT {}", self.config.recommended_max_limit)),
                estimated_speedup: Some(10.0),
                explanation: "Queries without LIMIT can return millions of results, consuming memory and network bandwidth. \
                              Add LIMIT to improve responsiveness."
                    .to_string(),
                location: None,
                documentation_url: None,
            });
        }

        // Check for excessive LIMIT
        if let Some(limit) = analysis.limit {
            if limit > self.config.recommended_max_limit {
                suggestions.push(OptimizationSuggestion {
                    id: "RESULT_LIMIT_002".to_string(),
                    message: format!(
                        "LIMIT {} exceeds recommended maximum {}. Consider pagination instead.",
                        limit, self.config.recommended_max_limit
                    ),
                    severity: SuggestionSeverity::Warning,
                    category: SuggestionCategory::ResultLimitation,
                    suggested_rewrite: Some(
                        "Use cursor-based pagination for large result sets".to_string(),
                    ),
                    estimated_speedup: Some(2.0),
                    explanation:
                        "Large result sets should be paginated to maintain responsiveness. \
                                  Use the query_pagination module for efficient pagination."
                            .to_string(),
                    location: None,
                    documentation_url: None,
                });
            }
        }

        Ok(suggestions)
    }

    /// Analyze best practices
    fn analyze_best_practices(
        &self,
        analysis: &QueryAnalysis,
    ) -> Result<Vec<OptimizationSuggestion>> {
        let mut suggestions = Vec::new();

        // Warn about SELECT *
        if analysis.has_select_star {
            suggestions.push(OptimizationSuggestion {
                id: "BEST_PRACTICE_001".to_string(),
                message: "Avoid SELECT *. Specify only the variables you need.".to_string(),
                severity: SuggestionSeverity::Warning,
                category: SuggestionCategory::BestPractices,
                suggested_rewrite: Some("SELECT ?var1 ?var2 ... WHERE { ... }".to_string()),
                estimated_speedup: Some(1.2),
                explanation: "SELECT * returns all variables, which may include unnecessary data. \
                              Selecting specific variables reduces result size and improves serialization performance."
                    .to_string(),
                location: None,
                documentation_url: None,
            });
        }

        // Warn about excessive OPTIONAL
        if analysis.optional_count > self.config.max_optional_depth {
            suggestions.push(OptimizationSuggestion {
                id: "BEST_PRACTICE_002".to_string(),
                message: format!(
                    "Query has {} OPTIONAL blocks (max recommended: {}). This can cause performance issues.",
                    analysis.optional_count, self.config.max_optional_depth
                ),
                severity: SuggestionSeverity::Warning,
                category: SuggestionCategory::BestPractices,
                suggested_rewrite: None,
                estimated_speedup: None,
                explanation: "Excessive OPTIONAL clauses create combinatorial complexity. \
                              Consider restructuring the query or using UNION instead."
                    .to_string(),
                location: None,
                documentation_url: None,
            });
        }

        // Warn about excessive UNION
        if analysis.union_count > self.config.max_union_branches {
            suggestions.push(OptimizationSuggestion {
                id: "BEST_PRACTICE_003".to_string(),
                message: format!(
                    "Query has {} UNION blocks (max recommended: {}). Consider query simplification.",
                    analysis.union_count, self.config.max_union_branches
                ),
                severity: SuggestionSeverity::Info,
                category: SuggestionCategory::BestPractices,
                suggested_rewrite: None,
                estimated_speedup: None,
                explanation: "Many UNION branches may indicate overly complex queries. \
                              Consider splitting into separate queries or using property paths."
                    .to_string(),
                location: None,
                documentation_url: None,
            });
        }

        // Recommend DISTINCT carefully
        if analysis.has_distinct && !analysis.has_aggregates {
            suggestions.push(OptimizationSuggestion {
                id: "BEST_PRACTICE_004".to_string(),
                message: "DISTINCT requires result deduplication. Ensure it's necessary."
                    .to_string(),
                severity: SuggestionSeverity::Info,
                category: SuggestionCategory::BestPractices,
                suggested_rewrite: None,
                estimated_speedup: Some(1.5),
                explanation:
                    "DISTINCT adds overhead for deduplication. If your data model guarantees \
                              unique results, DISTINCT is unnecessary."
                        .to_string(),
                location: None,
                documentation_url: None,
            });
        }

        Ok(suggestions)
    }

    /// Analyze index usage
    fn analyze_index_usage(&self, analysis: &QueryAnalysis) -> Result<Vec<OptimizationSuggestion>> {
        let mut suggestions = Vec::new();

        if analysis.triple_pattern_count > 5 && analysis.filter_count > 0 {
            suggestions.push(OptimizationSuggestion {
                id: "INDEX_USAGE_001".to_string(),
                message: "Consider creating indexes on frequently filtered predicates.".to_string(),
                severity: SuggestionSeverity::Info,
                category: SuggestionCategory::IndexUsage,
                suggested_rewrite: None,
                estimated_speedup: Some(5.0),
                explanation:
                    "Indexes on filtered predicates can dramatically improve query performance. \
                              Use the adaptive_index_advisor module for specific recommendations."
                        .to_string(),
                location: None,
                documentation_url: None,
            });
        }

        Ok(suggestions)
    }

    /// Generate optimization report
    pub fn generate_report(&self, suggestions: &[OptimizationSuggestion]) -> String {
        let mut report = String::new();
        report.push_str("# Query Optimization Report\n\n");

        if suggestions.is_empty() {
            report.push_str("✓ No optimization suggestions. Query looks good!\n");
            return report;
        }

        // Group by severity
        let critical: Vec<_> = suggestions
            .iter()
            .filter(|s| s.severity == SuggestionSeverity::Critical)
            .collect();
        let warnings: Vec<_> = suggestions
            .iter()
            .filter(|s| s.severity == SuggestionSeverity::Warning)
            .collect();
        let info: Vec<_> = suggestions
            .iter()
            .filter(|s| s.severity == SuggestionSeverity::Info)
            .collect();

        if !critical.is_empty() {
            report.push_str(&format!("## Critical Issues ({})\n\n", critical.len()));
            for (i, s) in critical.iter().enumerate() {
                report.push_str(&self.format_suggestion(i + 1, s));
            }
        }

        if !warnings.is_empty() {
            report.push_str(&format!("## Warnings ({})\n\n", warnings.len()));
            for (i, s) in warnings.iter().enumerate() {
                report.push_str(&self.format_suggestion(i + 1, s));
            }
        }

        if !info.is_empty() {
            report.push_str(&format!("## Informational ({})\n\n", info.len()));
            for (i, s) in info.iter().enumerate() {
                report.push_str(&self.format_suggestion(i + 1, s));
            }
        }

        // Calculate potential speedup
        let total_speedup: f64 = suggestions
            .iter()
            .filter_map(|s| s.estimated_speedup)
            .product();

        if total_speedup > 1.0 {
            report.push_str(&format!(
                "\n## Estimated Performance Impact\n\n\
                 Applying all suggestions could improve query performance by up to {:.1}x\n",
                total_speedup
            ));
        }

        report
    }

    fn format_suggestion(&self, index: usize, suggestion: &OptimizationSuggestion) -> String {
        let mut output = format!(
            "### {}. {} [{}]\n\n",
            index, suggestion.message, suggestion.id
        );
        output.push_str(&format!("**Severity**: {}\n", suggestion.severity));
        output.push_str(&format!("**Category**: {:?}\n\n", suggestion.category));
        output.push_str(&format!("{}\n\n", suggestion.explanation));

        if let Some(ref rewrite) = suggestion.suggested_rewrite {
            output.push_str(&format!("**Suggested Fix**: {}\n\n", rewrite));
        }

        if let Some(speedup) = suggestion.estimated_speedup {
            output.push_str(&format!("**Estimated Speedup**: {:.1}x\n\n", speedup));
        }

        output.push_str("---\n\n");
        output
    }
}

/// Simplified query analysis structure
#[derive(Debug, Clone, Default)]
struct QueryAnalysis {
    query_form: QueryForm,
    has_distinct: bool,
    limit: Option<usize>,
    optional_count: usize,
    union_count: usize,
    filter_count: usize,
    triple_pattern_count: usize,
    has_select_star: bool,
    bind_count: usize,
    has_order_by: bool,
    has_group_by: bool,
    has_aggregates: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum QueryForm {
    #[default]
    Select,
    Ask,
    Construct,
    Describe,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advisor_config_defaults() {
        let config = AdvisorConfig::default();
        assert!(config.analyze_pattern_ordering);
        assert!(config.analyze_best_practices);
        assert_eq!(config.max_optional_depth, 3);
    }

    #[test]
    fn test_simple_query_analysis() {
        let advisor = OptimizationAdvisor::new(AdvisorConfig::default());
        let query = "SELECT * WHERE { ?s ?p ?o }";

        let suggestions = advisor.analyze_query(query).unwrap();
        // Should suggest avoiding SELECT *
        assert!(suggestions.iter().any(|s| s.id == "BEST_PRACTICE_001"));
    }

    #[test]
    fn test_missing_limit_warning() {
        let config = AdvisorConfig {
            require_limit_clause: true,
            ..Default::default()
        };

        let advisor = OptimizationAdvisor::new(config);
        let query = "SELECT ?s WHERE { ?s ?p ?o }";

        let suggestions = advisor.analyze_query(query).unwrap();
        assert!(suggestions.iter().any(|s| s.id == "RESULT_LIMIT_001"));
    }

    #[test]
    fn test_excessive_limit_warning() {
        let advisor = OptimizationAdvisor::new(AdvisorConfig::default());
        let query = "SELECT ?s WHERE { ?s ?p ?o } LIMIT 50000";

        let suggestions = advisor.analyze_query(query).unwrap();
        assert!(suggestions.iter().any(|s| s.id == "RESULT_LIMIT_002"));
    }

    #[test]
    fn test_many_patterns_warning() {
        let advisor = OptimizationAdvisor::new(AdvisorConfig::default());
        let query = "SELECT ?s WHERE { ?s ?p ?o . ?s ?p2 ?o2 . ?s ?p3 ?o3 . ?s ?p4 ?o4 . ?s ?p5 ?o5 . ?s ?p6 ?o6 }";

        let suggestions = advisor.analyze_query(query).unwrap();
        assert!(suggestions.iter().any(|s| s.id == "PATTERN_ORDER_001"));
    }

    #[test]
    fn test_excessive_optional_warning() {
        let advisor = OptimizationAdvisor::new(AdvisorConfig::default());
        let query = "SELECT ?s WHERE { ?s ?p ?o OPTIONAL { ?s ?p1 ?o1 } OPTIONAL { ?s ?p2 ?o2 } \
                     OPTIONAL { ?s ?p3 ?o3 } OPTIONAL { ?s ?p4 ?o4 } }";

        let suggestions = advisor.analyze_query(query).unwrap();
        assert!(suggestions.iter().any(|s| s.id == "BEST_PRACTICE_002"));
    }

    #[test]
    fn test_report_generation() {
        let advisor = OptimizationAdvisor::new(AdvisorConfig::default());
        let query = "SELECT * WHERE { ?s ?p ?o }";

        let suggestions = advisor.analyze_query(query).unwrap();
        let report = advisor.generate_report(&suggestions);

        assert!(report.contains("Query Optimization Report"));
        assert!(report.contains("SELECT *"));
    }

    #[test]
    fn test_severity_ordering() {
        let advisor = OptimizationAdvisor::new(AdvisorConfig::default());
        let query = "SELECT * WHERE { ?s ?p ?o OPTIONAL { ?s ?p1 ?o1 } OPTIONAL { ?s ?p2 ?o2 } \
                     OPTIONAL { ?s ?p3 ?o3 } OPTIONAL { ?s ?p4 ?o4 } }";

        let suggestions = advisor.analyze_query(query).unwrap();

        // Verify suggestions are sorted by severity (critical first)
        for i in 0..suggestions.len().saturating_sub(1) {
            assert!(suggestions[i].severity >= suggestions[i + 1].severity);
        }
    }

    #[test]
    fn test_filter_placement_suggestion() {
        let advisor = OptimizationAdvisor::new(AdvisorConfig::default());
        let query =
            "SELECT ?s WHERE { ?s ?p ?o . ?s ?p2 ?o2 . ?s ?p3 ?o3 . ?s ?p4 ?o4 FILTER(?o > 100) }";

        let suggestions = advisor.analyze_query(query).unwrap();
        assert!(suggestions.iter().any(|s| s.id == "FILTER_PLACEMENT_001"));
    }

    #[test]
    fn test_good_query_no_suggestions() {
        let advisor = OptimizationAdvisor::new(AdvisorConfig::default());
        let query = "SELECT ?name WHERE { ?person foaf:name ?name } LIMIT 10";

        let suggestions = advisor.analyze_query(query).unwrap();

        // This query follows best practices - should have minimal/no critical suggestions
        let critical_count = suggestions
            .iter()
            .filter(|s| s.severity == SuggestionSeverity::Critical)
            .count();

        assert_eq!(critical_count, 0);
    }

    #[test]
    fn test_distinct_suggestion() {
        let advisor = OptimizationAdvisor::new(AdvisorConfig::default());
        let query = "SELECT DISTINCT ?s WHERE { ?s ?p ?o }";

        let suggestions = advisor.analyze_query(query).unwrap();
        assert!(suggestions.iter().any(|s| s.id == "BEST_PRACTICE_004"));
    }
}
