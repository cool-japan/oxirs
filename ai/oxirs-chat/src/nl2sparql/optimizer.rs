//! SPARQL query optimization component with rule-based rewriting capabilities.

use anyhow::Result;
use regex::Regex;
use tracing::info;

use super::types::{OptimizationHint, OptimizationHintType};

/// An individual query-rewriting optimization rule.
struct OptimizationRule {
    name: String,
    pattern: Regex,
    replacement: String,
    description: String,
    estimated_improvement: f32,
}

/// SPARQL optimization component with query rewriting capabilities
pub struct SPARQLOptimizer {
    optimization_rules: Vec<OptimizationRule>,
}

impl Default for SPARQLOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl SPARQLOptimizer {
    pub fn new() -> Self {
        let optimization_rules = vec![
            OptimizationRule {
                name: "redundant_distinct".to_string(),
                pattern: Regex::new(r"(?i)SELECT\s+DISTINCT\s+DISTINCT")
                    .expect("hardcoded regex should be valid"),
                replacement: "SELECT DISTINCT".to_string(),
                description: "Remove redundant DISTINCT clauses".to_string(),
                estimated_improvement: 0.1,
            },
            OptimizationRule {
                name: "limit_optimization".to_string(),
                pattern: Regex::new(r"(?i)ORDER\s+BY\s+[^}]+}\s*$")
                    .expect("hardcoded regex should be valid"),
                replacement: "$0 LIMIT 1000".to_string(),
                description: "Add default LIMIT for safety".to_string(),
                estimated_improvement: 0.3,
            },
        ];

        Self { optimization_rules }
    }

    pub fn optimize(&self, query: &str) -> Result<(String, Vec<OptimizationHint>)> {
        let mut optimized_query = query.to_string();
        let mut hints = Vec::new();

        for rule in &self.optimization_rules {
            if rule.pattern.is_match(&optimized_query) {
                optimized_query = rule
                    .pattern
                    .replace_all(&optimized_query, &rule.replacement)
                    .to_string();
                hints.push(OptimizationHint {
                    hint_type: OptimizationHintType::SimplifyExpression,
                    description: rule.description.clone(),
                    estimated_improvement: Some(rule.estimated_improvement),
                });
            }
        }

        let additional_hints = self.analyze_query_structure(&optimized_query)?;
        hints.extend(additional_hints);

        optimized_query = self.rewrite_query_patterns(optimized_query)?;

        Ok((optimized_query, hints))
    }

    fn analyze_query_structure(&self, query: &str) -> Result<Vec<OptimizationHint>> {
        let mut hints = Vec::new();
        let query_upper = query.to_uppercase();

        if !query_upper.contains("LIMIT") && query_upper.contains("SELECT") {
            hints.push(OptimizationHint {
                hint_type: OptimizationHintType::AddIndex,
                description: "Consider adding LIMIT clause to prevent large result sets"
                    .to_string(),
                estimated_improvement: Some(0.5),
            });
        }

        if query_upper.contains("OPTIONAL") && query_upper.contains("FILTER") {
            let optional_pos = query_upper.find("OPTIONAL").unwrap_or(0);
            let filter_pos = query_upper.find("FILTER").unwrap_or(0);

            if filter_pos > optional_pos {
                hints.push(OptimizationHint {
                    hint_type: OptimizationHintType::ReorderTriples,
                    description:
                        "Consider moving FILTER clauses before OPTIONAL for better performance"
                            .to_string(),
                    estimated_improvement: Some(0.3),
                });
            }
        }

        let triple_count = query_upper.matches(" . ").count();
        if triple_count > 5 && !query_upper.contains("FILTER") {
            hints.push(OptimizationHint {
                hint_type: OptimizationHintType::UseFilter,
                description:
                    "Multiple triple patterns without filters may create Cartesian products"
                        .to_string(),
                estimated_improvement: Some(0.7),
            });
        }

        Ok(hints)
    }

    fn rewrite_query_patterns(&self, query: String) -> Result<String> {
        let mut rewritten = query;

        let union_pattern = Regex::new(r"(?i)\{\s*(.+?)\s*\}\s*UNION\s*\{\s*(.+?)\s*\}")?;
        if union_pattern.is_match(&rewritten) {
            info!("Detected UNION pattern that could potentially be optimized");
        }

        let filter_pattern =
            Regex::new(r#"(?i)FILTER\s*\(\s*regex\s*\(\s*\?(\w+)\s*,\s*"([^"]+)"\s*\)\s*\)"#)?;
        rewritten = filter_pattern
            .replace_all(&rewritten, |caps: &regex::Captures| {
                format!(
                    "FILTER(CONTAINS(LCASE(?{}), LCASE(\"{}\")))",
                    &caps[1], &caps[2]
                )
            })
            .to_string();

        Ok(rewritten)
    }

    /// Get optimization recommendations for a query
    pub fn get_recommendations(&self, query: &str) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();
        let query_upper = query.to_uppercase();

        if !query_upper.contains("PREFIX") && query.contains(':') {
            recommendations.push("Add PREFIX declarations for better readability".to_string());
        }

        if query.len() > 500 && !query_upper.contains("LIMIT") {
            recommendations.push("Add LIMIT clause for large queries".to_string());
        }

        if query_upper.contains("SELECT *") {
            recommendations
                .push("Select specific variables instead of * for better performance".to_string());
        }

        if query_upper.matches("OPTIONAL").count() > 3 {
            recommendations.push("Consider restructuring multiple OPTIONAL clauses".to_string());
        }

        Ok(recommendations)
    }
}
