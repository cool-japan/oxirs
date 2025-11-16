//! Query Refinement System
//!
//! Helps users refine and improve their queries through interactive suggestions.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Query refinement suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementSuggestion {
    /// Suggestion type
    pub suggestion_type: RefinementType,
    /// Original query part
    pub original: String,
    /// Suggested improvement
    pub suggested: String,
    /// Reason for suggestion
    pub reason: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Example usage
    pub example: Option<String>,
}

/// Types of refinement suggestions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RefinementType {
    /// Add filters to narrow results
    AddFilter,
    /// Add sorting/ordering
    AddOrdering,
    /// Limit result count
    AddLimit,
    /// Add grouping/aggregation
    AddAggregation,
    /// Simplify complex query
    Simplify,
    /// Add constraints
    AddConstraints,
    /// Improve performance
    OptimizePerformance,
    /// Clarify ambiguity
    ClarifyAmbiguity,
    /// Add joins
    AddJoin,
    /// Use better predicates
    BetterPredicates,
}

/// Query analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryAnalysis {
    /// Original query
    pub original_query: String,
    /// Query complexity (1-10)
    pub complexity: u8,
    /// Estimated result count
    pub estimated_results: Option<usize>,
    /// Performance issues detected
    pub performance_issues: Vec<String>,
    /// Ambiguities detected
    pub ambiguities: Vec<String>,
    /// Missing optimizations
    pub missing_optimizations: Vec<String>,
}

/// Query refinement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementConfig {
    /// Maximum number of suggestions
    pub max_suggestions: usize,
    /// Minimum confidence threshold
    pub min_confidence: f32,
    /// Enable performance suggestions
    pub suggest_performance: bool,
    /// Enable clarity suggestions
    pub suggest_clarity: bool,
    /// Enable feature suggestions
    pub suggest_features: bool,
}

impl Default for RefinementConfig {
    fn default() -> Self {
        Self {
            max_suggestions: 5,
            min_confidence: 0.6,
            suggest_performance: true,
            suggest_clarity: true,
            suggest_features: true,
        }
    }
}

/// Query refiner
pub struct QueryRefiner {
    config: RefinementConfig,
}

impl QueryRefiner {
    /// Create a new query refiner
    pub fn new(config: RefinementConfig) -> Self {
        info!("Initialized query refiner");
        Self { config }
    }

    /// Analyze a query and suggest refinements
    pub fn refine(&self, query: &str) -> Result<Vec<RefinementSuggestion>> {
        debug!("Analyzing query for refinements: {}", query);

        let analysis = self.analyze_query(query)?;
        let mut suggestions = Vec::new();

        // Performance suggestions
        if self.config.suggest_performance {
            suggestions.extend(self.suggest_performance_improvements(&analysis)?);
        }

        // Clarity suggestions
        if self.config.suggest_clarity {
            suggestions.extend(self.suggest_clarity_improvements(&analysis)?);
        }

        // Feature suggestions
        if self.config.suggest_features {
            suggestions.extend(self.suggest_feature_additions(&analysis)?);
        }

        // Filter by confidence
        suggestions.retain(|s| s.confidence >= self.config.min_confidence);

        // Sort by confidence
        suggestions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        // Limit results
        suggestions.truncate(self.config.max_suggestions);

        info!("Generated {} refinement suggestions", suggestions.len());

        Ok(suggestions)
    }

    /// Analyze query for issues
    fn analyze_query(&self, query: &str) -> Result<QueryAnalysis> {
        let lowercase = query.to_lowercase();

        let mut performance_issues = Vec::new();
        let mut ambiguities = Vec::new();
        let mut missing_optimizations = Vec::new();

        // Check for missing LIMIT
        if !lowercase.contains("limit") {
            missing_optimizations
                .push("No LIMIT clause - query may return too many results".to_string());
        }

        // Check for missing ORDER BY
        if (lowercase.contains("list") || lowercase.contains("show"))
            && !lowercase.contains("order")
            && !lowercase.contains("sort")
        {
            missing_optimizations
                .push("Results not ordered - consider adding ORDER BY".to_string());
        }

        // Check for vague terms
        if lowercase.contains("thing")
            || lowercase.contains("stuff")
            || lowercase.contains("something")
        {
            ambiguities.push("Vague terms detected - be more specific".to_string());
        }

        // Check for performance issues
        if lowercase.contains("all") && !lowercase.contains("limit") {
            performance_issues.push("Requesting 'all' without LIMIT may be slow".to_string());
        }

        // Estimate complexity
        let complexity = self.estimate_complexity(query);

        Ok(QueryAnalysis {
            original_query: query.to_string(),
            complexity,
            estimated_results: None,
            performance_issues,
            ambiguities,
            missing_optimizations,
        })
    }

    /// Estimate query complexity
    fn estimate_complexity(&self, query: &str) -> u8 {
        let mut complexity = 1u8;

        // More words = more complex
        complexity += (query.split_whitespace().count() / 10).min(3) as u8;

        // Complex keywords
        let complex_keywords = [
            "aggregate",
            "group",
            "having",
            "union",
            "optional",
            "filter",
            "minus",
        ];
        for keyword in &complex_keywords {
            if query.to_lowercase().contains(keyword) {
                complexity += 1;
            }
        }

        complexity.min(10)
    }

    /// Suggest performance improvements
    fn suggest_performance_improvements(
        &self,
        analysis: &QueryAnalysis,
    ) -> Result<Vec<RefinementSuggestion>> {
        let mut suggestions = Vec::new();

        // Suggest LIMIT
        if !analysis.original_query.to_lowercase().contains("limit") {
            suggestions.push(RefinementSuggestion {
                suggestion_type: RefinementType::AddLimit,
                original: analysis.original_query.clone(),
                suggested: format!("{} LIMIT 100", analysis.original_query.trim()),
                reason: "Add LIMIT to prevent returning too many results".to_string(),
                confidence: 0.9,
                example: Some("SELECT * WHERE { ?s ?p ?o } LIMIT 100".to_string()),
            });
        }

        // Suggest indexing hints
        for issue in &analysis.performance_issues {
            if issue.contains("slow") {
                suggestions.push(RefinementSuggestion {
                    suggestion_type: RefinementType::OptimizePerformance,
                    original: analysis.original_query.clone(),
                    suggested: "Consider adding filters to narrow the search".to_string(),
                    reason: issue.clone(),
                    confidence: 0.75,
                    example: Some(
                        "SELECT * WHERE { ?s rdf:type :Movie . ?s :year 2023 } LIMIT 100"
                            .to_string(),
                    ),
                });
            }
        }

        Ok(suggestions)
    }

    /// Suggest clarity improvements
    fn suggest_clarity_improvements(
        &self,
        analysis: &QueryAnalysis,
    ) -> Result<Vec<RefinementSuggestion>> {
        let mut suggestions = Vec::new();

        for ambiguity in &analysis.ambiguities {
            suggestions.push(RefinementSuggestion {
                suggestion_type: RefinementType::ClarifyAmbiguity,
                original: analysis.original_query.clone(),
                suggested: "Replace vague terms with specific entity types or properties"
                    .to_string(),
                reason: ambiguity.clone(),
                confidence: 0.8,
                example: Some(
                    "Instead of 'show me things', use 'show me movies' or 'show me people'"
                        .to_string(),
                ),
            });
        }

        // Suggest being more specific
        if analysis.original_query.len() < 20 {
            suggestions.push(RefinementSuggestion {
                suggestion_type: RefinementType::ClarifyAmbiguity,
                original: analysis.original_query.clone(),
                suggested: "Add more details to your query".to_string(),
                reason: "Short query may be too vague".to_string(),
                confidence: 0.65,
                example: Some("Instead of 'movies', try 'movies released in 2023'".to_string()),
            });
        }

        Ok(suggestions)
    }

    /// Suggest feature additions
    fn suggest_feature_additions(
        &self,
        analysis: &QueryAnalysis,
    ) -> Result<Vec<RefinementSuggestion>> {
        let mut suggestions = Vec::new();
        let lowercase = analysis.original_query.to_lowercase();

        // Suggest ORDER BY
        if !lowercase.contains("order") && !lowercase.contains("sort") {
            suggestions.push(RefinementSuggestion {
                suggestion_type: RefinementType::AddOrdering,
                original: analysis.original_query.clone(),
                suggested: "Add ORDER BY to sort results".to_string(),
                reason: "Ordered results are easier to browse".to_string(),
                confidence: 0.7,
                example: Some("SELECT * WHERE { ?s ?p ?o } ORDER BY ?s LIMIT 100".to_string()),
            });
        }

        // Suggest aggregation
        if (lowercase.contains("list") || lowercase.contains("show"))
            && !lowercase.contains("count")
        {
            suggestions.push(RefinementSuggestion {
                suggestion_type: RefinementType::AddAggregation,
                original: analysis.original_query.clone(),
                suggested: "Consider counting results instead of listing all".to_string(),
                reason: "Aggregation provides quick overview".to_string(),
                confidence: 0.6,
                example: Some(
                    "SELECT (COUNT(?s) as ?count) WHERE { ?s rdf:type :Movie }".to_string(),
                ),
            });
        }

        // Suggest filters
        if lowercase.contains("all") || lowercase.contains("everything") {
            suggestions.push(RefinementSuggestion {
                suggestion_type: RefinementType::AddFilter,
                original: analysis.original_query.clone(),
                suggested: "Add filters to narrow down results".to_string(),
                reason: "Filtering improves result relevance".to_string(),
                confidence: 0.75,
                example: Some("FILTER (?year > 2020)".to_string()),
            });
        }

        Ok(suggestions)
    }

    /// Apply a refinement suggestion
    pub fn apply_suggestion(
        &self,
        query: &str,
        suggestion: &RefinementSuggestion,
    ) -> Result<String> {
        match suggestion.suggestion_type {
            RefinementType::AddLimit => {
                if !query.to_lowercase().contains("limit") {
                    Ok(format!("{} LIMIT 100", query.trim()))
                } else {
                    Ok(query.to_string())
                }
            }
            RefinementType::AddOrdering => {
                if !query.to_lowercase().contains("order by") {
                    // Insert ORDER BY before LIMIT if present
                    if let Some(limit_pos) = query.to_lowercase().find("limit") {
                        let mut refined = query.to_string();
                        refined.insert_str(limit_pos, "ORDER BY ?s ");
                        Ok(refined)
                    } else {
                        Ok(format!("{} ORDER BY ?s", query.trim()))
                    }
                } else {
                    Ok(query.to_string())
                }
            }
            _ => {
                // For other types, return the suggested query
                Ok(suggestion.suggested.clone())
            }
        }
    }

    /// Get interactive refinement session
    pub fn start_refinement_session(&self, initial_query: &str) -> RefinementSession {
        RefinementSession {
            original_query: initial_query.to_string(),
            current_query: initial_query.to_string(),
            applied_suggestions: Vec::new(),
            iteration: 0,
        }
    }
}

/// Interactive refinement session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementSession {
    /// Original query
    pub original_query: String,
    /// Current refined query
    pub current_query: String,
    /// Applied suggestions
    pub applied_suggestions: Vec<RefinementSuggestion>,
    /// Iteration count
    pub iteration: usize,
}

impl RefinementSession {
    /// Apply a suggestion to the session
    pub fn apply(&mut self, suggestion: RefinementSuggestion, refined_query: String) {
        self.current_query = refined_query;
        self.applied_suggestions.push(suggestion);
        self.iteration += 1;
    }

    /// Get refinement summary
    pub fn summary(&self) -> String {
        format!(
            "Refined query in {} iterations, applied {} suggestions",
            self.iteration,
            self.applied_suggestions.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_analysis() {
        let refiner = QueryRefiner::new(RefinementConfig::default());
        let analysis = refiner.analyze_query("show me all things").unwrap();

        assert!(!analysis.performance_issues.is_empty());
        assert!(!analysis.ambiguities.is_empty());
    }

    #[test]
    fn test_limit_suggestion() {
        let refiner = QueryRefiner::new(RefinementConfig::default());
        let suggestions = refiner.refine("SELECT * WHERE { ?s ?p ?o }").unwrap();

        assert!(suggestions
            .iter()
            .any(|s| s.suggestion_type == RefinementType::AddLimit));
    }

    #[test]
    fn test_apply_limit() {
        let refiner = QueryRefiner::new(RefinementConfig::default());
        let suggestion = RefinementSuggestion {
            suggestion_type: RefinementType::AddLimit,
            original: "SELECT * WHERE { ?s ?p ?o }".to_string(),
            suggested: "SELECT * WHERE { ?s ?p ?o } LIMIT 100".to_string(),
            reason: "Add limit".to_string(),
            confidence: 0.9,
            example: None,
        };

        let refined = refiner
            .apply_suggestion("SELECT * WHERE { ?s ?p ?o }", &suggestion)
            .unwrap();
        assert!(refined.contains("LIMIT"));
    }

    #[test]
    fn test_refinement_session() {
        let refiner = QueryRefiner::new(RefinementConfig::default());
        let mut session = refiner.start_refinement_session("show me movies");

        let suggestion = RefinementSuggestion {
            suggestion_type: RefinementType::AddLimit,
            original: "show me movies".to_string(),
            suggested: "show me movies LIMIT 100".to_string(),
            reason: "Add limit".to_string(),
            confidence: 0.9,
            example: None,
        };

        session.apply(suggestion, "show me movies LIMIT 100".to_string());
        assert_eq!(session.iteration, 1);
        assert_eq!(session.applied_suggestions.len(), 1);
    }
}
