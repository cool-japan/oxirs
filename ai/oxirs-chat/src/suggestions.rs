//! Query Suggestions System
//!
//! Provides intelligent query suggestions based on conversation context,
//! schema information, and user patterns.

use anyhow::Result;
use oxirs_core::Store;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

/// Query suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuerySuggestion {
    /// Suggestion text
    pub text: String,
    /// Suggestion type
    pub suggestion_type: SuggestionType,
    /// Relevance score (0.0 - 1.0)
    pub relevance: f32,
    /// Category
    pub category: String,
    /// Example SPARQL query (if applicable)
    pub sparql_example: Option<String>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Type of suggestion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestionType {
    /// Complete query suggestion
    Query,
    /// Query continuation
    Continuation,
    /// Entity suggestion
    Entity,
    /// Property/predicate suggestion
    Property,
    /// Filter suggestion
    Filter,
    /// Aggregation suggestion
    Aggregation,
    /// Follow-up question
    FollowUp,
}

/// Suggestion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestionConfig {
    /// Maximum number of suggestions to return
    pub max_suggestions: usize,
    /// Minimum relevance score
    pub min_relevance: f32,
    /// Enable schema-based suggestions
    pub enable_schema_suggestions: bool,
    /// Enable pattern-based suggestions
    pub enable_pattern_suggestions: bool,
    /// Enable history-based suggestions
    pub enable_history_suggestions: bool,
    /// Enable follow-up suggestions
    pub enable_followup_suggestions: bool,
}

impl Default for SuggestionConfig {
    fn default() -> Self {
        Self {
            max_suggestions: 5,
            min_relevance: 0.3,
            enable_schema_suggestions: true,
            enable_pattern_suggestions: true,
            enable_history_suggestions: true,
            enable_followup_suggestions: true,
        }
    }
}

/// Query suggestion engine
pub struct SuggestionEngine {
    config: SuggestionConfig,
    store: Arc<dyn Store>,
    query_patterns: Vec<QueryPattern>,
    user_history: Vec<String>,
}

/// Predefined query pattern
#[derive(Debug, Clone)]
struct QueryPattern {
    template: String,
    category: String,
    keywords: Vec<String>,
    sparql_template: Option<String>,
}

impl SuggestionEngine {
    /// Create a new suggestion engine
    pub fn new(config: SuggestionConfig, store: Arc<dyn Store>) -> Result<Self> {
        let query_patterns = Self::load_query_patterns();

        info!(
            "Initialized suggestion engine with {} patterns",
            query_patterns.len()
        );

        Ok(Self {
            config,
            store,
            query_patterns,
            user_history: Vec::new(),
        })
    }

    /// Load predefined query patterns
    fn load_query_patterns() -> Vec<QueryPattern> {
        vec![
            QueryPattern {
                template: "Show me all {entities} from {time_range}".to_string(),
                category: "Exploration".to_string(),
                keywords: vec!["show", "all", "list"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
                sparql_template: Some(
                    "SELECT * WHERE { ?s a ?type . FILTER(?time > {start}) }".to_string(),
                ),
            },
            QueryPattern {
                template: "Find {entities} related to {topic}".to_string(),
                category: "Search".to_string(),
                keywords: vec!["find", "search", "related"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
                sparql_template: Some("SELECT * WHERE { ?s ?p {topic} }".to_string()),
            },
            QueryPattern {
                template: "Count the total number of {entities}".to_string(),
                category: "Analytics".to_string(),
                keywords: vec!["count", "total", "number"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
                sparql_template: Some(
                    "SELECT (COUNT(?s) as ?count) WHERE { ?s a ?type }".to_string(),
                ),
            },
            QueryPattern {
                template: "What are the properties of {entity}?".to_string(),
                category: "Exploration".to_string(),
                keywords: vec!["properties", "attributes", "fields"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
                sparql_template: Some("SELECT ?p ?o WHERE { {entity} ?p ?o }".to_string()),
            },
            QueryPattern {
                template: "Show me {entities} ordered by {property}".to_string(),
                category: "Sorting".to_string(),
                keywords: vec!["sorted", "ordered", "ranked"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
                sparql_template: Some(
                    "SELECT * WHERE { ?s a ?type } ORDER BY ?property".to_string(),
                ),
            },
            QueryPattern {
                template: "Which {entities} have {property} equal to {value}?".to_string(),
                category: "Filter".to_string(),
                keywords: vec!["which", "where", "filter"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
                sparql_template: Some("SELECT * WHERE { ?s ?p {value} }".to_string()),
            },
            QueryPattern {
                template: "Compare {entity1} and {entity2}".to_string(),
                category: "Comparison".to_string(),
                keywords: vec!["compare", "difference", "versus"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
                sparql_template: None,
            },
            QueryPattern {
                template: "What is the average {property} of {entities}?".to_string(),
                category: "Analytics".to_string(),
                keywords: vec!["average", "mean", "aggregate"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
                sparql_template: Some(
                    "SELECT (AVG(?value) as ?avg) WHERE { ?s ?p ?value }".to_string(),
                ),
            },
        ]
    }

    /// Generate suggestions based on partial input
    pub fn suggest(
        &self,
        partial_query: &str,
        context: &SuggestionContext,
    ) -> Result<Vec<QuerySuggestion>> {
        debug!("Generating suggestions for: {}", partial_query);

        let mut suggestions = Vec::new();

        // Pattern-based suggestions
        if self.config.enable_pattern_suggestions {
            suggestions.extend(self.pattern_based_suggestions(partial_query)?);
        }

        // Schema-based suggestions
        if self.config.enable_schema_suggestions {
            suggestions.extend(self.schema_based_suggestions(partial_query, context)?);
        }

        // History-based suggestions
        if self.config.enable_history_suggestions {
            suggestions.extend(self.history_based_suggestions(partial_query)?);
        }

        // Follow-up suggestions
        if self.config.enable_followup_suggestions && !context.last_query.is_empty() {
            suggestions.extend(self.followup_suggestions(&context.last_query)?);
        }

        // Filter by relevance
        suggestions.retain(|s| s.relevance >= self.config.min_relevance);

        // Sort by relevance
        suggestions.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit results
        suggestions.truncate(self.config.max_suggestions);

        info!("Generated {} suggestions", suggestions.len());

        Ok(suggestions)
    }

    /// Generate pattern-based suggestions
    fn pattern_based_suggestions(&self, partial: &str) -> Result<Vec<QuerySuggestion>> {
        let mut suggestions = Vec::new();
        let lowercase = partial.to_lowercase();

        for pattern in &self.query_patterns {
            // Calculate relevance based on keyword matching
            let mut relevance: f32 = 0.0;
            for keyword in &pattern.keywords {
                if lowercase.contains(keyword) {
                    relevance += 0.3;
                }
            }

            if relevance > 0.0 {
                suggestions.push(QuerySuggestion {
                    text: pattern.template.clone(),
                    suggestion_type: SuggestionType::Query,
                    relevance: relevance.min(1.0),
                    category: pattern.category.clone(),
                    sparql_example: pattern.sparql_template.clone(),
                    metadata: HashMap::new(),
                });
            }
        }

        Ok(suggestions)
    }

    /// Generate schema-based suggestions
    fn schema_based_suggestions(
        &self,
        partial: &str,
        context: &SuggestionContext,
    ) -> Result<Vec<QuerySuggestion>> {
        let mut suggestions = Vec::new();

        // Suggest entity types from schema
        if let Some(classes) = &context.available_classes {
            for class in classes {
                if class.to_lowercase().contains(&partial.to_lowercase()) {
                    suggestions.push(QuerySuggestion {
                        text: format!("Show me all {}", class),
                        suggestion_type: SuggestionType::Entity,
                        relevance: 0.8,
                        category: "Entity Type".to_string(),
                        sparql_example: Some(format!("SELECT * WHERE {{ ?s a {} }}", class)),
                        metadata: HashMap::new(),
                    });
                }
            }
        }

        // Suggest properties
        if let Some(properties) = &context.available_properties {
            for property in properties {
                if property.to_lowercase().contains(&partial.to_lowercase()) {
                    suggestions.push(QuerySuggestion {
                        text: format!("Filter by {}", property),
                        suggestion_type: SuggestionType::Property,
                        relevance: 0.7,
                        category: "Property".to_string(),
                        sparql_example: None,
                        metadata: HashMap::new(),
                    });
                }
            }
        }

        Ok(suggestions)
    }

    /// Generate history-based suggestions
    fn history_based_suggestions(&self, partial: &str) -> Result<Vec<QuerySuggestion>> {
        let mut suggestions = Vec::new();

        for historical_query in &self.user_history {
            if historical_query
                .to_lowercase()
                .starts_with(&partial.to_lowercase())
            {
                suggestions.push(QuerySuggestion {
                    text: historical_query.clone(),
                    suggestion_type: SuggestionType::Query,
                    relevance: 0.6,
                    category: "Recent".to_string(),
                    sparql_example: None,
                    metadata: [("source".to_string(), "history".to_string())]
                        .into_iter()
                        .collect(),
                });
            }
        }

        Ok(suggestions)
    }

    /// Generate follow-up suggestions
    fn followup_suggestions(&self, _last_query: &str) -> Result<Vec<QuerySuggestion>> {
        let suggestions = vec![
            // Suggest refinements
            QuerySuggestion {
                text: "Refine the previous query".to_string(),
                suggestion_type: SuggestionType::FollowUp,
                relevance: 0.7,
                category: "Follow-up".to_string(),
                sparql_example: None,
                metadata: HashMap::new(),
            },
            // Suggest related queries
            QuerySuggestion {
                text: "Show me related entities".to_string(),
                suggestion_type: SuggestionType::FollowUp,
                relevance: 0.6,
                category: "Follow-up".to_string(),
                sparql_example: None,
                metadata: HashMap::new(),
            },
            // Suggest aggregations
            QuerySuggestion {
                text: "Count the results".to_string(),
                suggestion_type: SuggestionType::FollowUp,
                relevance: 0.5,
                category: "Follow-up".to_string(),
                sparql_example: None,
                metadata: HashMap::new(),
            },
        ];

        Ok(suggestions)
    }

    /// Add a query to user history
    pub fn add_to_history(&mut self, query: String) {
        self.user_history.push(query);

        // Keep only recent history
        if self.user_history.len() > 100 {
            self.user_history.remove(0);
        }
    }

    /// Clear user history
    pub fn clear_history(&mut self) {
        self.user_history.clear();
    }
}

/// Context for generating suggestions
#[derive(Debug, Clone, Default)]
pub struct SuggestionContext {
    /// Last query executed
    pub last_query: String,
    /// Available entity classes in the dataset
    pub available_classes: Option<Vec<String>>,
    /// Available properties in the dataset
    pub available_properties: Option<Vec<String>>,
    /// Current conversation topic
    pub conversation_topic: Option<String>,
    /// User preferences
    pub user_preferences: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::ConcreteStore;

    #[test]
    fn test_pattern_based_suggestions() {
        let store = Arc::new(ConcreteStore::new().unwrap());
        let engine = SuggestionEngine::new(SuggestionConfig::default(), store).unwrap();

        let context = SuggestionContext::default();
        let suggestions = engine.suggest("show me all", &context).unwrap();

        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.text.contains("Show me all")));
    }

    #[test]
    fn test_history_suggestions() {
        let store = Arc::new(ConcreteStore::new().unwrap());
        let mut engine = SuggestionEngine::new(SuggestionConfig::default(), store).unwrap();

        engine.add_to_history("What movies were released in 2023?".to_string());

        let context = SuggestionContext::default();
        let suggestions = engine.suggest("What", &context).unwrap();

        assert!(suggestions.iter().any(|s| s.text.contains("movies")));
    }

    #[test]
    fn test_followup_suggestions() {
        let store = Arc::new(ConcreteStore::new().unwrap());
        let engine = SuggestionEngine::new(SuggestionConfig::default(), store).unwrap();

        let context = SuggestionContext {
            last_query: "Show me all movies".to_string(),
            ..Default::default()
        };

        let suggestions = engine.suggest("", &context).unwrap();

        assert!(suggestions
            .iter()
            .any(|s| s.suggestion_type == SuggestionType::FollowUp));
    }
}
