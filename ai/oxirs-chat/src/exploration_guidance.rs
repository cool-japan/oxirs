//! Data Exploration Guidance System
//!
//! Provides intelligent guidance for users exploring knowledge graphs and datasets.
//! Suggests relevant queries, identifies interesting patterns, and helps users
//! discover insights in their data.

use anyhow::Result;
use oxirs_core::Store;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use tracing::{debug, info};

use crate::schema_introspection::SchemaIntrospector;

/// Exploration guidance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationConfig {
    /// Maximum suggestions to generate
    pub max_suggestions: usize,
    /// Enable entity recommendations
    pub enable_entity_recommendations: bool,
    /// Enable relationship discovery
    pub enable_relationship_discovery: bool,
    /// Enable pattern suggestions
    pub enable_pattern_suggestions: bool,
    /// Minimum relevance score
    pub min_relevance_score: f32,
}

impl Default for ExplorationConfig {
    fn default() -> Self {
        Self {
            max_suggestions: 10,
            enable_entity_recommendations: true,
            enable_relationship_discovery: true,
            enable_pattern_suggestions: true,
            min_relevance_score: 0.5,
        }
    }
}

/// Exploration suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationSuggestion {
    /// Suggestion ID
    pub id: String,
    /// Suggestion type
    pub suggestion_type: SuggestionType,
    /// Title/summary
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Relevance score (0.0-1.0)
    pub relevance: f32,
    /// Example SPARQL query
    pub example_query: Option<String>,
    /// Related entities/concepts
    pub related_concepts: Vec<String>,
    /// Tags for categorization
    pub tags: Vec<String>,
}

/// Type of exploration suggestion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestionType {
    /// Entity to explore
    Entity,
    /// Relationship to investigate
    Relationship,
    /// Pattern to analyze
    Pattern,
    /// Aggregation/statistics
    Aggregation,
    /// Temporal analysis
    Temporal,
    /// Comparison query
    Comparison,
    /// Navigation path
    Navigation,
}

/// Exploration context
#[derive(Debug, Clone, Default)]
pub struct ExplorationContext {
    /// Previously viewed entities
    pub viewed_entities: HashSet<String>,
    /// Previously executed queries
    pub query_history: Vec<String>,
    /// Current focus area
    pub focus_area: Option<String>,
    /// User interests
    pub interests: Vec<String>,
}

/// Data exploration guidance engine
pub struct ExplorationGuidance {
    config: ExplorationConfig,
    store: Arc<dyn Store>,
    schema_introspector: SchemaIntrospector,
}

impl ExplorationGuidance {
    /// Create new exploration guidance engine
    pub fn new(config: ExplorationConfig, store: Arc<dyn Store>) -> Self {
        info!("Initialized exploration guidance engine");

        let schema_introspector = SchemaIntrospector::new(store.clone());

        Self {
            config,
            store,
            schema_introspector,
        }
    }

    /// Generate exploration suggestions
    pub async fn generate_suggestions(
        &self,
        context: &ExplorationContext,
    ) -> Result<Vec<ExplorationSuggestion>> {
        debug!("Generating exploration suggestions");

        let mut suggestions = Vec::new();

        // Discover schema if not already done
        let schema = self.schema_introspector.discover_schema().await?;

        // Entity recommendations
        if self.config.enable_entity_recommendations {
            let entity_suggestions = self.suggest_entities(&schema, context).await?;
            suggestions.extend(entity_suggestions);
        }

        // Relationship discovery
        if self.config.enable_relationship_discovery {
            let relationship_suggestions = self.suggest_relationships(&schema, context).await?;
            suggestions.extend(relationship_suggestions);
        }

        // Pattern suggestions
        if self.config.enable_pattern_suggestions {
            let pattern_suggestions = self.suggest_patterns(&schema, context).await?;
            suggestions.extend(pattern_suggestions);
        }

        // Filter by relevance
        suggestions.retain(|s| s.relevance >= self.config.min_relevance_score);

        // Sort by relevance
        suggestions.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap());

        // Limit to max suggestions
        suggestions.truncate(self.config.max_suggestions);

        info!("Generated {} exploration suggestions", suggestions.len());

        Ok(suggestions)
    }

    /// Suggest interesting entities to explore
    async fn suggest_entities(
        &self,
        schema: &crate::schema_introspection::DiscoveredSchema,
        context: &ExplorationContext,
    ) -> Result<Vec<ExplorationSuggestion>> {
        let mut suggestions = Vec::new();

        for class in &schema.classes {
            // Skip if already viewed
            if context.viewed_entities.contains(&class.uri) {
                continue;
            }

            let relevance = self.calculate_entity_relevance(class, context);
            let label = class.label.clone().unwrap_or_else(|| class.uri.clone());

            if relevance >= self.config.min_relevance_score {
                suggestions.push(ExplorationSuggestion {
                    id: format!("entity_{}", uuid::Uuid::new_v4()),
                    suggestion_type: SuggestionType::Entity,
                    title: format!("Explore {} entities", label),
                    description: format!(
                        "Investigate {} instances of {}. This class has {} instances.",
                        class.instance_count, label, class.instance_count
                    ),
                    relevance,
                    example_query: Some(format!(
                        "SELECT ?entity WHERE {{ ?entity a <{}> }} LIMIT 10",
                        class.uri
                    )),
                    related_concepts: vec![label],
                    tags: vec!["entity".to_string(), "exploration".to_string()],
                });
            }
        }

        Ok(suggestions)
    }

    /// Suggest interesting relationships
    async fn suggest_relationships(
        &self,
        schema: &crate::schema_introspection::DiscoveredSchema,
        context: &ExplorationContext,
    ) -> Result<Vec<ExplorationSuggestion>> {
        let mut suggestions = Vec::new();

        for property in &schema.properties {
            let relevance = self.calculate_relationship_relevance(property, context);
            let label = property
                .label
                .clone()
                .unwrap_or_else(|| property.uri.clone());

            if relevance >= self.config.min_relevance_score {
                suggestions.push(ExplorationSuggestion {
                    id: format!("relationship_{}", uuid::Uuid::new_v4()),
                    suggestion_type: SuggestionType::Relationship,
                    title: format!("Investigate {} relationship", label),
                    description: format!(
                        "Explore the '{}' relationship connecting entities. This property is used {} times.",
                        label, property.usage_count
                    ),
                    relevance,
                    example_query: Some(format!(
                        "SELECT ?subject ?object WHERE {{ ?subject <{}> ?object }} LIMIT 10",
                        property.uri
                    )),
                    related_concepts: vec![label],
                    tags: vec!["relationship".to_string(), "connection".to_string()],
                });
            }
        }

        Ok(suggestions)
    }

    /// Suggest interesting patterns
    async fn suggest_patterns(
        &self,
        schema: &crate::schema_introspection::DiscoveredSchema,
        _context: &ExplorationContext,
    ) -> Result<Vec<ExplorationSuggestion>> {
        let mut suggestions = Vec::new();

        // Suggest aggregation patterns
        for class in schema.classes.iter().take(3) {
            let label = class.label.clone().unwrap_or_else(|| class.uri.clone());

            suggestions.push(ExplorationSuggestion {
                id: format!("pattern_{}", uuid::Uuid::new_v4()),
                suggestion_type: SuggestionType::Aggregation,
                title: format!("Count {} by category", label),
                description: format!(
                    "Analyze the distribution of {} instances across different categories",
                    label
                ),
                relevance: 0.7,
                example_query: Some(format!(
                    "SELECT ?category (COUNT(?entity) as ?count) WHERE {{ ?entity a <{}> . ?entity ?p ?category }} GROUP BY ?category",
                    class.uri
                )),
                related_concepts: vec![label, "aggregation".to_string()],
                tags: vec!["pattern".to_string(), "statistics".to_string()],
            });
        }

        // Suggest temporal patterns
        suggestions.push(ExplorationSuggestion {
            id: format!("temporal_{}", uuid::Uuid::new_v4()),
            suggestion_type: SuggestionType::Temporal,
            title: "Analyze temporal trends".to_string(),
            description: "Investigate how data changes over time".to_string(),
            relevance: 0.6,
            example_query: Some(
                "SELECT ?date (COUNT(?entity) as ?count) WHERE { ?entity ?p ?date FILTER(isLiteral(?date)) } GROUP BY ?date ORDER BY ?date"
                    .to_string(),
            ),
            related_concepts: vec!["time".to_string(), "trends".to_string()],
            tags: vec!["temporal".to_string(), "analysis".to_string()],
        });

        Ok(suggestions)
    }

    /// Calculate relevance score for an entity
    fn calculate_entity_relevance(
        &self,
        class: &crate::schema_introspection::RdfClass,
        context: &ExplorationContext,
    ) -> f32 {
        let mut score: f32 = 0.5; // Base score

        // Boost if matches user interests
        if let Some(label) = &class.label {
            if context
                .interests
                .iter()
                .any(|interest| label.to_lowercase().contains(&interest.to_lowercase()))
            {
                score += 0.3;
            }
        }

        // Boost based on instance count (more instances = more interesting)
        if class.instance_count > 100 {
            score += 0.1;
        }

        // Penalize if no instances
        if class.instance_count == 0 {
            score -= 0.3;
        }

        score.clamp(0.0, 1.0)
    }

    /// Calculate relevance score for a relationship
    fn calculate_relationship_relevance(
        &self,
        property: &crate::schema_introspection::RdfProperty,
        context: &ExplorationContext,
    ) -> f32 {
        let mut score: f32 = 0.5; // Base score

        // Boost if matches user interests
        if let Some(label) = &property.label {
            if context
                .interests
                .iter()
                .any(|interest| label.to_lowercase().contains(&interest.to_lowercase()))
            {
                score += 0.3;
            }
        }

        // Boost based on usage count
        if property.usage_count > 50 {
            score += 0.15;
        }

        score.clamp(0.0, 1.0)
    }

    /// Get next steps based on current exploration
    pub async fn get_next_steps(
        &self,
        current_query: &str,
        context: &ExplorationContext,
    ) -> Result<Vec<String>> {
        debug!("Generating next steps for: {}", current_query);

        let mut next_steps = Vec::new();

        // Suggest drilling down
        next_steps.push("Drill down into specific instances".to_string());

        // Suggest broadening search
        next_steps.push("Broaden search to related entities".to_string());

        // Suggest comparison
        next_steps.push("Compare with similar entities".to_string());

        // Suggest temporal analysis
        if !context.query_history.is_empty() {
            next_steps.push("Analyze changes over time".to_string());
        }

        Ok(next_steps)
    }

    /// Get exploration summary
    pub async fn get_exploration_summary(
        &self,
        context: &ExplorationContext,
    ) -> Result<ExplorationSummary> {
        Ok(ExplorationSummary {
            entities_explored: context.viewed_entities.len(),
            queries_executed: context.query_history.len(),
            focus_areas: context
                .focus_area
                .clone()
                .map(|f| vec![f])
                .unwrap_or_default(),
            suggested_next_actions: self.get_next_steps("", context).await?,
        })
    }
}

/// Exploration summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationSummary {
    /// Number of entities explored
    pub entities_explored: usize,
    /// Number of queries executed
    pub queries_executed: usize,
    /// Focus areas
    pub focus_areas: Vec<String>,
    /// Suggested next actions
    pub suggested_next_actions: Vec<String>,
}

#[cfg(test)]
mod tests {
    // Tests omitted - require concrete Store implementation
    // Integration tests should be in tests/ directory with actual store setup
}
