// Copyright (c) 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Automatic Schema Suggestions
//!
//! This module provides AI-powered automatic schema suggestions based on
//! query patterns, RDF data analysis, and machine learning recommendations.

use anyhow::Result;
use scirs2_core::ndarray_ext::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Schema suggestion type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestionType {
    /// Suggest adding a new type
    NewType,
    /// Suggest adding a field to existing type
    NewField,
    /// Suggest adding an index
    AddIndex,
    /// Suggest deprecating a field
    DeprecateField,
    /// Suggest relationship between types
    AddRelation,
    /// Suggest optimization
    Optimization,
}

/// Schema suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaSuggestion {
    /// Type of suggestion
    pub suggestion_type: SuggestionType,
    /// Target type name
    pub target_type: String,
    /// Field name (if applicable)
    pub field_name: Option<String>,
    /// Suggested schema change
    pub suggestion: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Rationale for the suggestion
    pub rationale: String,
    /// Impact score (higher = more impact)
    pub impact_score: f32,
}

impl SchemaSuggestion {
    /// Create a new schema suggestion
    pub fn new(suggestion_type: SuggestionType, target_type: String, suggestion: String) -> Self {
        Self {
            suggestion_type,
            target_type,
            field_name: None,
            suggestion,
            confidence: 0.8,
            rationale: String::new(),
            impact_score: 0.5,
        }
    }

    /// Set field name
    pub fn with_field(mut self, field_name: String) -> Self {
        self.field_name = Some(field_name);
        self
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    /// Set rationale
    pub fn with_rationale(mut self, rationale: String) -> Self {
        self.rationale = rationale;
        self
    }

    /// Set impact score
    pub fn with_impact_score(mut self, score: f32) -> Self {
        self.impact_score = score;
        self
    }
}

/// Query pattern for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPattern {
    /// Fields frequently queried together
    pub field_combinations: Vec<Vec<String>>,
    /// Query frequency
    pub frequency: usize,
    /// Average execution time
    pub avg_execution_time_ms: f64,
}

/// RDF data statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RdfDataStats {
    /// Number of triples
    pub triple_count: usize,
    /// Predicate usage frequency
    pub predicate_frequency: HashMap<String, usize>,
    /// Object types
    pub object_types: HashMap<String, usize>,
}

/// Schema suggestion engine
pub struct SchemaSuggestionEngine {
    /// Query pattern analyzer
    pattern_analyzer: Arc<RwLock<PatternAnalyzer>>,
    /// ML-based suggestion model
    ml_model: Arc<RwLock<SuggestionModel>>,
    /// Historical suggestions
    suggestion_history: Arc<RwLock<Vec<SchemaSuggestion>>>,
    /// RDF data statistics
    rdf_stats: Arc<RwLock<RdfDataStats>>,
}

/// Pattern analyzer for finding common query patterns
#[derive(Debug, Clone)]
pub struct PatternAnalyzer {
    /// Observed query patterns
    patterns: Vec<QueryPattern>,
    /// Minimum frequency for pattern detection
    min_frequency: usize,
}

impl PatternAnalyzer {
    /// Create a new pattern analyzer
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            min_frequency: 5,
        }
    }

    /// Add a query pattern
    pub fn add_pattern(&mut self, pattern: QueryPattern) {
        self.patterns.push(pattern);
    }

    /// Analyze patterns and generate suggestions
    pub fn analyze(&self) -> Vec<SchemaSuggestion> {
        let mut suggestions = Vec::new();

        // Find frequently co-queried fields
        for pattern in &self.patterns {
            if pattern.frequency >= self.min_frequency && !pattern.field_combinations.is_empty() {
                let fields = pattern
                    .field_combinations
                    .first()
                    .expect("collection validated to be non-empty");
                if fields.len() > 1 {
                    // Suggest adding a composite type or index
                    suggestions.push(
                        SchemaSuggestion::new(
                            SuggestionType::AddIndex,
                            "Query".to_string(),
                            format!("Add index on fields: {}", fields.join(", ")),
                        )
                        .with_confidence(0.85)
                        .with_rationale(format!(
                            "These fields are frequently queried together ({} times)",
                            pattern.frequency
                        ))
                        .with_impact_score(0.7),
                    );
                }
            }

            // Suggest optimization for slow queries
            if pattern.avg_execution_time_ms > 1000.0 {
                suggestions.push(
                    SchemaSuggestion::new(
                        SuggestionType::Optimization,
                        "Query".to_string(),
                        "Consider adding materialized view or cache".to_string(),
                    )
                    .with_confidence(0.75)
                    .with_rationale(format!(
                        "Average execution time is {:.2}ms",
                        pattern.avg_execution_time_ms
                    ))
                    .with_impact_score(0.8),
                );
            }
        }

        suggestions
    }

    /// Set minimum frequency threshold
    pub fn set_min_frequency(&mut self, frequency: usize) {
        self.min_frequency = frequency;
    }
}

impl Default for PatternAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// ML-based suggestion model
#[derive(Debug, Clone)]
pub struct SuggestionModel {
    /// Feature embeddings for schema elements
    #[allow(dead_code)]
    embeddings: HashMap<String, Array1<f32>>,
}

impl SuggestionModel {
    /// Create a new suggestion model
    pub fn new() -> Self {
        Self {
            embeddings: HashMap::new(),
        }
    }

    /// Predict schema suggestions based on ML
    pub fn predict(&mut self, rdf_stats: &RdfDataStats) -> Vec<SchemaSuggestion> {
        let mut suggestions = Vec::new();

        // Analyze predicate frequency for new field suggestions
        for (predicate, &frequency) in &rdf_stats.predicate_frequency {
            if frequency > 100 {
                // Suggest adding this as a GraphQL field
                let confidence = (frequency as f32 / rdf_stats.triple_count as f32).min(1.0);

                suggestions.push(
                    SchemaSuggestion::new(
                        SuggestionType::NewField,
                        "RdfResource".to_string(),
                        format!("Add field '{}' to schema", predicate),
                    )
                    .with_field(predicate.clone())
                    .with_confidence(confidence)
                    .with_rationale(format!("Predicate appears {} times in RDF data", frequency))
                    .with_impact_score(0.6),
                );
            }
        }

        // Suggest new types based on object types
        for (object_type, &count) in &rdf_stats.object_types {
            if count > 50 {
                suggestions.push(
                    SchemaSuggestion::new(
                        SuggestionType::NewType,
                        object_type.clone(),
                        format!("Create GraphQL type for '{}'", object_type),
                    )
                    .with_confidence(0.8)
                    .with_rationale(format!("{} instances found in RDF data", count))
                    .with_impact_score(0.75),
                );
            }
        }

        suggestions
    }

    /// Train model with feedback (placeholder for future ML training)
    pub fn train(&mut self, _examples: Vec<(RdfDataStats, Vec<SchemaSuggestion>)>) -> Result<()> {
        // In production, this would train a neural network
        Ok(())
    }
}

impl Default for SuggestionModel {
    fn default() -> Self {
        Self::new()
    }
}

impl SchemaSuggestionEngine {
    /// Create a new schema suggestion engine
    pub fn new() -> Self {
        Self {
            pattern_analyzer: Arc::new(RwLock::new(PatternAnalyzer::new())),
            ml_model: Arc::new(RwLock::new(SuggestionModel::new())),
            suggestion_history: Arc::new(RwLock::new(Vec::new())),
            rdf_stats: Arc::new(RwLock::new(RdfDataStats::default())),
        }
    }

    /// Update RDF statistics
    pub async fn update_rdf_stats(&self, stats: RdfDataStats) -> Result<()> {
        let mut rdf_stats = self.rdf_stats.write().await;
        *rdf_stats = stats;
        Ok(())
    }

    /// Add observed query pattern
    pub async fn add_query_pattern(&self, pattern: QueryPattern) -> Result<()> {
        let mut analyzer = self.pattern_analyzer.write().await;
        analyzer.add_pattern(pattern);
        Ok(())
    }

    /// Generate schema suggestions
    pub async fn generate_suggestions(&self) -> Result<Vec<SchemaSuggestion>> {
        let mut all_suggestions = Vec::new();

        // Get suggestions from pattern analysis
        let analyzer = self.pattern_analyzer.read().await;
        let pattern_suggestions = analyzer.analyze();
        all_suggestions.extend(pattern_suggestions);

        // Get suggestions from ML model
        let mut ml_model = self.ml_model.write().await;
        let rdf_stats = self.rdf_stats.read().await;
        let ml_suggestions = ml_model.predict(&rdf_stats);
        all_suggestions.extend(ml_suggestions);

        // Sort by impact score and confidence
        all_suggestions.sort_by(|a, b| {
            let score_a = a.impact_score * a.confidence;
            let score_b = b.impact_score * b.confidence;
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Store in history
        let mut history = self.suggestion_history.write().await;
        history.extend(all_suggestions.clone());

        Ok(all_suggestions)
    }

    /// Get suggestion history
    pub async fn get_history(&self) -> Vec<SchemaSuggestion> {
        let history = self.suggestion_history.read().await;
        history.clone()
    }

    /// Set minimum pattern frequency
    pub async fn set_min_frequency(&self, frequency: usize) -> Result<()> {
        let mut analyzer = self.pattern_analyzer.write().await;
        analyzer.set_min_frequency(frequency);
        Ok(())
    }

    /// Train ML model with feedback
    pub async fn train_model(
        &self,
        examples: Vec<(RdfDataStats, Vec<SchemaSuggestion>)>,
    ) -> Result<()> {
        let mut model = self.ml_model.write().await;
        model.train(examples)
    }
}

impl Default for SchemaSuggestionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_suggestion_creation() {
        let suggestion = SchemaSuggestion::new(
            SuggestionType::NewField,
            "User".to_string(),
            "Add email field".to_string(),
        );
        assert_eq!(suggestion.target_type, "User");
        assert_eq!(suggestion.suggestion_type, SuggestionType::NewField);
    }

    #[test]
    fn test_schema_suggestion_builder() {
        let suggestion = SchemaSuggestion::new(
            SuggestionType::NewField,
            "User".to_string(),
            "Add field".to_string(),
        )
        .with_field("email".to_string())
        .with_confidence(0.9)
        .with_rationale("Common field".to_string())
        .with_impact_score(0.8);

        assert_eq!(suggestion.field_name, Some("email".to_string()));
        assert_eq!(suggestion.confidence, 0.9);
        assert_eq!(suggestion.impact_score, 0.8);
    }

    #[test]
    fn test_pattern_analyzer() {
        let mut analyzer = PatternAnalyzer::new();
        let pattern = QueryPattern {
            field_combinations: vec![vec!["name".to_string(), "email".to_string()]],
            frequency: 10,
            avg_execution_time_ms: 50.0,
        };

        analyzer.add_pattern(pattern);
        let suggestions = analyzer.analyze();
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_pattern_analyzer_slow_query() {
        let mut analyzer = PatternAnalyzer::new();
        let pattern = QueryPattern {
            field_combinations: vec![vec!["data".to_string()]],
            frequency: 5,
            avg_execution_time_ms: 2000.0,
        };

        analyzer.add_pattern(pattern);
        let suggestions = analyzer.analyze();
        assert!(suggestions
            .iter()
            .any(|s| s.suggestion_type == SuggestionType::Optimization));
    }

    #[test]
    fn test_suggestion_model() {
        let mut model = SuggestionModel::new();
        let mut stats = RdfDataStats::default();
        stats.predicate_frequency.insert("name".to_string(), 150);
        stats.triple_count = 1000;

        let suggestions = model.predict(&stats);
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_suggestion_model_new_type() {
        let mut model = SuggestionModel::new();
        let mut stats = RdfDataStats::default();
        stats.object_types.insert("Person".to_string(), 100);

        let suggestions = model.predict(&stats);
        assert!(suggestions
            .iter()
            .any(|s| s.suggestion_type == SuggestionType::NewType));
    }

    #[tokio::test]
    async fn test_engine_creation() {
        let engine = SchemaSuggestionEngine::new();
        let history = engine.get_history().await;
        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn test_update_rdf_stats() {
        let engine = SchemaSuggestionEngine::new();
        let stats = RdfDataStats {
            triple_count: 1000,
            predicate_frequency: HashMap::new(),
            object_types: HashMap::new(),
        };

        engine.update_rdf_stats(stats).await.unwrap();
    }

    #[tokio::test]
    async fn test_add_query_pattern() {
        let engine = SchemaSuggestionEngine::new();
        let pattern = QueryPattern {
            field_combinations: vec![vec!["id".to_string()]],
            frequency: 5,
            avg_execution_time_ms: 100.0,
        };

        engine.add_query_pattern(pattern).await.unwrap();
    }

    #[tokio::test]
    async fn test_generate_suggestions() {
        let engine = SchemaSuggestionEngine::new();

        // Add RDF stats
        let mut stats = RdfDataStats::default();
        stats.predicate_frequency.insert("email".to_string(), 200);
        stats.triple_count = 1000;
        engine.update_rdf_stats(stats).await.unwrap();

        // Add pattern
        let pattern = QueryPattern {
            field_combinations: vec![vec!["name".to_string(), "email".to_string()]],
            frequency: 10,
            avg_execution_time_ms: 100.0,
        };
        engine.add_query_pattern(pattern).await.unwrap();

        // Generate suggestions
        let suggestions = engine.generate_suggestions().await.unwrap();
        assert!(!suggestions.is_empty());
    }

    #[tokio::test]
    async fn test_set_min_frequency() {
        let engine = SchemaSuggestionEngine::new();
        engine.set_min_frequency(10).await.unwrap();
    }

    #[tokio::test]
    async fn test_train_model() {
        let engine = SchemaSuggestionEngine::new();
        let examples = vec![];
        engine.train_model(examples).await.unwrap();
    }

    #[test]
    fn test_rdf_stats_default() {
        let stats = RdfDataStats::default();
        assert_eq!(stats.triple_count, 0);
        assert!(stats.predicate_frequency.is_empty());
    }

    #[test]
    fn test_suggestion_sorting() {
        let mut suggestions = [
            SchemaSuggestion::new(
                SuggestionType::NewField,
                "A".to_string(),
                "test".to_string(),
            )
            .with_impact_score(0.5)
            .with_confidence(0.5),
            SchemaSuggestion::new(
                SuggestionType::NewField,
                "B".to_string(),
                "test".to_string(),
            )
            .with_impact_score(0.9)
            .with_confidence(0.9),
        ];

        suggestions.sort_by(|a, b| {
            let score_a = a.impact_score * a.confidence;
            let score_b = b.impact_score * b.confidence;
            score_b.partial_cmp(&score_a).unwrap()
        });

        assert_eq!(suggestions[0].target_type, "B");
    }
}
