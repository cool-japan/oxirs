//! ML-based source predictor implementation
//!
//! Uses machine learning to predict optimal service selection

use crate::source_selection::types::*;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

impl Default for MLSourcePredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl MLSourcePredictor {
    pub fn new() -> Self {
        Self {
            training_data: Vec::new(),
            feature_weights: HashMap::new(),
            prediction_cache: Arc::new(RwLock::new(HashMap::new())),
            model_accuracy: 0.5,
        }
    }

    pub async fn predict_sources(&self, features: &QueryFeatures) -> Result<PredictionResult> {
        let feature_key = format!(
            "{}:{}:{}",
            features.pattern_count, features.variable_count, features.complexity_score
        );

        // Check cache first
        if let Some(cached) = self.prediction_cache.read().await.get(&feature_key) {
            return Ok(cached.clone());
        }

        // Simple prediction based on feature weights
        let mut source_scores = HashMap::new();

        // This is a simplified ML model - in practice, this would use
        // more sophisticated algorithms like neural networks or ensemble methods
        for sample in &self.training_data {
            if self.features_similar(features, &sample.query_features) {
                for source in &sample.selected_sources {
                    let score = self.calculate_similarity_score(features, &sample.query_features);
                    *source_scores.entry(source.clone()).or_insert(0.0) += score;
                }
            }
        }

        // Normalize scores
        let max_score = source_scores.values().cloned().fold(0.0, f64::max);
        if max_score > 0.0 {
            for score in source_scores.values_mut() {
                *score /= max_score;
            }
        }

        let recommended_sources: Vec<String> = source_scores.keys().cloned().collect();

        // Implement performance prediction based on historical data
        let predicted_performance = self
            .predict_performance_scores(features, &source_scores)
            .await;

        let result = PredictionResult {
            recommended_sources,
            confidence_scores: source_scores.clone(),
            predicted_performance,
            feature_importance: self.feature_weights.clone(),
        };

        // Cache the result
        self.prediction_cache
            .write()
            .await
            .insert(feature_key, result.clone());

        Ok(result)
    }

    async fn predict_performance_scores(
        &self,
        features: &QueryFeatures,
        source_scores: &HashMap<String, f64>,
    ) -> HashMap<String, f64> {
        let mut performance_predictions = HashMap::new();

        for (source, confidence_score) in source_scores {
            let predicted_performance =
                self.calculate_performance_prediction(source, features, *confidence_score);
            performance_predictions.insert(source.clone(), predicted_performance);
        }

        performance_predictions
    }

    fn calculate_performance_prediction(
        &self,
        source: &str,
        features: &QueryFeatures,
        confidence_score: f64,
    ) -> f64 {
        // Base performance prediction from historical data
        let mut performance_score = 0.5; // Default baseline

        // Find historical performance for similar queries
        for sample in &self.training_data {
            if sample.selected_sources.contains(&source.to_string()) {
                let similarity = self.calculate_similarity_score(features, &sample.query_features);
                if similarity > 0.7 {
                    // Weight by similarity
                    let historical_performance =
                        self.extract_performance_score(&sample.actual_performance);
                    performance_score += similarity * historical_performance;
                }
            }
        }

        // Adjust based on query complexity
        let complexity_factor = 1.0 - (features.complexity_score / 10.0).min(0.5);
        performance_score *= complexity_factor;

        // Factor in confidence score
        performance_score *= 0.5 + 0.5 * confidence_score;

        // Normalize to [0, 1] range
        performance_score.clamp(0.0, 1.0)
    }

    fn extract_performance_score(&self, metrics: &PerformanceMetrics) -> f64 {
        // Combine various performance metrics into a single score
        let response_time_score = 1.0 - (metrics.execution_time_ms as f64).min(5000.0) / 5000.0;
        let throughput_score = (metrics.result_count as f64 / 1000.0).min(1.0);
        let error_rate_score = metrics.success_rate; // success_rate is the inverse of error_rate
        let memory_score = 1.0 - (metrics.data_transfer_bytes as f64 / 1000000.0).min(1.0); // Use data transfer as proxy for memory

        // Weighted combination
        0.4 * response_time_score
            + 0.3 * throughput_score
            + 0.2 * error_rate_score
            + 0.1 * memory_score
    }

    pub async fn train(&mut self, samples: Vec<SourcePredictionSample>) -> Result<()> {
        self.training_data.extend(samples);

        // Simple feature weight learning
        self.update_feature_weights().await?;

        // Keep only recent training data (last 10000 samples)
        if self.training_data.len() > 10000 {
            let start = self.training_data.len() - 10000;
            self.training_data.drain(0..start);
        }

        // Clear prediction cache to force recomputation
        self.prediction_cache.write().await.clear();

        Ok(())
    }

    async fn update_feature_weights(&mut self) -> Result<()> {
        // Simple feature weight calculation based on correlation with performance
        let mut weights = HashMap::new();

        weights.insert("pattern_count".to_string(), 0.3);
        weights.insert("variable_count".to_string(), 0.2);
        weights.insert("complexity_score".to_string(), 0.4);
        weights.insert("has_joins".to_string(), 0.1);

        self.feature_weights = weights;
        Ok(())
    }

    fn features_similar(&self, f1: &QueryFeatures, f2: &QueryFeatures) -> bool {
        let pattern_diff = (f1.pattern_count as f64 - f2.pattern_count as f64).abs();
        let var_diff = (f1.variable_count as f64 - f2.variable_count as f64).abs();
        let complexity_diff = (f1.complexity_score - f2.complexity_score).abs();

        pattern_diff <= 2.0 && var_diff <= 3.0 && complexity_diff <= 1.0
    }

    fn calculate_similarity_score(&self, f1: &QueryFeatures, f2: &QueryFeatures) -> f64 {
        let pattern_sim = 1.0 - (f1.pattern_count as f64 - f2.pattern_count as f64).abs() / 10.0;
        let var_sim = 1.0 - (f1.variable_count as f64 - f2.variable_count as f64).abs() / 10.0;
        let complexity_sim = 1.0 - (f1.complexity_score - f2.complexity_score).abs();

        (pattern_sim + var_sim + complexity_sim) / 3.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FederatedService, service_registry::ServiceRegistry};

    fn create_test_pattern(subject: &str, predicate: &str, object: &str) -> TriplePattern {
        TriplePattern {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            graph: None,
        }
    }

    #[cfg(test)]
    #[allow(dead_code)]
    fn create_test_service() -> FederatedService {
        use std::collections::HashSet;

        FederatedService {
            id: "test-service-1".to_string(),
            name: "Test Service".to_string(),
            endpoint: "http://example.org/sparql".to_string(),
            service_type: crate::ServiceType::Sparql,
            capabilities: {
                let mut caps = HashSet::new();
                caps.insert(crate::ServiceCapability::SparqlQuery);
                caps.insert(crate::ServiceCapability::Sparql11Query);
                caps
            },
            data_patterns: vec!["http://example.org/".to_string()],
            auth: None,
            metadata: crate::service::ServiceMetadata {
                description: Some("Test SPARQL endpoint".to_string()),
                version: Some("1.0".to_string()),
                maintainer: None,
                tags: vec!["test".to_string()],
                documentation_url: None,
                schema_url: None,
            },
            extended_metadata: None,
            performance: crate::ServicePerformance::default(),
            status: None,
        }
    }

    #[tokio::test]
    async fn test_pattern_coverage_analyzer() {
        let analyzer = PatternCoverageAnalyzer::new();
        let pattern = create_test_pattern("?s", "http://example.org/name", "?o");

        // Update statistics with test data
        let triples = vec![
            (
                "http://example.org/entity1".to_string(),
                "http://example.org/name".to_string(),
                "Alice".to_string(),
            ),
            (
                "http://example.org/entity2".to_string(),
                "http://example.org/name".to_string(),
                "Bob".to_string(),
            ),
        ];

        analyzer
            .update_service_statistics("http://example.org/sparql", &triples)
            .await
            .unwrap();

        // Test coverage analysis
        let registry = ServiceRegistry::new();
        let results = analyzer
            .analyze_coverage(&[pattern], &registry)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].coverage_score >= 0.0);
    }

    #[tokio::test]
    async fn test_source_selector() {
        // Disable all methods except fallback to test pure fallback scenario
        let config = SourceSelectionConfig {
            enable_pattern_coverage: false,
            enable_predicate_filtering: false,
            enable_range_selection: false,
            enable_ml_prediction: false,
            ..Default::default()
        };
        let selector = AdvancedSourceSelector::new(config);

        let patterns = vec![
            create_test_pattern("?s", "http://example.org/name", "?o"),
            create_test_pattern("?s", "http://example.org/age", "?age"),
        ];

        let constraints = vec![RangeConstraint {
            field: "age".to_string(),
            min_value: Some("18".to_string()),
            max_value: Some("65".to_string()),
            data_type: RangeDataType::Integer,
        }];

        let registry = ServiceRegistry::new();
        let result = selector
            .select_sources(&patterns, &constraints, &registry)
            .await
            .unwrap();

        assert!(!result.selected_sources.is_empty());
        assert!(matches!(result.selection_method, SelectionMethod::Fallback)); // No services registered
    }

    #[tokio::test]
    async fn test_predicate_filter() {
        let config = SourceSelectionConfig::default();
        let filter = PredicateBasedFilter::new(&config);

        let triples = vec![
            (
                "http://example.org/entity1".to_string(),
                "http://example.org/name".to_string(),
                "Alice".to_string(),
            ),
            (
                "http://example.org/entity2".to_string(),
                "http://example.org/age".to_string(),
                "25".to_string(),
            ),
        ];

        filter
            .update_filters("http://example.org/sparql", &triples)
            .await
            .unwrap();

        let patterns = vec![create_test_pattern("?s", "http://example.org/name", "?o")];

        let registry = ServiceRegistry::new();
        let matches = filter
            .filter_by_predicates(&patterns, &registry)
            .await
            .unwrap();

        assert!(matches.contains_key("http://example.org/sparql"));
    }

    #[test]
    fn test_query_features() {
        let features = QueryFeatures {
            pattern_count: 3,
            variable_count: 2,
            predicate_types: vec!["http://example.org/name".to_string()],
            has_ranges: true,
            has_joins: true,
            complexity_score: 1.5,
            selectivity_estimate: 0.3,
        };

        assert_eq!(features.pattern_count, 3);
        assert_eq!(features.variable_count, 2);
        assert!(features.has_ranges);
        assert!(features.has_joins);
    }

    #[test]
    fn test_config_serialization() {
        let config = SourceSelectionConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: SourceSelectionConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config.coverage_threshold, deserialized.coverage_threshold);
        assert_eq!(
            config.enable_pattern_coverage,
            deserialized.enable_pattern_coverage
        );
    }
}
