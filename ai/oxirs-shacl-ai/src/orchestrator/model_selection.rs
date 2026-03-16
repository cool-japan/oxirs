//! Model selection for AI orchestrator

use super::types::{DataCharacteristics, PerformanceRequirements};
pub use super::types::{ModelPerformanceMetrics, ModelSelectionResult, ModelSelectionStrategy};

/// Advanced model selector that chooses the best model configuration
/// based on data characteristics and performance requirements.
pub struct AdvancedModelSelector {
    strategy: ModelSelectionStrategy,
    performance_requirements: PerformanceRequirements,
    /// Historical model performance data (model_name → metrics)
    performance_history: std::collections::HashMap<String, Vec<f64>>,
    /// Model weights for ensemble selection
    model_weights: std::collections::HashMap<String, f64>,
}

impl Default for AdvancedModelSelector {
    fn default() -> Self {
        Self {
            strategy: ModelSelectionStrategy::PerformanceBased,
            performance_requirements: PerformanceRequirements::default(),
            performance_history: std::collections::HashMap::new(),
            model_weights: std::collections::HashMap::new(),
        }
    }
}

impl AdvancedModelSelector {
    /// Create a new model selector with the given strategy
    pub fn new(strategy: ModelSelectionStrategy) -> Self {
        Self {
            strategy,
            ..Default::default()
        }
    }

    /// Set performance requirements
    pub fn with_requirements(mut self, requirements: PerformanceRequirements) -> Self {
        self.performance_requirements = requirements;
        self
    }

    /// Record a performance observation for a model
    pub fn record_performance(&mut self, model_name: &str, f1_score: f64) {
        self.performance_history
            .entry(model_name.to_string())
            .or_default()
            .push(f1_score);
    }

    /// Select models given the current data characteristics
    pub fn select(&self, characteristics: &DataCharacteristics) -> ModelSelectionResult {
        let selected_models = self.select_models(characteristics);
        let confidence = self.compute_selection_confidence(&selected_models);
        let rationale = self.build_rationale(characteristics);

        ModelSelectionResult {
            selected_models,
            selection_confidence: confidence,
            expected_performance: self.estimate_performance(),
            selection_rationale: rationale,
        }
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn select_models(&self, characteristics: &DataCharacteristics) -> Vec<String> {
        match &self.strategy {
            ModelSelectionStrategy::PerformanceBased => self.select_by_performance(),
            ModelSelectionStrategy::DataAdaptive => self.select_by_characteristics(characteristics),
            ModelSelectionStrategy::EnsembleWeighted => self.select_ensemble(),
            ModelSelectionStrategy::ReinforcementLearning => {
                vec!["rl_policy_model".to_string()]
            }
            ModelSelectionStrategy::MetaLearning => {
                vec!["meta_learner".to_string()]
            }
            ModelSelectionStrategy::Hybrid(strategies) => {
                let mut models = Vec::new();
                for s in strategies {
                    let sub_selector = AdvancedModelSelector::new(s.clone());
                    models.extend(sub_selector.select_models(characteristics));
                }
                models.sort();
                models.dedup();
                models
            }
        }
    }

    fn select_by_performance(&self) -> Vec<String> {
        let mut sorted: Vec<(&String, f64)> = self
            .performance_history
            .iter()
            .map(|(name, scores)| {
                let avg = if scores.is_empty() {
                    0.0
                } else {
                    scores.iter().sum::<f64>() / scores.len() as f64
                };
                (name, avg)
            })
            .collect();

        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if sorted.is_empty() {
            vec!["graph_transformer".to_string(), "rule_learner".to_string()]
        } else {
            sorted
                .into_iter()
                .take(3)
                .map(|(name, _)| name.clone())
                .collect()
        }
    }

    fn select_by_characteristics(&self, chars: &DataCharacteristics) -> Vec<String> {
        let mut models = Vec::new();

        if chars.graph_size < 10_000 {
            models.push("rule_learner".to_string());
        }
        if chars.complexity_score > 0.5 {
            models.push("graph_transformer".to_string());
        }
        if chars.semantic_richness > 0.3 {
            models.push("semantic_model".to_string());
        }

        if models.is_empty() {
            models.push("graph_transformer".to_string());
        }

        models
    }

    fn select_ensemble(&self) -> Vec<String> {
        let mut models: Vec<String> = self.model_weights.keys().cloned().collect();
        if models.is_empty() {
            models = vec![
                "graph_transformer".to_string(),
                "rule_learner".to_string(),
                "statistical_miner".to_string(),
            ];
        }
        models
    }

    fn compute_selection_confidence(&self, models: &[String]) -> f64 {
        if models.is_empty() {
            return 0.0;
        }
        // Confidence based on historical performance data availability
        let models_with_history = models
            .iter()
            .filter(|m| self.performance_history.contains_key(*m))
            .count();
        (models_with_history as f64 / models.len() as f64) * 0.5 + 0.5
    }

    fn estimate_performance(&self) -> ModelPerformanceMetrics {
        ModelPerformanceMetrics {
            accuracy: 0.85,
            precision: 0.83,
            recall: 0.81,
            f1_score: 0.82,
            training_time: std::time::Duration::from_secs(60),
            inference_time: std::time::Duration::from_millis(10),
            memory_usage: 256.0,
            confidence_calibration: 0.9,
            robustness_score: 0.8,
        }
    }

    fn build_rationale(&self, characteristics: &DataCharacteristics) -> String {
        format!(
            "Selected using {:?} strategy for graph with {} triples, \
            complexity {:.2}, semantic richness {:.2}",
            self.strategy,
            characteristics.graph_size,
            characteristics.complexity_score,
            characteristics.semantic_richness
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_based_selection_empty_history() {
        let selector = AdvancedModelSelector::new(ModelSelectionStrategy::PerformanceBased);
        let chars = DataCharacteristics::default();
        let result = selector.select(&chars);
        assert!(!result.selected_models.is_empty());
        assert!(result.selection_confidence > 0.0);
    }

    #[test]
    fn test_performance_based_selection_with_history() {
        let mut selector = AdvancedModelSelector::new(ModelSelectionStrategy::PerformanceBased);
        selector.record_performance("model_a", 0.9);
        selector.record_performance("model_b", 0.7);
        let chars = DataCharacteristics::default();
        let result = selector.select(&chars);
        // model_a has better performance and should be selected
        assert!(result.selected_models.contains(&"model_a".to_string()));
    }

    #[test]
    fn test_data_adaptive_selection_small_graph() {
        let selector = AdvancedModelSelector::new(ModelSelectionStrategy::DataAdaptive);
        let chars = DataCharacteristics {
            graph_size: 500,
            complexity_score: 0.2,
            ..Default::default()
        };
        let result = selector.select(&chars);
        assert!(result.selected_models.contains(&"rule_learner".to_string()));
    }

    #[test]
    fn test_data_adaptive_selection_complex_graph() {
        let selector = AdvancedModelSelector::new(ModelSelectionStrategy::DataAdaptive);
        let chars = DataCharacteristics {
            graph_size: 50_000,
            complexity_score: 0.8,
            ..Default::default()
        };
        let result = selector.select(&chars);
        assert!(result
            .selected_models
            .contains(&"graph_transformer".to_string()));
    }

    #[test]
    fn test_ensemble_selection() {
        let selector = AdvancedModelSelector::new(ModelSelectionStrategy::EnsembleWeighted);
        let chars = DataCharacteristics::default();
        let result = selector.select(&chars);
        assert!(!result.selected_models.is_empty());
    }

    #[test]
    fn test_selection_rationale_non_empty() {
        let selector = AdvancedModelSelector::new(ModelSelectionStrategy::PerformanceBased);
        let chars = DataCharacteristics::default();
        let result = selector.select(&chars);
        assert!(!result.selection_rationale.is_empty());
    }

    #[test]
    fn test_record_performance() {
        let mut selector = AdvancedModelSelector::new(ModelSelectionStrategy::PerformanceBased);
        selector.record_performance("my_model", 0.85);
        selector.record_performance("my_model", 0.87);
        let history = selector
            .performance_history
            .get("my_model")
            .expect("should succeed");
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_with_requirements() {
        let requirements = PerformanceRequirements {
            min_accuracy: 0.95,
            ..Default::default()
        };
        let selector = AdvancedModelSelector::new(ModelSelectionStrategy::PerformanceBased)
            .with_requirements(requirements);
        assert!((selector.performance_requirements.min_accuracy - 0.95).abs() < 1e-9);
    }
}
