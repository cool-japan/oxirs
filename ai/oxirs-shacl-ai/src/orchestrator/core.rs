//! Core AI orchestrator implementation

use std::sync::{Arc, RwLock};
use std::time::Instant;

use super::config::ShapeOrchestratorConfig;
use super::data_analysis::{DataAnalyzer, DataStatistics};
use super::model_selection::AdvancedModelSelector;
use super::types::{
    ComprehensiveLearningResult, ConfidentShape, ImplementationEffort, LearningMetadata,
    OptimizationRecommendation, OrchestrationMetrics, PredictiveInsights, QualityAnalysis,
    ShapeOrchestratorStats,
};
use crate::ShaclAiError;

/// AI orchestrator that coordinates multiple shape learning models
pub struct ShapeOrchestrator {
    config: ShapeOrchestratorConfig,
    model_selector: AdvancedModelSelector,
    data_analyzer: DataAnalyzer,
    stats: Arc<RwLock<ShapeOrchestratorStats>>,
}

impl ShapeOrchestrator {
    /// Create a new orchestrator with default configuration
    pub fn new() -> Self {
        let config = ShapeOrchestratorConfig::default();
        let selector = AdvancedModelSelector::new(config.selection_strategy.clone());
        Self {
            model_selector: selector,
            config,
            data_analyzer: DataAnalyzer::new(),
            stats: Arc::new(RwLock::new(ShapeOrchestratorStats::default())),
        }
    }

    /// Create an orchestrator with a custom configuration
    pub fn with_config(config: ShapeOrchestratorConfig) -> Self {
        let selector = AdvancedModelSelector::new(config.selection_strategy.clone());
        Self {
            model_selector: selector,
            config,
            data_analyzer: DataAnalyzer::new(),
            stats: Arc::new(RwLock::new(ShapeOrchestratorStats::default())),
        }
    }

    /// Execute a comprehensive learning run on the provided dataset statistics
    pub fn learn(
        &mut self,
        stats: &DataStatistics,
    ) -> Result<ComprehensiveLearningResult, ShaclAiError> {
        let start = Instant::now();

        // Analyse the data
        let characteristics = self.data_analyzer.analyse(stats);

        // Select appropriate models
        let selection = self.model_selector.select(&characteristics);

        // Simulate shape discovery (placeholder for real ML pipeline)
        let shapes = self.discover_shapes(stats, &selection.selected_models);

        // Quality analysis
        let quality = self.analyse_quality(&shapes, self.config.min_confidence);

        // Generate recommendations
        let recommendations = self.generate_recommendations(&quality);

        // Generate predictions
        let predictions = self.generate_predictions(&shapes);

        let duration_ms = start.elapsed().as_millis() as u64;
        let metadata = LearningMetadata {
            duration_ms,
            triples_analysed: stats.triple_count,
            models_used: selection.selected_models.len(),
            orchestrator_version: "0.3.0".to_string(),
        };

        // Update statistics
        self.update_stats(shapes.len(), quality.quality_score);

        Ok(ComprehensiveLearningResult {
            shapes,
            quality,
            recommendations,
            predictions,
            metadata,
        })
    }

    /// Get current orchestrator statistics
    pub fn stats(&self) -> Result<ShapeOrchestratorStats, ShaclAiError> {
        self.stats
            .read()
            .map(|s| s.clone())
            .map_err(|e| ShaclAiError::ModelTraining(e.to_string()))
    }

    /// Get metrics for the last orchestration run
    pub fn get_metrics(&self) -> OrchestrationMetrics {
        let stats = self.stats.read().map(|s| s.clone()).unwrap_or_default();
        OrchestrationMetrics {
            shape_count: stats.total_shapes_discovered,
            avg_confidence: stats.avg_quality_score,
            quality_score: stats.avg_quality_score,
            duration_ms: 0,
            models_used: Vec::new(),
        }
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn discover_shapes(&self, stats: &DataStatistics, models: &[String]) -> Vec<ConfidentShape> {
        // Deterministic placeholder: generate one shape per model per type
        let shape_count = (stats.type_count.min(5)).max(1);
        let mut shapes = Vec::new();

        for i in 0..shape_count {
            let model_name = models
                .get(i % models.len().max(1))
                .cloned()
                .unwrap_or_else(|| "default".to_string());

            shapes.push(ConfidentShape {
                shape_iri: format!("urn:shacl:shape:{i}"),
                confidence: 0.7 + (i as f64 * 0.05).min(0.25),
                target_class: format!("urn:class:type{i}"),
                constraints: {
                    let mut m = std::collections::HashMap::new();
                    m.insert("sh:minCount".to_string(), "1".to_string());
                    m
                },
                evidence_count: stats.triple_count / (i + 1).max(1),
            });

            let _ = model_name; // suppress unused warning
        }

        shapes
    }

    fn analyse_quality(&self, shapes: &[ConfidentShape], min_confidence: f64) -> QualityAnalysis {
        if shapes.is_empty() {
            return QualityAnalysis {
                avg_confidence: 0.0,
                high_confidence_fraction: 0.0,
                estimated_fpr: 0.5,
                estimated_fnr: 0.5,
                quality_score: 0.0,
                verdict: "No shapes discovered".to_string(),
            };
        }

        let avg_confidence = shapes.iter().map(|s| s.confidence).sum::<f64>() / shapes.len() as f64;
        let high_confidence = shapes
            .iter()
            .filter(|s| s.confidence >= min_confidence)
            .count();
        let high_confidence_fraction = high_confidence as f64 / shapes.len() as f64;
        let quality_score = (avg_confidence * 0.6 + high_confidence_fraction * 0.4).min(1.0);

        let verdict = if quality_score >= 0.85 {
            "Excellent"
        } else if quality_score >= 0.7 {
            "Good"
        } else if quality_score >= 0.5 {
            "Fair"
        } else {
            "Poor"
        };

        QualityAnalysis {
            avg_confidence,
            high_confidence_fraction,
            estimated_fpr: 1.0 - avg_confidence,
            estimated_fnr: 1.0 - high_confidence_fraction,
            quality_score,
            verdict: verdict.to_string(),
        }
    }

    fn generate_recommendations(
        &self,
        quality: &QualityAnalysis,
    ) -> Vec<OptimizationRecommendation> {
        let mut recs = Vec::new();

        if quality.avg_confidence < 0.75 {
            recs.push(OptimizationRecommendation {
                title: "Increase training data".to_string(),
                description: "Add more labelled examples to improve confidence scores".to_string(),
                expected_improvement: 0.1,
                effort: ImplementationEffort::Medium,
                priority: 80,
            });
        }

        if quality.estimated_fpr > 0.3 {
            recs.push(OptimizationRecommendation {
                title: "Tune confidence threshold".to_string(),
                description: "Raise the minimum confidence to reduce false positives".to_string(),
                expected_improvement: 0.05,
                effort: ImplementationEffort::Low,
                priority: 70,
            });
        }

        recs
    }

    fn generate_predictions(&self, shapes: &[ConfidentShape]) -> PredictiveInsights {
        let high_risk: Vec<String> = shapes
            .iter()
            .filter(|s| s.confidence < 0.75)
            .map(|s| s.shape_iri.clone())
            .collect();

        let predicted_violations = high_risk.len() * 2; // rough estimate

        PredictiveInsights {
            predicted_violations,
            violation_confidence: 0.7,
            high_risk_shapes: high_risk,
            recommendations: vec![
                "Review low-confidence shapes manually".to_string(),
                "Add more training examples for uncertain constraints".to_string(),
            ],
        }
    }

    fn update_stats(&self, shape_count: usize, quality_score: f64) {
        if let Ok(mut stats) = self.stats.write() {
            stats.total_runs += 1;
            stats.successful_runs += 1;
            stats.total_shapes_discovered += shape_count;
            stats.avg_shapes_per_run =
                stats.total_shapes_discovered as f64 / stats.total_runs as f64;
            // Exponential moving average of quality score
            if stats.total_runs == 1 {
                stats.avg_quality_score = quality_score;
            } else {
                stats.avg_quality_score = stats.avg_quality_score * 0.9 + quality_score * 0.1;
            }
        }
    }
}

impl Default for ShapeOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stats(triple_count: usize, type_count: usize) -> DataStatistics {
        DataStatistics {
            triple_count,
            type_count,
            subject_count: triple_count / 3,
            predicate_count: 10,
            object_count: triple_count / 2,
            ..Default::default()
        }
    }

    #[test]
    fn test_orchestrator_creation() {
        let orch = ShapeOrchestrator::new();
        let stats = orch.stats().expect("stats should succeed");
        assert_eq!(stats.total_runs, 0);
    }

    #[test]
    fn test_orchestrator_with_config() {
        let config = ShapeOrchestratorConfig::high_performance();
        let orch = ShapeOrchestrator::with_config(config);
        let stats = orch.stats().expect("stats should succeed");
        assert_eq!(stats.total_runs, 0);
    }

    #[test]
    fn test_learn_basic() {
        let mut orch = ShapeOrchestrator::new();
        let data_stats = make_stats(1000, 3);
        let result = orch.learn(&data_stats).expect("learn should succeed");
        assert!(!result.shapes.is_empty());
        assert!(result.quality.quality_score >= 0.0);
        assert!(result.quality.quality_score <= 1.0);
    }

    #[test]
    fn test_learn_updates_stats() {
        let mut orch = ShapeOrchestrator::new();
        let data_stats = make_stats(500, 2);
        orch.learn(&data_stats).expect("learn should succeed");
        let stats = orch.stats().expect("stats should succeed");
        assert_eq!(stats.total_runs, 1);
        assert_eq!(stats.successful_runs, 1);
    }

    #[test]
    fn test_learn_multiple_runs() {
        let mut orch = ShapeOrchestrator::new();
        let data_stats = make_stats(1000, 3);
        orch.learn(&data_stats).expect("first run");
        orch.learn(&data_stats).expect("second run");
        let stats = orch.stats().expect("stats");
        assert_eq!(stats.total_runs, 2);
    }

    #[test]
    fn test_quality_analysis_non_zero_shapes() {
        let mut orch = ShapeOrchestrator::new();
        let data_stats = make_stats(2000, 5);
        let result = orch.learn(&data_stats).expect("learn");
        assert!(result.quality.avg_confidence > 0.0);
        assert!(!result.quality.verdict.is_empty());
    }

    #[test]
    fn test_learn_metadata() {
        let mut orch = ShapeOrchestrator::new();
        let data_stats = make_stats(1000, 3);
        let result = orch.learn(&data_stats).expect("learn");
        assert_eq!(result.metadata.triples_analysed, 1000);
        assert!(!result.metadata.orchestrator_version.is_empty());
    }

    #[test]
    fn test_learn_predictions() {
        let mut orch = ShapeOrchestrator::new();
        let data_stats = make_stats(1000, 3);
        let result = orch.learn(&data_stats).expect("learn");
        assert!(result.predictions.violation_confidence >= 0.0);
        assert!(!result.predictions.recommendations.is_empty());
    }

    #[test]
    fn test_default_orchestrator() {
        let orch = ShapeOrchestrator::default();
        assert_eq!(orch.config.max_ensemble_size, 5);
    }
}
