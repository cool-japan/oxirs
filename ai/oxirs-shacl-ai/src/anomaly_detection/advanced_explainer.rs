//! Advanced Explainability for Anomaly Detection
//!
//! This module provides state-of-the-art explainability techniques for anomaly detection,
//! including SHAP-style attribution, natural language generation, and visual explanations.

use super::explainer::{ContributingFactor, Counterfactual, ExplanationReport, RuleViolation};
use super::types::{Anomaly, AnomalyScore, AnomalyType};
use crate::{Result, ShaclAiError};

use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::random::Random;
use scirs2_core::Rng; // Import the Rng trait for gen() method
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Advanced anomaly explainer with multiple explanation techniques
#[derive(Debug)]
pub struct AdvancedAnomalyExplainer {
    /// Configuration
    config: ExplainerConfig,

    /// SHAP explainer for feature attribution
    shap_explainer: ShapExplainer,

    /// Natural language generator
    nlg_generator: NaturalLanguageGenerator,

    /// Decision tree explainer
    tree_explainer: DecisionTreeExplainer,

    /// Explanation cache
    explanation_cache: HashMap<String, CachedExplanation>,

    /// Random number generator
    rng: Random,
}

/// Configuration for advanced explainer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainerConfig {
    /// Enable SHAP explanations
    pub enable_shap: bool,

    /// Enable natural language explanations
    pub enable_nlg: bool,

    /// Enable decision tree explanations
    pub enable_decision_trees: bool,

    /// Number of samples for SHAP
    pub shap_samples: usize,

    /// Minimum feature importance threshold
    pub min_importance_threshold: f64,

    /// Maximum explanation length (characters)
    pub max_explanation_length: usize,

    /// Enable caching
    pub enable_caching: bool,

    /// Cache size limit
    pub cache_size_limit: usize,

    /// Explanation detail level
    pub detail_level: ExplanationDetailLevel,

    /// Include visual data
    pub include_visual_data: bool,
}

impl Default for ExplainerConfig {
    fn default() -> Self {
        Self {
            enable_shap: true,
            enable_nlg: true,
            enable_decision_trees: true,
            shap_samples: 1000,
            min_importance_threshold: 0.05,
            max_explanation_length: 2000,
            enable_caching: true,
            cache_size_limit: 1000,
            detail_level: ExplanationDetailLevel::Detailed,
            include_visual_data: true,
        }
    }
}

/// Explanation detail level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExplanationDetailLevel {
    /// Brief summary only
    Brief,
    /// Standard detail
    Standard,
    /// Detailed explanation
    Detailed,
    /// Comprehensive with all analyses
    Comprehensive,
}

/// Cached explanation
#[derive(Debug, Clone)]
struct CachedExplanation {
    report: AdvancedExplanationReport,
    timestamp: std::time::Instant,
}

impl AdvancedAnomalyExplainer {
    /// Create a new advanced explainer
    pub fn new(config: ExplainerConfig) -> Self {
        Self {
            config: config.clone(),
            shap_explainer: ShapExplainer::new(config.shap_samples),
            nlg_generator: NaturalLanguageGenerator::new(config.max_explanation_length),
            tree_explainer: DecisionTreeExplainer::new(),
            explanation_cache: HashMap::new(),
            rng: Random::default(),
        }
    }

    /// Generate comprehensive explanation for an anomaly
    pub fn explain_comprehensive(
        &mut self,
        anomaly: &Anomaly,
        reference_data: Option<&Array2<f64>>,
    ) -> Result<AdvancedExplanationReport> {
        // Check cache first
        if self.config.enable_caching {
            if let Some(cached) = self.explanation_cache.get(&anomaly.id) {
                // Return cached if fresh (within 5 minutes)
                if cached.timestamp.elapsed().as_secs() < 300 {
                    return Ok(cached.report.clone());
                }
            }
        }

        let mut report = AdvancedExplanationReport {
            anomaly_id: anomaly.id.clone(),
            anomaly_type: anomaly.anomaly_type,
            severity: anomaly.score.score,
            primary_explanation: String::new(),
            detailed_explanations: Vec::new(),
            feature_attributions: HashMap::new(),
            natural_language_summary: String::new(),
            decision_path: Vec::new(),
            remediation_suggestions: Vec::new(),
            confidence_breakdown: ConfidenceBreakdown::default(),
            visualization_data: None,
        };

        // Generate SHAP explanations if enabled
        if self.config.enable_shap {
            if let Some(ref_data) = reference_data {
                let attributions = self.shap_explainer.compute_shap_values(anomaly, ref_data)?;
                report.feature_attributions = attributions.clone();

                // Add SHAP explanation
                report.detailed_explanations.push(DetailedExplanation {
                    technique: ExplanationTechnique::SHAP,
                    title: "Feature Attribution Analysis".to_string(),
                    content: self.format_shap_explanation(&attributions),
                    confidence: 0.9,
                });
            }
        }

        // Generate decision tree explanation if enabled
        if self.config.enable_decision_trees {
            let decision_path = self.tree_explainer.explain_decision(anomaly)?;
            report.decision_path = decision_path.clone();

            report.detailed_explanations.push(DetailedExplanation {
                technique: ExplanationTechnique::DecisionTree,
                title: "Decision Path Analysis".to_string(),
                content: self.format_decision_path(&decision_path),
                confidence: 0.85,
            });
        }

        // Generate natural language summary if enabled
        if self.config.enable_nlg {
            let nl_summary = self.nlg_generator.generate_explanation(
                anomaly,
                &report.feature_attributions,
                &report.decision_path,
            )?;
            report.natural_language_summary = nl_summary.clone();
            report.primary_explanation = nl_summary;
        }

        // Generate remediation suggestions
        report.remediation_suggestions =
            self.generate_remediation_suggestions(anomaly, &report.feature_attributions)?;

        // Calculate confidence breakdown
        report.confidence_breakdown = self.calculate_confidence_breakdown(anomaly);

        // Generate visualization data if enabled
        if self.config.include_visual_data {
            report.visualization_data =
                Some(self.generate_visualization_data(anomaly, &report.feature_attributions)?);
        }

        // Cache the result
        if self.config.enable_caching {
            self.cache_explanation(anomaly.id.clone(), report.clone());
        }

        Ok(report)
    }

    /// Format SHAP explanation
    fn format_shap_explanation(&self, attributions: &HashMap<String, f64>) -> String {
        let mut sorted_attrs: Vec<_> = attributions.iter().collect();
        sorted_attrs.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        let mut explanation = String::from("Top contributing features:\n");
        for (i, (feature, &value)) in sorted_attrs.iter().take(5).enumerate() {
            let direction = if value > 0.0 {
                "increases"
            } else {
                "decreases"
            };
            explanation.push_str(&format!(
                "{}. {} {} anomaly score by {:.3}\n",
                i + 1,
                feature,
                direction,
                value.abs()
            ));
        }

        explanation
    }

    /// Format decision path
    fn format_decision_path(&self, path: &[DecisionNode]) -> String {
        let mut explanation = String::from("Decision path:\n");
        for (i, node) in path.iter().enumerate() {
            explanation.push_str(&format!(
                "{}. {} (split: {}, threshold: {:.3})\n",
                i + 1,
                node.description,
                node.split_feature,
                node.threshold
            ));
        }

        explanation
    }

    /// Generate remediation suggestions
    fn generate_remediation_suggestions(
        &self,
        anomaly: &Anomaly,
        attributions: &HashMap<String, f64>,
    ) -> Result<Vec<RemediationSuggestion>> {
        let mut suggestions = Vec::new();

        // Find most impactful features
        let mut sorted_attrs: Vec<_> = attributions.iter().collect();
        sorted_attrs.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        for (feature, &attribution) in sorted_attrs.iter().take(3) {
            suggestions.push(RemediationSuggestion {
                feature: feature.to_string(),
                current_impact: attribution,
                suggested_action: if attribution > 0.0 {
                    format!("Reduce {} to decrease anomaly score", feature)
                } else {
                    format!("Increase {} to decrease anomaly score", feature)
                },
                expected_improvement: attribution.abs() * 0.7,
                priority: if attribution.abs() > 0.5 {
                    Priority::High
                } else {
                    Priority::Medium
                },
            });
        }

        Ok(suggestions)
    }

    /// Calculate confidence breakdown
    fn calculate_confidence_breakdown(&self, anomaly: &Anomaly) -> ConfidenceBreakdown {
        ConfidenceBreakdown {
            overall_confidence: anomaly.score.confidence,
            detection_confidence: anomaly.score.confidence,
            attribution_confidence: 0.85,
            explanation_confidence: 0.80,
            prediction_confidence: 0.75,
        }
    }

    /// Generate visualization data
    fn generate_visualization_data(
        &self,
        anomaly: &Anomaly,
        attributions: &HashMap<String, f64>,
    ) -> Result<VisualizationData> {
        Ok(VisualizationData {
            feature_importance_plot: self.create_importance_plot(attributions),
            anomaly_score_timeline: vec![anomaly.score.score],
            comparison_distribution: self.create_distribution_data(),
        })
    }

    /// Create feature importance plot data
    fn create_importance_plot(&self, attributions: &HashMap<String, f64>) -> Vec<(String, f64)> {
        let mut sorted: Vec<_> = attributions.iter().map(|(k, &v)| (k.clone(), v)).collect();
        sorted.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        sorted.truncate(10);
        sorted
    }

    /// Create distribution comparison data
    fn create_distribution_data(&self) -> Vec<f64> {
        // Placeholder - in production would use actual reference distribution
        vec![0.1, 0.3, 0.5, 0.7, 0.9]
    }

    /// Cache an explanation
    fn cache_explanation(&mut self, anomaly_id: String, report: AdvancedExplanationReport) {
        // Evict old entries if cache is full
        if self.explanation_cache.len() >= self.config.cache_size_limit {
            // Remove oldest entry
            if let Some(oldest_key) = self.explanation_cache.keys().next().cloned() {
                self.explanation_cache.remove(&oldest_key);
            }
        }

        self.explanation_cache.insert(
            anomaly_id,
            CachedExplanation {
                report,
                timestamp: std::time::Instant::now(),
            },
        );
    }

    /// Get configuration
    pub fn config(&self) -> &ExplainerConfig {
        &self.config
    }
}

/// Advanced explanation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedExplanationReport {
    /// Anomaly ID
    pub anomaly_id: String,

    /// Anomaly type
    pub anomaly_type: AnomalyType,

    /// Severity score
    pub severity: f64,

    /// Primary explanation (brief)
    pub primary_explanation: String,

    /// Detailed explanations from different techniques
    pub detailed_explanations: Vec<DetailedExplanation>,

    /// Feature attributions (SHAP values)
    pub feature_attributions: HashMap<String, f64>,

    /// Natural language summary
    pub natural_language_summary: String,

    /// Decision path
    pub decision_path: Vec<DecisionNode>,

    /// Remediation suggestions
    pub remediation_suggestions: Vec<RemediationSuggestion>,

    /// Confidence breakdown
    pub confidence_breakdown: ConfidenceBreakdown,

    /// Visualization data
    pub visualization_data: Option<VisualizationData>,
}

/// Detailed explanation from a specific technique
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedExplanation {
    pub technique: ExplanationTechnique,
    pub title: String,
    pub content: String,
    pub confidence: f64,
}

/// Explanation technique
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExplanationTechnique {
    SHAP,
    LIME,
    DecisionTree,
    NaturalLanguage,
    Counterfactual,
}

/// Decision node in explanation tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionNode {
    pub description: String,
    pub split_feature: String,
    pub threshold: f64,
    pub samples: usize,
    pub is_anomaly: bool,
}

/// Remediation suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationSuggestion {
    pub feature: String,
    pub current_impact: f64,
    pub suggested_action: String,
    pub expected_improvement: f64,
    pub priority: Priority,
}

/// Priority level
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// Confidence breakdown
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConfidenceBreakdown {
    pub overall_confidence: f64,
    pub detection_confidence: f64,
    pub attribution_confidence: f64,
    pub explanation_confidence: f64,
    pub prediction_confidence: f64,
}

/// Visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationData {
    pub feature_importance_plot: Vec<(String, f64)>,
    pub anomaly_score_timeline: Vec<f64>,
    pub comparison_distribution: Vec<f64>,
}

/// SHAP explainer for feature attribution
#[derive(Debug)]
struct ShapExplainer {
    num_samples: usize,
    rng: Random,
}

impl ShapExplainer {
    fn new(num_samples: usize) -> Self {
        Self {
            num_samples,
            rng: Random::default(),
        }
    }

    /// Compute SHAP values for features
    fn compute_shap_values(
        &mut self,
        anomaly: &Anomaly,
        reference_data: &Array2<f64>,
    ) -> Result<HashMap<String, f64>> {
        let mut shap_values = HashMap::new();

        // Simplified SHAP computation - in production would use proper Shapley value calculation
        let num_features = reference_data.ncols().min(10);

        for feature_idx in 0..num_features {
            // Compute marginal contribution
            let contribution =
                self.compute_marginal_contribution(anomaly, feature_idx, reference_data)?;

            shap_values.insert(format!("feature_{}", feature_idx), contribution);
        }

        Ok(shap_values)
    }

    /// Compute marginal contribution of a feature
    fn compute_marginal_contribution(
        &mut self,
        anomaly: &Anomaly,
        feature_idx: usize,
        reference_data: &Array2<f64>,
    ) -> Result<f64> {
        let mut total_contribution = 0.0;
        let samples = self.num_samples.min(reference_data.nrows());

        for _ in 0..samples {
            // Sample a random row
            let row_idx = (self.rng.random::<f64>() * reference_data.nrows() as f64) as usize
                % reference_data.nrows();

            // Compute contribution (simplified)
            if feature_idx < reference_data.ncols() {
                let value = reference_data[[row_idx, feature_idx]];
                total_contribution += value * anomaly.score.score;
            }
        }

        Ok(total_contribution / samples as f64)
    }
}

/// Natural language explanation generator
#[derive(Debug)]
struct NaturalLanguageGenerator {
    max_length: usize,
}

impl NaturalLanguageGenerator {
    fn new(max_length: usize) -> Self {
        Self { max_length }
    }

    /// Generate natural language explanation
    fn generate_explanation(
        &self,
        anomaly: &Anomaly,
        attributions: &HashMap<String, f64>,
        decision_path: &[DecisionNode],
    ) -> Result<String> {
        let mut explanation = format!(
            "This {} anomaly was detected with {:.1}% confidence. ",
            self.format_anomaly_type(&anomaly.anomaly_type),
            anomaly.score.confidence * 100.0
        );

        // Add top contributing factors
        if !attributions.is_empty() {
            let mut sorted_attrs: Vec<_> = attributions.iter().collect();
            sorted_attrs.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

            explanation.push_str("The primary contributing factors are: ");
            for (i, (feature, &value)) in sorted_attrs.iter().take(3).enumerate() {
                if i > 0 {
                    explanation.push_str(", ");
                }
                let impact = if value > 0.0 {
                    "increases"
                } else {
                    "decreases"
                };
                explanation.push_str(&format!("{} (which {} the score)", feature, impact));
            }
            explanation.push_str(". ");
        }

        // Add decision path summary
        if !decision_path.is_empty() {
            explanation.push_str(&format!(
                "The anomaly was identified through {} decision steps. ",
                decision_path.len()
            ));
        }

        // Truncate if too long
        if explanation.len() > self.max_length {
            explanation.truncate(self.max_length - 3);
            explanation.push_str("...");
        }

        Ok(explanation)
    }

    fn format_anomaly_type(&self, anomaly_type: &AnomalyType) -> &str {
        match anomaly_type {
            AnomalyType::Outlier => "outlier",
            AnomalyType::ContextualAnomaly => "contextual",
            AnomalyType::CollectiveAnomaly => "collective",
            AnomalyType::NovelPattern => "novel pattern",
            AnomalyType::DataDistributionDrift => "distribution drift",
            AnomalyType::ConstraintViolationPattern => "constraint violation",
        }
    }
}

/// Decision tree explainer
#[derive(Debug)]
struct DecisionTreeExplainer;

impl DecisionTreeExplainer {
    fn new() -> Self {
        Self
    }

    /// Explain decision using decision tree
    fn explain_decision(&self, anomaly: &Anomaly) -> Result<Vec<DecisionNode>> {
        let mut decision_path = Vec::new();

        // Simplified decision tree - in production would use actual trained tree
        decision_path.push(DecisionNode {
            description: "Root node: Anomaly score evaluation".to_string(),
            split_feature: "anomaly_score".to_string(),
            threshold: 0.5,
            samples: 1000,
            is_anomaly: anomaly.score.score > 0.5,
        });

        if anomaly.score.score > 0.7 {
            decision_path.push(DecisionNode {
                description: "High severity branch".to_string(),
                split_feature: "severity".to_string(),
                threshold: 0.7,
                samples: 200,
                is_anomaly: true,
            });
        } else {
            decision_path.push(DecisionNode {
                description: "Moderate severity branch".to_string(),
                split_feature: "severity".to_string(),
                threshold: 0.5,
                samples: 800,
                is_anomaly: anomaly.score.score > 0.5,
            });
        }

        Ok(decision_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_advanced_explainer_creation() {
        let config = ExplainerConfig::default();
        let explainer = AdvancedAnomalyExplainer::new(config);

        assert!(explainer.config.enable_shap);
        assert!(explainer.config.enable_nlg);
    }

    #[test]
    fn test_shap_explainer() {
        let mut shap = ShapExplainer::new(100);
        let reference_data = Array2::zeros((10, 5));

        let anomaly = Anomaly {
            id: "test".to_string(),
            anomaly_type: AnomalyType::Outlier,
            description: "Test anomaly".to_string(),
            score: AnomalyScore {
                score: 0.8,
                confidence: 0.9,
                factors: HashMap::new(),
                threshold: 0.7,
            },
            affected_entities: Vec::new(),
            timestamp: chrono::Utc::now(),
            context: HashMap::new(),
            recommendations: Vec::new(),
        };

        let result = shap.compute_shap_values(&anomaly, &reference_data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_nlg_generator() {
        let nlg = NaturalLanguageGenerator::new(2000);

        let anomaly = Anomaly {
            id: "test".to_string(),
            anomaly_type: AnomalyType::ContextualAnomaly,
            description: "Test anomaly".to_string(),
            score: AnomalyScore {
                score: 0.85,
                confidence: 0.92,
                factors: HashMap::new(),
                threshold: 0.75,
            },
            affected_entities: Vec::new(),
            timestamp: chrono::Utc::now(),
            context: HashMap::new(),
            recommendations: Vec::new(),
        };

        let mut attributions = HashMap::new();
        attributions.insert("feature_1".to_string(), 0.5);
        attributions.insert("feature_2".to_string(), -0.3);

        let explanation = nlg
            .generate_explanation(&anomaly, &attributions, &[])
            .unwrap();
        assert!(!explanation.is_empty());
        assert!(explanation.contains("contextual"));
    }
}
