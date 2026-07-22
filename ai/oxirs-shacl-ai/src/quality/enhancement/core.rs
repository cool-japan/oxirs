//! Core quality enhancement engine and configuration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::quality::QualityReport;
use crate::Result;

/// Quality enhancement recommendations engine
#[derive(Debug)]
pub struct QualityEnhancementEngine {
    config: EnhancementConfig,
    recommendation_models: RecommendationModels,
    enhancement_history: Vec<EnhancementAction>,
    statistics: EnhancementStatistics,
}

/// Configuration for quality enhancement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancementConfig {
    /// Enable data enhancement recommendations
    pub enable_data_enhancement: bool,

    /// Enable process optimization recommendations
    pub enable_process_optimization: bool,

    /// Enable automated improvements
    pub enable_automated_improvements: bool,

    /// Enhancement priority threshold
    pub priority_threshold: f64,

    /// Maximum recommendations per category
    pub max_recommendations_per_category: usize,

    /// Minimum confidence for recommendations
    pub min_recommendation_confidence: f64,

    /// Enable cost-benefit analysis
    pub enable_cost_benefit_analysis: bool,

    /// Enable impact prediction
    pub enable_impact_prediction: bool,

    /// Enhancement strategy preference
    pub strategy_preference: EnhancementStrategy,
}

impl Default for EnhancementConfig {
    fn default() -> Self {
        Self {
            enable_data_enhancement: true,
            enable_process_optimization: true,
            enable_automated_improvements: true,
            priority_threshold: 0.7,
            max_recommendations_per_category: 10,
            min_recommendation_confidence: 0.75,
            enable_cost_benefit_analysis: true,
            enable_impact_prediction: true,
            strategy_preference: EnhancementStrategy::Balanced,
        }
    }
}

/// Enhancement strategy preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnhancementStrategy {
    Conservative, // Focus on low-risk, high-confidence improvements
    Balanced,     // Balance between impact and risk
    Aggressive,   // Focus on high-impact improvements
    Automated,    // Prefer automated solutions
    Manual,       // Prefer manual interventions
}

/// Recommendation models for different enhancement types
#[derive(Debug, Default)]
pub struct RecommendationModels {
    pub data_enhancement_model: Option<DataEnhancementModel>,
    pub process_optimization_model: Option<ProcessOptimizationModel>,
    pub automation_model: Option<AutomationModel>,
    pub impact_prediction_model: Option<ImpactPredictionModel>,
    pub cost_benefit_model: Option<CostBenefitModel>,
}

/// Enhancement action history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancementAction {
    pub action_id: String,
    pub action_type: EnhancementActionType,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub status: ActionStatus,
    pub metadata: HashMap<String, String>,
}

/// Enhancement action types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnhancementActionType {
    DataQualityImprovement,
    ProcessOptimization,
    AutomatedValidation,
    SchemaEnhancement,
    PerformanceOptimization,
    ErrorCorrection,
}

/// Action execution status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

/// Enhancement statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancementStatistics {
    pub total_recommendations: u64,
    pub successful_improvements: u64,
    pub average_improvement_score: f64,
    pub total_cost_savings: f64,
    pub processing_time_improvements: f64,
}

/// Data enhancement model
#[derive(Debug)]
pub struct DataEnhancementModel {
    pub model_version: String,
    pub accuracy: f64,
}

/// Process optimization model
#[derive(Debug)]
pub struct ProcessOptimizationModel {
    pub model_version: String,
    pub optimization_score: f64,
}

/// Automation model
#[derive(Debug)]
pub struct AutomationModel {
    pub model_version: String,
    pub automation_level: f64,
}

/// Impact prediction model
#[derive(Debug)]
pub struct ImpactPredictionModel {
    pub model_version: String,
    pub prediction_accuracy: f64,
}

/// Cost-benefit analysis model
#[derive(Debug)]
pub struct CostBenefitModel {
    pub model_version: String,
    pub analysis_accuracy: f64,
}

impl QualityEnhancementEngine {
    /// Create new quality enhancement engine
    pub fn new(config: EnhancementConfig) -> Self {
        Self {
            config,
            recommendation_models: RecommendationModels::default(),
            enhancement_history: Vec::new(),
            statistics: EnhancementStatistics::default(),
        }
    }

    /// Generate enhancement recommendations
    pub async fn generate_recommendations(
        &self,
        quality_report: &QualityReport,
    ) -> Result<Vec<EnhancementRecommendation>> {
        let mut recommendations = Vec::new();

        if self.config.enable_data_enhancement {
            recommendations.extend(
                self.generate_data_enhancement_recommendations(quality_report)
                    .await?,
            );
        }

        if self.config.enable_process_optimization {
            recommendations.extend(
                self.generate_process_optimization_recommendations(quality_report)
                    .await?,
            );
        }

        if self.config.enable_automated_improvements {
            recommendations.extend(
                self.generate_automation_recommendations(quality_report)
                    .await?,
            );
        }

        // Filter by confidence threshold
        recommendations.retain(|r| r.confidence >= self.config.min_recommendation_confidence);

        Ok(recommendations)
    }

    /// Get configuration
    pub fn config(&self) -> &EnhancementConfig {
        &self.config
    }

    /// Get statistics
    pub fn statistics(&self) -> &EnhancementStatistics {
        &self.statistics
    }

    // Private helper methods

    /// Map a dimension score in `[0, 1]` to a priority: the worse the score, the
    /// higher the priority. Returns `None` when the dimension is healthy enough
    /// that no recommendation is warranted.
    fn priority_for_score(score: f64) -> Option<Priority> {
        if score < 0.5 {
            Some(Priority::Critical)
        } else if score < 0.7 {
            Some(Priority::High)
        } else if score < 0.85 {
            Some(Priority::Medium)
        } else {
            None
        }
    }

    /// Data-enhancement recommendations derived from the actual scored
    /// dimensions of `quality_report` (completeness, consistency, accuracy,
    /// duplicates, schema adherence). Dimensions that are already healthy
    /// produce no recommendation.
    async fn generate_data_enhancement_recommendations(
        &self,
        quality_report: &QualityReport,
    ) -> Result<Vec<EnhancementRecommendation>> {
        let mut recs = Vec::new();

        // Each entry: (id, dimension score, title, description, estimated impact).
        let dimensions: &[(&str, f64, &str, &str)] = &[
            (
                "data_completeness",
                quality_report.completeness_score,
                "Improve data completeness",
                "Populate missing required properties flagged by the completeness assessment",
            ),
            (
                "data_consistency",
                quality_report.consistency_score,
                "Improve data consistency",
                "Reconcile inconsistent values detected across the dataset",
            ),
            (
                "data_accuracy",
                quality_report.accuracy_score,
                "Improve data accuracy",
                "Correct values that fail datatype/range accuracy checks",
            ),
            (
                "schema_adherence",
                quality_report.schema_adherence_score,
                "Improve schema adherence",
                "Align data with the expected schema/shape definitions",
            ),
        ];

        for (id, score, title, description) in dimensions {
            if let Some(priority) = Self::priority_for_score(*score) {
                recs.push(EnhancementRecommendation {
                    id: (*id).to_string(),
                    title: (*title).to_string(),
                    description: format!("{description} (current score: {score:.2})"),
                    category: EnhancementCategory::DataQuality,
                    priority,
                    // Confidence scales with how far below perfect the score is.
                    confidence: (0.6 + (1.0 - score) * 0.4).clamp(0.0, 1.0),
                    estimated_impact: (1.0 - score).clamp(0.0, 1.0),
                    implementation_effort: ImplementationEffort::Medium,
                    automated: false,
                });
            }
        }

        // Duplicates are a ratio (higher is worse), so handle separately.
        if quality_report.duplicate_ratio > 0.05 {
            let ratio = quality_report.duplicate_ratio;
            recs.push(EnhancementRecommendation {
                id: "data_deduplication".to_string(),
                title: "Deduplicate records".to_string(),
                description: format!(
                    "Merge or remove duplicate entities (duplicate ratio: {ratio:.2})"
                ),
                category: EnhancementCategory::DataQuality,
                priority: if ratio > 0.3 {
                    Priority::Critical
                } else if ratio > 0.15 {
                    Priority::High
                } else {
                    Priority::Medium
                },
                confidence: (0.6 + ratio * 0.4).clamp(0.0, 1.0),
                estimated_impact: ratio.clamp(0.0, 1.0),
                implementation_effort: ImplementationEffort::Medium,
                automated: false,
            });
        }

        Ok(recs)
    }

    /// Process-optimization recommendations. Conformance and overall issue
    /// volume drive whether validation-workflow changes are worthwhile.
    async fn generate_process_optimization_recommendations(
        &self,
        quality_report: &QualityReport,
    ) -> Result<Vec<EnhancementRecommendation>> {
        let mut recs = Vec::new();

        if let Some(priority) = Self::priority_for_score(quality_report.conformance_score) {
            recs.push(EnhancementRecommendation {
                id: "process_conformance".to_string(),
                title: "Optimize validation workflow".to_string(),
                description: format!(
                    "Tighten the validation pipeline to raise shape conformance (current score: {:.2}, {} open issues)",
                    quality_report.conformance_score,
                    quality_report.issues.len()
                ),
                category: EnhancementCategory::ProcessOptimization,
                priority,
                confidence: (0.6 + (1.0 - quality_report.conformance_score) * 0.4).clamp(0.0, 1.0),
                estimated_impact: (1.0 - quality_report.conformance_score).clamp(0.0, 1.0),
                implementation_effort: ImplementationEffort::High,
                automated: false,
            });
        }

        Ok(recs)
    }

    /// Automation recommendations. Only proposed when there is a meaningful
    /// volume of recurring issues that automation could address.
    async fn generate_automation_recommendations(
        &self,
        quality_report: &QualityReport,
    ) -> Result<Vec<EnhancementRecommendation>> {
        let mut recs = Vec::new();

        let issue_count = quality_report.issues.len();
        if issue_count >= 5 {
            // Impact grows with issue volume but saturates.
            let impact = (issue_count as f64 / 50.0).clamp(0.3, 1.0);
            recs.push(EnhancementRecommendation {
                id: "automation_error_correction".to_string(),
                title: "Automated error correction".to_string(),
                description: format!(
                    "Implement automated correction for the {issue_count} recurring quality issues detected"
                ),
                category: EnhancementCategory::Automation,
                priority: if issue_count >= 20 {
                    Priority::High
                } else {
                    Priority::Medium
                },
                confidence: 0.75,
                estimated_impact: impact,
                implementation_effort: ImplementationEffort::High,
                automated: true,
            });
        }

        Ok(recs)
    }
}

/// Enhancement recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancementRecommendation {
    pub id: String,
    pub title: String,
    pub description: String,
    pub category: EnhancementCategory,
    pub priority: Priority,
    pub confidence: f64,
    pub estimated_impact: f64,
    pub implementation_effort: ImplementationEffort,
    pub automated: bool,
}

/// Enhancement categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnhancementCategory {
    DataQuality,
    ProcessOptimization,
    Automation,
    Performance,
    Security,
    Usability,
}

/// Priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// Implementation effort levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Low,
    Medium,
    High,
    VeryHigh,
}

impl Default for QualityEnhancementEngine {
    fn default() -> Self {
        Self::new(EnhancementConfig::default())
    }
}

impl Default for EnhancementStatistics {
    fn default() -> Self {
        Self {
            total_recommendations: 0,
            successful_improvements: 0,
            average_improvement_score: 0.0,
            total_cost_savings: 0.0,
            processing_time_improvements: 0.0,
        }
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;
    use crate::quality::QualityReport;

    fn perfect_report() -> QualityReport {
        let mut r = QualityReport::new();
        r.completeness_score = 1.0;
        r.consistency_score = 1.0;
        r.accuracy_score = 1.0;
        r.conformance_score = 1.0;
        r.schema_adherence_score = 1.0;
        r.duplicate_ratio = 0.0;
        r
    }

    fn broken_report() -> QualityReport {
        let mut r = QualityReport::new();
        r.completeness_score = 0.3;
        r.consistency_score = 0.4;
        r.accuracy_score = 0.35;
        r.conformance_score = 0.4;
        r.schema_adherence_score = 0.3;
        r.duplicate_ratio = 0.4;
        r
    }

    /// Regression: a perfect report must yield no data-enhancement
    /// recommendations (the old code always emitted the same 3 canned recs).
    #[tokio::test]
    async fn regression_perfect_report_yields_no_recommendations() {
        let engine = QualityEnhancementEngine::default();
        let recs = engine
            .generate_recommendations(&perfect_report())
            .await
            .expect("recommendation generation should succeed");
        assert!(
            recs.is_empty(),
            "a perfect report should produce no recommendations, got {recs:?}"
        );
    }

    /// Regression: a badly broken report must surface recommendations that
    /// reflect the actually-failing dimensions.
    #[tokio::test]
    async fn regression_broken_report_yields_targeted_recommendations() {
        let engine = QualityEnhancementEngine::default();
        let recs = engine
            .generate_recommendations(&broken_report())
            .await
            .expect("recommendation generation should succeed");
        assert!(!recs.is_empty(), "broken report must yield recommendations");
        // A deduplication recommendation must appear because duplicate_ratio is high.
        assert!(
            recs.iter().any(|r| r.id == "data_deduplication"),
            "expected a deduplication recommendation, got {recs:?}"
        );
    }
}
