//! Data types and structures for predictive analytics

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Comprehensive predictive insights result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveInsights {
    pub forecasts: Vec<Forecast>,
    pub recommendations: Vec<IntelligentRecommendation>,
    pub trend_predictions: Vec<TrendPrediction>,
    pub risk_assessments: Vec<RiskAssessment>,
    pub improvement_opportunities: Vec<ImprovementOpportunity>,
    pub generation_timestamp: chrono::DateTime<chrono::Utc>,
    pub generation_time: Duration,
}

impl PredictiveInsights {
    pub fn new() -> Self {
        Self {
            forecasts: Vec::new(),
            recommendations: Vec::new(),
            trend_predictions: Vec::new(),
            risk_assessments: Vec::new(),
            improvement_opportunities: Vec::new(),
            generation_timestamp: chrono::Utc::now(),
            generation_time: Duration::from_secs(0),
        }
    }
}

impl Default for PredictiveInsights {
    fn default() -> Self {
        Self::new()
    }
}

/// Forecast result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Forecast {
    pub category: String,
    pub metric_name: String,
    pub current_value: f64,
    pub predicted_values: Vec<ForecastPoint>,
    pub confidence: f64,
    pub trend: Option<TrendInfo>,
    pub time_horizon: Duration,
    pub methodology: String,
}

/// Individual forecast point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastPoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub value: f64,
    pub confidence_interval: ConfidenceInterval,
}

/// Confidence interval for forecasts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub confidence_level: f64,
}

/// Trend information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendInfo {
    pub direction: TrendDirection,
    pub magnitude: f64,
    pub stability: f64,
    pub seasonal_component: Option<SeasonalInfo>,
}

/// Trend direction
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Cyclical,
    Volatile,
}

/// Seasonal information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalInfo {
    pub period: Duration,
    pub amplitude: f64,
    pub phase: f64,
}

/// Intelligent recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntelligentRecommendation {
    pub id: String,
    pub title: String,
    pub description: String,
    pub category: RecommendationCategory,
    pub priority: RecommendationPriority,
    pub score: f64,
    pub confidence: f64,
    pub expected_impact: ExpectedImpact,
    pub implementation_effort: ImplementationEffort,
    pub prerequisites: Vec<String>,
    pub implementation_steps: Vec<ImplementationStep>,
    pub success_metrics: Vec<SuccessMetric>,
    pub risks: Vec<String>,
    pub alternatives: Vec<Alternative>,
}

/// Recommendation categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Performance,
    Quality,
    Optimization,
    Configuration,
    ShapeDesign,
    DataCleaning,
    ProcessImprovement,
}

/// Recommendation priorities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}

/// Expected impact of a recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedImpact {
    pub performance_improvement: f64,
    pub quality_improvement: f64,
    pub efficiency_gain: f64,
    pub cost_reduction: f64,
}

/// Implementation effort estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationEffort {
    pub time_estimate: Duration,
    pub complexity: EffortComplexity,
    pub required_expertise: Vec<String>,
    pub resource_requirements: Vec<String>,
}

/// Effort complexity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffortComplexity {
    Simple,
    Moderate,
    Complex,
    Expert,
}

/// Implementation step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationStep {
    pub step_number: usize,
    pub title: String,
    pub description: String,
    pub estimated_time: Duration,
    pub prerequisites: Vec<String>,
}

/// Success metric for measuring recommendation effectiveness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessMetric {
    pub name: String,
    pub target_value: f64,
    pub measurement_method: String,
    pub timeframe: Duration,
}

/// Alternative recommendation option
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alternative {
    pub title: String,
    pub description: String,
    pub score: f64,
    pub pros: Vec<String>,
    pub cons: Vec<String>,
}

/// Trend prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendPrediction {
    pub metric: String,
    pub current_trend: TrendDirection,
    pub predicted_trend: TrendDirection,
    pub confidence: f64,
    pub turning_point: Option<chrono::DateTime<chrono::Utc>>,
    pub factors: Vec<String>,
}

/// Risk assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub risk_category: String,
    pub risk_level: RiskLevel,
    pub probability: f64,
    pub impact: f64,
    pub description: String,
    pub mitigation_strategies: Vec<String>,
    pub early_warning_indicators: Vec<String>,
}

/// Risk levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Improvement opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementOpportunity {
    pub area: String,
    pub description: String,
    pub potential_benefit: f64,
    pub implementation_difficulty: EffortComplexity,
    pub prerequisites: Vec<String>,
    pub success_indicators: Vec<String>,
}

/// Time series data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesDataPoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub value: f64,
    pub metadata: std::collections::HashMap<String, String>,
}

/// Detected trend information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedTrend {
    pub direction: TrendDirection,
    pub magnitude: f64,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub duration: Duration,
    pub statistical_significance: f64,
}

/// Statistics for predictive analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveAnalyticsStatistics {
    pub total_forecasts_generated: u64,
    pub total_recommendations_generated: u64,
    pub average_forecast_accuracy: f64,
    pub average_recommendation_acceptance_rate: f64,
    pub processing_time_statistics: ProcessingTimeStats,
}

/// Processing time statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingTimeStats {
    pub average_processing_time: Duration,
    pub min_processing_time: Duration,
    pub max_processing_time: Duration,
    pub total_processing_time: Duration,
}
