//! Configuration for predictive analytics

use serde::{Deserialize, Serialize};

/// Configuration for predictive analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveAnalyticsConfig {
    /// Enable time series forecasting
    pub enable_forecasting: bool,

    /// Enable recommendation systems
    pub enable_recommendations: bool,

    /// Forecasting horizon in days
    pub forecasting_horizon_days: u32,

    /// Minimum historical data points for forecasting
    pub min_historical_points: usize,

    /// Confidence threshold for predictions
    pub prediction_confidence_threshold: f64,

    /// Enable trend analysis
    pub enable_trend_analysis: bool,

    /// Trend detection sensitivity
    pub trend_detection_sensitivity: f64,

    /// Enable seasonality detection
    pub enable_seasonality_detection: bool,

    /// Recommendation scoring threshold
    pub recommendation_score_threshold: f64,

    /// Maximum recommendations per category
    pub max_recommendations_per_category: usize,

    /// Enable adaptive learning
    pub enable_adaptive_learning: bool,
}

impl Default for PredictiveAnalyticsConfig {
    fn default() -> Self {
        Self {
            enable_forecasting: true,
            enable_recommendations: true,
            forecasting_horizon_days: 30,
            min_historical_points: 20,
            prediction_confidence_threshold: 0.7,
            enable_trend_analysis: true,
            trend_detection_sensitivity: 0.1,
            enable_seasonality_detection: true,
            recommendation_score_threshold: 0.6,
            max_recommendations_per_category: 10,
            enable_adaptive_learning: true,
        }
    }
}

/// Recommendation engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationConfig {
    /// Enable performance recommendations
    pub enable_performance_recommendations: bool,

    /// Enable quality recommendations
    pub enable_quality_recommendations: bool,

    /// Enable optimization recommendations
    pub enable_optimization_recommendations: bool,

    /// Enable proactive recommendations
    pub enable_proactive_recommendations: bool,

    /// Recommendation scoring algorithm
    pub scoring_algorithm: ScoringAlgorithm,

    /// Personalization level
    pub personalization_level: PersonalizationLevel,

    /// Enable collaborative filtering
    pub enable_collaborative_filtering: bool,

    /// Enable content-based filtering
    pub enable_content_based_filtering: bool,
}

impl Default for RecommendationConfig {
    fn default() -> Self {
        Self {
            enable_performance_recommendations: true,
            enable_quality_recommendations: true,
            enable_optimization_recommendations: true,
            enable_proactive_recommendations: true,
            scoring_algorithm: ScoringAlgorithm::Hybrid,
            personalization_level: PersonalizationLevel::Medium,
            enable_collaborative_filtering: true,
            enable_content_based_filtering: true,
        }
    }
}

/// Scoring algorithms for recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScoringAlgorithm {
    ContentBased,
    CollaborativeFiltering,
    Hybrid,
    MatrixFactorization,
    DeepLearning,
}

/// Personalization levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PersonalizationLevel {
    Low,
    Medium,
    High,
    Adaptive,
}