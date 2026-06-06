//! Types, enums, config structs, and trait definitions for forecasting models

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Time series data point for forecasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesDataPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub metadata: HashMap<String, String>,
}

/// Time series for various metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    pub metric_name: String,
    pub unit: String,
    pub data_points: Vec<TimeSeriesDataPoint>,
    pub collection_interval: Duration,
    pub last_updated: DateTime<Utc>,
}

/// Forecasting model types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForecastingModelType {
    /// Linear regression for simple trends
    LinearRegression,
    /// ARIMA for time series with seasonality
    ARIMA,
    /// Exponential smoothing
    ExponentialSmoothing,
    /// Neural networks for complex patterns
    NeuralNetwork,
    /// Random forest for feature-based prediction
    RandomForest,
    /// Ensemble combining multiple models
    Ensemble,
}

/// Forecasting horizon
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForecastingHorizon {
    /// Short-term (1-7 days)
    ShortTerm,
    /// Medium-term (1-4 weeks)
    MediumTerm,
    /// Long-term (1-12 months)
    LongTerm,
    /// Real-time (next few hours)
    RealTime,
}

/// Model accuracy metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAccuracyMetrics {
    pub mean_absolute_error: f64,
    pub mean_squared_error: f64,
    pub root_mean_squared_error: f64,
    pub mean_absolute_percentage_error: f64,
    pub r_squared: f64,
    pub accuracy_score: f64,
}

/// Forecast result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResult {
    pub forecast_id: Uuid,
    pub metric_name: String,
    pub model_type: ForecastingModelType,
    pub horizon: ForecastingHorizon,
    pub predictions: Vec<ForecastedDataPoint>,
    pub confidence_intervals: Vec<ConfidenceInterval>,
    pub accuracy_estimate: f64,
    pub generated_at: DateTime<Utc>,
    pub valid_until: DateTime<Utc>,
}

/// Forecasted data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastedDataPoint {
    pub timestamp: DateTime<Utc>,
    pub predicted_value: f64,
    pub confidence: f64,
    pub factors: HashMap<String, f64>,
}

/// Confidence interval for predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub timestamp: DateTime<Utc>,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub confidence_level: f64,
}

/// Workload projections for resource forecasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadProjections {
    pub validation_requests_per_hour: f64,
    pub average_shape_complexity: f64,
    pub concurrent_users: u32,
    pub data_growth_rate: f64,
}

/// Resource forecast result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceForecast {
    pub forecast_id: Uuid,
    pub cpu_forecast: ForecastResult,
    pub memory_forecast: ForecastResult,
    pub storage_forecast: ForecastResult,
    pub network_forecast: ForecastResult,
    pub scaling_recommendations: Vec<ScalingRecommendation>,
    pub cost_projections: CostProjections,
    pub generated_at: DateTime<Utc>,
}

/// Scaling recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingRecommendation {
    pub resource_type: ResourceType,
    pub action: ScalingAction,
    pub target_capacity: f64,
    pub confidence: f64,
    pub reasoning: String,
    pub estimated_cost_impact: f64,
}

/// Resource type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceType {
    CPU,
    Memory,
    Storage,
    Network,
}

/// Scaling action
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalingAction {
    ScaleUp,
    ScaleDown,
    Maintain,
    Optimize,
}

/// Cost projections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostProjections {
    pub current_monthly_cost: f64,
    pub projected_monthly_cost: f64,
    pub cost_breakdown: HashMap<String, f64>,
    pub optimization_opportunities: Vec<String>,
}

/// Risk forecast result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskForecast {
    pub forecast_id: Uuid,
    pub quality_risks: ForecastResult,
    pub performance_risks: ForecastResult,
    pub security_risks: ForecastResult,
    pub overall_risk_level: RiskLevel,
    pub mitigation_strategies: Vec<MitigationStrategy>,
    pub generated_at: DateTime<Utc>,
}

/// Risk level
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Mitigation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MitigationStrategy {
    pub strategy_id: Uuid,
    pub risk_type: RiskType,
    pub description: String,
    pub priority: StrategyPriority,
    pub estimated_effort: f64,
    pub expected_risk_reduction: f64,
}

/// Risk type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskType {
    Quality,
    Performance,
    Security,
    Compliance,
}

/// Strategy priority
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrategyPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Comprehensive forecast result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveForecast {
    pub forecast_id: Uuid,
    pub quality_forecasts: HashMap<String, ForecastResult>,
    pub resource_forecast: ResourceForecast,
    pub risk_forecast: RiskForecast,
    pub recommendations: Vec<ForecastingRecommendation>,
    pub confidence_score: f64,
    pub generated_at: DateTime<Utc>,
}

/// Forecasting recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastingRecommendation {
    pub recommendation_id: Uuid,
    pub category: RecommendationCategory,
    pub title: String,
    pub description: String,
    pub priority: RecommendationPriority,
    pub expected_impact: f64,
    pub implementation_effort: f64,
}

/// Recommendation category
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Performance,
    Cost,
    Quality,
    Security,
    Scalability,
}

/// Recommendation priority
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}
