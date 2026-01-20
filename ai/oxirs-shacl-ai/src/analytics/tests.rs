//! Tests for analytics module

use super::*;

#[test]
fn test_analytics_engine_creation() {
    let engine = AnalyticsEngine::new();
    assert!(engine.config().enable_analytics);
    assert!(engine.config().enable_performance_analytics);
}

#[test]
fn test_analytics_config_default() {
    let config = AnalyticsConfig::default();
    assert!(config.enable_analytics);
    assert!(config.enable_performance_analytics);
    assert!(config.enable_quality_analytics);
    assert!(config.enable_validation_analytics);
    assert!(config.enable_trend_analysis);
}

#[test]
fn test_validation_insights_creation() {
    let insights = ValidationInsights::new();
    assert!(insights.validation_insights.is_empty());
    assert!(insights.performance_insights.is_empty());
    assert!(insights.quality_insights.is_empty());
    assert!(insights.trend_analysis.is_none());
}

#[test]
fn test_trend_direction() {
    use TrendDirection::*;

    assert_eq!(Increasing, Increasing);
    assert_ne!(Increasing, Decreasing);
    assert_ne!(Stable, Increasing);
}

#[test]
fn test_overall_health() {
    use OverallHealth::*;

    assert_eq!(Good, Good);
    assert_ne!(Good, Poor);
    assert_ne!(Critical, Excellent);
}

#[test]
fn test_dashboard_data_creation() {
    let dashboard = DashboardData::new();
    assert!(dashboard.performance_charts.is_empty());
    assert!(dashboard.trend_indicators.is_empty());
    assert!(dashboard.alerts.is_empty());
}
