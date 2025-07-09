//! Core predictive analytics engine

use std::time::Duration;

use oxirs_core::Store;
use oxirs_shacl::Shape;

use super::{
    config::PredictiveAnalyticsConfig,
    forecasting::ForecastingModels,
    recommendation::RecommendationEngine,
    time_series::{TimeSeriesProcessor, TrendAnalyzer},
    types::{PredictiveAnalyticsStatistics, PredictiveInsights},
};
use crate::{
    analytics::{PerformanceAnalysis, ValidationInsights},
    quality::QualityReport,
    Result,
};

/// Predictive analytics engine with forecasting and recommendation capabilities
#[derive(Debug)]
pub struct PredictiveAnalyticsEngine {
    config: PredictiveAnalyticsConfig,
    forecasting_models: ForecastingModels,
    recommendation_engine: RecommendationEngine,
    time_series_processor: TimeSeriesProcessor,
    trend_analyzer: TrendAnalyzer,
    statistics: PredictiveAnalyticsStatistics,
}

impl PredictiveAnalyticsEngine {
    /// Create a new predictive analytics engine
    pub fn new(config: PredictiveAnalyticsConfig) -> Self {
        Self {
            forecasting_models: ForecastingModels::new(&config),
            recommendation_engine: RecommendationEngine::new(&config),
            time_series_processor: TimeSeriesProcessor::new(),
            trend_analyzer: TrendAnalyzer::new(),
            statistics: PredictiveAnalyticsStatistics::new(),
            config,
        }
    }

    /// Generate comprehensive predictive insights
    pub async fn generate_predictive_insights(
        &mut self,
        store: &dyn Store,
        shapes: &[Shape],
        quality_report: &QualityReport,
        performance_analysis: &PerformanceAnalysis,
        validation_insights: &ValidationInsights,
    ) -> Result<PredictiveInsights> {
        let mut insights = PredictiveInsights::new();

        if self.config.enable_forecasting {
            let forecasts = self
                .forecasting_models
                .generate_forecasts(quality_report, performance_analysis)
                .await?;
            insights.forecasts = forecasts;
        }

        if self.config.enable_recommendations {
            let recommendations = self
                .recommendation_engine
                .generate_recommendations(store, shapes, quality_report, performance_analysis)
                .await?;
            insights.recommendations = recommendations;
        }

        if self.config.enable_trend_analysis {
            let trend_predictions = self
                .trend_analyzer
                .analyze_trends(validation_insights)
                .await?;
            insights.trend_predictions = trend_predictions;
        }

        self.update_statistics(&insights).await?;

        Ok(insights)
    }

    /// Get predictive analytics statistics
    pub fn get_statistics(&self) -> &PredictiveAnalyticsStatistics {
        &self.statistics
    }

    /// Update internal statistics
    async fn update_statistics(&mut self, _insights: &PredictiveInsights) -> Result<()> {
        // Simplified implementation - would update internal statistics
        Ok(())
    }
}

impl Default for PredictiveAnalyticsEngine {
    fn default() -> Self {
        Self::new(PredictiveAnalyticsConfig::default())
    }
}

impl PredictiveAnalyticsStatistics {
    pub fn new() -> Self {
        Self {
            total_forecasts_generated: 0,
            total_recommendations_generated: 0,
            average_forecast_accuracy: 0.0,
            average_recommendation_acceptance_rate: 0.0,
            processing_time_statistics: super::types::ProcessingTimeStats {
                average_processing_time: Duration::from_secs(0),
                min_processing_time: Duration::from_secs(0),
                max_processing_time: Duration::from_secs(0),
                total_processing_time: Duration::from_secs(0),
            },
        }
    }
}

impl Default for PredictiveAnalyticsStatistics {
    fn default() -> Self {
        Self::new()
    }
}
