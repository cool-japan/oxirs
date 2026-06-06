//! Tests for forecasting models

#[cfg(test)]
mod tests {
    use crate::forecasting_models_engines::{ForecastingManager, QualityForecastingModel};
    use crate::forecasting_models_types::{
        ForecastingHorizon, ForecastingModelType, TimeSeries, TimeSeriesDataPoint,
    };
    use chrono::{Duration, Utc};
    use std::collections::HashMap;

    #[test]
    fn test_forecasting_model_creation() {
        let model = QualityForecastingModel::new(
            ForecastingModelType::LinearRegression,
            vec!["test_metric".to_string()],
            vec!["feature1".to_string()],
        );

        assert_eq!(model.model_type, ForecastingModelType::LinearRegression);
        assert_eq!(model.target_metrics.len(), 1);
        assert_eq!(model.features.len(), 1);
    }

    #[test]
    fn test_time_series_data_point() {
        let data_point = TimeSeriesDataPoint {
            timestamp: Utc::now(),
            value: 42.0,
            metadata: HashMap::new(),
        };

        assert_eq!(data_point.value, 42.0);
    }

    #[test]
    fn test_linear_regression_training() {
        let mut model = QualityForecastingModel::new(
            ForecastingModelType::LinearRegression,
            vec!["test".to_string()],
            vec![],
        );

        let data_points = vec![
            TimeSeriesDataPoint {
                timestamp: Utc::now(),
                value: 1.0,
                metadata: HashMap::new(),
            },
            TimeSeriesDataPoint {
                timestamp: Utc::now(),
                value: 2.0,
                metadata: HashMap::new(),
            },
            TimeSeriesDataPoint {
                timestamp: Utc::now(),
                value: 3.0,
                metadata: HashMap::new(),
            },
        ];

        let series = TimeSeries {
            metric_name: "test".to_string(),
            unit: "units".to_string(),
            data_points,
            collection_interval: Duration::hours(1),
            last_updated: Utc::now(),
        };

        let result = model.train(vec![series]);
        assert!(result.is_ok());

        // After training, the slope parameter should be stored
        let slope = model.model_parameters.get("test_slope");
        assert!(slope.is_some());
        let slope_val = *slope.unwrap();
        assert!(
            slope_val > 0.0,
            "Slope should be positive for increasing series"
        );
    }

    #[test]
    fn test_forecasting_manager() {
        let mut manager = ForecastingManager::new();

        let result = manager.create_quality_model(
            "test_model".to_string(),
            ForecastingModelType::LinearRegression,
            vec!["metric1".to_string()],
            vec!["feature1".to_string()],
        );

        assert!(result.is_ok());
        assert!(manager.quality_models.contains_key("test_model"));
    }

    #[test]
    fn test_forecast_horizon_enum() {
        assert_eq!(ForecastingHorizon::ShortTerm, ForecastingHorizon::ShortTerm);
        assert_ne!(ForecastingHorizon::ShortTerm, ForecastingHorizon::LongTerm);
    }
}
