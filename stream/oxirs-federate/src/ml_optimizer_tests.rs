//! ML Optimizer — Test suite

#[cfg(test)]
mod tests {
    use crate::ml_optimizer::{
        LinearRegressionModel, MLOptimizer, NeuralNetworkModel, PerformanceOutcome, QueryFeatures,
    };
    use std::time::SystemTime;

    #[tokio::test]
    async fn test_ml_optimizer_creation() {
        let optimizer = MLOptimizer::new();
        let stats = optimizer.get_statistics().await;
        assert_eq!(stats.total_predictions, 0);
    }

    #[tokio::test]
    async fn test_performance_prediction() {
        let optimizer = MLOptimizer::new();
        let features = QueryFeatures {
            pattern_count: 5,
            join_count: 2,
            filter_count: 1,
            complexity_score: 3.0,
            selectivity: 0.5,
            service_count: 2,
            avg_service_latency: 100.0,
            data_size_estimate: 1024,
            query_depth: 2,
            has_optional: false,
            has_union: false,
            has_aggregation: false,
            variable_count: 3,
        };

        let prediction = optimizer
            .predict_performance(&features)
            .await
            .expect("async operation should succeed");
        assert!(prediction >= 0.0);
    }

    #[tokio::test]
    async fn test_source_selection_recommendation() {
        let optimizer = MLOptimizer::new();
        let features = QueryFeatures {
            pattern_count: 3,
            join_count: 1,
            filter_count: 1,
            complexity_score: 2.0,
            selectivity: 0.8,
            service_count: 3,
            avg_service_latency: 50.0,
            data_size_estimate: 512,
            query_depth: 1,
            has_optional: false,
            has_union: false,
            has_aggregation: false,
            variable_count: 2,
        };

        let services = vec![
            "service1".to_string(),
            "service2".to_string(),
            "service3".to_string(),
        ];
        let recommendation = optimizer
            .recommend_source_selection(&features, &services)
            .await
            .expect("operation should succeed");

        assert!(!recommendation.recommended_services.is_empty());
    }

    #[tokio::test]
    async fn test_anomaly_detection() {
        let optimizer = MLOptimizer::new();
        let features = QueryFeatures {
            pattern_count: 10,
            join_count: 5,
            filter_count: 3,
            complexity_score: 8.0,
            selectivity: 0.1,
            service_count: 5,
            avg_service_latency: 200.0,
            data_size_estimate: 10240,
            query_depth: 3,
            has_optional: true,
            has_union: true,
            has_aggregation: true,
            variable_count: 8,
        };

        let outcome = PerformanceOutcome {
            execution_time_ms: 5000.0,
            memory_usage_bytes: 100 * 1024 * 1024,
            network_io_ms: 1000.0,
            cpu_usage_percent: 80.0,
            success_rate: 0.9,
            error_count: 1,
            cache_hit_rate: 0.2,
            timestamp: SystemTime::now(),
        };

        let detection = optimizer
            .detect_anomalies(&features, &outcome)
            .await
            .expect("operation should succeed");
        assert!(detection.anomaly_score >= 0.0);
    }

    #[test]
    fn test_linear_regression_model() {
        let model = LinearRegressionModel::new(5);
        assert_eq!(model.weights.len(), 5);
        assert_eq!(model.bias, 0.0);
        assert_eq!(model.iterations, 0);
    }

    #[test]
    fn test_neural_network_model() {
        let model = NeuralNetworkModel::new(5, 3, 0.01);
        assert_eq!(model.weights_input_hidden.len(), 3);
        assert_eq!(model.weights_input_hidden[0].len(), 5);
        assert_eq!(model.weights_hidden_output.len(), 3);
        assert_eq!(model.bias_hidden.len(), 3);
        assert_eq!(model.iterations, 0);
        assert_eq!(model.learning_rate, 0.01);
    }

    #[tokio::test]
    async fn test_ensemble_prediction() {
        let optimizer = MLOptimizer::new();
        let features = QueryFeatures {
            pattern_count: 3,
            join_count: 1,
            filter_count: 1,
            complexity_score: 2.5,
            selectivity: 0.6,
            service_count: 2,
            avg_service_latency: 75.0,
            data_size_estimate: 2048,
            query_depth: 1,
            has_optional: false,
            has_union: false,
            has_aggregation: false,
            variable_count: 4,
        };

        let prediction = optimizer
            .predict_performance(&features)
            .await
            .expect("async operation should succeed");
        assert!(prediction >= 0.0);

        let prediction2 = optimizer
            .predict_performance(&features)
            .await
            .expect("async operation should succeed");
        assert_eq!(prediction, prediction2);
    }
}
