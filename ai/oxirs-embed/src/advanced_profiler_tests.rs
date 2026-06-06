//! Advanced Profiler Tests

#[cfg(test)]
mod tests {
    use super::super::*;
    use std::collections::HashMap;

    #[test]
    fn test_profiler_config_default() {
        let config = ProfilerConfig::default();
        assert_eq!(config.max_sessions, 10);
        assert_eq!(config.sampling_rate, 0.01);
        assert!(config.enable_memory_profiling);
        assert!(config.enable_cpu_profiling);
    }

    #[test]
    fn test_profiling_session_creation() {
        let session = ProfilingSession {
            session_id: "test-session".to_string(),
            name: "Test Session".to_string(),
            start_time: chrono::Utc::now(),
            end_time: None,
            status: SessionStatus::Active,
            metrics: Vec::new(),
            tags: HashMap::new(),
        };

        assert_eq!(session.session_id, "test-session");
        assert_eq!(session.name, "Test Session");
        assert!(matches!(session.status, SessionStatus::Active));
    }

    #[test]
    fn test_metric_data_point_creation() {
        let metric = MetricDataPoint {
            timestamp: chrono::Utc::now(),
            metric_name: "cpu_usage".to_string(),
            value: 75.5,
            unit: "percent".to_string(),
            metadata: HashMap::new(),
            thread_id: Some("thread-1".to_string()),
            component: "embedding_service".to_string(),
        };

        assert_eq!(metric.metric_name, "cpu_usage");
        assert_eq!(metric.value, 75.5);
        assert_eq!(metric.unit, "percent");
    }

    #[test]
    fn test_performance_collector() {
        let mut collector = PerformanceCollector::new();

        let metric = MetricDataPoint {
            timestamp: chrono::Utc::now(),
            metric_name: "test_metric".to_string(),
            value: 100.0,
            unit: "units".to_string(),
            metadata: HashMap::new(),
            thread_id: None,
            component: "test".to_string(),
        };

        collector.add_metric(metric);
        assert_eq!(collector.stats.total_points, 1);
        assert_eq!(collector.buffer.len(), 1);
    }

    #[test]
    fn test_performance_tracker() {
        let mut collector = PerformanceCollector::new();
        let tracker_id = collector.start_tracker("test_tracker".to_string());

        assert_eq!(tracker_id, "test_tracker");
        assert!(collector.trackers.contains_key("test_tracker"));

        let tracker = collector.stop_tracker("test_tracker");
        assert!(tracker.is_some());
        assert!(matches!(
            tracker.expect("should succeed").state,
            TrackerState::Stopped
        ));
    }

    #[test]
    fn test_anomaly_creation() {
        let anomaly = PerformanceAnomaly {
            id: "test-anomaly".to_string(),
            anomaly_type: AnomalyType::LatencySpike,
            severity: AnomalySeverity::High,
            detected_at: chrono::Utc::now(),
            affected_metrics: vec!["latency".to_string()],
            anomaly_score: 0.9,
            context: AnomalyContext {
                component: "test_component".to_string(),
                related_events: Vec::new(),
                environmental_factors: HashMap::new(),
                potential_causes: Vec::new(),
            },
        };

        assert_eq!(anomaly.id, "test-anomaly");
        assert!(matches!(anomaly.anomaly_type, AnomalyType::LatencySpike));
        assert!(matches!(anomaly.severity, AnomalySeverity::High));
    }

    #[test]
    fn test_optimization_recommendation() {
        let recommendation = OptimizationRecommendation {
            id: "test-rec".to_string(),
            recommendation_type: RecommendationType::ResourceScaling,
            priority: RecommendationPriority::High,
            component: "test_component".to_string(),
            current_state: "Current state".to_string(),
            recommended_state: "Recommended state".to_string(),
            expected_improvement: ExpectedImprovement {
                latency_improvement_percent: 20.0,
                throughput_improvement_percent: 15.0,
                resource_savings_percent: 10.0,
                cost_reduction_percent: 5.0,
                confidence: 0.8,
            },
            implementation_effort: ImplementationEffort {
                estimated_hours: 8.0,
                required_skills: vec!["DevOps".to_string()],
                complexity: ComplexityLevel::Medium,
                dependencies: Vec::new(),
            },
            risk_assessment: RiskAssessment {
                risk_level: RiskLevel::Low,
                potential_impacts: Vec::new(),
                mitigation_strategies: Vec::new(),
                rollback_plan: "Rollback plan".to_string(),
            },
            description: "Test recommendation".to_string(),
            implementation_steps: Vec::new(),
        };

        assert_eq!(recommendation.id, "test-rec");
        assert!(matches!(
            recommendation.recommendation_type,
            RecommendationType::ResourceScaling
        ));
        assert_eq!(
            recommendation
                .expected_improvement
                .latency_improvement_percent,
            20.0
        );
    }

    #[tokio::test]
    async fn test_profiler_session_lifecycle() {
        let config = ProfilerConfig::default();
        let profiler = AdvancedProfiler::new(config);

        let session_id = profiler
            .start_session("Test Session".to_string(), HashMap::new())
            .await
            .expect("should succeed");
        assert!(!session_id.is_empty());

        let session = profiler
            .stop_session(&session_id)
            .await
            .expect("should succeed");
        assert!(matches!(session.status, SessionStatus::Completed));
        assert!(session.end_time.is_some());
    }

    #[tokio::test]
    async fn test_metric_recording() {
        let config = ProfilerConfig::default();
        let profiler = AdvancedProfiler::new(config);

        let metric = MetricDataPoint {
            timestamp: chrono::Utc::now(),
            metric_name: "test_metric".to_string(),
            value: 50.0,
            unit: "ms".to_string(),
            metadata: HashMap::new(),
            thread_id: None,
            component: "test".to_string(),
        };

        let result = profiler.record_metric(metric).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_pattern_detection_components() {
        let detector = PatternDetector::new();
        assert!(!detector.templates.is_empty());

        let template = &detector.templates[0];
        assert_eq!(template.name, "Memory Leak Pattern");
        assert!(!template.signature.characteristics.is_empty());
    }

    #[test]
    fn test_anomaly_detection_components() {
        let detector = AnomalyDetector::new();
        assert!(!detector.algorithms.is_empty());

        let algorithm = &detector.algorithms[0];
        assert_eq!(algorithm.name, "Statistical Outlier");
        assert!(matches!(
            algorithm.algorithm_type,
            AnomalyAlgorithmType::StatisticalOutlier
        ));
    }

    #[test]
    fn test_recommendation_rules() {
        let recommender = OptimizationRecommender::new();
        assert!(!recommender.rules.is_empty());

        let rule = &recommender.rules[0];
        assert_eq!(rule.name, "High Memory Usage");
        assert!(!rule.conditions.is_empty());
    }
}
