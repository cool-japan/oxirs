//! Performance optimization functionality

use crate::performance_analytics::{
    config::PerformanceOptimizationConfig,
    types::{OptimizationRecommendation, PerformanceStatistics, ResourceUtilization},
};
use std::time::SystemTime;
use uuid::Uuid;

/// Performance optimizer with intelligent recommendation generation
#[derive(Debug)]
pub struct PerformanceOptimizer {
    config: PerformanceOptimizationConfig,
    /// Historical performance data for trend analysis
    performance_history: Vec<PerformanceSnapshot>,
    /// Current system state
    current_state: Option<SystemState>,
}

/// System performance snapshot
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    pub timestamp: SystemTime,
    pub performance_stats: PerformanceStatistics,
    pub resource_utilization: ResourceUtilization,
}

/// Current system state for optimization analysis
#[derive(Debug, Clone)]
pub struct SystemState {
    pub avg_response_time_ms: f64,
    pub memory_usage_percent: f64,
    pub cpu_usage_percent: f64,
    pub error_rate_percent: f64,
    pub throughput_rps: f64,
    pub cache_hit_rate: f64,
}

impl PerformanceOptimizer {
    pub fn new() -> Self {
        Self {
            config: PerformanceOptimizationConfig::default(),
            performance_history: Vec::new(),
            current_state: None,
        }
    }

    pub fn with_config(config: PerformanceOptimizationConfig) -> Self {
        Self {
            config,
            performance_history: Vec::new(),
            current_state: None,
        }
    }

    /// Update the current system state
    pub fn update_system_state(&mut self, state: SystemState) {
        self.current_state = Some(state);
    }

    /// Add performance snapshot for historical analysis
    pub fn add_performance_snapshot(&mut self, snapshot: PerformanceSnapshot) {
        self.performance_history.push(snapshot);

        // Keep only recent history (last 100 snapshots)
        if self.performance_history.len() > 100 {
            self.performance_history.remove(0);
        }
    }

    /// Generate intelligent performance optimization recommendations
    pub fn generate_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        if let Some(state) = &self.current_state {
            // Analyze response time performance (using 5000ms as threshold)
            if state.avg_response_time_ms > 5000.0 {
                recommendations
                    .push(self.create_response_time_recommendation(state.avg_response_time_ms));
            }

            // Analyze memory usage
            if state.memory_usage_percent > self.config.target_memory_percent {
                recommendations.push(
                    self.create_memory_optimization_recommendation(state.memory_usage_percent),
                );
            }

            // Analyze CPU usage
            if state.cpu_usage_percent > self.config.target_cpu_percent {
                recommendations
                    .push(self.create_cpu_optimization_recommendation(state.cpu_usage_percent));
            }

            // Analyze error rate
            if state.error_rate_percent > 1.0 {
                recommendations
                    .push(self.create_error_rate_recommendation(state.error_rate_percent));
            }

            // Analyze cache performance
            if state.cache_hit_rate < 0.8 {
                recommendations
                    .push(self.create_cache_optimization_recommendation(state.cache_hit_rate));
            }

            // Analyze throughput optimization
            if state.throughput_rps < 100.0 {
                recommendations
                    .push(self.create_throughput_optimization_recommendation(state.throughput_rps));
            }

            // Generate trend-based recommendations
            recommendations.extend(self.generate_trend_recommendations());

            // Generate advanced optimization recommendations
            recommendations.extend(self.generate_advanced_recommendations(state));
        }

        recommendations
    }

    /// Create response time optimization recommendation
    fn create_response_time_recommendation(
        &self,
        response_time: f64,
    ) -> OptimizationRecommendation {
        let severity = if response_time > 10000.0 {
            "critical"
        } else if response_time > 5000.0 {
            "high"
        } else {
            "medium"
        };

        OptimizationRecommendation {
            id: Uuid::new_v4().to_string(),
            recommendation_type: "Response Time Optimization".to_string(),
            description: format!(
                "Response time is {response_time}ms, which exceeds the threshold of 5000ms. Consider implementing caching, optimizing database queries, or scaling resources."
            ),
            expected_improvement_percent: if response_time > 10000.0 { 60.0 } else { 40.0 },
            implementation_complexity: if response_time > 10000.0 { 8 } else { 6 },
            priority: if severity == "critical" { 9 } else if severity == "high" { 7 } else { 5 },
            estimated_hours: if response_time > 10000.0 { 16.0 } else { 8.0 },
            tags: vec!["performance".to_string(), "response-time".to_string(), severity.to_string()],
            created_at: SystemTime::now(),
        }
    }

    /// Create memory optimization recommendation
    fn create_memory_optimization_recommendation(
        &self,
        memory_usage: f64,
    ) -> OptimizationRecommendation {
        OptimizationRecommendation {
            id: Uuid::new_v4().to_string(),
            recommendation_type: "Memory Optimization".to_string(),
            description: format!(
                "Memory usage is {memory_usage:.1}%, which is above optimal levels. Consider implementing memory pooling, garbage collection tuning, or reducing memory footprint."
            ),
            expected_improvement_percent: 30.0,
            implementation_complexity: 7,
            priority: if memory_usage > 90.0 { 8 } else { 6 },
            estimated_hours: 12.0,
            tags: vec!["memory".to_string(), "optimization".to_string(), "resource".to_string()],
            created_at: SystemTime::now(),
        }
    }

    /// Create CPU optimization recommendation
    fn create_cpu_optimization_recommendation(&self, cpu_usage: f64) -> OptimizationRecommendation {
        OptimizationRecommendation {
            id: Uuid::new_v4().to_string(),
            recommendation_type: "CPU Optimization".to_string(),
            description: format!(
                "CPU usage is {cpu_usage:.1}%, indicating high computational load. Consider implementing parallel processing, algorithm optimization, or load balancing."
            ),
            expected_improvement_percent: 45.0,
            implementation_complexity: 6,
            priority: if cpu_usage > 95.0 { 9 } else { 7 },
            estimated_hours: 10.0,
            tags: vec!["cpu".to_string(), "parallel".to_string(), "optimization".to_string()],
            created_at: SystemTime::now(),
        }
    }

    /// Create error rate optimization recommendation
    fn create_error_rate_recommendation(&self, error_rate: f64) -> OptimizationRecommendation {
        OptimizationRecommendation {
            id: Uuid::new_v4().to_string(),
            recommendation_type: "Error Rate Reduction".to_string(),
            description: format!(
                "Error rate is {error_rate:.2}%, indicating reliability issues. Implement better error handling, input validation, and monitoring."
            ),
            expected_improvement_percent: 70.0,
            implementation_complexity: 5,
            priority: 8,
            estimated_hours: 6.0,
            tags: vec!["reliability".to_string(), "error-handling".to_string(), "quality".to_string()],
            created_at: SystemTime::now(),
        }
    }

    /// Create cache optimization recommendation
    fn create_cache_optimization_recommendation(
        &self,
        hit_rate: f64,
    ) -> OptimizationRecommendation {
        OptimizationRecommendation {
            id: Uuid::new_v4().to_string(),
            recommendation_type: "Cache Optimization".to_string(),
            description: format!(
                "Cache hit rate is {:.1}%, which is below optimal. Consider tuning cache size, TTL values, or implementing smarter eviction policies.",
                hit_rate * 100.0
            ),
            expected_improvement_percent: 35.0,
            implementation_complexity: 4,
            priority: 6,
            estimated_hours: 4.0,
            tags: vec!["cache".to_string(), "performance".to_string(), "optimization".to_string()],
            created_at: SystemTime::now(),
        }
    }

    /// Create throughput optimization recommendation
    fn create_throughput_optimization_recommendation(
        &self,
        throughput: f64,
    ) -> OptimizationRecommendation {
        OptimizationRecommendation {
            id: Uuid::new_v4().to_string(),
            recommendation_type: "Throughput Enhancement".to_string(),
            description: format!(
                "Throughput is {throughput:.1} RPS, which may benefit from optimization. Consider connection pooling, async processing, or scaling strategies."
            ),
            expected_improvement_percent: 50.0,
            implementation_complexity: 7,
            priority: 6,
            estimated_hours: 14.0,
            tags: vec!["throughput".to_string(), "scaling".to_string(), "async".to_string()],
            created_at: SystemTime::now(),
        }
    }

    /// Generate recommendations based on performance trends
    fn generate_trend_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        if self.performance_history.len() >= 10 {
            // Analyze response time trend
            let recent_response_times: Vec<f64> = self
                .performance_history
                .iter()
                .rev()
                .take(10)
                .map(|s| s.performance_stats.avg_response_time_ms)
                .collect();

            if self.is_degrading_trend(&recent_response_times) {
                recommendations.push(OptimizationRecommendation {
                    id: Uuid::new_v4().to_string(),
                    recommendation_type: "Trend Analysis - Response Time".to_string(),
                    description: "Response time shows a degrading trend over recent measurements. Proactive optimization recommended.".to_string(),
                    expected_improvement_percent: 25.0,
                    implementation_complexity: 5,
                    priority: 7,
                    estimated_hours: 8.0,
                    tags: vec!["trend".to_string(), "proactive".to_string(), "response-time".to_string()],
                    created_at: SystemTime::now(),
                });
            }
        }

        recommendations
    }

    /// Generate advanced optimization recommendations
    fn generate_advanced_recommendations(
        &self,
        state: &SystemState,
    ) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Advanced pattern-based recommendations
        if state.memory_usage_percent > 80.0 && state.cpu_usage_percent > 80.0 {
            recommendations.push(OptimizationRecommendation {
                id: Uuid::new_v4().to_string(),
                recommendation_type: "Resource Bottleneck Resolution".to_string(),
                description: "Both memory and CPU usage are high, indicating potential resource bottleneck. Consider vertical scaling or resource optimization.".to_string(),
                expected_improvement_percent: 55.0,
                implementation_complexity: 8,
                priority: 9,
                estimated_hours: 20.0,
                tags: vec!["bottleneck".to_string(), "scaling".to_string(), "resource".to_string()],
                created_at: SystemTime::now(),
            });
        }

        // AI-driven optimization recommendation
        if self.config.aggressiveness > 0.7 {
            recommendations.push(OptimizationRecommendation {
                id: Uuid::new_v4().to_string(),
                recommendation_type: "AI-Driven Optimization".to_string(),
                description: "Enable advanced AI optimization features including quantum-enhanced pattern recognition and neural cost estimation.".to_string(),
                expected_improvement_percent: 40.0,
                implementation_complexity: 9,
                priority: 5,
                estimated_hours: 24.0,
                tags: vec!["ai".to_string(), "advanced".to_string(), "experimental".to_string()],
                created_at: SystemTime::now(),
            });
        }

        recommendations
    }

    /// Check if a trend is degrading
    fn is_degrading_trend(&self, values: &[f64]) -> bool {
        if values.len() < 3 {
            return false;
        }

        let mut increasing_count = 0;
        for i in 1..values.len() {
            if values[i] > values[i - 1] {
                increasing_count += 1;
            }
        }

        // Consider degrading if more than 60% of measurements are increasing
        (increasing_count as f64 / (values.len() - 1) as f64) > 0.6
    }

    /// Get performance optimization configuration
    pub fn get_config(&self) -> &PerformanceOptimizationConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: PerformanceOptimizationConfig) {
        self.config = config;
    }
}

impl Default for PerformanceOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::performance_analytics::types::{PerformanceStatistics, ResourceUtilization};

    fn create_test_system_state() -> SystemState {
        SystemState {
            avg_response_time_ms: 3000.0,
            memory_usage_percent: 65.0, // Below 70% default target
            cpu_usage_percent: 55.0,    // Below 60% default target
            error_rate_percent: 0.5,    // Below 1% threshold
            throughput_rps: 150.0,      // Above 100 RPS threshold
            cache_hit_rate: 0.85,       // Above 0.8 threshold
        }
    }

    fn create_high_load_system_state() -> SystemState {
        SystemState {
            avg_response_time_ms: 8000.0,
            memory_usage_percent: 85.0,
            cpu_usage_percent: 90.0,
            error_rate_percent: 2.5,
            throughput_rps: 50.0,
            cache_hit_rate: 0.60,
        }
    }

    #[test]
    fn test_performance_optimizer_creation() {
        let optimizer = PerformanceOptimizer::new();
        assert!(optimizer.current_state.is_none());
        assert!(optimizer.performance_history.is_empty());
    }

    #[test]
    fn test_system_state_update() {
        let mut optimizer = PerformanceOptimizer::new();
        let state = create_test_system_state();

        optimizer.update_system_state(state);
        assert!(optimizer.current_state.is_some());

        let stored_state = optimizer.current_state.as_ref().unwrap();
        assert_eq!(stored_state.avg_response_time_ms, 3000.0);
        assert_eq!(stored_state.memory_usage_percent, 65.0);
    }

    #[test]
    fn test_normal_performance_no_recommendations() {
        let mut optimizer = PerformanceOptimizer::new();
        let state = create_test_system_state();
        optimizer.update_system_state(state);

        let recommendations = optimizer.generate_recommendations();
        // With normal performance, should generate no recommendations
        assert!(
            recommendations.is_empty(),
            "Normal performance should not generate recommendations"
        );
    }

    #[test]
    fn test_high_response_time_recommendation() {
        let mut optimizer = PerformanceOptimizer::new();
        let mut state = create_test_system_state();
        state.avg_response_time_ms = 7000.0; // Above 5000ms threshold
        optimizer.update_system_state(state);

        let recommendations = optimizer.generate_recommendations();
        assert!(
            !recommendations.is_empty(),
            "High response time should generate recommendations"
        );

        let response_time_rec = recommendations
            .iter()
            .find(|r| r.recommendation_type == "Response Time Optimization");
        assert!(
            response_time_rec.is_some(),
            "Should contain response time optimization recommendation"
        );

        let rec = response_time_rec.unwrap();
        assert!(rec.expected_improvement_percent > 0.0);
        assert!(rec.priority >= 5);
        assert!(!rec.description.is_empty());
    }

    #[test]
    fn test_high_memory_usage_recommendation() {
        let mut optimizer = PerformanceOptimizer::new();
        let mut state = create_test_system_state();
        state.memory_usage_percent = 85.0; // Above default target
        optimizer.update_system_state(state);

        let recommendations = optimizer.generate_recommendations();
        let memory_rec = recommendations
            .iter()
            .find(|r| r.recommendation_type == "Memory Optimization");
        assert!(
            memory_rec.is_some(),
            "High memory usage should generate memory optimization recommendation"
        );

        let rec = memory_rec.unwrap();
        assert_eq!(rec.expected_improvement_percent, 30.0);
        assert!(rec.implementation_complexity <= 10);
    }

    #[test]
    fn test_high_cpu_usage_recommendation() {
        let mut optimizer = PerformanceOptimizer::new();
        let mut state = create_test_system_state();
        state.cpu_usage_percent = 85.0; // Above default target
        optimizer.update_system_state(state);

        let recommendations = optimizer.generate_recommendations();
        let cpu_rec = recommendations
            .iter()
            .find(|r| r.recommendation_type == "CPU Optimization");
        assert!(
            cpu_rec.is_some(),
            "High CPU usage should generate CPU optimization recommendation"
        );

        let rec = cpu_rec.unwrap();
        assert_eq!(rec.expected_improvement_percent, 45.0);
        assert!(rec.tags.contains(&"cpu".to_string()));
    }

    #[test]
    fn test_high_error_rate_recommendation() {
        let mut optimizer = PerformanceOptimizer::new();
        let mut state = create_test_system_state();
        state.error_rate_percent = 2.0; // Above 1% threshold
        optimizer.update_system_state(state);

        let recommendations = optimizer.generate_recommendations();
        let error_rec = recommendations
            .iter()
            .find(|r| r.recommendation_type == "Error Rate Reduction");
        assert!(
            error_rec.is_some(),
            "High error rate should generate error rate reduction recommendation"
        );

        let rec = error_rec.unwrap();
        assert_eq!(rec.expected_improvement_percent, 70.0);
        assert_eq!(rec.priority, 8);
    }

    #[test]
    fn test_low_cache_hit_rate_recommendation() {
        let mut optimizer = PerformanceOptimizer::new();
        let mut state = create_test_system_state();
        state.cache_hit_rate = 0.7; // Below 0.8 threshold
        optimizer.update_system_state(state);

        let recommendations = optimizer.generate_recommendations();
        let cache_rec = recommendations
            .iter()
            .find(|r| r.recommendation_type == "Cache Optimization");
        assert!(
            cache_rec.is_some(),
            "Low cache hit rate should generate cache optimization recommendation"
        );

        let rec = cache_rec.unwrap();
        assert_eq!(rec.expected_improvement_percent, 35.0);
        assert!(rec.tags.contains(&"cache".to_string()));
    }

    #[test]
    fn test_low_throughput_recommendation() {
        let mut optimizer = PerformanceOptimizer::new();
        let mut state = create_test_system_state();
        state.throughput_rps = 80.0; // Below 100 RPS threshold
        optimizer.update_system_state(state);

        let recommendations = optimizer.generate_recommendations();
        let throughput_rec = recommendations
            .iter()
            .find(|r| r.recommendation_type == "Throughput Enhancement");
        assert!(
            throughput_rec.is_some(),
            "Low throughput should generate throughput enhancement recommendation"
        );

        let rec = throughput_rec.unwrap();
        assert_eq!(rec.expected_improvement_percent, 50.0);
        assert!(rec.tags.contains(&"throughput".to_string()));
    }

    #[test]
    fn test_multiple_issues_generate_multiple_recommendations() {
        let mut optimizer = PerformanceOptimizer::new();
        let state = create_high_load_system_state(); // Multiple issues
        optimizer.update_system_state(state);

        let recommendations = optimizer.generate_recommendations();
        assert!(
            recommendations.len() >= 5,
            "Multiple issues should generate multiple recommendations"
        );

        // Should have response time, memory, CPU, error rate, and cache recommendations
        let rec_types: Vec<&str> = recommendations
            .iter()
            .map(|r| r.recommendation_type.as_str())
            .collect();

        assert!(rec_types.contains(&"Response Time Optimization"));
        assert!(rec_types.contains(&"Memory Optimization"));
        assert!(rec_types.contains(&"CPU Optimization"));
        assert!(rec_types.contains(&"Error Rate Reduction"));
        assert!(rec_types.contains(&"Cache Optimization"));
    }

    #[test]
    fn test_resource_bottleneck_recommendation() {
        let mut optimizer = PerformanceOptimizer::new();
        let mut state = create_test_system_state();
        state.memory_usage_percent = 85.0; // Above 80%
        state.cpu_usage_percent = 85.0; // Above 80%
        optimizer.update_system_state(state);

        let recommendations = optimizer.generate_recommendations();
        let bottleneck_rec = recommendations
            .iter()
            .find(|r| r.recommendation_type == "Resource Bottleneck Resolution");
        assert!(
            bottleneck_rec.is_some(),
            "High memory and CPU should generate bottleneck recommendation"
        );

        let rec = bottleneck_rec.unwrap();
        assert_eq!(rec.expected_improvement_percent, 55.0);
        assert_eq!(rec.priority, 9);
        assert!(rec.tags.contains(&"bottleneck".to_string()));
    }

    #[test]
    fn test_ai_optimization_recommendation_with_high_aggressiveness() {
        let config = PerformanceOptimizationConfig {
            aggressiveness: 0.8, // Above 0.7 threshold
            ..Default::default()
        };
        let mut optimizer = PerformanceOptimizer::with_config(config);
        let state = create_test_system_state();
        optimizer.update_system_state(state);

        let recommendations = optimizer.generate_recommendations();
        let ai_rec = recommendations
            .iter()
            .find(|r| r.recommendation_type == "AI-Driven Optimization");
        assert!(
            ai_rec.is_some(),
            "High aggressiveness should generate AI optimization recommendation"
        );

        let rec = ai_rec.unwrap();
        assert_eq!(rec.expected_improvement_percent, 40.0);
        assert!(rec.tags.contains(&"ai".to_string()));
        assert!(rec.tags.contains(&"experimental".to_string()));
    }

    #[test]
    fn test_performance_snapshot_history() {
        let mut optimizer = PerformanceOptimizer::new();

        for _i in 0..5 {
            let snapshot = PerformanceSnapshot {
                timestamp: SystemTime::now(),
                performance_stats: PerformanceStatistics::default(),
                resource_utilization: ResourceUtilization::default(),
            };
            optimizer.add_performance_snapshot(snapshot);
        }

        assert_eq!(optimizer.performance_history.len(), 5);

        // Add more than the limit (100) to test cleanup
        for _i in 0..98 {
            let snapshot = PerformanceSnapshot {
                timestamp: SystemTime::now(),
                performance_stats: PerformanceStatistics::default(),
                resource_utilization: ResourceUtilization::default(),
            };
            optimizer.add_performance_snapshot(snapshot);
        }

        assert_eq!(
            optimizer.performance_history.len(),
            100,
            "Should maintain history limit of 100"
        );
    }

    #[test]
    fn test_degrading_trend_detection() {
        let optimizer = PerformanceOptimizer::new();

        // Test degrading trend (increasing values)
        let degrading_values = vec![100.0, 120.0, 150.0, 180.0, 200.0];
        assert!(
            optimizer.is_degrading_trend(&degrading_values),
            "Should detect degrading trend"
        );

        // Test improving trend (decreasing values)
        let improving_values = vec![200.0, 180.0, 150.0, 120.0, 100.0];
        assert!(
            !optimizer.is_degrading_trend(&improving_values),
            "Should not detect degrading trend for improving values"
        );

        // Test stable trend
        let stable_values = vec![100.0, 102.0, 98.0, 101.0, 99.0];
        assert!(
            !optimizer.is_degrading_trend(&stable_values),
            "Should not detect degrading trend for stable values"
        );

        // Test insufficient data
        let insufficient_values = vec![100.0, 120.0];
        assert!(
            !optimizer.is_degrading_trend(&insufficient_values),
            "Should not detect trend with insufficient data"
        );
    }

    #[test]
    fn test_config_access_and_update() {
        let mut optimizer = PerformanceOptimizer::new();
        let original_config = optimizer.get_config().clone();

        let new_config = PerformanceOptimizationConfig {
            aggressiveness: 0.9,
            enable_auto_optimization: true,
            ..Default::default()
        };

        optimizer.update_config(new_config);

        assert_eq!(optimizer.get_config().aggressiveness, 0.9);
        assert!(optimizer.get_config().enable_auto_optimization);
        assert_ne!(
            optimizer.get_config().aggressiveness,
            original_config.aggressiveness
        );
    }

    #[test]
    fn test_recommendation_fields_are_properly_set() {
        let mut optimizer = PerformanceOptimizer::new();
        let mut state = create_test_system_state();
        state.avg_response_time_ms = 6000.0; // Trigger response time recommendation
        optimizer.update_system_state(state);

        let recommendations = optimizer.generate_recommendations();
        let rec = &recommendations[0];

        // Verify all required fields are set
        assert!(!rec.id.is_empty(), "Recommendation ID should not be empty");
        assert!(
            !rec.recommendation_type.is_empty(),
            "Recommendation type should not be empty"
        );
        assert!(
            !rec.description.is_empty(),
            "Description should not be empty"
        );
        assert!(
            rec.expected_improvement_percent > 0.0,
            "Expected improvement should be positive"
        );
        assert!(
            rec.implementation_complexity >= 1 && rec.implementation_complexity <= 10,
            "Implementation complexity should be 1-10"
        );
        assert!(
            rec.priority >= 1 && rec.priority <= 10,
            "Priority should be 1-10"
        );
        assert!(
            rec.estimated_hours > 0.0,
            "Estimated hours should be positive"
        );
        assert!(!rec.tags.is_empty(), "Tags should not be empty");
        // created_at timestamp should be recent (within last minute)
        let now = SystemTime::now();
        let duration = now.duration_since(rec.created_at).unwrap();
        assert!(
            duration.as_secs() < 60,
            "Created timestamp should be recent"
        );
    }
}
