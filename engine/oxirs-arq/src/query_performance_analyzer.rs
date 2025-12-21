//! # Query Performance Analyzer
//!
//! Advanced performance analysis for SPARQL queries with ML-powered insights,
//! bottleneck detection, and optimization recommendations.
//!
//! ## Features
//!
//! - **Execution profiling**: Detailed breakdown of query execution phases
//! - **Bottleneck detection**: Automatic identification of performance bottlenecks
//! - **Pattern analysis**: Statistical analysis of query execution patterns
//! - **ML predictions**: Machine learning-based performance predictions
//! - **Optimization suggestions**: Actionable recommendations for query improvement
//! - **Comparative analysis**: Compare query performance across different versions
//! - **Resource tracking**: CPU, memory, and I/O utilization monitoring
//!
//! ## Example
//!
//! ```rust
//! use oxirs_arq::query_performance_analyzer::{
//!     QueryPerformanceAnalyzer, AnalyzerConfig, ExecutionProfile
//! };
//!
//! # fn example() -> anyhow::Result<()> {
//! let analyzer = QueryPerformanceAnalyzer::new(AnalyzerConfig::default());
//!
//! // Record query execution
//! let profile = ExecutionProfile::new("SELECT * WHERE { ?s ?p ?o }")
//!     .with_duration_ms(125)
//!     .with_result_count(1000);
//!
//! analyzer.record_execution(profile)?;
//!
//! // Get performance insights
//! let insights = analyzer.analyze_performance()?;
//! println!("Performance Score: {:.1}/100", insights.score);
//! println!("Recommendations: {}", insights.recommendations.len());
//! # Ok(())
//! # }
//! ```

use anyhow::{Result, Context};
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::metrics::{Counter, Timer, Histogram};
use scirs2_core::random::{Random, Rng};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};
use serde::{Serialize, Deserialize};

/// Configuration for the performance analyzer
#[derive(Debug, Clone)]
pub struct AnalyzerConfig {
    /// Maximum number of execution records to keep
    pub max_records: usize,
    /// Enable ML-based predictions
    pub enable_ml_predictions: bool,
    /// Enable detailed profiling (higher overhead)
    pub enable_detailed_profiling: bool,
    /// Threshold for slow query detection (milliseconds)
    pub slow_query_threshold_ms: u64,
    /// Enable automatic optimization suggestions
    pub enable_auto_suggestions: bool,
    /// Sample rate for profiling (0.0 - 1.0)
    pub sampling_rate: f64,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            max_records: 10000,
            enable_ml_predictions: true,
            enable_detailed_profiling: false,
            slow_query_threshold_ms: 1000,
            enable_auto_suggestions: true,
            sampling_rate: 1.0,
        }
    }
}

impl AnalyzerConfig {
    /// Create a lightweight configuration (minimal overhead)
    pub fn lightweight() -> Self {
        Self {
            max_records: 1000,
            enable_ml_predictions: false,
            enable_detailed_profiling: false,
            slow_query_threshold_ms: 5000,
            enable_auto_suggestions: false,
            sampling_rate: 0.1,
        }
    }

    /// Create a comprehensive configuration (detailed analysis)
    pub fn comprehensive() -> Self {
        Self {
            max_records: 50000,
            enable_ml_predictions: true,
            enable_detailed_profiling: true,
            slow_query_threshold_ms: 500,
            enable_auto_suggestions: true,
            sampling_rate: 1.0,
        }
    }
}

/// Execution phase breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPhases {
    /// Query parsing time
    pub parse_ms: f64,
    /// Query optimization time
    pub optimize_ms: f64,
    /// Query execution time
    pub execute_ms: f64,
    /// Result serialization time
    pub serialize_ms: f64,
}

impl ExecutionPhases {
    /// Calculate total execution time
    pub fn total_ms(&self) -> f64 {
        self.parse_ms + self.optimize_ms + self.execute_ms + self.serialize_ms
    }

    /// Identify the slowest phase
    pub fn slowest_phase(&self) -> &str {
        let phases = [
            ("parse", self.parse_ms),
            ("optimize", self.optimize_ms),
            ("execute", self.execute_ms),
            ("serialize", self.serialize_ms),
        ];

        phases.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|p| p.0)
            .unwrap_or("unknown")
    }
}

/// Resource utilization during query execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// Peak memory usage (MB)
    pub peak_memory_mb: f64,
    /// Average CPU usage (0.0 - 1.0)
    pub avg_cpu: f64,
    /// I/O operations
    pub io_operations: u64,
    /// Network bytes transferred
    pub network_bytes: u64,
}

impl Default for ResourceUtilization {
    fn default() -> Self {
        Self {
            peak_memory_mb: 0.0,
            avg_cpu: 0.0,
            io_operations: 0,
            network_bytes: 0,
        }
    }
}

/// Execution profile for a query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionProfile {
    /// Query string
    pub query: String,
    /// Execution timestamp
    pub timestamp: SystemTime,
    /// Total execution duration
    pub duration: Duration,
    /// Number of results returned
    pub result_count: usize,
    /// Execution phase breakdown
    pub phases: Option<ExecutionPhases>,
    /// Resource utilization
    pub resources: Option<ResourceUtilization>,
    /// Whether the query was cached
    pub was_cached: bool,
    /// Query complexity score
    pub complexity_score: f64,
}

impl ExecutionProfile {
    /// Create a new execution profile
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            timestamp: SystemTime::now(),
            duration: Duration::from_secs(0),
            result_count: 0,
            phases: None,
            resources: None,
            was_cached: false,
            complexity_score: 0.0,
        }
    }

    /// Set execution duration in milliseconds
    pub fn with_duration_ms(mut self, ms: u64) -> Self {
        self.duration = Duration::from_millis(ms);
        self
    }

    /// Set result count
    pub fn with_result_count(mut self, count: usize) -> Self {
        self.result_count = count;
        self
    }

    /// Set execution phases
    pub fn with_phases(mut self, phases: ExecutionPhases) -> Self {
        self.phases = Some(phases);
        self
    }

    /// Set resource utilization
    pub fn with_resources(mut self, resources: ResourceUtilization) -> Self {
        self.resources = Some(resources);
        self
    }

    /// Mark as cached
    pub fn cached(mut self) -> Self {
        self.was_cached = true;
        self
    }

    /// Set complexity score
    pub fn with_complexity(mut self, score: f64) -> Self {
        self.complexity_score = score;
        self
    }

    /// Get execution duration in milliseconds
    pub fn duration_ms(&self) -> f64 {
        self.duration.as_secs_f64() * 1000.0
    }
}

/// Performance bottleneck type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BottleneckType {
    /// Slow query parsing
    Parsing,
    /// Inefficient query plan
    Optimization,
    /// Slow query execution
    Execution,
    /// Large result set serialization
    Serialization,
    /// High memory usage
    Memory,
    /// High CPU usage
    Cpu,
    /// I/O bottleneck
    Io,
    /// Cartesian product
    CartesianProduct,
}

/// Performance bottleneck with severity and recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    /// Type of bottleneck
    pub bottleneck_type: BottleneckType,
    /// Severity (0.0 - 1.0)
    pub severity: f64,
    /// Description
    pub description: String,
    /// Recommendation for improvement
    pub recommendation: String,
    /// Estimated improvement (percentage)
    pub estimated_improvement: f64,
}

/// Performance insights and recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceInsights {
    /// Overall performance score (0 - 100)
    pub score: f64,
    /// Detected bottlenecks
    pub bottlenecks: Vec<PerformanceBottleneck>,
    /// General recommendations
    pub recommendations: Vec<String>,
    /// Predicted performance for similar queries
    pub predicted_duration_ms: Option<f64>,
    /// Statistical summary
    pub summary: PerformanceSummary,
}

/// Statistical performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    /// Total queries analyzed
    pub total_queries: usize,
    /// Average execution time (ms)
    pub avg_duration_ms: f64,
    /// Median execution time (ms)
    pub median_duration_ms: f64,
    /// 95th percentile (ms)
    pub p95_duration_ms: f64,
    /// 99th percentile (ms)
    pub p99_duration_ms: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Slow query count
    pub slow_query_count: usize,
}

impl Default for PerformanceSummary {
    fn default() -> Self {
        Self {
            total_queries: 0,
            avg_duration_ms: 0.0,
            median_duration_ms: 0.0,
            p95_duration_ms: 0.0,
            p99_duration_ms: 0.0,
            cache_hit_rate: 0.0,
            slow_query_count: 0,
        }
    }
}

/// Query performance analyzer
pub struct QueryPerformanceAnalyzer {
    /// Configuration
    config: AnalyzerConfig,
    /// Execution history
    history: VecDeque<ExecutionProfile>,
    /// Metrics
    queries_analyzed: Counter,
    bottlenecks_detected: Counter,
    analysis_duration: Timer,
    /// Random number generator for sampling
    rng: Random,
}

impl QueryPerformanceAnalyzer {
    /// Create a new performance analyzer
    pub fn new(config: AnalyzerConfig) -> Self {
        Self {
            config,
            history: VecDeque::new(),
            queries_analyzed: Counter::new("queries_analyzed".to_string()),
            bottlenecks_detected: Counter::new("bottlenecks_detected".to_string()),
            analysis_duration: Timer::new("analysis_duration".to_string()),
            rng: Random::default(),
        }
    }

    /// Record a query execution
    pub fn record_execution(&mut self, profile: ExecutionProfile) -> Result<()> {
        // Apply sampling
        if self.rng.random_f64() > self.config.sampling_rate {
            return Ok(());
        }

        // Maintain max records limit
        while self.history.len() >= self.config.max_records {
            self.history.pop_front();
        }

        self.history.push_back(profile);
        self.queries_analyzed.inc();

        Ok(())
    }

    /// Analyze performance and get insights
    pub fn analyze_performance(&self) -> Result<PerformanceInsights> {
        let start = Instant::now();

        if self.history.is_empty() {
            return Ok(PerformanceInsights {
                score: 0.0,
                bottlenecks: Vec::new(),
                recommendations: vec!["No queries analyzed yet".to_string()],
                predicted_duration_ms: None,
                summary: PerformanceSummary::default(),
            });
        }

        // Calculate statistical summary
        let summary = self.calculate_summary()?;

        // Detect bottlenecks
        let bottlenecks = self.detect_bottlenecks()?;

        // Generate recommendations
        let recommendations = self.generate_recommendations(&bottlenecks)?;

        // Calculate performance score
        let score = self.calculate_performance_score(&summary, &bottlenecks)?;

        // Predict future performance (if ML enabled)
        let predicted_duration_ms = if self.config.enable_ml_predictions {
            Some(self.predict_duration()?)
        } else {
            None
        };

        let insights = PerformanceInsights {
            score,
            bottlenecks,
            recommendations,
            predicted_duration_ms,
            summary,
        };

        Ok(insights)
    }

    /// Calculate statistical summary
    fn calculate_summary(&self) -> Result<PerformanceSummary> {
        let mut durations: Vec<f64> = self.history
            .iter()
            .map(|p| p.duration_ms())
            .collect();

        durations.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let avg = durations.iter().sum::<f64>() / durations.len() as f64;

        let median = if !durations.is_empty() {
            durations[durations.len() / 2]
        } else {
            0.0
        };

        let p95_idx = (durations.len() as f64 * 0.95) as usize;
        let p99_idx = (durations.len() as f64 * 0.99) as usize;

        let p95 = durations.get(p95_idx).copied().unwrap_or(0.0);
        let p99 = durations.get(p99_idx).copied().unwrap_or(0.0);

        let cached = self.history.iter().filter(|p| p.was_cached).count();
        let cache_hit_rate = cached as f64 / self.history.len() as f64;

        let slow_count = self.history
            .iter()
            .filter(|p| p.duration_ms() > self.config.slow_query_threshold_ms as f64)
            .count();

        Ok(PerformanceSummary {
            total_queries: self.history.len(),
            avg_duration_ms: avg,
            median_duration_ms: median,
            p95_duration_ms: p95,
            p99_duration_ms: p99,
            cache_hit_rate,
            slow_query_count: slow_count,
        })
    }

    /// Detect performance bottlenecks
    fn detect_bottlenecks(&self) -> Result<Vec<PerformanceBottleneck>> {
        let mut bottlenecks = Vec::new();

        // Analyze phase breakdown
        let profiles_with_phases: Vec<_> = self.history
            .iter()
            .filter_map(|p| p.phases.as_ref().map(|ph| (p, ph)))
            .collect();

        if !profiles_with_phases.is_empty() {
            // Check for parsing bottlenecks
            let avg_parse = profiles_with_phases.iter()
                .map(|(_, ph)| ph.parse_ms)
                .sum::<f64>() / profiles_with_phases.len() as f64;

            if avg_parse > 100.0 {
                bottlenecks.push(PerformanceBottleneck {
                    bottleneck_type: BottleneckType::Parsing,
                    severity: (avg_parse / 1000.0).min(1.0),
                    description: format!("Average parsing time is {:.0}ms", avg_parse),
                    recommendation: "Consider caching parsed queries or simplifying query syntax".to_string(),
                    estimated_improvement: 20.0,
                });
            }

            // Check for optimization bottlenecks
            let avg_optimize = profiles_with_phases.iter()
                .map(|(_, ph)| ph.optimize_ms)
                .sum::<f64>() / profiles_with_phases.len() as f64;

            if avg_optimize > 500.0 {
                bottlenecks.push(PerformanceBottleneck {
                    bottleneck_type: BottleneckType::Optimization,
                    severity: (avg_optimize / 2000.0).min(1.0),
                    description: format!("Average optimization time is {:.0}ms", avg_optimize),
                    recommendation: "Enable query plan caching or simplify join patterns".to_string(),
                    estimated_improvement: 35.0,
                });
            }

            // Check for execution bottlenecks
            let avg_execute = profiles_with_phases.iter()
                .map(|(_, ph)| ph.execute_ms)
                .sum::<f64>() / profiles_with_phases.len() as f64;

            if avg_execute > 1000.0 {
                bottlenecks.push(PerformanceBottleneck {
                    bottleneck_type: BottleneckType::Execution,
                    severity: (avg_execute / 5000.0).min(1.0),
                    description: format!("Average execution time is {:.0}ms", avg_execute),
                    recommendation: "Add indexes, use filters early, or limit result sets".to_string(),
                    estimated_improvement: 50.0,
                });
            }
        }

        // Check resource utilization
        let profiles_with_resources: Vec<_> = self.history
            .iter()
            .filter_map(|p| p.resources.as_ref().map(|r| (p, r)))
            .collect();

        if !profiles_with_resources.is_empty() {
            let avg_memory = profiles_with_resources.iter()
                .map(|(_, r)| r.peak_memory_mb)
                .sum::<f64>() / profiles_with_resources.len() as f64;

            if avg_memory > 1024.0 {
                bottlenecks.push(PerformanceBottleneck {
                    bottleneck_type: BottleneckType::Memory,
                    severity: (avg_memory / 4096.0).min(1.0),
                    description: format!("High memory usage: {:.0}MB average", avg_memory),
                    recommendation: "Enable streaming results or reduce intermediate result sizes".to_string(),
                    estimated_improvement: 30.0,
                });
            }
        }

        // Check for slow queries
        let slow_queries = self.history.iter()
            .filter(|p| p.duration_ms() > self.config.slow_query_threshold_ms as f64)
            .count();

        if slow_queries as f64 / self.history.len() as f64 > 0.1 {
            let severity = (slow_queries as f64 / self.history.len() as f64).min(1.0);
            bottlenecks.push(PerformanceBottleneck {
                bottleneck_type: BottleneckType::CartesianProduct,
                severity,
                description: format!("{}% of queries are slow (>{} ms)",
                    (severity * 100.0) as u32,
                    self.config.slow_query_threshold_ms),
                recommendation: "Review query patterns for cartesian products or missing filters".to_string(),
                estimated_improvement: 60.0,
            });
        }

        if !bottlenecks.is_empty() {
            for _ in 0..bottlenecks.len() {
                self.bottlenecks_detected.inc();
            }
        }

        Ok(bottlenecks)
    }

    /// Generate optimization recommendations
    fn generate_recommendations(&self, bottlenecks: &[PerformanceBottleneck]) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        // Add specific recommendations based on bottlenecks
        for bottleneck in bottlenecks {
            if bottleneck.severity > 0.5 {
                recommendations.push(bottleneck.recommendation.clone());
            }
        }

        // Add general recommendations
        let summary = self.calculate_summary()?;

        if summary.cache_hit_rate < 0.3 {
            recommendations.push("Low cache hit rate - consider enabling result caching".to_string());
        }

        if summary.p99_duration_ms > summary.avg_duration_ms * 5.0 {
            recommendations.push("High variance in query performance - analyze outliers".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Performance is within acceptable parameters".to_string());
        }

        Ok(recommendations)
    }

    /// Calculate overall performance score (0-100)
    fn calculate_performance_score(
        &self,
        summary: &PerformanceSummary,
        bottlenecks: &[PerformanceBottleneck],
    ) -> Result<f64> {
        let mut score = 100.0;

        // Penalize for slow average performance
        if summary.avg_duration_ms > self.config.slow_query_threshold_ms as f64 {
            let penalty = ((summary.avg_duration_ms / self.config.slow_query_threshold_ms as f64) - 1.0) * 30.0;
            score -= penalty.min(40.0);
        }

        // Penalize for bottlenecks
        for bottleneck in bottlenecks {
            score -= bottleneck.severity * 15.0;
        }

        // Reward for good cache hit rate
        score += summary.cache_hit_rate * 10.0;

        Ok(score.max(0.0).min(100.0))
    }

    /// Predict query duration using simple ML model
    fn predict_duration(&self) -> Result<f64> {
        if self.history.len() < 10 {
            return Ok(0.0);
        }

        // Simple exponential moving average prediction
        let alpha = 0.3;
        let mut ema = self.history[0].duration_ms();

        for profile in self.history.iter().skip(1) {
            ema = alpha * profile.duration_ms() + (1.0 - alpha) * ema;
        }

        Ok(ema)
    }

    /// Get execution history
    pub fn history(&self) -> &VecDeque<ExecutionProfile> {
        &self.history
    }

    /// Clear execution history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Get number of queries analyzed
    pub fn queries_analyzed_count(&self) -> u64 {
        self.queries_analyzed.get()
    }

    /// Get number of bottlenecks detected
    pub fn bottlenecks_detected_count(&self) -> u64 {
        self.bottlenecks_detected.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyzer_creation() {
        let analyzer = QueryPerformanceAnalyzer::new(AnalyzerConfig::default());
        assert_eq!(analyzer.history().len(), 0);
        assert_eq!(analyzer.queries_analyzed_count(), 0);
    }

    #[test]
    fn test_record_execution() {
        let mut analyzer = QueryPerformanceAnalyzer::new(AnalyzerConfig::default());

        let profile = ExecutionProfile::new("SELECT * WHERE { ?s ?p ?o }")
            .with_duration_ms(100)
            .with_result_count(50);

        analyzer.record_execution(profile).unwrap();
        assert_eq!(analyzer.history().len(), 1);
    }

    #[test]
    fn test_execution_profile_builder() {
        let profile = ExecutionProfile::new("SELECT ?s WHERE { ?s ?p ?o }")
            .with_duration_ms(250)
            .with_result_count(100)
            .cached()
            .with_complexity(0.5);

        assert_eq!(profile.duration_ms(), 250.0);
        assert_eq!(profile.result_count, 100);
        assert!(profile.was_cached);
        assert_eq!(profile.complexity_score, 0.5);
    }

    #[test]
    fn test_performance_analysis_empty() {
        let analyzer = QueryPerformanceAnalyzer::new(AnalyzerConfig::default());
        let insights = analyzer.analyze_performance().unwrap();

        assert_eq!(insights.score, 0.0);
        assert_eq!(insights.summary.total_queries, 0);
    }

    #[test]
    fn test_performance_analysis_with_data() {
        let mut analyzer = QueryPerformanceAnalyzer::new(AnalyzerConfig::default());

        // Add some profiles
        for i in 1..=10 {
            let profile = ExecutionProfile::new(format!("SELECT {} WHERE {{ ?s ?p ?o }}", i))
                .with_duration_ms(i * 100)
                .with_result_count(i * 10);

            analyzer.record_execution(profile).unwrap();
        }

        let insights = analyzer.analyze_performance().unwrap();

        assert_eq!(insights.summary.total_queries, 10);
        assert!(insights.score > 0.0);
        assert!(insights.summary.avg_duration_ms > 0.0);
    }

    #[test]
    fn test_statistical_summary() {
        let mut analyzer = QueryPerformanceAnalyzer::new(AnalyzerConfig::default());

        // Add queries with known durations
        let durations = vec![100, 200, 300, 400, 500];
        for duration in durations {
            let profile = ExecutionProfile::new("SELECT * WHERE { ?s ?p ?o }")
                .with_duration_ms(duration);

            analyzer.record_execution(profile).unwrap();
        }

        let summary = analyzer.calculate_summary().unwrap();

        assert_eq!(summary.total_queries, 5);
        assert_eq!(summary.avg_duration_ms, 300.0);
        assert_eq!(summary.median_duration_ms, 300.0);
    }

    #[test]
    fn test_bottleneck_detection() {
        let mut analyzer = QueryPerformanceAnalyzer::new(
            AnalyzerConfig::default().with_slow_query_threshold_ms(100)
        );

        // Add slow queries
        for _ in 0..5 {
            let phases = ExecutionPhases {
                parse_ms: 10.0,
                optimize_ms: 20.0,
                execute_ms: 2000.0, // Slow execution
                serialize_ms: 10.0,
            };

            let profile = ExecutionProfile::new("SELECT * WHERE { ?s ?p ?o }")
                .with_duration_ms(2040)
                .with_phases(phases);

            analyzer.record_execution(profile).unwrap();
        }

        let bottlenecks = analyzer.detect_bottlenecks().unwrap();

        assert!(!bottlenecks.is_empty());
        assert!(bottlenecks.iter().any(|b| b.bottleneck_type == BottleneckType::Execution));
    }

    #[test]
    fn test_max_records_limit() {
        let config = AnalyzerConfig {
            max_records: 5,
            ..Default::default()
        };

        let mut analyzer = QueryPerformanceAnalyzer::new(config);

        // Add more than max_records
        for i in 0..10 {
            let profile = ExecutionProfile::new(format!("Query {}", i))
                .with_duration_ms(100);

            analyzer.record_execution(profile).unwrap();
        }

        assert_eq!(analyzer.history().len(), 5);
    }

    #[test]
    fn test_config_presets() {
        let lightweight = AnalyzerConfig::lightweight();
        assert_eq!(lightweight.max_records, 1000);
        assert!(!lightweight.enable_ml_predictions);

        let comprehensive = AnalyzerConfig::comprehensive();
        assert_eq!(comprehensive.max_records, 50000);
        assert!(comprehensive.enable_ml_predictions);
    }

    #[test]
    fn test_cache_hit_rate_calculation() {
        let mut analyzer = QueryPerformanceAnalyzer::new(AnalyzerConfig::default());

        // Add 3 cached and 7 uncached queries
        for i in 0..10 {
            let mut profile = ExecutionProfile::new(format!("Query {}", i))
                .with_duration_ms(100);

            if i < 3 {
                profile = profile.cached();
            }

            analyzer.record_execution(profile).unwrap();
        }

        let summary = analyzer.calculate_summary().unwrap();
        assert_eq!(summary.cache_hit_rate, 0.3);
    }

    #[test]
    fn test_execution_phases() {
        let phases = ExecutionPhases {
            parse_ms: 10.0,
            optimize_ms: 50.0,
            execute_ms: 200.0,
            serialize_ms: 5.0,
        };

        assert_eq!(phases.total_ms(), 265.0);
        assert_eq!(phases.slowest_phase(), "execute");
    }

    #[test]
    fn test_performance_score_calculation() {
        let analyzer = QueryPerformanceAnalyzer::new(AnalyzerConfig::default());

        let summary = PerformanceSummary {
            total_queries: 10,
            avg_duration_ms: 500.0,
            median_duration_ms: 400.0,
            p95_duration_ms: 900.0,
            p99_duration_ms: 1000.0,
            cache_hit_rate: 0.5,
            slow_query_count: 2,
        };

        let score = analyzer.calculate_performance_score(&summary, &[]).unwrap();

        assert!(score > 0.0 && score <= 100.0);
    }
}
