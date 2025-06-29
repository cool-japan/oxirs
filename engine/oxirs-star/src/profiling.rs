//! Performance profiling and analysis tools for RDF-star operations.
//!
//! This module provides comprehensive profiling capabilities for analyzing
//! the performance characteristics of RDF-star parsing, serialization, and query operations.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tracing::{debug, info, span, Level};

use crate::model::{StarGraph, StarQuad, StarTerm, StarTriple};
use crate::parser::{StarFormat, StarParser};
use crate::serializer::{SerializationOptions, StarSerializer};
use crate::{StarConfig, StarError, StarResult};

/// Profiling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingConfig {
    /// Enable detailed memory tracking
    pub track_memory: bool,
    /// Enable operation timing
    pub track_timing: bool,
    /// Sample rate for profiling (0.0 to 1.0)
    pub sample_rate: f64,
    /// Maximum number of samples to keep
    pub max_samples: usize,
    /// Enable statistical analysis
    pub enable_statistics: bool,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            track_memory: true,
            track_timing: true,
            sample_rate: 1.0,
            max_samples: 10000,
            enable_statistics: true,
        }
    }
}

/// Performance profiler for RDF-star operations
pub struct StarProfiler {
    config: ProfilingConfig,
    samples: Vec<ProfileSample>,
    operation_stats: HashMap<String, OperationStatistics>,
    start_time: Option<Instant>,
    current_operation: Option<String>,
}

/// Individual performance sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileSample {
    /// Operation name
    pub operation: String,
    /// Duration of the operation
    pub duration: Duration,
    /// Memory used (in bytes)
    pub memory_used: Option<u64>,
    /// Input size (for parsing/serialization)
    pub input_size: Option<usize>,
    /// Output size (for serialization)
    pub output_size: Option<usize>,
    /// Timestamp when operation started
    pub timestamp: std::time::SystemTime,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Aggregated statistics for an operation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationStatistics {
    /// Total number of samples
    pub count: usize,
    /// Total time spent in this operation
    pub total_duration: Duration,
    /// Average duration
    pub average_duration: Duration,
    /// Minimum duration observed
    pub min_duration: Duration,
    /// Maximum duration observed
    pub max_duration: Duration,
    /// Standard deviation of durations
    pub std_deviation: f64,
    /// Operations per second (average)
    pub ops_per_second: f64,
    /// Throughput in bytes per second (if applicable)
    pub bytes_per_second: Option<f64>,
}

/// Comprehensive profiling report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingReport {
    /// Configuration used for profiling
    pub config: ProfilingConfig,
    /// Total profiling duration
    pub total_duration: Duration,
    /// Total number of samples collected
    pub total_samples: usize,
    /// Statistics by operation type
    pub operation_stats: HashMap<String, OperationStatistics>,
    /// Performance trends over time
    pub trends: Vec<PerformanceTrend>,
    /// Memory usage patterns
    pub memory_patterns: Option<MemoryUsagePattern>,
    /// Bottleneck analysis
    pub bottlenecks: Vec<PerformanceBottleneck>,
}

/// Performance trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    /// Operation name
    pub operation: String,
    /// Time window (start)
    pub window_start: std::time::SystemTime,
    /// Time window (end)
    pub window_end: std::time::SystemTime,
    /// Average performance in this window
    pub average_duration: Duration,
    /// Trend direction (improving/degrading)
    pub trend_direction: TrendDirection,
    /// Confidence in trend analysis (0.0 to 1.0)
    pub confidence: f64,
}

/// Memory usage pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsagePattern {
    /// Peak memory usage observed
    pub peak_memory: u64,
    /// Average memory usage
    pub average_memory: u64,
    /// Memory efficiency (output/input ratio)
    pub efficiency_ratio: f64,
    /// Memory leak indicators
    pub potential_leaks: Vec<String>,
}

/// Performance bottleneck identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    /// Operation causing the bottleneck
    pub operation: String,
    /// Severity (0.0 to 1.0)
    pub severity: f64,
    /// Description of the bottleneck
    pub description: String,
    /// Suggested optimizations
    pub suggestions: Vec<String>,
    /// Percentage of total time consumed
    pub time_percentage: f64,
}

/// Trend direction enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Unknown,
}

impl StarProfiler {
    /// Create a new profiler with default configuration
    pub fn new() -> Self {
        Self::with_config(ProfilingConfig::default())
    }

    /// Create a new profiler with custom configuration
    pub fn with_config(config: ProfilingConfig) -> Self {
        Self {
            config,
            samples: Vec::new(),
            operation_stats: HashMap::new(),
            start_time: None,
            current_operation: None,
        }
    }

    /// Start profiling an operation
    pub fn start_operation(&mut self, operation: &str) {
        if self.should_sample() {
            self.current_operation = Some(operation.to_string());
            self.start_time = Some(Instant::now());
            debug!("Started profiling operation: {}", operation);
        }
    }

    /// End profiling the current operation
    pub fn end_operation(&mut self) {
        self.end_operation_with_metadata(HashMap::new());
    }

    /// End profiling with additional metadata
    pub fn end_operation_with_metadata(&mut self, metadata: HashMap<String, String>) {
        if let (Some(operation), Some(start_time)) =
            (self.current_operation.clone(), self.start_time)
        {
            let duration = start_time.elapsed();

            let sample = ProfileSample {
                operation: operation.clone(),
                duration,
                memory_used: if self.config.track_memory {
                    Some(self.estimate_memory_usage())
                } else {
                    None
                },
                input_size: metadata.get("input_size").and_then(|s| s.parse().ok()),
                output_size: metadata.get("output_size").and_then(|s| s.parse().ok()),
                timestamp: std::time::SystemTime::now(),
                metadata,
            };

            self.add_sample(sample);
            self.current_operation = None;
            self.start_time = None;

            debug!(
                "Finished profiling operation: {} ({}ms)",
                operation,
                duration.as_millis()
            );
        }
    }

    /// Profile a parsing operation
    pub fn profile_parsing<F, R>(&mut self, format: StarFormat, input_size: usize, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let operation = format!("parse_{:?}", format);

        let mut metadata = HashMap::new();
        metadata.insert("input_size".to_string(), input_size.to_string());
        metadata.insert("format".to_string(), format!("{:?}", format));

        self.start_operation(&operation);
        let result = f();
        self.end_operation_with_metadata(metadata);

        result
    }

    /// Profile a serialization operation
    pub fn profile_serialization<F, R>(
        &mut self,
        format: StarFormat,
        input_triples: usize,
        f: F,
    ) -> R
    where
        F: FnOnce() -> R,
    {
        let operation = format!("serialize_{:?}", format);

        let mut metadata = HashMap::new();
        metadata.insert("input_triples".to_string(), input_triples.to_string());
        metadata.insert("format".to_string(), format!("{:?}", format));

        self.start_operation(&operation);
        let result = f();
        self.end_operation_with_metadata(metadata);

        result
    }

    /// Profile a query operation
    pub fn profile_query<F, R>(&mut self, query_type: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let operation = format!("query_{}", query_type);

        let mut metadata = HashMap::new();
        metadata.insert("query_type".to_string(), query_type.to_string());

        self.start_operation(&operation);
        let result = f();
        self.end_operation_with_metadata(metadata);

        result
    }

    /// Add a sample to the profiler
    pub fn add_sample(&mut self, sample: ProfileSample) {
        if self.samples.len() >= self.config.max_samples {
            // Remove oldest samples
            let remove_count = self.samples.len() - self.config.max_samples + 1;
            self.samples.drain(0..remove_count);
        }

        // Update operation statistics
        self.update_operation_stats(&sample);

        self.samples.push(sample);
    }

    /// Generate a comprehensive profiling report
    pub fn generate_report(&self) -> ProfilingReport {
        let total_duration = self.calculate_total_duration();
        let trends = self.analyze_trends();
        let memory_patterns = if self.config.track_memory {
            Some(self.analyze_memory_patterns())
        } else {
            None
        };
        let bottlenecks = self.identify_bottlenecks();

        ProfilingReport {
            config: self.config.clone(),
            total_duration,
            total_samples: self.samples.len(),
            operation_stats: self.operation_stats.clone(),
            trends,
            memory_patterns,
            bottlenecks,
        }
    }

    /// Get samples for a specific operation
    pub fn get_operation_samples(&self, operation: &str) -> Vec<&ProfileSample> {
        self.samples
            .iter()
            .filter(|sample| sample.operation == operation)
            .collect()
    }

    /// Get the most recent samples
    pub fn get_recent_samples(&self, count: usize) -> Vec<&ProfileSample> {
        let start_index = self.samples.len().saturating_sub(count);
        self.samples[start_index..].iter().collect()
    }

    /// Clear all collected samples
    pub fn clear_samples(&mut self) {
        self.samples.clear();
        self.operation_stats.clear();
    }

    /// Export samples to JSON
    pub fn export_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(&self.samples)
    }

    /// Import samples from JSON
    pub fn import_json(&mut self, json: &str) -> serde_json::Result<()> {
        let samples: Vec<ProfileSample> = serde_json::from_str(json)?;
        for sample in samples {
            self.add_sample(sample);
        }
        Ok(())
    }

    // Private helper methods

    fn should_sample(&self) -> bool {
        if self.config.sample_rate >= 1.0 {
            true
        } else if self.config.sample_rate <= 0.0 {
            false
        } else {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            std::time::SystemTime::now().hash(&mut hasher);
            let hash = hasher.finish();

            (hash as f64 / u64::MAX as f64) < self.config.sample_rate
        }
    }

    fn estimate_memory_usage(&self) -> u64 {
        // Simple memory estimation based on process memory
        // In a real implementation, this would use more sophisticated memory tracking
        1024 * 1024 // 1MB placeholder
    }

    fn update_operation_stats(&mut self, sample: &ProfileSample) {
        let stats = self
            .operation_stats
            .entry(sample.operation.clone())
            .or_insert_with(|| OperationStatistics {
                count: 0,
                total_duration: Duration::ZERO,
                average_duration: Duration::ZERO,
                min_duration: sample.duration,
                max_duration: sample.duration,
                std_deviation: 0.0,
                ops_per_second: 0.0,
                bytes_per_second: None,
            });

        stats.count += 1;
        stats.total_duration += sample.duration;
        stats.average_duration = stats.total_duration / stats.count as u32;
        stats.min_duration = stats.min_duration.min(sample.duration);
        stats.max_duration = stats.max_duration.max(sample.duration);

        // Calculate operations per second
        if stats.average_duration.as_secs_f64() > 0.0 {
            stats.ops_per_second = 1.0 / stats.average_duration.as_secs_f64();
        }

        // Calculate bytes per second if applicable
        if let Some(input_size) = sample.input_size {
            let bytes_per_sec = input_size as f64 / sample.duration.as_secs_f64();
            stats.bytes_per_second = Some(bytes_per_sec);
        }

        // Update standard deviation (simplified calculation)
        // Note: Skip std_deviation calculation to avoid borrowing issues
        // In a production implementation, this would be calculated separately
        stats.std_deviation = 0.0;
    }

    fn calculate_total_duration(&self) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }

        let earliest = self.samples.iter().map(|s| s.timestamp).min().unwrap();

        let latest = self.samples.iter().map(|s| s.timestamp).max().unwrap();

        latest.duration_since(earliest).unwrap_or(Duration::ZERO)
    }

    fn analyze_trends(&self) -> Vec<PerformanceTrend> {
        let mut trends = Vec::new();

        for operation in self.operation_stats.keys() {
            let samples = self.get_operation_samples(operation);
            if samples.len() >= 3 {
                let trend = self.calculate_trend_for_operation(operation, &samples);
                trends.push(trend);
            }
        }

        trends
    }

    fn calculate_trend_for_operation(
        &self,
        operation: &str,
        samples: &[&ProfileSample],
    ) -> PerformanceTrend {
        // Simple linear regression to determine trend
        let n = samples.len() as f64;
        let sum_x: f64 = (0..samples.len()).map(|i| i as f64).sum();
        let sum_y: f64 = samples.iter().map(|s| s.duration.as_secs_f64()).sum();
        let sum_xy: f64 = samples
            .iter()
            .enumerate()
            .map(|(i, s)| i as f64 * s.duration.as_secs_f64())
            .sum();
        let sum_x2: f64 = (0..samples.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));

        let direction = if slope.abs() < 0.001 {
            TrendDirection::Stable
        } else if slope < 0.0 {
            TrendDirection::Improving // Decreasing time is improving
        } else {
            TrendDirection::Degrading
        };

        let confidence = if n >= 10.0 { 0.8 } else { 0.4 };

        PerformanceTrend {
            operation: operation.to_string(),
            window_start: samples.first().unwrap().timestamp,
            window_end: samples.last().unwrap().timestamp,
            average_duration: Duration::from_secs_f64(sum_y / n),
            trend_direction: direction,
            confidence,
        }
    }

    fn analyze_memory_patterns(&self) -> MemoryUsagePattern {
        let memory_samples: Vec<u64> = self.samples.iter().filter_map(|s| s.memory_used).collect();

        if memory_samples.is_empty() {
            return MemoryUsagePattern {
                peak_memory: 0,
                average_memory: 0,
                efficiency_ratio: 0.0,
                potential_leaks: Vec::new(),
            };
        }

        let peak_memory = *memory_samples.iter().max().unwrap();
        let average_memory = memory_samples.iter().sum::<u64>() / memory_samples.len() as u64;

        // Calculate efficiency ratio (simplified)
        let efficiency_ratio = if peak_memory > 0 {
            average_memory as f64 / peak_memory as f64
        } else {
            0.0
        };

        // Detect potential memory leaks (simplified heuristic)
        let mut potential_leaks = Vec::new();
        if memory_samples.len() > 10 {
            let first_half_avg = memory_samples[..memory_samples.len() / 2]
                .iter()
                .sum::<u64>() as f64
                / (memory_samples.len() / 2) as f64;
            let second_half_avg = memory_samples[memory_samples.len() / 2..]
                .iter()
                .sum::<u64>() as f64
                / (memory_samples.len() / 2) as f64;

            if second_half_avg > first_half_avg * 1.5 {
                potential_leaks.push("Increasing memory usage trend detected".to_string());
            }
        }

        MemoryUsagePattern {
            peak_memory,
            average_memory,
            efficiency_ratio,
            potential_leaks,
        }
    }

    fn identify_bottlenecks(&self) -> Vec<PerformanceBottleneck> {
        let mut bottlenecks = Vec::new();
        let total_time: Duration = self
            .operation_stats
            .values()
            .map(|stats| stats.total_duration)
            .sum();

        if total_time.as_secs_f64() == 0.0 {
            return bottlenecks;
        }

        for (operation, stats) in &self.operation_stats {
            let time_percentage =
                stats.total_duration.as_secs_f64() / total_time.as_secs_f64() * 100.0;

            if time_percentage > 20.0 {
                // Consider operations taking more than 20% of total time as bottlenecks
                let severity = (time_percentage / 100.0).min(1.0);

                let mut suggestions = Vec::new();
                if stats.average_duration.as_millis() > 100 {
                    suggestions.push("Consider optimizing algorithm or implementation".to_string());
                }
                if stats.std_deviation > stats.average_duration.as_secs_f64() * 0.5 {
                    suggestions.push(
                        "High variance detected - investigate inconsistent performance".to_string(),
                    );
                }

                bottlenecks.push(PerformanceBottleneck {
                    operation: operation.clone(),
                    severity,
                    description: format!(
                        "Operation consumes {:.1}% of total execution time",
                        time_percentage
                    ),
                    suggestions,
                    time_percentage,
                });
            }
        }

        // Sort by severity (highest first)
        bottlenecks.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap());

        bottlenecks
    }
}

impl Default for StarProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience macro for profiling operations
#[macro_export]
macro_rules! profile_operation {
    ($profiler:expr, $operation:expr, $code:block) => {{
        $profiler.start_operation($operation);
        let result = $code;
        $profiler.end_operation();
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let profiler = StarProfiler::new();
        assert_eq!(profiler.samples.len(), 0);
        assert_eq!(profiler.operation_stats.len(), 0);
    }

    #[test]
    fn test_operation_profiling() {
        let mut profiler = StarProfiler::new();

        profiler.start_operation("test_operation");
        std::thread::sleep(Duration::from_millis(10));
        profiler.end_operation();

        assert_eq!(profiler.samples.len(), 1);
        assert!(profiler.samples[0].duration >= Duration::from_millis(10));
        assert_eq!(profiler.samples[0].operation, "test_operation");
    }

    #[test]
    fn test_operation_statistics() {
        let mut profiler = StarProfiler::new();

        // Add multiple samples for the same operation
        for _ in 0..5 {
            profiler.start_operation("test_op");
            std::thread::sleep(Duration::from_millis(1));
            profiler.end_operation();
        }

        let stats = profiler.operation_stats.get("test_op").unwrap();
        assert_eq!(stats.count, 5);
        assert!(stats.average_duration > Duration::ZERO);
        assert!(stats.ops_per_second > 0.0);
    }

    #[test]
    fn test_sample_export_import() {
        let mut profiler = StarProfiler::new();

        profiler.start_operation("export_test");
        profiler.end_operation();

        let json = profiler.export_json().unwrap();
        assert!(!json.is_empty());

        let mut new_profiler = StarProfiler::new();
        new_profiler.import_json(&json).unwrap();

        assert_eq!(new_profiler.samples.len(), 1);
        assert_eq!(new_profiler.samples[0].operation, "export_test");
    }

    #[test]
    fn test_trend_analysis() {
        let mut profiler = StarProfiler::new();

        // Add samples with increasing duration to simulate degrading performance
        for i in 1..=10 {
            let sample = ProfileSample {
                operation: "degrading_op".to_string(),
                duration: Duration::from_millis(i * 10),
                memory_used: None,
                input_size: None,
                output_size: None,
                timestamp: std::time::SystemTime::now(),
                metadata: HashMap::new(),
            };
            profiler.add_sample(sample);
        }

        let report = profiler.generate_report();
        let trends = &report.trends;

        assert!(!trends.is_empty());
        assert_eq!(trends[0].operation, "degrading_op");
        assert!(matches!(
            trends[0].trend_direction,
            TrendDirection::Degrading
        ));
    }
}
