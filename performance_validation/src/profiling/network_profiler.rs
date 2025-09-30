//! Network Performance Profiler
//!
//! Monitors network I/O, latency, and bandwidth utilization.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, Ordering};

/// Network profiler for monitoring I/O and performance
#[derive(Debug)]
pub struct NetworkProfiler {
    operation_metrics: HashMap<String, NetworkOperationMetrics>,
    global_metrics: NetworkGlobalMetrics,
    start_time: Instant,
    baseline_stats: NetworkBaseline,
}

/// Per-operation network metrics
#[derive(Debug, Clone)]
pub struct NetworkOperationMetrics {
    pub start_time: Instant,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub requests_sent: u32,
    pub responses_received: u32,
    pub connection_attempts: u32,
    pub connection_successes: u32,
    pub total_latency: Duration,
    pub min_latency: Duration,
    pub max_latency: Duration,
    pub timeouts: u32,
    pub errors: u32,
}

/// Global network metrics
#[derive(Debug, Default)]
pub struct NetworkGlobalMetrics {
    pub total_operations: AtomicU64,
    pub total_bytes_sent: AtomicU64,
    pub total_bytes_received: AtomicU64,
    pub total_requests: AtomicU64,
    pub total_responses: AtomicU64,
    pub total_connections: AtomicU64,
    pub total_timeouts: AtomicU64,
    pub total_errors: AtomicU64,
    pub cumulative_latency: Duration,
}

/// Network baseline measurements
#[derive(Debug, Default)]
pub struct NetworkBaseline {
    pub baseline_latency: Duration,
    pub baseline_bandwidth_mbps: f32,
    pub baseline_packet_loss: f32,
}

/// Network performance report
#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkReport {
    pub duration: Duration,
    pub total_bytes_transferred: u64,
    pub average_bandwidth_mbps: f32,
    pub peak_bandwidth_mbps: f32,
    pub total_operations: u64,
    pub success_rate: f32,
    pub average_latency: Duration,
    pub latency_percentiles: LatencyPercentiles,
    pub connection_metrics: ConnectionMetrics,
    pub error_analysis: ErrorAnalysis,
    pub operation_breakdown: HashMap<String, NetworkOperationSummary>,
    pub network_recommendations: Vec<String>,
}

/// Latency percentile measurements
#[derive(Debug, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    pub p50: Duration,
    pub p90: Duration,
    pub p95: Duration,
    pub p99: Duration,
}

/// Connection-related metrics
#[derive(Debug, Serialize, Deserialize)]
pub struct ConnectionMetrics {
    pub total_connections: u64,
    pub successful_connections: u64,
    pub connection_success_rate: f32,
    pub average_connection_time: Duration,
}

/// Error analysis
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorAnalysis {
    pub total_errors: u64,
    pub timeout_errors: u64,
    pub connection_errors: u64,
    pub error_rate: f32,
    pub most_common_error_type: String,
}

/// Summary of network metrics for an operation
#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkOperationSummary {
    pub bytes_transferred: u64,
    pub average_latency: Duration,
    pub throughput_mbps: f32,
    pub success_rate: f32,
    pub error_count: u32,
    pub performance_score: f32,
}

impl NetworkProfiler {
    /// Create a new network profiler
    pub fn new() -> Self {
        let baseline_stats = Self::measure_baseline();

        Self {
            operation_metrics: HashMap::new(),
            global_metrics: NetworkGlobalMetrics::default(),
            start_time: Instant::now(),
            baseline_stats,
        }
    }

    /// Measure baseline network performance
    fn measure_baseline() -> NetworkBaseline {
        // In a real implementation, this would perform actual network tests
        // For now, return simulated baseline measurements
        NetworkBaseline {
            baseline_latency: Duration::from_millis(50), // 50ms baseline latency
            baseline_bandwidth_mbps: 100.0, // 100 Mbps baseline bandwidth
            baseline_packet_loss: 0.01, // 1% baseline packet loss
        }
    }

    /// Start monitoring an operation
    pub fn start_operation(&mut self, operation_name: &str) {
        let metrics = NetworkOperationMetrics {
            start_time: Instant::now(),
            bytes_sent: 0,
            bytes_received: 0,
            requests_sent: 0,
            responses_received: 0,
            connection_attempts: 0,
            connection_successes: 0,
            total_latency: Duration::ZERO,
            min_latency: Duration::from_secs(u64::MAX),
            max_latency: Duration::ZERO,
            timeouts: 0,
            errors: 0,
        };

        self.operation_metrics.insert(operation_name.to_string(), metrics);
    }

    /// Finish monitoring an operation
    pub fn finish_operation(&mut self, operation_name: &str) -> NetworkOperationMetrics {
        if let Some(metrics) = self.operation_metrics.remove(operation_name) {
            // Update global metrics
            self.global_metrics.total_operations.fetch_add(1, Ordering::Relaxed);
            self.global_metrics.total_bytes_sent.fetch_add(metrics.bytes_sent, Ordering::Relaxed);
            self.global_metrics.total_bytes_received.fetch_add(metrics.bytes_received, Ordering::Relaxed);
            self.global_metrics.total_requests.fetch_add(metrics.requests_sent as u64, Ordering::Relaxed);
            self.global_metrics.total_responses.fetch_add(metrics.responses_received as u64, Ordering::Relaxed);
            self.global_metrics.total_connections.fetch_add(metrics.connection_attempts as u64, Ordering::Relaxed);
            self.global_metrics.total_timeouts.fetch_add(metrics.timeouts as u64, Ordering::Relaxed);
            self.global_metrics.total_errors.fetch_add(metrics.errors as u64, Ordering::Relaxed);

            // Note: In a real implementation, cumulative_latency would need atomic duration handling
            // self.global_metrics.cumulative_latency += metrics.total_latency;

            metrics
        } else {
            // Return default metrics if operation not found
            NetworkOperationMetrics {
                start_time: Instant::now(),
                bytes_sent: 0,
                bytes_received: 0,
                requests_sent: 0,
                responses_received: 0,
                connection_attempts: 0,
                connection_successes: 0,
                total_latency: Duration::ZERO,
                min_latency: Duration::ZERO,
                max_latency: Duration::ZERO,
                timeouts: 0,
                errors: 0,
            }
        }
    }

    /// Record data sent
    pub fn record_data_sent(&mut self, operation_name: &str, bytes: u64) {
        if let Some(metrics) = self.operation_metrics.get_mut(operation_name) {
            metrics.bytes_sent += bytes;
        }
    }

    /// Record data received
    pub fn record_data_received(&mut self, operation_name: &str, bytes: u64) {
        if let Some(metrics) = self.operation_metrics.get_mut(operation_name) {
            metrics.bytes_received += bytes;
        }
    }

    /// Record request sent
    pub fn record_request(&mut self, operation_name: &str) {
        if let Some(metrics) = self.operation_metrics.get_mut(operation_name) {
            metrics.requests_sent += 1;
        }
    }

    /// Record response received with latency
    pub fn record_response(&mut self, operation_name: &str, latency: Duration) {
        if let Some(metrics) = self.operation_metrics.get_mut(operation_name) {
            metrics.responses_received += 1;
            metrics.total_latency += latency;
            metrics.min_latency = metrics.min_latency.min(latency);
            metrics.max_latency = metrics.max_latency.max(latency);
        }
    }

    /// Record connection attempt
    pub fn record_connection_attempt(&mut self, operation_name: &str, success: bool) {
        if let Some(metrics) = self.operation_metrics.get_mut(operation_name) {
            metrics.connection_attempts += 1;
            if success {
                metrics.connection_successes += 1;
            }
        }
    }

    /// Record timeout
    pub fn record_timeout(&mut self, operation_name: &str) {
        if let Some(metrics) = self.operation_metrics.get_mut(operation_name) {
            metrics.timeouts += 1;
        }
    }

    /// Record error
    pub fn record_error(&mut self, operation_name: &str) {
        if let Some(metrics) = self.operation_metrics.get_mut(operation_name) {
            metrics.errors += 1;
        }
    }

    /// Generate comprehensive network report
    pub fn generate_report(&self) -> NetworkReport {
        let runtime = self.start_time.elapsed();
        let total_operations = self.global_metrics.total_operations.load(Ordering::Relaxed);
        let total_bytes_sent = self.global_metrics.total_bytes_sent.load(Ordering::Relaxed);
        let total_bytes_received = self.global_metrics.total_bytes_received.load(Ordering::Relaxed);
        let total_bytes_transferred = total_bytes_sent + total_bytes_received;

        // Calculate bandwidth
        let average_bandwidth_mbps = if runtime.as_secs() > 0 {
            (total_bytes_transferred as f32 * 8.0) / (runtime.as_secs() as f32 * 1_000_000.0)
        } else {
            0.0
        };

        // Calculate peak bandwidth (simplified - would need time-windowed measurements)
        let peak_bandwidth_mbps = average_bandwidth_mbps * 1.5; // Estimate

        // Calculate success rate
        let total_requests = self.global_metrics.total_requests.load(Ordering::Relaxed);
        let total_responses = self.global_metrics.total_responses.load(Ordering::Relaxed);
        let success_rate = if total_requests > 0 {
            (total_responses as f32 / total_requests as f32) * 100.0
        } else {
            100.0
        };

        // Calculate average latency
        let average_latency = if total_responses > 0 {
            // This is simplified - in reality we'd need to properly aggregate durations
            Duration::from_millis(
                self.operation_metrics.values()
                    .map(|m| if m.responses_received > 0 {
                        m.total_latency.as_millis() / m.responses_received as u128
                    } else {
                        0
                    })
                    .sum::<u128>() / self.operation_metrics.len().max(1) as u128
            )
        } else {
            Duration::ZERO
        };

        // Calculate latency percentiles (simplified)
        let latency_percentiles = self.calculate_latency_percentiles();

        // Calculate connection metrics
        let total_connections = self.global_metrics.total_connections.load(Ordering::Relaxed);
        let successful_connections = self.operation_metrics.values()
            .map(|m| m.connection_successes as u64)
            .sum::<u64>();

        let connection_metrics = ConnectionMetrics {
            total_connections,
            successful_connections,
            connection_success_rate: if total_connections > 0 {
                (successful_connections as f32 / total_connections as f32) * 100.0
            } else {
                100.0
            },
            average_connection_time: Duration::from_millis(100), // Simplified estimate
        };

        // Calculate error analysis
        let total_errors = self.global_metrics.total_errors.load(Ordering::Relaxed);
        let timeout_errors = self.global_metrics.total_timeouts.load(Ordering::Relaxed);
        let connection_errors = total_connections - successful_connections;

        let error_analysis = ErrorAnalysis {
            total_errors,
            timeout_errors,
            connection_errors,
            error_rate: if total_requests > 0 {
                (total_errors as f32 / total_requests as f32) * 100.0
            } else {
                0.0
            },
            most_common_error_type: if timeout_errors > connection_errors {
                "Timeout".to_string()
            } else {
                "Connection".to_string()
            },
        };

        // Generate operation summaries
        let operation_breakdown: HashMap<String, NetworkOperationSummary> = self.operation_metrics.iter()
            .map(|(name, metrics)| {
                let bytes_transferred = metrics.bytes_sent + metrics.bytes_received;
                let duration = metrics.start_time.elapsed();
                let throughput_mbps = if duration.as_secs() > 0 {
                    (bytes_transferred as f32 * 8.0) / (duration.as_secs() as f32 * 1_000_000.0)
                } else {
                    0.0
                };

                let success_rate = if metrics.requests_sent > 0 {
                    (metrics.responses_received as f32 / metrics.requests_sent as f32) * 100.0
                } else {
                    100.0
                };

                let average_latency = if metrics.responses_received > 0 {
                    Duration::from_millis(metrics.total_latency.as_millis() / metrics.responses_received as u128)
                } else {
                    Duration::ZERO
                };

                let performance_score = Self::calculate_performance_score(metrics, &self.baseline_stats);

                let summary = NetworkOperationSummary {
                    bytes_transferred,
                    average_latency,
                    throughput_mbps,
                    success_rate,
                    error_count: metrics.errors,
                    performance_score,
                };
                (name.clone(), summary)
            })
            .collect();

        let recommendations = self.generate_recommendations(&connection_metrics, &error_analysis, average_bandwidth_mbps);

        NetworkReport {
            duration: runtime,
            total_bytes_transferred,
            average_bandwidth_mbps,
            peak_bandwidth_mbps,
            total_operations,
            success_rate,
            average_latency,
            latency_percentiles,
            connection_metrics,
            error_analysis,
            operation_breakdown,
            network_recommendations: recommendations,
        }
    }

    /// Calculate latency percentiles (simplified implementation)
    fn calculate_latency_percentiles(&self) -> LatencyPercentiles {
        let mut all_latencies: Vec<Duration> = Vec::new();

        for metrics in self.operation_metrics.values() {
            if metrics.responses_received > 0 {
                let avg_latency = Duration::from_millis(
                    metrics.total_latency.as_millis() / metrics.responses_received as u128
                );
                all_latencies.push(avg_latency);
            }
        }

        all_latencies.sort();

        let len = all_latencies.len();
        if len == 0 {
            return LatencyPercentiles {
                p50: Duration::ZERO,
                p90: Duration::ZERO,
                p95: Duration::ZERO,
                p99: Duration::ZERO,
            };
        }

        LatencyPercentiles {
            p50: all_latencies[len * 50 / 100],
            p90: all_latencies[len * 90 / 100],
            p95: all_latencies[len * 95 / 100],
            p99: all_latencies[len * 99 / 100],
        }
    }

    /// Calculate performance score for an operation
    fn calculate_performance_score(metrics: &NetworkOperationMetrics, baseline: &NetworkBaseline) -> f32 {
        let mut score = 1.0;

        // Latency score
        if metrics.responses_received > 0 {
            let avg_latency = Duration::from_millis(metrics.total_latency.as_millis() / metrics.responses_received as u128);
            let latency_ratio = avg_latency.as_millis() as f32 / baseline.baseline_latency.as_millis() as f32;
            score *= (2.0 - latency_ratio).max(0.1); // Better latency = higher score
        }

        // Success rate score
        if metrics.requests_sent > 0 {
            let success_rate = metrics.responses_received as f32 / metrics.requests_sent as f32;
            score *= success_rate;
        }

        // Error penalty
        if metrics.errors > 0 || metrics.timeouts > 0 {
            let error_penalty = (metrics.errors + metrics.timeouts) as f32 * 0.1;
            score -= error_penalty;
        }

        score.max(0.0).min(1.0)
    }

    /// Generate network optimization recommendations
    fn generate_recommendations(
        &self,
        connection_metrics: &ConnectionMetrics,
        error_analysis: &ErrorAnalysis,
        bandwidth_mbps: f32,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Connection recommendations
        if connection_metrics.connection_success_rate < 95.0 {
            recommendations.push("Low connection success rate detected. Check network stability and retry logic.".to_string());
        }

        // Error rate recommendations
        if error_analysis.error_rate > 5.0 {
            recommendations.push("High error rate detected. Review error handling and network resilience.".to_string());
        }

        // Timeout recommendations
        if error_analysis.timeout_errors > error_analysis.total_errors / 2 {
            recommendations.push("High timeout rate detected. Consider increasing timeout values or improving network performance.".to_string());
        }

        // Bandwidth recommendations
        if bandwidth_mbps < self.baseline_stats.baseline_bandwidth_mbps * 0.5 {
            recommendations.push("Low bandwidth utilization detected. Consider optimizing data transfer or checking network capacity.".to_string());
        }

        // Latency recommendations
        let has_high_latency = self.operation_metrics.values()
            .any(|m| m.responses_received > 0 &&
                 Duration::from_millis(m.total_latency.as_millis() / m.responses_received as u128) >
                 Duration::from_millis(200));

        if has_high_latency {
            recommendations.push("High latency detected in some operations. Consider using connection pooling or CDN.".to_string());
        }

        recommendations
    }

    /// Reset all profiling data
    pub fn reset(&mut self) {
        self.operation_metrics.clear();
        self.global_metrics = NetworkGlobalMetrics::default();
        self.start_time = Instant::now();
    }

    /// Get real-time network statistics
    pub fn get_realtime_stats(&self) -> RealtimeNetworkStats {
        let runtime = self.start_time.elapsed();
        let total_bytes = self.global_metrics.total_bytes_sent.load(Ordering::Relaxed) +
                         self.global_metrics.total_bytes_received.load(Ordering::Relaxed);

        let current_bandwidth_mbps = if runtime.as_secs() > 0 {
            (total_bytes as f32 * 8.0) / (runtime.as_secs() as f32 * 1_000_000.0)
        } else {
            0.0
        };

        RealtimeNetworkStats {
            active_operations: self.operation_metrics.len(),
            total_operations: self.global_metrics.total_operations.load(Ordering::Relaxed),
            current_bandwidth_mbps,
            bytes_transferred: total_bytes,
            active_connections: self.operation_metrics.values()
                .map(|m| m.connection_attempts - m.connection_successes)
                .sum::<u32>() as u64,
        }
    }

    /// Check if network is bottleneck
    pub fn is_network_bottleneck(&self) -> bool {
        let error_rate = if self.global_metrics.total_requests.load(Ordering::Relaxed) > 0 {
            self.global_metrics.total_errors.load(Ordering::Relaxed) as f32 /
            self.global_metrics.total_requests.load(Ordering::Relaxed) as f32 * 100.0
        } else {
            0.0
        };

        error_rate > 10.0 || self.has_high_latency()
    }

    /// Check if operations have high latency
    fn has_high_latency(&self) -> bool {
        self.operation_metrics.values()
            .any(|m| m.responses_received > 0 &&
                 Duration::from_millis(m.total_latency.as_millis() / m.responses_received as u128) >
                 Duration::from_millis(500))
    }
}

/// Real-time network statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct RealtimeNetworkStats {
    pub active_operations: usize,
    pub total_operations: u64,
    pub current_bandwidth_mbps: f32,
    pub bytes_transferred: u64,
    pub active_connections: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_network_profiler_creation() {
        let profiler = NetworkProfiler::new();
        assert!(profiler.baseline_stats.baseline_latency > Duration::ZERO);
    }

    #[test]
    fn test_operation_profiling() {
        let mut profiler = NetworkProfiler::new();

        profiler.start_operation("test_op");
        profiler.record_data_sent("test_op", 1024);
        profiler.record_data_received("test_op", 2048);
        profiler.record_request("test_op");
        profiler.record_response("test_op", Duration::from_millis(100));
        let metrics = profiler.finish_operation("test_op");

        assert_eq!(metrics.bytes_sent, 1024);
        assert_eq!(metrics.bytes_received, 2048);
        assert_eq!(metrics.requests_sent, 1);
        assert_eq!(metrics.responses_received, 1);
        assert_eq!(metrics.total_latency, Duration::from_millis(100));
    }

    #[test]
    fn test_connection_tracking() {
        let mut profiler = NetworkProfiler::new();

        profiler.start_operation("conn_test");
        profiler.record_connection_attempt("conn_test", true);
        profiler.record_connection_attempt("conn_test", false);
        let metrics = profiler.finish_operation("conn_test");

        assert_eq!(metrics.connection_attempts, 2);
        assert_eq!(metrics.connection_successes, 1);
    }

    #[test]
    fn test_error_tracking() {
        let mut profiler = NetworkProfiler::new();

        profiler.start_operation("error_test");
        profiler.record_timeout("error_test");
        profiler.record_error("error_test");
        let metrics = profiler.finish_operation("error_test");

        assert_eq!(metrics.timeouts, 1);
        assert_eq!(metrics.errors, 1);
    }

    #[test]
    fn test_report_generation() {
        let mut profiler = NetworkProfiler::new();

        profiler.start_operation("test_op");
        profiler.record_data_sent("test_op", 1000);
        profiler.record_request("test_op");
        profiler.record_response("test_op", Duration::from_millis(50));
        thread::sleep(Duration::from_millis(10));
        profiler.finish_operation("test_op");

        let report = profiler.generate_report();
        assert_eq!(report.total_operations, 1);
        assert!(report.duration >= Duration::from_millis(10));
        assert_eq!(report.total_bytes_transferred, 1000);
        assert_eq!(report.success_rate, 100.0);
    }

    #[test]
    fn test_performance_score_calculation() {
        let metrics = NetworkOperationMetrics {
            start_time: Instant::now(),
            bytes_sent: 1000,
            bytes_received: 2000,
            requests_sent: 10,
            responses_received: 10,
            connection_attempts: 1,
            connection_successes: 1,
            total_latency: Duration::from_millis(500),
            min_latency: Duration::from_millis(50),
            max_latency: Duration::from_millis(100),
            timeouts: 0,
            errors: 0,
        };

        let baseline = NetworkBaseline {
            baseline_latency: Duration::from_millis(50),
            baseline_bandwidth_mbps: 100.0,
            baseline_packet_loss: 0.01,
        };

        let score = NetworkProfiler::calculate_performance_score(&metrics, &baseline);
        assert!(score >= 0.0 && score <= 1.0);
    }
}