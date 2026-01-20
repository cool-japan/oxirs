//! # Comprehensive Performance Metrics
//!
//! Advanced performance monitoring with SciRS2 statistical analysis for
//! distributed RDF cluster operations. Provides deep insights into consensus,
//! replication, query execution, and resource utilization.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;

use crate::raft::OxirsNodeId;

/// Performance metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetricsConfig {
    /// Enable detailed metrics collection
    pub enabled: bool,
    /// Sample retention window (seconds)
    pub retention_window_secs: u64,
    /// Histogram bucket count
    pub histogram_buckets: usize,
    /// Enable percentile calculations
    pub enable_percentiles: bool,
    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,
    /// Anomaly detection threshold (standard deviations)
    pub anomaly_threshold: f64,
}

impl Default for PerformanceMetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            retention_window_secs: 3600, // 1 hour
            histogram_buckets: 100,
            enable_percentiles: true,
            enable_anomaly_detection: true,
            anomaly_threshold: 3.0,
        }
    }
}

/// RDF operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RdfOperationType {
    /// Insert triple
    Insert,
    /// Delete triple
    Delete,
    /// Query triples
    Query,
    /// SPARQL query
    SparqlQuery,
    /// Transaction begin
    BeginTransaction,
    /// Transaction commit
    CommitTransaction,
    /// Transaction rollback
    RollbackTransaction,
}

/// Operation metrics sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationSample {
    /// Operation type
    pub operation_type: RdfOperationType,
    /// Duration in microseconds
    pub duration_micros: u64,
    /// Result size (bytes or row count)
    pub result_size: usize,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Success flag
    pub success: bool,
    /// Node ID
    pub node_id: OxirsNodeId,
}

/// Histogram for latency distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Histogram {
    /// Bucket boundaries (microseconds)
    pub buckets: Vec<u64>,
    /// Bucket counts
    pub counts: Vec<u64>,
    /// Total count
    pub total_count: u64,
    /// Min value
    pub min: u64,
    /// Max value
    pub max: u64,
}

impl Histogram {
    /// Create a new histogram with logarithmic buckets
    pub fn new(bucket_count: usize) -> Self {
        let mut buckets = Vec::with_capacity(bucket_count);

        // Logarithmic scale from 1us to 1 hour
        let min_log = 0.0; // log10(1)
        let max_log = 7.0; // log10(10_000_000) ~ 1 hour
        let step = (max_log - min_log) / bucket_count as f64;

        for i in 0..bucket_count {
            let log_value = min_log + (i as f64 * step);
            buckets.push(10f64.powf(log_value) as u64);
        }

        Self {
            buckets,
            counts: vec![0; bucket_count],
            total_count: 0,
            min: u64::MAX,
            max: 0,
        }
    }

    /// Record a value
    pub fn record(&mut self, value: u64) {
        self.total_count += 1;
        self.min = self.min.min(value);
        self.max = self.max.max(value);

        // Find the appropriate bucket
        for (i, &boundary) in self.buckets.iter().enumerate() {
            if value <= boundary {
                self.counts[i] += 1;
                return;
            }
        }

        // If value exceeds all buckets, add to last bucket
        if let Some(last) = self.counts.last_mut() {
            *last += 1;
        }
    }

    /// Calculate percentile
    pub fn percentile(&self, p: f64) -> u64 {
        if self.total_count == 0 {
            return 0;
        }

        let target_count = (self.total_count as f64 * p / 100.0) as u64;
        let mut cumulative = 0u64;

        for (i, &count) in self.counts.iter().enumerate() {
            cumulative += count;
            if cumulative >= target_count {
                return self.buckets[i];
            }
        }

        self.max
    }
}

/// Statistical summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSummary {
    /// Count
    pub count: u64,
    /// Mean
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum
    pub min: f64,
    /// Maximum
    pub max: f64,
    /// P50 (median)
    pub p50: f64,
    /// P95
    pub p95: f64,
    /// P99
    pub p99: f64,
    /// P99.9
    pub p999: f64,
}

/// Operation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationMetrics {
    /// Total operations
    pub total_count: u64,
    /// Successful operations
    pub success_count: u64,
    /// Failed operations
    pub failure_count: u64,
    /// Latency histogram
    pub latency_histogram: Histogram,
    /// Statistical summary
    pub stats: StatisticalSummary,
    /// Anomaly count
    pub anomaly_count: u64,
}

/// Consensus metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsensusMetrics {
    /// Total leader elections
    pub total_elections: u64,
    /// Successful elections
    pub successful_elections: u64,
    /// Average election duration (ms)
    pub avg_election_duration_ms: f64,
    /// Total log entries replicated
    pub total_log_entries: u64,
    /// Log replication rate (entries/sec)
    pub replication_rate: f64,
    /// Total Raft proposals
    pub total_proposals: u64,
    /// Successful proposals
    pub successful_proposals: u64,
    /// Average proposal latency (ms)
    pub avg_proposal_latency_ms: f64,
}

/// Network metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Average latency (ms)
    pub avg_latency_ms: f64,
    /// Packet loss rate
    pub packet_loss_rate: f64,
    /// Connection errors
    pub connection_errors: u64,
    /// Active connections
    pub active_connections: usize,
}

/// Query metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryMetrics {
    /// Total queries
    pub total_queries: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Average query duration (ms)
    pub avg_query_duration_ms: f64,
    /// Average result size
    pub avg_result_size: f64,
    /// Slow queries (>1s)
    pub slow_query_count: u64,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// CPU usage (0.0-1.0)
    pub cpu_usage: f64,
    /// Memory usage (bytes)
    pub memory_usage: u64,
    /// Memory usage percentage
    pub memory_usage_percent: f64,
    /// Disk I/O read (bytes/sec)
    pub disk_read_bps: u64,
    /// Disk I/O write (bytes/sec)
    pub disk_write_bps: u64,
    /// Network I/O read (bytes/sec)
    pub network_read_bps: u64,
    /// Network I/O write (bytes/sec)
    pub network_write_bps: u64,
}

/// Comprehensive performance metrics
pub struct PerformanceMetrics {
    config: PerformanceMetricsConfig,
    /// Operation samples by type
    operation_samples: Arc<RwLock<BTreeMap<RdfOperationType, VecDeque<OperationSample>>>>,
    /// Consensus metrics
    consensus_metrics: Arc<RwLock<ConsensusMetrics>>,
    /// Network metrics
    network_metrics: Arc<RwLock<NetworkMetrics>>,
    /// Query metrics
    query_metrics: Arc<RwLock<QueryMetrics>>,
    /// Resource metrics history
    resource_metrics: Arc<RwLock<VecDeque<(SystemTime, ResourceMetrics)>>>,
    /// Anomalies detected
    anomalies: Arc<RwLock<Vec<AnomalyReport>>>,
}

/// Anomaly report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyReport {
    /// Anomaly type
    pub anomaly_type: String,
    /// Metric name
    pub metric_name: String,
    /// Expected value
    pub expected_value: f64,
    /// Actual value
    pub actual_value: f64,
    /// Deviation (standard deviations)
    pub deviation: f64,
    /// Timestamp
    pub timestamp: SystemTime,
}

impl PerformanceMetrics {
    /// Create a new performance metrics collector
    pub fn new(config: PerformanceMetricsConfig) -> Self {
        Self {
            config,
            operation_samples: Arc::new(RwLock::new(BTreeMap::new())),
            consensus_metrics: Arc::new(RwLock::new(ConsensusMetrics::default())),
            network_metrics: Arc::new(RwLock::new(NetworkMetrics::default())),
            query_metrics: Arc::new(RwLock::new(QueryMetrics::default())),
            resource_metrics: Arc::new(RwLock::new(VecDeque::new())),
            anomalies: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Record an operation
    pub async fn record_operation(&self, sample: OperationSample) {
        if !self.config.enabled {
            return;
        }

        {
            let mut samples = self.operation_samples.write().await;
            let queue = samples
                .entry(sample.operation_type)
                .or_insert_with(VecDeque::new);

            queue.push_back(sample.clone());

            // Cleanup old samples
            let cutoff = SystemTime::now()
                .checked_sub(Duration::from_secs(self.config.retention_window_secs))
                .unwrap_or(SystemTime::UNIX_EPOCH);

            while let Some(first) = queue.front() {
                if first.timestamp < cutoff {
                    queue.pop_front();
                } else {
                    break;
                }
            }
        } // Lock is dropped here

        // Anomaly detection (now safe - no lock held)
        if self.config.enable_anomaly_detection {
            self.detect_anomaly(&sample).await;
        }
    }

    /// Detect anomalies in operation samples
    async fn detect_anomaly(&self, sample: &OperationSample) {
        let samples = self.operation_samples.read().await;

        if let Some(queue) = samples.get(&sample.operation_type) {
            if queue.len() < 30 {
                return; // Need enough samples
            }

            // Calculate mean and std dev
            let durations: Vec<f64> = queue.iter().map(|s| s.duration_micros as f64).collect();

            let mean = durations.iter().sum::<f64>() / durations.len() as f64;
            let variance =
                durations.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / durations.len() as f64;
            let std_dev = variance.sqrt();

            let deviation = ((sample.duration_micros as f64 - mean) / std_dev).abs();

            if deviation > self.config.anomaly_threshold {
                let mut anomalies = self.anomalies.write().await;
                anomalies.push(AnomalyReport {
                    anomaly_type: "OperationLatency".to_string(),
                    metric_name: format!("{:?}", sample.operation_type),
                    expected_value: mean,
                    actual_value: sample.duration_micros as f64,
                    deviation,
                    timestamp: sample.timestamp,
                });

                // Keep only last 1000 anomalies
                if anomalies.len() > 1000 {
                    anomalies.remove(0);
                }
            }
        }
    }

    /// Get operation metrics
    pub async fn get_operation_metrics(
        &self,
        operation_type: RdfOperationType,
    ) -> Option<OperationMetrics> {
        let samples = self.operation_samples.read().await;
        let queue = samples.get(&operation_type)?;

        if queue.is_empty() {
            return None;
        }

        let total_count = queue.len() as u64;
        let success_count = queue.iter().filter(|s| s.success).count() as u64;
        let failure_count = total_count - success_count;

        // Build histogram
        let mut histogram = Histogram::new(self.config.histogram_buckets);
        let mut durations = Vec::new();

        for sample in queue.iter() {
            histogram.record(sample.duration_micros);
            durations.push(sample.duration_micros as f64);
        }

        // Calculate statistics
        let mean = durations.iter().sum::<f64>() / durations.len() as f64;
        let variance =
            durations.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / durations.len() as f64;
        let std_dev = variance.sqrt();

        let min = durations.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = durations.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let stats = StatisticalSummary {
            count: total_count,
            mean,
            std_dev,
            min,
            max,
            p50: histogram.percentile(50.0) as f64,
            p95: histogram.percentile(95.0) as f64,
            p99: histogram.percentile(99.0) as f64,
            p999: histogram.percentile(99.9) as f64,
        };

        let anomalies = self.anomalies.read().await;
        let anomaly_count = anomalies
            .iter()
            .filter(|a| a.metric_name == format!("{:?}", operation_type))
            .count() as u64;

        Some(OperationMetrics {
            total_count,
            success_count,
            failure_count,
            latency_histogram: histogram,
            stats,
            anomaly_count,
        })
    }

    /// Update consensus metrics
    pub async fn update_consensus_metrics<F>(&self, updater: F)
    where
        F: FnOnce(&mut ConsensusMetrics),
    {
        if !self.config.enabled {
            return;
        }

        let mut metrics = self.consensus_metrics.write().await;
        updater(&mut metrics);
    }

    /// Get consensus metrics
    pub async fn get_consensus_metrics(&self) -> ConsensusMetrics {
        self.consensus_metrics.read().await.clone()
    }

    /// Update network metrics
    pub async fn update_network_metrics<F>(&self, updater: F)
    where
        F: FnOnce(&mut NetworkMetrics),
    {
        if !self.config.enabled {
            return;
        }

        let mut metrics = self.network_metrics.write().await;
        updater(&mut metrics);
    }

    /// Get network metrics
    pub async fn get_network_metrics(&self) -> NetworkMetrics {
        self.network_metrics.read().await.clone()
    }

    /// Update query metrics
    pub async fn update_query_metrics<F>(&self, updater: F)
    where
        F: FnOnce(&mut QueryMetrics),
    {
        if !self.config.enabled {
            return;
        }

        let mut metrics = self.query_metrics.write().await;
        updater(&mut metrics);
    }

    /// Get query metrics
    pub async fn get_query_metrics(&self) -> QueryMetrics {
        self.query_metrics.read().await.clone()
    }

    /// Record resource metrics
    pub async fn record_resource_metrics(&self, metrics: ResourceMetrics) {
        if !self.config.enabled {
            return;
        }

        let mut resource_metrics = self.resource_metrics.write().await;
        resource_metrics.push_back((SystemTime::now(), metrics));

        // Cleanup old samples
        let cutoff = SystemTime::now()
            .checked_sub(Duration::from_secs(self.config.retention_window_secs))
            .unwrap_or(SystemTime::UNIX_EPOCH);

        while let Some((timestamp, _)) = resource_metrics.front() {
            if *timestamp < cutoff {
                resource_metrics.pop_front();
            } else {
                break;
            }
        }
    }

    /// Get latest resource metrics
    pub async fn get_latest_resource_metrics(&self) -> Option<ResourceMetrics> {
        let metrics = self.resource_metrics.read().await;
        metrics.back().map(|(_, m)| m.clone())
    }

    /// Get resource metrics history
    pub async fn get_resource_metrics_history(&self) -> Vec<(SystemTime, ResourceMetrics)> {
        self.resource_metrics.read().await.iter().cloned().collect()
    }

    /// Get anomalies
    pub async fn get_anomalies(&self) -> Vec<AnomalyReport> {
        self.anomalies.read().await.clone()
    }

    /// Clear all metrics
    pub async fn clear(&self) {
        self.operation_samples.write().await.clear();
        *self.consensus_metrics.write().await = ConsensusMetrics::default();
        *self.network_metrics.write().await = NetworkMetrics::default();
        *self.query_metrics.write().await = QueryMetrics::default();
        self.resource_metrics.write().await.clear();
        self.anomalies.write().await.clear();
    }
}

/// Helper function to measure operation duration
pub struct OperationTimer {
    start: Instant,
    operation_type: RdfOperationType,
    node_id: OxirsNodeId,
    metrics: Arc<PerformanceMetrics>,
}

impl OperationTimer {
    /// Start a new operation timer
    pub fn start(
        operation_type: RdfOperationType,
        node_id: OxirsNodeId,
        metrics: Arc<PerformanceMetrics>,
    ) -> Self {
        Self {
            start: Instant::now(),
            operation_type,
            node_id,
            metrics,
        }
    }

    /// Finish the operation and record metrics
    pub async fn finish(self, success: bool, result_size: usize) {
        let duration_micros = self.start.elapsed().as_micros() as u64;

        let sample = OperationSample {
            operation_type: self.operation_type,
            duration_micros,
            result_size,
            timestamp: SystemTime::now(),
            success,
            node_id: self.node_id,
        };

        self.metrics.record_operation(sample).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_creation() {
        let histogram = Histogram::new(10);
        assert_eq!(histogram.buckets.len(), 10);
        assert_eq!(histogram.counts.len(), 10);
        assert_eq!(histogram.total_count, 0);
    }

    #[test]
    fn test_histogram_record() {
        let mut histogram = Histogram::new(10);
        histogram.record(100);
        histogram.record(200);
        histogram.record(150);

        assert_eq!(histogram.total_count, 3);
        assert_eq!(histogram.min, 100);
        assert_eq!(histogram.max, 200);
    }

    #[tokio::test]
    async fn test_performance_metrics_creation() {
        let config = PerformanceMetricsConfig::default();
        let metrics = PerformanceMetrics::new(config);

        let consensus = metrics.get_consensus_metrics().await;
        assert_eq!(consensus.total_elections, 0);
    }

    #[tokio::test]
    async fn test_record_operation() {
        let config = PerformanceMetricsConfig::default();
        let metrics = PerformanceMetrics::new(config);

        let sample = OperationSample {
            operation_type: RdfOperationType::Insert,
            duration_micros: 1000,
            result_size: 100,
            timestamp: SystemTime::now(),
            success: true,
            node_id: 1,
        };

        metrics.record_operation(sample).await;

        let op_metrics = metrics
            .get_operation_metrics(RdfOperationType::Insert)
            .await;
        assert!(op_metrics.is_some());

        let op_metrics = op_metrics.unwrap();
        assert_eq!(op_metrics.total_count, 1);
        assert_eq!(op_metrics.success_count, 1);
    }

    #[tokio::test]
    async fn test_consensus_metrics_update() {
        let config = PerformanceMetricsConfig::default();
        let metrics = PerformanceMetrics::new(config);

        metrics
            .update_consensus_metrics(|m| {
                m.total_elections = 5;
                m.successful_elections = 4;
            })
            .await;

        let consensus = metrics.get_consensus_metrics().await;
        assert_eq!(consensus.total_elections, 5);
        assert_eq!(consensus.successful_elections, 4);
    }

    #[tokio::test]
    async fn test_network_metrics() {
        let config = PerformanceMetricsConfig::default();
        let metrics = PerformanceMetrics::new(config);

        metrics
            .update_network_metrics(|m| {
                m.bytes_sent = 1000;
                m.bytes_received = 2000;
                m.active_connections = 5;
            })
            .await;

        let network = metrics.get_network_metrics().await;
        assert_eq!(network.bytes_sent, 1000);
        assert_eq!(network.bytes_received, 2000);
        assert_eq!(network.active_connections, 5);
    }

    #[tokio::test]
    async fn test_query_metrics() {
        let config = PerformanceMetricsConfig::default();
        let metrics = PerformanceMetrics::new(config);

        metrics
            .update_query_metrics(|m| {
                m.total_queries = 100;
                m.cache_hits = 80;
                m.cache_misses = 20;
            })
            .await;

        let query = metrics.get_query_metrics().await;
        assert_eq!(query.total_queries, 100);
        assert_eq!(query.cache_hits, 80);
        assert_eq!(query.cache_misses, 20);
    }

    #[tokio::test]
    async fn test_resource_metrics() {
        let config = PerformanceMetricsConfig::default();
        let metrics = PerformanceMetrics::new(config);

        let resource = ResourceMetrics {
            cpu_usage: 0.5,
            memory_usage: 1024 * 1024 * 1024, // 1GB
            memory_usage_percent: 50.0,
            disk_read_bps: 1000,
            disk_write_bps: 2000,
            network_read_bps: 500,
            network_write_bps: 1500,
        };

        metrics.record_resource_metrics(resource.clone()).await;

        let latest = metrics.get_latest_resource_metrics().await;
        assert!(latest.is_some());

        let latest = latest.unwrap();
        assert_eq!(latest.cpu_usage, 0.5);
        assert_eq!(latest.memory_usage, 1024 * 1024 * 1024);
    }

    #[tokio::test]
    async fn test_clear_metrics() {
        let config = PerformanceMetricsConfig::default();
        let metrics = PerformanceMetrics::new(config);

        // Add some data
        metrics
            .update_consensus_metrics(|m| m.total_elections = 5)
            .await;
        metrics
            .update_network_metrics(|m| m.bytes_sent = 1000)
            .await;

        // Clear
        metrics.clear().await;

        let consensus = metrics.get_consensus_metrics().await;
        let network = metrics.get_network_metrics().await;

        assert_eq!(consensus.total_elections, 0);
        assert_eq!(network.bytes_sent, 0);
    }

    #[tokio::test]
    async fn test_operation_timer() {
        let config = PerformanceMetricsConfig::default();
        let metrics = Arc::new(PerformanceMetrics::new(config));

        let timer = OperationTimer::start(RdfOperationType::Query, 1, Arc::clone(&metrics));

        // Simulate some work
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        timer.finish(true, 42).await;

        let op_metrics = metrics.get_operation_metrics(RdfOperationType::Query).await;
        assert!(op_metrics.is_some());

        let op_metrics = op_metrics.unwrap();
        assert_eq!(op_metrics.total_count, 1);
        assert_eq!(op_metrics.success_count, 1);
    }
}
