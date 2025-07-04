//! Performance profiler for neural cost estimation

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::{config::*, types::*};
use crate::{Result, ShaclAiError};

/// Performance profiler
#[derive(Debug)]
pub struct PerformanceProfiler {
    /// Configuration
    config: PerformanceProfilingConfig,
    
    /// Performance metrics
    metrics: PerformanceMetrics,
    
    /// Resource usage history
    resource_history: Vec<ResourceSnapshot>,
    
    /// Timing measurements
    timing_measurements: HashMap<String, TimingStats>,
    
    /// Cache analysis data
    cache_analysis: CacheAnalysisData,
    
    /// I/O analysis data
    io_analysis: IoAnalysisData,
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_operations: usize,
    pub average_execution_time: Duration,
    pub peak_memory_usage: usize,
    pub average_cpu_usage: f64,
    pub cache_hit_ratio: f64,
    pub throughput: f64, // Operations per second
}

/// Resource snapshot
#[derive(Debug, Clone)]
pub struct ResourceSnapshot {
    pub timestamp: Instant,
    pub cpu_usage: f64,
    pub memory_usage: usize,
    pub disk_usage: usize,
    pub network_usage: usize,
    pub operation_type: String,
}

/// Timing statistics
#[derive(Debug, Clone)]
pub struct TimingStats {
    pub operation_name: String,
    pub total_calls: usize,
    pub total_time: Duration,
    pub average_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub percentile_95: Duration,
}

/// Cache analysis data
#[derive(Debug, Clone)]
pub struct CacheAnalysisData {
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub cache_evictions: usize,
    pub cache_utilization: f64,
    pub hit_patterns: HashMap<String, usize>,
}

/// I/O analysis data
#[derive(Debug, Clone)]
pub struct IoAnalysisData {
    pub disk_reads: usize,
    pub disk_writes: usize,
    pub network_requests: usize,
    pub total_bytes_read: usize,
    pub total_bytes_written: usize,
    pub average_latency: Duration,
}

impl PerformanceProfiler {
    pub fn new(config: PerformanceProfilingConfig) -> Self {
        Self {
            config,
            metrics: PerformanceMetrics::default(),
            resource_history: Vec::new(),
            timing_measurements: HashMap::new(),
            cache_analysis: CacheAnalysisData::default(),
            io_analysis: IoAnalysisData::default(),
        }
    }

    /// Start profiling an operation
    pub fn start_operation(&mut self, operation_name: &str) -> OperationProfiler {
        OperationProfiler::new(operation_name.to_string(), self.config.granularity.clone())
    }

    /// Record completed operation
    pub fn record_operation(&mut self, operation_profiler: OperationProfiler) -> Result<()> {
        let stats = operation_profiler.finish()?;
        
        // Update timing measurements
        let timing_entry = self.timing_measurements
            .entry(stats.operation_name.clone())
            .or_insert_with(|| TimingStats::new(stats.operation_name.clone()));
        
        timing_entry.add_measurement(stats.execution_time);
        
        // Update overall metrics
        self.update_overall_metrics(&stats);
        
        // Record resource snapshot if enabled
        if self.config.monitor_resources {
            self.record_resource_snapshot(&stats.operation_name)?;
        }
        
        Ok(())
    }

    /// Record cache operation
    pub fn record_cache_operation(&mut self, operation_type: CacheOperation, key: &str) {
        if !self.config.analyze_cache {
            return;
        }

        match operation_type {
            CacheOperation::Hit => {
                self.cache_analysis.cache_hits += 1;
                *self.cache_analysis.hit_patterns.entry(key.to_string()).or_insert(0) += 1;
            },
            CacheOperation::Miss => {
                self.cache_analysis.cache_misses += 1;
            },
            CacheOperation::Eviction => {
                self.cache_analysis.cache_evictions += 1;
            },
        }
        
        // Update cache utilization
        let total_operations = self.cache_analysis.cache_hits + self.cache_analysis.cache_misses;
        if total_operations > 0 {
            self.cache_analysis.hit_patterns.get(key);
        }
    }

    /// Record I/O operation
    pub fn record_io_operation(&mut self, operation: IoOperation) {
        if !self.config.analyze_io {
            return;
        }

        match operation.operation_type {
            IoOperationType::DiskRead => {
                self.io_analysis.disk_reads += 1;
                self.io_analysis.total_bytes_read += operation.bytes;
            },
            IoOperationType::DiskWrite => {
                self.io_analysis.disk_writes += 1;
                self.io_analysis.total_bytes_written += operation.bytes;
            },
            IoOperationType::NetworkRequest => {
                self.io_analysis.network_requests += 1;
            },
        }

        // Update average latency
        let total_ops = self.io_analysis.disk_reads + self.io_analysis.disk_writes + self.io_analysis.network_requests;
        if total_ops > 0 {
            self.io_analysis.average_latency = Duration::from_nanos(
                (self.io_analysis.average_latency.as_nanos() as u64 * (total_ops - 1) as u64 + 
                 operation.latency.as_nanos() as u64) / total_ops as u64
            );
        }
    }

    fn update_overall_metrics(&mut self, stats: &OperationStats) {
        self.metrics.total_operations += 1;
        
        // Update average execution time
        let n = self.metrics.total_operations as u64;
        let current_avg_nanos = self.metrics.average_execution_time.as_nanos() as u64;
        let new_avg_nanos = (current_avg_nanos * (n - 1) + stats.execution_time.as_nanos() as u64) / n;
        self.metrics.average_execution_time = Duration::from_nanos(new_avg_nanos);
        
        // Update peak memory usage
        self.metrics.peak_memory_usage = self.metrics.peak_memory_usage.max(stats.peak_memory_usage);
        
        // Update throughput (operations per second)
        if !self.resource_history.is_empty() {
            let time_span = self.resource_history.last().unwrap().timestamp
                .duration_since(self.resource_history.first().unwrap().timestamp);
            self.metrics.throughput = self.metrics.total_operations as f64 / time_span.as_secs_f64();
        }
    }

    fn record_resource_snapshot(&mut self, operation_type: &str) -> Result<()> {
        let snapshot = ResourceSnapshot {
            timestamp: Instant::now(),
            cpu_usage: self.get_current_cpu_usage(),
            memory_usage: self.get_current_memory_usage(),
            disk_usage: self.get_current_disk_usage(),
            network_usage: self.get_current_network_usage(),
            operation_type: operation_type.to_string(),
        };

        self.resource_history.push(snapshot);

        // Keep only recent snapshots
        if self.resource_history.len() > 10000 {
            self.resource_history.drain(0..1000);
        }

        Ok(())
    }

    fn get_current_cpu_usage(&self) -> f64 {
        // Simplified CPU usage (in practice, would use system APIs)
        0.3 + 0.4 * fastrand::f64()
    }

    fn get_current_memory_usage(&self) -> usize {
        // Simplified memory usage (in practice, would use system APIs)
        (500 + (200.0 * fastrand::f64()) as usize) * 1024 * 1024 // 500-700 MB
    }

    fn get_current_disk_usage(&self) -> usize {
        // Simplified disk usage
        (10 + (50.0 * fastrand::f64()) as usize) * 1024 * 1024 // 10-60 MB
    }

    fn get_current_network_usage(&self) -> usize {
        // Simplified network usage
        (1 + (10.0 * fastrand::f64()) as usize) * 1024 // 1-11 KB
    }

    /// Get performance summary
    pub fn get_performance_summary(&self) -> PerformanceSummary {
        PerformanceSummary {
            overall_metrics: self.metrics.clone(),
            top_slow_operations: self.get_slowest_operations(5),
            resource_utilization: self.calculate_resource_utilization(),
            cache_performance: self.cache_analysis.clone(),
            io_performance: self.io_analysis.clone(),
        }
    }

    fn get_slowest_operations(&self, count: usize) -> Vec<TimingStats> {
        let mut operations: Vec<_> = self.timing_measurements.values().cloned().collect();
        operations.sort_by(|a, b| b.average_time.cmp(&a.average_time));
        operations.into_iter().take(count).collect()
    }

    fn calculate_resource_utilization(&self) -> ResourceUtilization {
        if self.resource_history.is_empty() {
            return ResourceUtilization::default();
        }

        let avg_cpu = self.resource_history.iter()
            .map(|s| s.cpu_usage)
            .sum::<f64>() / self.resource_history.len() as f64;

        let avg_memory = self.resource_history.iter()
            .map(|s| s.memory_usage)
            .sum::<usize>() / self.resource_history.len();

        ResourceUtilization {
            average_cpu_usage: avg_cpu,
            average_memory_usage: avg_memory,
            peak_cpu_usage: self.resource_history.iter()
                .map(|s| s.cpu_usage)
                .fold(0.0, f64::max),
            peak_memory_usage: self.resource_history.iter()
                .map(|s| s.memory_usage)
                .max()
                .unwrap_or(0),
        }
    }

    /// Reset profiler statistics
    pub fn reset(&mut self) {
        self.metrics = PerformanceMetrics::default();
        self.resource_history.clear();
        self.timing_measurements.clear();
        self.cache_analysis = CacheAnalysisData::default();
        self.io_analysis = IoAnalysisData::default();
    }

    /// Get configuration
    pub fn get_config(&self) -> &PerformanceProfilingConfig {
        &self.config
    }
}

/// Operation profiler for individual operations
#[derive(Debug)]
pub struct OperationProfiler {
    operation_name: String,
    start_time: Instant,
    start_memory: usize,
    granularity: ProfilingGranularity,
}

/// Operation statistics
#[derive(Debug, Clone)]
pub struct OperationStats {
    pub operation_name: String,
    pub execution_time: Duration,
    pub peak_memory_usage: usize,
    pub cpu_usage: f64,
}

/// Cache operation types
#[derive(Debug, Clone)]
pub enum CacheOperation {
    Hit,
    Miss,
    Eviction,
}

/// I/O operation
#[derive(Debug, Clone)]
pub struct IoOperation {
    pub operation_type: IoOperationType,
    pub bytes: usize,
    pub latency: Duration,
}

/// I/O operation types
#[derive(Debug, Clone)]
pub enum IoOperationType {
    DiskRead,
    DiskWrite,
    NetworkRequest,
}

/// Performance summary
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    pub overall_metrics: PerformanceMetrics,
    pub top_slow_operations: Vec<TimingStats>,
    pub resource_utilization: ResourceUtilization,
    pub cache_performance: CacheAnalysisData,
    pub io_performance: IoAnalysisData,
}

/// Resource utilization
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    pub average_cpu_usage: f64,
    pub average_memory_usage: usize,
    pub peak_cpu_usage: f64,
    pub peak_memory_usage: usize,
}

impl OperationProfiler {
    pub fn new(operation_name: String, granularity: ProfilingGranularity) -> Self {
        Self {
            operation_name,
            start_time: Instant::now(),
            start_memory: Self::get_current_memory_usage(),
            granularity,
        }
    }

    pub fn finish(self) -> Result<OperationStats> {
        let execution_time = self.start_time.elapsed();
        let current_memory = Self::get_current_memory_usage();
        let peak_memory_usage = current_memory.max(self.start_memory);

        Ok(OperationStats {
            operation_name: self.operation_name,
            execution_time,
            peak_memory_usage,
            cpu_usage: 0.5, // Simplified
        })
    }

    fn get_current_memory_usage() -> usize {
        // Simplified memory usage measurement
        500 * 1024 * 1024 // 500 MB
    }
}

impl TimingStats {
    pub fn new(operation_name: String) -> Self {
        Self {
            operation_name,
            total_calls: 0,
            total_time: Duration::from_nanos(0),
            average_time: Duration::from_nanos(0),
            min_time: Duration::from_secs(u64::MAX),
            max_time: Duration::from_nanos(0),
            percentile_95: Duration::from_nanos(0),
        }
    }

    pub fn add_measurement(&mut self, duration: Duration) {
        self.total_calls += 1;
        self.total_time += duration;
        self.average_time = self.total_time / self.total_calls as u32;
        self.min_time = self.min_time.min(duration);
        self.max_time = self.max_time.max(duration);
        
        // Simplified 95th percentile calculation
        self.percentile_95 = Duration::from_nanos(
            (self.max_time.as_nanos() as f64 * 0.95) as u64
        );
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_operations: 0,
            average_execution_time: Duration::from_nanos(0),
            peak_memory_usage: 0,
            average_cpu_usage: 0.0,
            cache_hit_ratio: 0.0,
            throughput: 0.0,
        }
    }
}

impl Default for CacheAnalysisData {
    fn default() -> Self {
        Self {
            cache_hits: 0,
            cache_misses: 0,
            cache_evictions: 0,
            cache_utilization: 0.0,
            hit_patterns: HashMap::new(),
        }
    }
}

impl Default for IoAnalysisData {
    fn default() -> Self {
        Self {
            disk_reads: 0,
            disk_writes: 0,
            network_requests: 0,
            total_bytes_read: 0,
            total_bytes_written: 0,
            average_latency: Duration::from_nanos(0),
        }
    }
}

impl Default for ResourceUtilization {
    fn default() -> Self {
        Self {
            average_cpu_usage: 0.0,
            average_memory_usage: 0,
            peak_cpu_usage: 0.0,
            peak_memory_usage: 0,
        }
    }
}