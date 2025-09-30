//! CPU Performance Profiler
//!
//! Monitors CPU utilization, thread performance, and SIMD operations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, Ordering};
use sysinfo::{CpuExt, System, SystemExt};

/// CPU profiler for monitoring performance and utilization
#[derive(Debug)]
pub struct CpuProfiler {
    system: System,
    operation_metrics: HashMap<String, CpuOperationMetrics>,
    global_metrics: CpuGlobalMetrics,
    start_time: Instant,
}

/// Per-operation CPU metrics
#[derive(Debug, Clone)]
pub struct CpuOperationMetrics {
    pub start_time: Instant,
    pub total_duration: Duration,
    pub average_utilization: f32,
    pub peak_utilization: f32,
    pub thread_count: u32,
    pub simd_operations: u64,
    pub cache_misses: u64,
}

/// Global CPU metrics
#[derive(Debug, Default)]
pub struct CpuGlobalMetrics {
    pub total_operations: AtomicU64,
    pub total_cpu_time: Duration,
    pub average_utilization: f32,
    pub peak_utilization: f32,
    pub core_count: u32,
    pub simd_capability: SimdCapability,
}

/// SIMD capability detection
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct SimdCapability {
    pub avx2_available: bool,
    pub avx512_available: bool,
    pub neon_available: bool,
    pub sse_available: bool,
}

/// CPU performance report
#[derive(Debug, Serialize, Deserialize)]
pub struct CpuReport {
    pub duration: Duration,
    pub average_utilization: f32,
    pub peak_utilization: f32,
    pub core_count: u32,
    pub total_operations: u64,
    pub simd_capability: SimdCapability,
    pub per_core_utilization: Vec<f32>,
    pub operation_breakdown: HashMap<String, CpuOperationSummary>,
}

/// Summary of CPU metrics for an operation
#[derive(Debug, Serialize, Deserialize)]
pub struct CpuOperationSummary {
    pub total_duration: Duration,
    pub average_utilization: f32,
    pub peak_utilization: f32,
    pub simd_operations: u64,
    pub efficiency_score: f32,
}

impl CpuProfiler {
    /// Create a new CPU profiler
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_cpu();

        let simd_capability = Self::detect_simd_capability();
        let core_count = system.cpus().len() as u32;

        Self {
            system,
            operation_metrics: HashMap::new(),
            global_metrics: CpuGlobalMetrics {
                core_count,
                simd_capability,
                ..Default::default()
            },
            start_time: Instant::now(),
        }
    }

    /// Detect available SIMD capabilities
    fn detect_simd_capability() -> SimdCapability {
        SimdCapability {
            avx2_available: cfg!(target_feature = "avx2") || std::arch::is_x86_feature_detected!("avx2"),
            avx512_available: cfg!(target_feature = "avx512f") || std::arch::is_x86_feature_detected!("avx512f"),
            neon_available: cfg!(target_arch = "aarch64") && std::arch::is_aarch64_feature_detected!("neon"),
            sse_available: cfg!(target_feature = "sse2") || std::arch::is_x86_feature_detected!("sse2"),
        }
    }

    /// Start monitoring an operation
    pub fn start_operation(&mut self, operation_name: &str) {
        self.system.refresh_cpu();

        let current_utilization = self.get_current_cpu_utilization();

        let metrics = CpuOperationMetrics {
            start_time: Instant::now(),
            total_duration: Duration::ZERO,
            average_utilization: current_utilization,
            peak_utilization: current_utilization,
            thread_count: self.get_current_thread_count(),
            simd_operations: 0,
            cache_misses: 0,
        };

        self.operation_metrics.insert(operation_name.to_string(), metrics);
    }

    /// Finish monitoring an operation
    pub fn finish_operation(&mut self, operation_name: &str) -> CpuOperationMetrics {
        if let Some(mut metrics) = self.operation_metrics.remove(operation_name) {
            self.system.refresh_cpu();

            let current_utilization = self.get_current_cpu_utilization();
            metrics.total_duration = metrics.start_time.elapsed();
            metrics.peak_utilization = metrics.peak_utilization.max(current_utilization);
            metrics.average_utilization = (metrics.average_utilization + current_utilization) / 2.0;

            // Update global metrics
            self.global_metrics.total_operations.fetch_add(1, Ordering::Relaxed);
            self.global_metrics.total_cpu_time += metrics.total_duration;
            self.global_metrics.peak_utilization = self.global_metrics.peak_utilization.max(metrics.peak_utilization);

            metrics
        } else {
            // Return default metrics if operation not found
            CpuOperationMetrics {
                start_time: Instant::now(),
                total_duration: Duration::ZERO,
                average_utilization: 0.0,
                peak_utilization: 0.0,
                thread_count: 1,
                simd_operations: 0,
                cache_misses: 0,
            }
        }
    }

    /// Get current CPU utilization
    fn get_current_cpu_utilization(&self) -> f32 {
        let total_usage: f32 = self.system.cpus().iter().map(|cpu| cpu.cpu_usage()).sum();
        total_usage / self.system.cpus().len() as f32
    }

    /// Get current thread count (approximation)
    fn get_current_thread_count(&self) -> u32 {
        // This is a simplification - in practice you'd use more sophisticated methods
        rayon::current_num_threads() as u32
    }

    /// Record SIMD operation
    pub fn record_simd_operation(&mut self, operation_name: &str, count: u64) {
        if let Some(metrics) = self.operation_metrics.get_mut(operation_name) {
            metrics.simd_operations += count;
        }
    }

    /// Record cache miss
    pub fn record_cache_miss(&mut self, operation_name: &str, count: u64) {
        if let Some(metrics) = self.operation_metrics.get_mut(operation_name) {
            metrics.cache_misses += count;
        }
    }

    /// Generate comprehensive CPU report
    pub fn generate_report(&mut self) -> CpuReport {
        self.system.refresh_cpu();

        let per_core_utilization: Vec<f32> = self.system.cpus().iter()
            .map(|cpu| cpu.cpu_usage())
            .collect();

        let current_utilization = self.get_current_cpu_utilization();
        let total_operations = self.global_metrics.total_operations.load(Ordering::Relaxed);

        // Calculate average utilization over the entire profiling period
        let runtime = self.start_time.elapsed();
        let average_utilization = if total_operations > 0 {
            self.global_metrics.average_utilization
        } else {
            current_utilization
        };

        // Generate operation summaries
        let operation_breakdown: HashMap<String, CpuOperationSummary> = self.operation_metrics.iter()
            .map(|(name, metrics)| {
                let efficiency_score = Self::calculate_efficiency_score(metrics);
                let summary = CpuOperationSummary {
                    total_duration: metrics.total_duration,
                    average_utilization: metrics.average_utilization,
                    peak_utilization: metrics.peak_utilization,
                    simd_operations: metrics.simd_operations,
                    efficiency_score,
                };
                (name.clone(), summary)
            })
            .collect();

        CpuReport {
            duration: runtime,
            average_utilization,
            peak_utilization: self.global_metrics.peak_utilization,
            core_count: self.global_metrics.core_count,
            total_operations,
            simd_capability: self.global_metrics.simd_capability.clone(),
            per_core_utilization,
            operation_breakdown,
        }
    }

    /// Calculate efficiency score for an operation
    fn calculate_efficiency_score(metrics: &CpuOperationMetrics) -> f32 {
        let base_score = metrics.average_utilization / 100.0;
        let simd_bonus = if metrics.simd_operations > 0 { 0.1 } else { 0.0 };
        let cache_penalty = (metrics.cache_misses as f32 / 1000.0).min(0.2);

        (base_score + simd_bonus - cache_penalty).max(0.0).min(1.0)
    }

    /// Reset all profiling data
    pub fn reset(&mut self) {
        self.operation_metrics.clear();
        self.global_metrics = CpuGlobalMetrics {
            core_count: self.global_metrics.core_count,
            simd_capability: self.global_metrics.simd_capability.clone(),
            ..Default::default()
        };
        self.start_time = Instant::now();
    }

    /// Get real-time CPU statistics
    pub fn get_realtime_stats(&mut self) -> RealtimeCpuStats {
        self.system.refresh_cpu();

        let per_core_usage: Vec<f32> = self.system.cpus().iter()
            .map(|cpu| cpu.cpu_usage())
            .collect();

        let average_usage = per_core_usage.iter().sum::<f32>() / per_core_usage.len() as f32;

        RealtimeCpuStats {
            average_utilization: average_usage,
            per_core_utilization: per_core_usage,
            active_operations: self.operation_metrics.len(),
            total_operations: self.global_metrics.total_operations.load(Ordering::Relaxed),
        }
    }

    /// Check if the system is CPU-bound
    pub fn is_cpu_bound(&mut self) -> bool {
        self.system.refresh_cpu();
        self.get_current_cpu_utilization() > 85.0
    }

    /// Get SIMD optimization recommendations
    pub fn get_simd_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if !self.global_metrics.simd_capability.avx2_available && !self.global_metrics.simd_capability.neon_available {
            recommendations.push("Consider enabling SIMD optimizations for better performance".to_string());
        }

        if self.global_metrics.simd_capability.avx512_available {
            recommendations.push("AVX-512 is available - consider using it for large vector operations".to_string());
        }

        // Check if SIMD operations are being used effectively
        let total_simd_ops: u64 = self.operation_metrics.values()
            .map(|m| m.simd_operations)
            .sum();

        if total_simd_ops == 0 && (self.global_metrics.simd_capability.avx2_available || self.global_metrics.simd_capability.neon_available) {
            recommendations.push("SIMD capabilities detected but not being used - consider vectorizing operations".to_string());
        }

        recommendations
    }
}

/// Real-time CPU statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct RealtimeCpuStats {
    pub average_utilization: f32,
    pub per_core_utilization: Vec<f32>,
    pub active_operations: usize,
    pub total_operations: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_cpu_profiler_creation() {
        let profiler = CpuProfiler::new();
        assert!(profiler.global_metrics.core_count > 0);
    }

    #[test]
    fn test_simd_detection() {
        let capability = CpuProfiler::detect_simd_capability();
        // At least one SIMD capability should be available on modern systems
        assert!(capability.avx2_available || capability.neon_available || capability.sse_available);
    }

    #[test]
    fn test_operation_profiling() {
        let mut profiler = CpuProfiler::new();

        profiler.start_operation("test_op");
        thread::sleep(Duration::from_millis(10));
        let metrics = profiler.finish_operation("test_op");

        assert!(metrics.total_duration >= Duration::from_millis(10));
        assert!(metrics.thread_count > 0);
    }

    #[test]
    fn test_report_generation() {
        let mut profiler = CpuProfiler::new();

        profiler.start_operation("test_op");
        thread::sleep(Duration::from_millis(10));
        profiler.finish_operation("test_op");

        let report = profiler.generate_report();
        assert!(report.total_operations > 0);
        assert!(report.duration >= Duration::from_millis(10));
        assert_eq!(report.per_core_utilization.len(), report.core_count as usize);
    }

    #[test]
    fn test_simd_recording() {
        let mut profiler = CpuProfiler::new();

        profiler.start_operation("simd_test");
        profiler.record_simd_operation("simd_test", 100);
        let metrics = profiler.finish_operation("simd_test");

        assert_eq!(metrics.simd_operations, 100);
    }

    #[test]
    fn test_efficiency_calculation() {
        let metrics = CpuOperationMetrics {
            start_time: Instant::now(),
            total_duration: Duration::from_millis(100),
            average_utilization: 75.0,
            peak_utilization: 90.0,
            thread_count: 4,
            simd_operations: 50,
            cache_misses: 10,
        };

        let efficiency = CpuProfiler::calculate_efficiency_score(&metrics);
        assert!(efficiency > 0.0 && efficiency <= 1.0);
    }
}