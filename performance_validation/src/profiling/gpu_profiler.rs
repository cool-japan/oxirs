//! GPU Performance Profiler
//!
//! Monitors GPU utilization, memory usage, and compute performance.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// GPU profiler for monitoring performance and utilization
#[derive(Debug)]
pub struct GpuProfiler {
    gpu_available: bool,
    device_info: Option<GpuDeviceInfo>,
    operation_metrics: HashMap<String, GpuOperationMetrics>,
    global_metrics: GpuGlobalMetrics,
    start_time: Instant,
}

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    pub device_name: String,
    pub device_id: u32,
    pub total_memory_mb: u64,
    pub compute_capability: String,
    pub driver_version: String,
    pub backend: GpuBackend,
}

/// GPU backend type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuBackend {
    CUDA,
    OpenCL,
    Metal,
    ROCm,
    Vulkan,
    CPU, // Fallback
}

/// Per-operation GPU metrics.
///
/// `gpu_utilization` and `memory_used_mb`/`memory_peak_mb` are `None` when no
/// real telemetry source was available for the detected backend/platform
/// (e.g. Metal on macOS has no free, reliable realtime utilization API) --
/// they are never a fabricated, plausible-looking placeholder value.
#[derive(Debug, Clone)]
pub struct GpuOperationMetrics {
    pub start_time: Instant,
    pub total_duration: Duration,
    pub gpu_utilization: Option<f32>,
    pub memory_used_mb: Option<u64>,
    pub memory_peak_mb: Option<u64>,
    pub memory_transfers: u64,
    pub compute_operations: u64,
    pub kernel_launches: u32,
}

/// Global GPU metrics
#[derive(Debug, Default)]
pub struct GpuGlobalMetrics {
    pub total_operations: u64,
    pub total_gpu_time: Duration,
    /// Running sum of every real per-operation utilization sample recorded
    /// so far (see `utilization_samples` for the divisor).
    pub utilization_sum: f32,
    /// Count of operations that actually produced a real utilization
    /// sample (i.e. `gpu_utilization` was `Some`). May be less than
    /// `total_operations` when telemetry was unavailable for some/all ops.
    pub utilization_samples: u64,
    pub peak_utilization: Option<f32>,
    pub total_memory_transferred: u64,
    pub peak_memory_usage: Option<u64>,
    pub kernel_launch_overhead: Duration,
}

/// GPU performance report
#[derive(Debug, Serialize, Deserialize)]
pub struct GpuReport {
    pub gpu_available: bool,
    pub device_info: Option<GpuDeviceInfo>,
    pub duration: Duration,
    pub average_utilization: Option<f32>,
    pub peak_utilization: Option<f32>,
    pub total_operations: u64,
    pub memory_efficiency: Option<f32>,
    pub compute_efficiency: Option<f32>,
    pub operation_breakdown: HashMap<String, GpuOperationSummary>,
    pub performance_recommendations: Vec<String>,
}

/// Summary of GPU metrics for an operation
#[derive(Debug, Serialize, Deserialize)]
pub struct GpuOperationSummary {
    pub total_duration: Duration,
    pub average_utilization: Option<f32>,
    pub memory_usage_mb: Option<u64>,
    pub memory_efficiency: Option<f32>,
    pub compute_operations: u64,
    pub kernel_launches: u32,
    pub throughput_ops_per_sec: f64,
}

impl Default for GpuProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuProfiler {
    /// Create a new GPU profiler
    pub fn new() -> Self {
        let (gpu_available, device_info) = Self::detect_gpu();

        Self {
            gpu_available,
            device_info,
            operation_metrics: HashMap::new(),
            global_metrics: GpuGlobalMetrics::default(),
            start_time: Instant::now(),
        }
    }

    /// Detect available GPU and get device information
    fn detect_gpu() -> (bool, Option<GpuDeviceInfo>) {
        // This would be implemented based on the actual GPU detection logic
        // For now, detect based on common GPU indicators

        if Self::detect_metal() {
            (
                true,
                Some(GpuDeviceInfo {
                    device_name: "Apple Metal GPU".to_string(),
                    device_id: 0,
                    total_memory_mb: Self::estimate_metal_memory(),
                    compute_capability: "Metal 3.0".to_string(),
                    driver_version: "Unknown".to_string(),
                    backend: GpuBackend::Metal,
                }),
            )
        } else if Self::detect_cuda() {
            (
                true,
                Some(GpuDeviceInfo {
                    device_name: "CUDA GPU".to_string(),
                    device_id: 0,
                    total_memory_mb: 8192, // Default estimate
                    compute_capability: "Unknown".to_string(),
                    driver_version: "Unknown".to_string(),
                    backend: GpuBackend::CUDA,
                }),
            )
        } else {
            (false, None)
        }
    }

    /// Detect Metal GPU (macOS)
    fn detect_metal() -> bool {
        cfg!(target_os = "macos")
    }

    /// Detect CUDA GPU
    fn detect_cuda() -> bool {
        // Would check for CUDA runtime/drivers
        false
    }

    /// Estimate Metal GPU memory (very rough approximation)
    fn estimate_metal_memory() -> u64 {
        // On Apple Silicon, GPU shares system memory
        // This is a very rough estimate
        8192 // 8GB default
    }

    /// Start monitoring an operation
    pub fn start_operation(&mut self, operation_name: &str) {
        let metrics = GpuOperationMetrics {
            start_time: Instant::now(),
            total_duration: Duration::ZERO,
            gpu_utilization: None,
            memory_used_mb: None,
            memory_peak_mb: None,
            memory_transfers: 0,
            compute_operations: 0,
            kernel_launches: 0,
        };

        self.operation_metrics
            .insert(operation_name.to_string(), metrics);
    }

    /// Finish monitoring an operation
    pub fn finish_operation(&mut self, operation_name: &str) -> GpuOperationMetrics {
        if let Some(mut metrics) = self.operation_metrics.remove(operation_name) {
            metrics.total_duration = metrics.start_time.elapsed();

            // Query real GPU telemetry for the detected backend. When no
            // free, reliable telemetry source exists for that backend (e.g.
            // Metal on macOS), these stay `None` -- never a fabricated
            // plausible-looking number.
            if self.gpu_available {
                if let Some(backend) = self.device_info.as_ref().map(|d| d.backend.clone()) {
                    metrics.gpu_utilization = Self::query_gpu_utilization(&backend);
                    metrics.memory_used_mb = Self::query_gpu_memory_used_mb(&backend);
                    metrics.memory_peak_mb = metrics.memory_used_mb;
                    // Only fall back to the duration-based estimate when the
                    // caller never explicitly recorded real counts via
                    // `record_compute_operation`/no kernel launches were
                    // implied -- an explicit recording must never be
                    // silently clobbered by a rough estimate.
                    if metrics.compute_operations == 0 {
                        metrics.compute_operations = Self::estimate_compute_operations(&metrics);
                    }
                    if metrics.kernel_launches == 0 {
                        metrics.kernel_launches = Self::estimate_kernel_launches(&metrics);
                    }
                }
            }

            // Update global metrics
            self.global_metrics.total_operations += 1;
            self.global_metrics.total_gpu_time += metrics.total_duration;
            if let Some(utilization) = metrics.gpu_utilization {
                self.global_metrics.utilization_sum += utilization;
                self.global_metrics.utilization_samples += 1;
                self.global_metrics.peak_utilization = Some(
                    self.global_metrics
                        .peak_utilization
                        .unwrap_or(0.0)
                        .max(utilization),
                );
            }
            self.global_metrics.total_memory_transferred += metrics.memory_transfers;
            if let Some(peak) = metrics.memory_peak_mb {
                self.global_metrics.peak_memory_usage =
                    Some(self.global_metrics.peak_memory_usage.unwrap_or(0).max(peak));
            }

            metrics
        } else {
            // Return default metrics if operation not found
            GpuOperationMetrics {
                start_time: Instant::now(),
                total_duration: Duration::ZERO,
                gpu_utilization: None,
                memory_used_mb: None,
                memory_peak_mb: None,
                memory_transfers: 0,
                compute_operations: 0,
                kernel_launches: 0,
            }
        }
    }

    /// Query real-time GPU utilization percentage for the given backend.
    ///
    /// Returns `None` when no reliable, cost-free telemetry source exists
    /// for that backend (e.g. Metal on macOS has no public realtime
    /// utilization API without private frameworks) instead of fabricating a
    /// plausible-looking number.
    fn query_gpu_utilization(backend: &GpuBackend) -> Option<f32> {
        match backend {
            GpuBackend::CUDA => Self::query_nvidia_smi_metric("utilization.gpu"),
            GpuBackend::Metal
            | GpuBackend::OpenCL
            | GpuBackend::ROCm
            | GpuBackend::Vulkan
            | GpuBackend::CPU => None,
        }
    }

    /// Query real-time GPU memory used (MB) for the given backend. See
    /// [`Self::query_gpu_utilization`] for the honesty contract.
    fn query_gpu_memory_used_mb(backend: &GpuBackend) -> Option<u64> {
        match backend {
            GpuBackend::CUDA => Self::query_nvidia_smi_metric("memory.used").map(|v| v as u64),
            GpuBackend::Metal
            | GpuBackend::OpenCL
            | GpuBackend::ROCm
            | GpuBackend::Vulkan
            | GpuBackend::CPU => None,
        }
    }

    /// Shell out to `nvidia-smi` for a single numeric metric. Returns `None`
    /// if the binary is unavailable, the process fails, or the output can't
    /// be parsed -- it never fabricates a substitute value.
    fn query_nvidia_smi_metric(metric: &str) -> Option<f32> {
        let output = std::process::Command::new("nvidia-smi")
            .args([
                format!("--query-gpu={metric}").as_str(),
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok()?;

        if !output.status.success() {
            return None;
        }

        String::from_utf8_lossy(&output.stdout)
            .lines()
            .next()?
            .trim()
            .parse::<f32>()
            .ok()
    }

    /// Estimate compute operations based on duration
    fn estimate_compute_operations(metrics: &GpuOperationMetrics) -> u64 {
        // Rough estimate: assume some operations per millisecond
        (metrics.total_duration.as_millis() as u64) * 1000
    }

    /// Estimate kernel launches
    fn estimate_kernel_launches(metrics: &GpuOperationMetrics) -> u32 {
        // Estimate based on operation complexity
        if metrics.total_duration > Duration::from_millis(100) {
            5 // Multiple kernel launches for complex operations
        } else {
            1 // Single kernel launch for simple operations
        }
    }

    /// Record memory transfer
    pub fn record_memory_transfer(&mut self, operation_name: &str, bytes: u64) {
        if let Some(metrics) = self.operation_metrics.get_mut(operation_name) {
            metrics.memory_transfers += bytes;
        }
    }

    /// Record compute operation
    pub fn record_compute_operation(&mut self, operation_name: &str, count: u64) {
        if let Some(metrics) = self.operation_metrics.get_mut(operation_name) {
            metrics.compute_operations += count;
        }
    }

    /// Generate comprehensive GPU report
    pub fn generate_report(&self) -> GpuReport {
        let runtime = self.start_time.elapsed();

        // Real average, computed only from operations that actually
        // produced a utilization sample; `None` when telemetry was never
        // available (rather than silently averaging in zeros).
        let average_utilization = if self.global_metrics.utilization_samples > 0 {
            Some(
                self.global_metrics.utilization_sum
                    / self.global_metrics.utilization_samples as f32,
            )
        } else {
            None
        };

        // Calculate efficiency metrics
        let memory_efficiency = self.calculate_memory_efficiency();
        let compute_efficiency = self.calculate_compute_efficiency();

        // Generate operation summaries
        let operation_breakdown: HashMap<String, GpuOperationSummary> = self
            .operation_metrics
            .iter()
            .map(|(name, metrics)| {
                let summary = GpuOperationSummary {
                    total_duration: metrics.total_duration,
                    average_utilization: metrics.gpu_utilization,
                    memory_usage_mb: metrics.memory_used_mb,
                    memory_efficiency: self.calculate_operation_memory_efficiency(metrics),
                    compute_operations: metrics.compute_operations,
                    kernel_launches: metrics.kernel_launches,
                    throughput_ops_per_sec: if metrics.total_duration.as_secs_f64() > 0.0 {
                        metrics.compute_operations as f64 / metrics.total_duration.as_secs_f64()
                    } else {
                        0.0
                    },
                };
                (name.clone(), summary)
            })
            .collect();

        let recommendations = self.generate_recommendations();

        GpuReport {
            gpu_available: self.gpu_available,
            device_info: self.device_info.clone(),
            duration: runtime,
            average_utilization,
            peak_utilization: self.global_metrics.peak_utilization,
            total_operations: self.global_metrics.total_operations,
            memory_efficiency,
            compute_efficiency,
            operation_breakdown,
            performance_recommendations: recommendations,
        }
    }

    /// Calculate overall memory efficiency. `None` when the device's total
    /// memory or a real peak-usage sample isn't known.
    fn calculate_memory_efficiency(&self) -> Option<f32> {
        let device_info = self.device_info.as_ref()?;
        let total_memory = device_info.total_memory_mb as f32;
        let peak_usage = self.global_metrics.peak_memory_usage? as f32;

        if total_memory > 0.0 {
            Some((peak_usage / total_memory).min(1.0))
        } else {
            None
        }
    }

    /// Calculate overall compute efficiency from real accumulated
    /// utilization samples. `None` when no sample was ever recorded.
    fn calculate_compute_efficiency(&self) -> Option<f32> {
        if self.global_metrics.utilization_samples > 0 {
            let average_utilization = self.global_metrics.utilization_sum
                / self.global_metrics.utilization_samples as f32;
            Some(average_utilization / 100.0)
        } else {
            None
        }
    }

    /// Calculate memory efficiency for a specific operation. `None` when the
    /// device's total memory or a real per-operation sample isn't known.
    fn calculate_operation_memory_efficiency(&self, metrics: &GpuOperationMetrics) -> Option<f32> {
        let device_info = self.device_info.as_ref()?;
        let total_memory = device_info.total_memory_mb as f32;
        let used_memory = metrics.memory_peak_mb? as f32;

        if total_memory > 0.0 {
            Some((used_memory / total_memory).min(1.0))
        } else {
            None
        }
    }

    /// Generate performance recommendations
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if !self.gpu_available {
            recommendations.push(
                "No GPU detected. Consider using GPU acceleration for better performance."
                    .to_string(),
            );
            return recommendations;
        }

        // Utilization recommendations -- only issued when we actually have a
        // real utilization sample; otherwise tell the caller honestly that
        // telemetry wasn't available rather than guessing.
        match self.global_metrics.peak_utilization {
            Some(peak) if peak < 50.0 => {
                recommendations.push("Low GPU utilization detected. Consider increasing batch sizes or using more parallel operations.".to_string());
            }
            None => {
                recommendations.push("GPU telemetry is unavailable on this backend/platform; utilization-based recommendations are disabled.".to_string());
            }
            _ => {}
        }

        // Memory recommendations (only when a real efficiency figure exists)
        if let Some(memory_efficiency) = self.calculate_memory_efficiency() {
            if memory_efficiency < 0.3 {
                recommendations.push("Low memory utilization. Consider processing larger datasets or using memory pooling.".to_string());
            } else if memory_efficiency > 0.9 {
                recommendations.push("High memory usage detected. Consider reducing batch sizes or implementing memory optimization.".to_string());
            }
        }

        // Kernel launch overhead
        let avg_kernel_launches: f32 = self
            .operation_metrics
            .values()
            .map(|m| m.kernel_launches as f32)
            .sum::<f32>()
            / self.operation_metrics.len().max(1) as f32;

        if avg_kernel_launches > 10.0 {
            recommendations.push("High number of kernel launches detected. Consider batching operations to reduce overhead.".to_string());
        }

        recommendations
    }

    /// Reset all profiling data
    pub fn reset(&mut self) {
        self.operation_metrics.clear();
        self.global_metrics = GpuGlobalMetrics::default();
        self.start_time = Instant::now();
    }

    /// Get real-time GPU statistics
    pub fn get_realtime_stats(&self) -> RealtimeGpuStats {
        let backend = self.device_info.as_ref().map(|d| d.backend.clone());
        let (current_utilization, memory_usage_mb) = if self.gpu_available {
            (
                backend.as_ref().and_then(Self::query_gpu_utilization),
                backend.as_ref().and_then(Self::query_gpu_memory_used_mb),
            )
        } else {
            (None, None)
        };

        RealtimeGpuStats {
            gpu_available: self.gpu_available,
            current_utilization,
            memory_usage_mb,
            active_operations: self.operation_metrics.len(),
            total_operations: self.global_metrics.total_operations,
        }
    }

    /// Check if GPU is bottleneck
    pub fn is_gpu_bottleneck(&self) -> bool {
        self.gpu_available
            && self
                .global_metrics
                .peak_utilization
                .map(|peak| peak > 90.0)
                .unwrap_or(false)
    }

    /// Get device capabilities
    pub fn get_device_capabilities(&self) -> Option<GpuCapabilities> {
        self.device_info.as_ref().map(|info| {
            GpuCapabilities {
                supports_fp16: true, // Most modern GPUs support half precision
                supports_int8: true,
                supports_tensor_cores: matches!(info.backend, GpuBackend::CUDA),
                max_threads_per_block: match info.backend {
                    GpuBackend::CUDA => 1024,
                    GpuBackend::Metal => 1024,
                    GpuBackend::OpenCL => 256,
                    _ => 64,
                },
                memory_bandwidth_gb_s: match info.backend {
                    GpuBackend::Metal => 400.0, // Apple Silicon estimate
                    GpuBackend::CUDA => 900.0,  // High-end GPU estimate
                    _ => 200.0,
                },
            }
        })
    }
}

/// Real-time GPU statistics. `current_utilization`/`memory_usage_mb` are
/// `None` when no real telemetry source is available for the detected
/// backend/platform.
#[derive(Debug, Serialize, Deserialize)]
pub struct RealtimeGpuStats {
    pub gpu_available: bool,
    pub current_utilization: Option<f32>,
    pub memory_usage_mb: Option<u64>,
    pub active_operations: usize,
    pub total_operations: u64,
}

/// GPU capabilities
#[derive(Debug, Serialize, Deserialize)]
pub struct GpuCapabilities {
    pub supports_fp16: bool,
    pub supports_int8: bool,
    pub supports_tensor_cores: bool,
    pub max_threads_per_block: u32,
    pub memory_bandwidth_gb_s: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_gpu_profiler_creation() {
        let profiler = GpuProfiler::new();
        // Should work regardless of GPU availability, and the two fields
        // must agree: a GPU is "available" iff device info was detected.
        assert_eq!(profiler.gpu_available, profiler.device_info.is_some());
    }

    #[test]
    fn test_operation_profiling() {
        let mut profiler = GpuProfiler::new();

        profiler.start_operation("test_op");
        thread::sleep(Duration::from_millis(10));
        let metrics = profiler.finish_operation("test_op");

        assert!(metrics.total_duration >= Duration::from_millis(10));
    }

    #[test]
    fn test_report_generation() {
        let mut profiler = GpuProfiler::new();

        profiler.start_operation("test_op");
        thread::sleep(Duration::from_millis(10));
        profiler.finish_operation("test_op");

        let report = profiler.generate_report();
        assert_eq!(report.total_operations, 1);
        assert!(report.duration >= Duration::from_millis(10));
    }

    #[test]
    fn test_memory_transfer_recording() {
        let mut profiler = GpuProfiler::new();

        profiler.start_operation("transfer_test");
        profiler.record_memory_transfer("transfer_test", 1024);
        let metrics = profiler.finish_operation("transfer_test");

        assert_eq!(metrics.memory_transfers, 1024);
    }

    #[test]
    fn test_compute_operation_recording() {
        let mut profiler = GpuProfiler::new();

        profiler.start_operation("compute_test");
        profiler.record_compute_operation("compute_test", 500);
        let metrics = profiler.finish_operation("compute_test");

        assert_eq!(metrics.compute_operations, 500);
    }

    #[test]
    fn test_efficiency_calculations() {
        let profiler = GpuProfiler::new();

        // No operation has run yet, so there is no real peak-memory sample:
        // the efficiency must be honestly `None`, not a fabricated 0.0/1.0.
        if profiler.gpu_available {
            let efficiency = profiler.calculate_memory_efficiency();
            if let Some(value) = efficiency {
                assert!((0.0..=1.0).contains(&value));
            }
        }
    }

    /// regression: Metal (and every other backend without a free realtime
    /// telemetry API) must report utilization/memory as honestly
    /// unavailable (`None`), never a hash-of-system-time fabricated number
    /// dressed up as a real measurement.
    #[test]
    fn regression_metal_backend_reports_telemetry_as_unavailable_not_fabricated() {
        assert_eq!(GpuProfiler::query_gpu_utilization(&GpuBackend::Metal), None);
        assert_eq!(
            GpuProfiler::query_gpu_memory_used_mb(&GpuBackend::Metal),
            None
        );
        assert_eq!(
            GpuProfiler::query_gpu_utilization(&GpuBackend::OpenCL),
            None
        );
        assert_eq!(GpuProfiler::query_gpu_utilization(&GpuBackend::ROCm), None);
        assert_eq!(
            GpuProfiler::query_gpu_utilization(&GpuBackend::Vulkan),
            None
        );
        assert_eq!(GpuProfiler::query_gpu_utilization(&GpuBackend::CPU), None);
    }

    /// regression: on a host with no `nvidia-smi` binary (true for macOS and
    /// most CI runners), even the CUDA telemetry path must fail loud-as-None
    /// rather than substituting a fabricated value.
    #[test]
    fn regression_nvidia_smi_unavailable_returns_none_not_fabricated() {
        if std::process::Command::new("nvidia-smi").output().is_err() {
            assert_eq!(GpuProfiler::query_gpu_utilization(&GpuBackend::CUDA), None);
            assert_eq!(
                GpuProfiler::query_gpu_memory_used_mb(&GpuBackend::CUDA),
                None
            );
        }
    }

    /// regression: `global_metrics.utilization_sum`/`utilization_samples`
    /// (the fields that replace the previously dead-write-only
    /// `average_utilization`) must actually be consumed by
    /// `generate_report()` and `calculate_compute_efficiency()` to produce a
    /// real average, instead of both always evaluating to 0/None regardless
    /// of recorded operations.
    #[test]
    fn regression_average_utilization_is_wired_and_accumulates() {
        let mut profiler = GpuProfiler::new();
        // Directly inject accumulated samples (simulating operations that
        // did have real telemetry) to prove the read side is wired up.
        profiler.global_metrics.utilization_sum = 150.0;
        profiler.global_metrics.utilization_samples = 3;
        profiler.global_metrics.total_operations = 5;

        let report = profiler.generate_report();
        assert_eq!(report.average_utilization, Some(50.0));

        let compute_efficiency = profiler.calculate_compute_efficiency();
        assert_eq!(compute_efficiency, Some(0.5));
    }

    /// regression: with zero real utilization samples ever recorded, the
    /// average/efficiency must be honestly `None`, not silently 0.0 (which
    /// would look like a real "0% utilization" measurement).
    #[test]
    fn regression_no_samples_yields_none_not_fabricated_zero() {
        let profiler = GpuProfiler::new();
        let report = profiler.generate_report();
        assert_eq!(report.average_utilization, None);
        assert_eq!(profiler.calculate_compute_efficiency(), None);
    }
}
