//! GPU Monitoring for AI Engine
//!
//! This module provides GPU utilization monitoring across different platforms:
//! - NVIDIA GPUs via NVML (NVIDIA Management Library)
//! - Future support for Apple Metal, AMD ROCm, etc.
//!
//! NOTE (COOLJAPAN Pure Rust Policy v2): the real NVML (`nvml-wrapper`) backend
//! has been quarantined into the `oxirs-gpu-monitor` crate (`publish = false`) so
//! that `oxirs-core`'s published `--all-features` surface stays free of the
//! `nvml-wrapper-sys` C FFI. `GpuMonitor` below is a Pure-Rust stub with an
//! unchanged public API (it reports "no GPU" / zeros). Binaries that need live
//! NVIDIA telemetry should use `oxirs_gpu_monitor::NvmlGpuMonitor`, which returns
//! the same `GpuStats` type defined here.

use anyhow::Result;
use std::sync::{Arc, Mutex, OnceLock};

/// GPU monitoring statistics
#[derive(Debug, Clone, Default)]
pub struct GpuStats {
    /// GPU utilization percentage (0.0-100.0)
    pub utilization: f32,

    /// Memory utilization percentage (0.0-100.0)
    pub memory_utilization: f32,

    /// Temperature in Celsius
    pub temperature: f32,

    /// Power usage in watts
    pub power_usage: f32,

    /// Available memory in MB
    pub memory_free_mb: u64,

    /// Total memory in MB
    pub memory_total_mb: u64,
}

/// GPU monitor that provides cross-platform GPU statistics
pub struct GpuMonitor {}

static GPU_MONITOR: OnceLock<Arc<Mutex<GpuMonitor>>> = OnceLock::new();

impl GpuMonitor {
    /// Create a new GPU monitor
    pub fn new() -> Self {
        Self::with_device(0)
    }

    /// Create a new GPU monitor with specific device index
    pub fn with_device(device_index: u32) -> Self {
        let _ = device_index; // Suppress unused variable warning
        Self {}
    }

    /// Get the global GPU monitor instance
    pub fn global() -> Arc<Mutex<GpuMonitor>> {
        GPU_MONITOR
            .get_or_init(|| Arc::new(Mutex::new(GpuMonitor::new())))
            .clone()
    }

    /// Get current GPU statistics
    pub fn get_stats(&self) -> Result<GpuStats> {
        // GPU monitoring not enabled (Pure-Rust stub; see module docs / oxirs-gpu-monitor)
        Ok(GpuStats::default())
    }

    /// Get GPU utilization percentage (0.0-100.0)
    pub fn get_utilization(&self) -> f32 {
        self.get_stats()
            .map(|stats| stats.utilization)
            .unwrap_or(0.0)
    }

    /// Check if GPU is available
    pub fn is_available(&self) -> bool {
        false
    }

    /// Get number of available GPUs
    pub fn device_count() -> u32 {
        0
    }
}

impl Default for GpuMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_monitor_creation() {
        let monitor = GpuMonitor::new();
        // Should not panic even if GPU is not available
        let _ = monitor.is_available();
    }

    #[test]
    fn test_gpu_stats() {
        let monitor = GpuMonitor::new();
        let stats = monitor.get_stats();
        assert!(stats.is_ok());

        if monitor.is_available() {
            let stats = stats.expect("stats should be available");
            assert!(stats.utilization >= 0.0 && stats.utilization <= 100.0);
        }
    }

    #[test]
    fn test_device_count() {
        let _count = GpuMonitor::device_count();
        // Device count is u32, always >= 0 (no assertion needed)
        // Just verify the method doesn't panic
    }

    #[test]
    fn test_global_monitor() {
        let monitor1 = GpuMonitor::global();
        let monitor2 = GpuMonitor::global();

        // Should return the same instance
        assert!(Arc::ptr_eq(&monitor1, &monitor2));
    }
}
