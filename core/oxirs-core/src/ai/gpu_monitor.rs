//! GPU Monitoring for AI Engine
//!
//! This module provides GPU utilization monitoring across different platforms:
//! - NVIDIA GPUs via NVML (NVIDIA Management Library)
//! - Future support for Apple Metal, AMD ROCm, etc.

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
pub struct GpuMonitor {
    #[cfg(feature = "gpu")]
    nvml: Option<Arc<Mutex<nvml_wrapper::Nvml>>>,

    #[cfg(feature = "gpu")]
    device_index: u32,
}

static GPU_MONITOR: OnceLock<Arc<Mutex<GpuMonitor>>> = OnceLock::new();

impl GpuMonitor {
    /// Create a new GPU monitor
    pub fn new() -> Self {
        Self::with_device(0)
    }

    /// Create a new GPU monitor with specific device index
    pub fn with_device(device_index: u32) -> Self {
        #[cfg(feature = "gpu")]
        {
            // Try to initialize NVML
            let nvml = match nvml_wrapper::Nvml::init() {
                Ok(nvml) => Some(Arc::new(Mutex::new(nvml))),
                Err(e) => {
                    tracing::warn!("Failed to initialize NVML: {}. GPU monitoring disabled.", e);
                    None
                }
            };

            Self { nvml, device_index }
        }

        #[cfg(not(feature = "gpu"))]
        {
            let _ = device_index; // Suppress unused variable warning
            Self {}
        }
    }

    /// Get the global GPU monitor instance
    pub fn global() -> Arc<Mutex<GpuMonitor>> {
        GPU_MONITOR
            .get_or_init(|| Arc::new(Mutex::new(GpuMonitor::new())))
            .clone()
    }

    /// Get current GPU statistics
    pub fn get_stats(&self) -> Result<GpuStats> {
        #[cfg(feature = "gpu")]
        {
            if let Some(nvml_arc) = &self.nvml {
                let nvml = nvml_arc
                    .lock()
                    .map_err(|e| anyhow::anyhow!("Failed to lock NVML: {}", e))?;

                // Get device
                let device = nvml.device_by_index(self.device_index)?;

                // Get utilization rates
                let utilization = device.utilization_rates()?;

                // Get memory info
                let memory_info = device.memory_info()?;

                // Get temperature (optional, may fail on some GPUs)
                let temperature = device
                    .temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
                    .unwrap_or(0);

                // Get power usage (optional, may fail on some GPUs)
                let power_usage = device.power_usage().unwrap_or(0) as f32 / 1000.0; // Convert mW to W

                Ok(GpuStats {
                    utilization: utilization.gpu as f32,
                    memory_utilization: (memory_info.used as f32 / memory_info.total as f32)
                        * 100.0,
                    temperature: temperature as f32,
                    power_usage,
                    memory_free_mb: memory_info.free / (1024 * 1024),
                    memory_total_mb: memory_info.total / (1024 * 1024),
                })
            } else {
                // No GPU available
                Ok(GpuStats::default())
            }
        }

        #[cfg(not(feature = "gpu"))]
        {
            // GPU monitoring not enabled
            Ok(GpuStats::default())
        }
    }

    /// Get GPU utilization percentage (0.0-100.0)
    pub fn get_utilization(&self) -> f32 {
        self.get_stats()
            .map(|stats| stats.utilization)
            .unwrap_or(0.0)
    }

    /// Check if GPU is available
    pub fn is_available(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.nvml.is_some()
        }

        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    /// Get number of available GPUs
    pub fn device_count() -> u32 {
        #[cfg(feature = "gpu")]
        {
            if let Ok(nvml) = nvml_wrapper::Nvml::init() {
                nvml.device_count().unwrap_or(0)
            } else {
                0
            }
        }

        #[cfg(not(feature = "gpu"))]
        {
            0
        }
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
            let stats = stats.unwrap();
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
