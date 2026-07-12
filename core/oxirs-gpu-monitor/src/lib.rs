//! Quarantined NVIDIA NVML GPU-telemetry adapter for `oxirs-core`.
//!
//! # Why this crate exists
//!
//! Under the COOLJAPAN **Pure Rust Policy v2** (purity is measured on the full
//! `--all-features` dependency closure), a *published* foundational crate must
//! not drag in C FFI. The real NVIDIA telemetry backend, [`nvml-wrapper`], pulls
//! in `nvml-wrapper-sys` (a C FFI binding to `libnvidia-ml`). To keep the
//! published `oxirs-core` surface 100% Pure Rust while *preserving* the real
//! capability, the live NVML monitor has been **quarantined** into this crate.
//!
//! This crate is `publish = false`: it never ships to crates.io, so its C FFI
//! dependency never appears in the published Pure-Rust surface. Binaries that
//! actually want live NVIDIA GPU telemetry depend on this crate directly.
//!
//! Meanwhile, `oxirs-core` keeps [`oxirs_core::ai::gpu_monitor::GpuMonitor`] as a
//! Pure-Rust stub with a byte-identical public API (it reports zeros / "no GPU").
//!
//! # Relationship to `oxirs-core`
//!
//! [`NvmlGpuMonitor`] returns [`oxirs_core::ai::gpu_monitor::GpuStats`] — the very
//! same type the stub uses — so callers can swap the stub for this real monitor
//! without touching any of their downstream types.
//!
//! # Runtime behavior
//!
//! `nvml-wrapper` loads `libnvidia-ml` lazily via `dlopen` at runtime, so this
//! crate **compiles and links without an NVIDIA GPU or driver present**. When no
//! NVML library / device is available, the monitor degrades gracefully:
//! [`NvmlGpuMonitor::is_available`] returns `false` and [`NvmlGpuMonitor::get_stats`]
//! yields [`GpuStats::default`] (all zeros) — identical to the stub.
//!
//! [`nvml-wrapper`]: https://docs.rs/nvml-wrapper

use anyhow::Result;
use oxirs_core::ai::gpu_monitor::GpuStats;
use std::sync::{Arc, Mutex};

/// Real, NVML-backed GPU monitor for NVIDIA devices.
///
/// This is the live counterpart to the Pure-Rust `oxirs_core::ai::gpu_monitor::GpuMonitor`
/// stub. It wraps an [`nvml_wrapper::Nvml`] handle (lazily initialized; `None`
/// when no NVIDIA driver / device is present) and a target device index.
pub struct NvmlGpuMonitor {
    /// NVML handle, shared and interior-locked. `None` when NVML init failed
    /// (e.g. no NVIDIA driver/GPU on the host) — the monitor then reports "no GPU".
    nvml: Option<Arc<Mutex<nvml_wrapper::Nvml>>>,

    /// Index of the NVIDIA device this monitor reports on.
    device_index: u32,
}

impl NvmlGpuMonitor {
    /// Create a new NVML GPU monitor targeting device index `0`.
    pub fn new() -> Self {
        Self::with_device(0)
    }

    /// Create a new NVML GPU monitor targeting a specific device index.
    ///
    /// NVML is initialized eagerly here; if initialization fails (no driver /
    /// no GPU / library missing) the monitor is constructed in a disabled state
    /// rather than panicking, and a warning is logged via `tracing`.
    pub fn with_device(device_index: u32) -> Self {
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

    /// Get current GPU statistics from NVML.
    ///
    /// Returns [`GpuStats::default`] (all zeros) when NVML is unavailable. When a
    /// device is present, utilization / memory are required (errors propagate),
    /// while temperature and power are best-effort (default to `0` on failure).
    pub fn get_stats(&self) -> Result<GpuStats> {
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
                memory_utilization: (memory_info.used as f32 / memory_info.total as f32) * 100.0,
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

    /// Get GPU utilization percentage (0.0-100.0), or `0.0` if unavailable.
    pub fn get_utilization(&self) -> f32 {
        self.get_stats()
            .map(|stats| stats.utilization)
            .unwrap_or(0.0)
    }

    /// Check whether an NVML-backed GPU is available to this monitor.
    pub fn is_available(&self) -> bool {
        self.nvml.is_some()
    }

    /// Get the number of NVIDIA devices visible to NVML.
    ///
    /// Unlike the `oxirs-core` stub (which returns `0` unconditionally), this
    /// propagates NVML errors so callers can distinguish "no driver" from
    /// "zero devices".
    pub fn device_count() -> Result<u32> {
        let nvml = nvml_wrapper::Nvml::init()?;
        Ok(nvml.device_count()?)
    }
}

impl Default for NvmlGpuMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nvml_monitor_creation() {
        let monitor = NvmlGpuMonitor::new();
        // Should not panic even if no NVIDIA GPU is available.
        let _ = monitor.is_available();
    }

    #[test]
    fn test_nvml_get_stats() {
        let monitor = NvmlGpuMonitor::new();
        let stats = monitor.get_stats();
        // get_stats never errors when NVML is absent (returns default zeros).
        assert!(stats.is_ok());

        if monitor.is_available() {
            let stats = stats.expect("stats should be available when a GPU is present");
            assert!(stats.utilization >= 0.0 && stats.utilization <= 100.0);
        }
    }

    #[test]
    fn test_default_matches_new() {
        // Both construct device-0 monitors; on a GPU-less host both report no GPU.
        let a = NvmlGpuMonitor::default();
        let b = NvmlGpuMonitor::new();
        assert_eq!(a.is_available(), b.is_available());
    }

    #[test]
    fn test_device_count_does_not_panic() {
        // On a host without an NVIDIA driver, init() fails and device_count()
        // returns Err — that is expected and must not panic.
        let _ = NvmlGpuMonitor::device_count();
    }
}
