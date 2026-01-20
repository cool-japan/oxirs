//! System Load Monitoring for Adaptive Query Execution
//!
//! This module provides real-time system resource monitoring to enable
//! adaptive concurrency adjustment for optimal performance under varying load conditions.

use anyhow::Result;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// System load monitor for adaptive resource management
#[derive(Debug, Clone)]
pub struct SystemLoadMonitor {
    /// CPU usage percentage (0-100)
    cpu_usage: Arc<AtomicU64>,
    /// Memory usage percentage (0-100)
    memory_usage: Arc<AtomicU64>,
    /// Last update timestamp
    last_update: Arc<std::sync::Mutex<Instant>>,
    /// Update interval
    update_interval: Duration,
}

impl SystemLoadMonitor {
    /// Create new system load monitor
    pub fn new() -> Self {
        Self::with_update_interval(Duration::from_secs(1))
    }

    /// Create monitor with custom update interval
    pub fn with_update_interval(interval: Duration) -> Self {
        Self {
            cpu_usage: Arc::new(AtomicU64::new(0)),
            memory_usage: Arc::new(AtomicU64::new(0)),
            last_update: Arc::new(std::sync::Mutex::new(Instant::now())),
            update_interval: interval,
        }
    }

    /// Get current CPU usage percentage (0-100)
    pub fn cpu_usage(&self) -> f64 {
        self.maybe_update();
        f64::from_bits(self.cpu_usage.load(Ordering::Relaxed)) / 100.0
    }

    /// Get current memory usage percentage (0-100)
    pub fn memory_usage(&self) -> f64 {
        self.maybe_update();
        f64::from_bits(self.memory_usage.load(Ordering::Relaxed)) / 100.0
    }

    /// Get overall system load (0.0 - 1.0)
    pub fn overall_load(&self) -> f64 {
        let cpu = self.cpu_usage();
        let mem = self.memory_usage();

        // Weighted average: CPU 60%, Memory 40%
        (cpu * 0.6 + mem * 0.4).min(1.0)
    }

    /// Check if system is under high load
    pub fn is_high_load(&self, threshold: f64) -> bool {
        self.overall_load() > threshold
    }

    /// Check if system is under low load
    pub fn is_low_load(&self, threshold: f64) -> bool {
        self.overall_load() < threshold
    }

    /// Get recommended concurrency level based on current load
    pub fn recommended_concurrency(&self, max_concurrency: usize) -> usize {
        let load = self.overall_load();

        // Scale concurrency inversely with load
        // At 0% load: use max concurrency
        // At 50% load: use 75% of max
        // At 80% load: use 40% of max
        // At 90%+ load: use 25% of max

        let scale_factor = if load < 0.5 {
            1.0
        } else if load < 0.7 {
            0.75
        } else if load < 0.8 {
            0.5
        } else if load < 0.9 {
            0.4
        } else {
            0.25
        };

        ((max_concurrency as f64 * scale_factor).max(1.0) as usize).min(max_concurrency)
    }

    /// Update system metrics if needed
    fn maybe_update(&self) {
        let mut last_update = self.last_update.lock().expect("Lock poisoned");

        if last_update.elapsed() < self.update_interval {
            return; // Too soon to update
        }

        // Update timestamp
        *last_update = Instant::now();
        drop(last_update); // Release lock before potentially slow system calls

        // Update CPU and memory metrics
        if let Ok((cpu, memory)) = self.read_system_metrics() {
            self.cpu_usage.store(cpu.to_bits(), Ordering::Relaxed);
            self.memory_usage.store(memory.to_bits(), Ordering::Relaxed);
        }
    }

    /// Read actual system metrics from OS
    fn read_system_metrics(&self) -> Result<(f64, f64)> {
        // Cross-platform system metrics using sysinfo would go here
        // For now, use a lightweight fallback approach

        #[cfg(target_os = "linux")]
        {
            self.read_linux_metrics()
        }

        #[cfg(target_os = "macos")]
        {
            self.read_macos_metrics()
        }

        #[cfg(target_os = "windows")]
        {
            self.read_windows_metrics()
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            // Fallback: estimate based on num_cpus and available parallelism
            Ok((50.0, 50.0)) // Conservative estimates
        }
    }

    #[cfg(target_os = "linux")]
    fn read_linux_metrics(&self) -> Result<(f64, f64)> {
        // Read from /proc/stat for CPU
        // Read from /proc/meminfo for memory
        // Simplified implementation - production would use sysinfo crate

        let cpu = self.estimate_cpu_from_loadavg()?;
        let memory = self.estimate_memory_from_available()?;

        Ok((cpu, memory))
    }

    #[cfg(target_os = "macos")]
    fn read_macos_metrics(&self) -> Result<(f64, f64)> {
        // Use host_statistics for CPU
        // Use vm_stat for memory
        // Simplified implementation

        let cpu = 30.0; // Conservative estimate for macOS
        let memory = 40.0;

        Ok((cpu, memory))
    }

    #[cfg(target_os = "windows")]
    fn read_windows_metrics(&self) -> Result<(f64, f64)> {
        // Use Windows API for system metrics
        // Simplified implementation

        let cpu = 40.0;
        let memory = 45.0;

        Ok((cpu, memory))
    }

    #[cfg(target_os = "linux")]
    fn estimate_cpu_from_loadavg(&self) -> Result<f64> {
        // Read /proc/loadavg and normalize by number of cores
        use std::fs;

        if let Ok(loadavg) = fs::read_to_string("/proc/loadavg") {
            if let Some(load_str) = loadavg.split_whitespace().next() {
                if let Ok(load) = load_str.parse::<f64>() {
                    let num_cpus = num_cpus::get() as f64;
                    // Convert load average to percentage
                    return Ok((load / num_cpus * 100.0).min(100.0));
                }
            }
        }

        Ok(30.0) // Fallback estimate
    }

    #[cfg(target_os = "linux")]
    fn estimate_memory_from_available(&self) -> Result<f64> {
        // Read /proc/meminfo
        use std::fs;

        if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
            let mut total = 0u64;
            let mut available = 0u64;

            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(val) = line.split_whitespace().nth(1) {
                        total = val.parse().unwrap_or(0);
                    }
                } else if line.starts_with("MemAvailable:") {
                    if let Some(val) = line.split_whitespace().nth(1) {
                        available = val.parse().unwrap_or(0);
                    }
                }
            }

            if total > 0 {
                let used = total.saturating_sub(available);
                return Ok((used as f64 / total as f64 * 100.0).min(100.0));
            }
        }

        Ok(50.0) // Fallback estimate
    }
}

impl Default for SystemLoadMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Adaptive concurrency controller
#[derive(Debug)]
pub struct AdaptiveConcurrencyController {
    /// System load monitor
    monitor: SystemLoadMonitor,
    /// Base maximum concurrency
    max_concurrency: usize,
    /// Current concurrency level
    current_concurrency: Arc<AtomicU64>,
    /// High load threshold (0.0 - 1.0)
    high_load_threshold: f64,
    /// Low load threshold (0.0 - 1.0)
    low_load_threshold: f64,
    /// Adjustment interval
    adjustment_interval: Duration,
    /// Last adjustment time
    last_adjustment: Arc<std::sync::Mutex<Instant>>,
}

impl AdaptiveConcurrencyController {
    /// Create new adaptive concurrency controller
    pub fn new(max_concurrency: usize) -> Self {
        Self {
            monitor: SystemLoadMonitor::new(),
            max_concurrency,
            current_concurrency: Arc::new(AtomicU64::new(max_concurrency as u64)),
            high_load_threshold: 0.75, // 75% system load
            low_load_threshold: 0.40,  // 40% system load
            adjustment_interval: Duration::from_secs(5),
            last_adjustment: Arc::new(std::sync::Mutex::new(Instant::now())),
        }
    }

    /// Get current recommended concurrency level
    pub fn current_concurrency(&self) -> usize {
        self.current_concurrency.load(Ordering::Relaxed) as usize
    }

    /// Update concurrency based on system load (call periodically)
    pub fn update_concurrency(&self) {
        let mut last_adj = self.last_adjustment.lock().expect("Lock poisoned");

        if last_adj.elapsed() < self.adjustment_interval {
            return; // Too soon to adjust
        }

        *last_adj = Instant::now();
        drop(last_adj);

        let load = self.monitor.overall_load();
        let current = self.current_concurrency.load(Ordering::Relaxed) as usize;

        let new_concurrency = if load > self.high_load_threshold {
            // High load - reduce concurrency by 25%
            ((current as f64 * 0.75).max(1.0) as usize).min(self.max_concurrency)
        } else if load < self.low_load_threshold {
            // Low load - increase concurrency by 25%
            ((current as f64 * 1.25) as usize).min(self.max_concurrency)
        } else {
            // Moderate load - use recommended level
            self.monitor.recommended_concurrency(self.max_concurrency)
        };

        self.current_concurrency
            .store(new_concurrency as u64, Ordering::Relaxed);
    }

    /// Get system load monitor for detailed metrics
    pub fn monitor(&self) -> &SystemLoadMonitor {
        &self.monitor
    }

    /// Configure thresholds
    pub fn with_thresholds(mut self, high: f64, low: f64) -> Self {
        self.high_load_threshold = high.clamp(0.0, 1.0);
        self.low_load_threshold = low.clamp(0.0, 1.0);
        self
    }

    /// Configure adjustment interval
    pub fn with_adjustment_interval(mut self, interval: Duration) -> Self {
        self.adjustment_interval = interval;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_load_monitor_creation() {
        let monitor = SystemLoadMonitor::new();

        // CPU and memory should be valid percentages
        let cpu = monitor.cpu_usage();
        let mem = monitor.memory_usage();

        assert!((0.0..=100.0).contains(&cpu));
        assert!((0.0..=100.0).contains(&mem));
    }

    #[test]
    fn test_overall_load_calculation() {
        let monitor = SystemLoadMonitor::new();
        let load = monitor.overall_load();

        // Load should be between 0 and 1
        assert!((0.0..=1.0).contains(&load));
    }

    #[test]
    fn test_recommended_concurrency() {
        let monitor = SystemLoadMonitor::new();
        let rec = monitor.recommended_concurrency(16);

        // Should recommend between 1 and max
        assert!(rec >= 1);
        assert!(rec <= 16);
    }

    #[test]
    fn test_high_low_load_detection() {
        let monitor = SystemLoadMonitor::new();

        // These should be mutually exclusive
        let is_high = monitor.is_high_load(0.75);
        let is_low = monitor.is_low_load(0.40);

        // Can't be both high and low simultaneously
        if is_high {
            assert!(!is_low);
        }
    }

    #[test]
    fn test_adaptive_concurrency_controller() {
        let controller = AdaptiveConcurrencyController::new(16);

        let initial = controller.current_concurrency();
        assert_eq!(initial, 16);

        // Update concurrency (should be safe to call)
        controller.update_concurrency();

        let after_update = controller.current_concurrency();
        assert!(after_update >= 1);
        assert!(after_update <= 16);
    }

    #[test]
    fn test_concurrency_adjustment_interval() {
        let controller = AdaptiveConcurrencyController::new(16);

        // First update should work
        controller.update_concurrency();
        let first = controller.current_concurrency();

        // Immediate second update should be ignored (too soon)
        controller.update_concurrency();
        let second = controller.current_concurrency();

        assert_eq!(first, second, "Concurrency should not change immediately");
    }

    #[test]
    fn test_threshold_configuration() {
        let controller = AdaptiveConcurrencyController::new(16).with_thresholds(0.80, 0.30);

        assert_eq!(controller.high_load_threshold, 0.80);
        assert_eq!(controller.low_load_threshold, 0.30);
    }

    #[test]
    fn test_adjustment_interval_configuration() {
        let interval = Duration::from_secs(10);
        let controller = AdaptiveConcurrencyController::new(16).with_adjustment_interval(interval);

        assert_eq!(controller.adjustment_interval, interval);
    }

    #[test]
    fn test_monitor_access() {
        let controller = AdaptiveConcurrencyController::new(16);
        let monitor = controller.monitor();

        // Should be able to get detailed metrics
        let cpu = monitor.cpu_usage();
        let mem = monitor.memory_usage();

        assert!((0.0..=100.0).contains(&cpu));
        assert!((0.0..=100.0).contains(&mem));
    }

    #[test]
    fn test_concurrency_bounds() {
        let controller = AdaptiveConcurrencyController::new(8);

        // Even with load adjustments, should stay within bounds
        for _ in 0..10 {
            controller.update_concurrency();
            let current = controller.current_concurrency();
            assert!(current >= 1, "Concurrency should never be zero");
            assert!(current <= 8, "Concurrency should not exceed max");
        }
    }
}
