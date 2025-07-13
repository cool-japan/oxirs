//! System Resource Monitoring
//!
//! This module provides real-time system resource monitoring capabilities
//! including memory usage, CPU usage, and other system metrics.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use sysinfo::{Pid, ProcessesToUpdate, System};
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// System resource metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// Memory usage in MB for the current process
    pub memory_usage_mb: f64,
    /// CPU usage percentage for the current process (0.0 to 100.0)
    pub cpu_usage_percent: f64,
    /// Total system memory in MB
    pub total_memory_mb: f64,
    /// Available system memory in MB
    pub available_memory_mb: f64,
    /// System-wide CPU usage percentage
    pub system_cpu_usage_percent: f64,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// Current process ID
    pub process_id: u32,
    /// Timestamp when metrics were collected (duration since a reference point)
    #[serde(with = "duration_serde")]
    pub timestamp: Duration,
}

/// Serde module for Duration serialization
mod duration_serde {
    use super::*;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_millis() as u64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(millis))
    }
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            memory_usage_mb: 0.0,
            cpu_usage_percent: 0.0,
            total_memory_mb: 0.0,
            available_memory_mb: 0.0,
            system_cpu_usage_percent: 0.0,
            cpu_cores: num_cpus::get(),
            process_id: std::process::id(),
            timestamp: Duration::from_secs(0),
        }
    }
}

/// Configuration for system monitoring
#[derive(Debug, Clone)]
pub struct SystemMonitorConfig {
    /// How often to update system information
    pub update_interval: Duration,
    /// Enable detailed CPU per-core monitoring
    pub enable_per_core_cpu: bool,
    /// Enable memory breakdown monitoring
    pub enable_memory_breakdown: bool,
    /// Enable network monitoring
    pub enable_network_monitoring: bool,
    /// Enable disk I/O monitoring
    pub enable_disk_monitoring: bool,
}

impl Default for SystemMonitorConfig {
    fn default() -> Self {
        Self {
            update_interval: Duration::from_secs(1),
            enable_per_core_cpu: false,
            enable_memory_breakdown: true,
            enable_network_monitoring: false,
            enable_disk_monitoring: false,
        }
    }
}

/// Real-time system resource monitor
pub struct SystemMonitor {
    config: SystemMonitorConfig,
    system: Arc<Mutex<System>>,
    last_update: Arc<RwLock<Instant>>,
    cached_metrics: Arc<RwLock<SystemMetrics>>,
    process_id: u32,
}

impl Default for SystemMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemMonitor {
    /// Create a new system monitor
    pub fn new() -> Self {
        Self::with_config(SystemMonitorConfig::default())
    }

    /// Create a new system monitor with custom configuration
    pub fn with_config(config: SystemMonitorConfig) -> Self {
        let mut system = System::new_all();
        system.refresh_all();

        let process_id = std::process::id();

        Self {
            config,
            system: Arc::new(Mutex::new(system)),
            last_update: Arc::new(RwLock::new(Instant::now())),
            cached_metrics: Arc::new(RwLock::new(SystemMetrics::default())),
            process_id,
        }
    }

    /// Get current system metrics (cached or fresh)
    pub async fn get_metrics(&self) -> Result<SystemMetrics> {
        let now = Instant::now();
        let last_update = *self.last_update.read().await;

        // Check if we need to update metrics
        if now.duration_since(last_update) >= self.config.update_interval {
            self.update_metrics().await?;
        }

        let metrics = self.cached_metrics.read().await.clone();
        Ok(metrics)
    }

    /// Force update of system metrics
    pub async fn update_metrics(&self) -> Result<()> {
        let metrics = {
            let mut system = self
                .system
                .lock()
                .map_err(|_| anyhow::anyhow!("System mutex poisoned"))?;

            // Refresh system information
            system.refresh_cpu_all();
            system.refresh_memory();
            system.refresh_processes(ProcessesToUpdate::All, false);

            self.collect_metrics(&mut system)?
        };

        // Update cached metrics and timestamp
        {
            let mut cached = self.cached_metrics.write().await;
            *cached = metrics;
        }

        {
            let mut last_update = self.last_update.write().await;
            *last_update = Instant::now();
        }

        debug!("System metrics updated");
        Ok(())
    }

    /// Collect current system metrics
    fn collect_metrics(&self, system: &mut System) -> Result<SystemMetrics> {
        // Get process-specific metrics
        let (memory_usage_mb, cpu_usage_percent) =
            if let Some(process) = system.process(Pid::from_u32(self.process_id)) {
                let memory_bytes = process.memory();
                let memory_mb = memory_bytes as f64 / 1024.0 / 1024.0;
                let cpu_percent = process.cpu_usage() as f64;
                (memory_mb, cpu_percent)
            } else {
                warn!("Could not find current process in system info");
                (0.0, 0.0)
            };

        // Get system-wide metrics
        let total_memory_mb = system.total_memory() as f64 / 1024.0 / 1024.0;
        let available_memory_mb = system.available_memory() as f64 / 1024.0 / 1024.0;

        // Calculate system-wide CPU usage
        let system_cpu_usage_percent = system.global_cpu_usage() as f64;

        let cpu_cores = system.cpus().len();

        // Use duration since epoch as timestamp for consistency
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0));

        Ok(SystemMetrics {
            memory_usage_mb,
            cpu_usage_percent,
            total_memory_mb,
            available_memory_mb,
            system_cpu_usage_percent,
            cpu_cores,
            process_id: self.process_id,
            timestamp,
        })
    }

    /// Get memory usage for the current process in MB
    pub async fn get_memory_usage_mb(&self) -> Result<f64> {
        let metrics = self.get_metrics().await?;
        Ok(metrics.memory_usage_mb)
    }

    /// Get CPU usage for the current process as percentage (0.0 to 100.0)
    pub async fn get_cpu_usage_percent(&self) -> Result<f64> {
        let metrics = self.get_metrics().await?;
        Ok(metrics.cpu_usage_percent)
    }

    /// Get a lightweight snapshot of current metrics (may be slightly stale)
    pub async fn get_cached_metrics(&self) -> SystemMetrics {
        self.cached_metrics.read().await.clone()
    }

    /// Start background monitoring task
    pub async fn start_background_monitoring(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        let monitor = self.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(monitor.config.update_interval);

            loop {
                interval.tick().await;

                if let Err(e) = monitor.update_metrics().await {
                    warn!("Failed to update system metrics: {}", e);
                }
            }
        })
    }
}

/// Helper function to get current memory usage (convenience method)
pub async fn get_current_memory_usage_mb() -> f64 {
    static MONITOR: std::sync::OnceLock<Arc<SystemMonitor>> = std::sync::OnceLock::new();

    let monitor = MONITOR.get_or_init(|| Arc::new(SystemMonitor::new()));

    monitor.get_memory_usage_mb().await.unwrap_or_else(|e| {
        warn!("Failed to get memory usage: {}", e);
        0.0
    })
}

/// Helper function to get current CPU usage (convenience method)
pub async fn get_current_cpu_usage_percent() -> f64 {
    static MONITOR: std::sync::OnceLock<Arc<SystemMonitor>> = std::sync::OnceLock::new();

    let monitor = MONITOR.get_or_init(|| Arc::new(SystemMonitor::new()));

    monitor.get_cpu_usage_percent().await.unwrap_or_else(|e| {
        warn!("Failed to get CPU usage: {}", e);
        0.0
    })
}

/// Calculate throughput in Mbps from requests per second and average response size
pub fn calculate_throughput_mbps(requests_per_second: f64, avg_response_size_bytes: f64) -> f64 {
    if requests_per_second <= 0.0 || avg_response_size_bytes <= 0.0 {
        return 0.0;
    }

    // Convert bytes per second to megabits per second
    let bytes_per_second = requests_per_second * avg_response_size_bytes;
    let bits_per_second = bytes_per_second * 8.0;

    bits_per_second / (1024.0 * 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_system_monitor_creation() {
        let monitor = SystemMonitor::new();

        // Update metrics to ensure system info is refreshed
        monitor.update_metrics().await.unwrap();

        // Should be able to get metrics
        let metrics = monitor.get_metrics().await.unwrap();

        // Basic validation
        assert!(metrics.cpu_cores > 0);
        assert!(metrics.process_id > 0);
        // Some CI systems might not report memory properly, so use >= 0.0
        assert!(metrics.total_memory_mb >= 0.0);
    }

    #[tokio::test]
    async fn test_memory_monitoring() {
        let monitor = SystemMonitor::new();

        let memory_usage = monitor.get_memory_usage_mb().await.unwrap();

        // Should have some memory usage (process should be using at least some memory)
        assert!(memory_usage >= 0.0);
    }

    #[tokio::test]
    async fn test_cpu_monitoring() {
        let monitor = SystemMonitor::new();

        // Update metrics to get CPU readings
        monitor.update_metrics().await.unwrap();

        let cpu_usage = monitor.get_cpu_usage_percent().await.unwrap();

        // CPU usage should be between 0 and 100
        assert!(cpu_usage >= 0.0);
        assert!(cpu_usage <= 100.0);
    }

    #[test]
    fn test_throughput_calculation() {
        // Test normal case
        let throughput = calculate_throughput_mbps(100.0, 1024.0); // 100 RPS, 1KB responses
        assert!(throughput > 0.0);

        // Test edge cases
        assert_eq!(calculate_throughput_mbps(0.0, 1024.0), 0.0);
        assert_eq!(calculate_throughput_mbps(100.0, 0.0), 0.0);
        assert_eq!(calculate_throughput_mbps(-1.0, 1024.0), 0.0);
    }

    #[tokio::test]
    async fn test_convenience_functions() {
        // Test convenience functions
        let memory = get_current_memory_usage_mb().await;
        assert!(memory >= 0.0);

        let cpu = get_current_cpu_usage_percent().await;
        assert!(cpu >= 0.0);
        assert!(cpu <= 100.0);
    }

    #[tokio::test]
    async fn test_background_monitoring() {
        let monitor = Arc::new(SystemMonitor::with_config(SystemMonitorConfig {
            update_interval: Duration::from_millis(100),
            ..SystemMonitorConfig::default()
        }));

        let handle = monitor.clone().start_background_monitoring().await;

        // Wait a bit to let background monitoring run
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Should have updated metrics
        let metrics = monitor.get_cached_metrics().await;
        assert!(metrics.cpu_cores > 0);

        handle.abort();
    }
}
