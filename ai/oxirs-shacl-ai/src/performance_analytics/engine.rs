//! Performance analytics engine implementation
//!
//! This module contains the main PerformanceAnalyticsEngine and core functionality.

use std::sync::{Arc, Mutex, RwLock};

use crate::performance_analytics::{
    alerts::AlertEngine, config::PerformanceAnalyticsConfig, dashboard::DashboardProvider,
    metrics::MetricsCollector, monitoring::RealTimeMonitor, optimization::PerformanceOptimizer,
    types::PerformanceStatistics,
};

/// Real-time performance analytics engine
#[derive(Debug)]
pub struct PerformanceAnalyticsEngine {
    config: PerformanceAnalyticsConfig,
    real_time_monitor: Arc<Mutex<RealTimeMonitor>>,
    performance_optimizer: PerformanceOptimizer,
    metrics_collector: MetricsCollector,
    alert_engine: AlertEngine,
    dashboard_provider: DashboardProvider,
    statistics: Arc<RwLock<PerformanceAnalyticsStatistics>>,
}

/// Performance analytics statistics
#[derive(Debug, Default)]
pub struct PerformanceAnalyticsStatistics {
    /// Current performance statistics
    pub current_stats: PerformanceStatistics,

    /// Total monitoring sessions
    pub total_sessions: u64,

    /// Total alerts generated
    pub total_alerts: u64,

    /// Total optimizations performed
    pub total_optimizations: u64,
}

impl PerformanceAnalyticsEngine {
    /// Create a new performance analytics engine
    pub fn new() -> Self {
        Self::with_config(PerformanceAnalyticsConfig::default())
    }

    /// Create a new performance analytics engine with custom configuration
    pub fn with_config(config: PerformanceAnalyticsConfig) -> Self {
        Self {
            real_time_monitor: Arc::new(Mutex::new(RealTimeMonitor::new())),
            performance_optimizer: PerformanceOptimizer::new(),
            metrics_collector: MetricsCollector::new(),
            alert_engine: AlertEngine::new(),
            dashboard_provider: DashboardProvider::new(),
            statistics: Arc::new(RwLock::new(PerformanceAnalyticsStatistics::default())),
            config,
        }
    }

    /// Start real-time monitoring
    pub fn start_monitoring(&self) -> crate::Result<()> {
        if !self.config.enable_real_time_monitoring {
            return Ok(());
        }

        let mut monitor = self.real_time_monitor.lock().unwrap();
        monitor.start()
    }

    /// Stop real-time monitoring
    pub fn stop_monitoring(&self) -> crate::Result<()> {
        let mut monitor = self.real_time_monitor.lock().unwrap();
        monitor.stop()
    }

    /// Get current performance statistics
    pub fn get_statistics(&self) -> PerformanceAnalyticsStatistics {
        self.statistics.read().unwrap().clone()
    }

    /// Update configuration
    pub fn update_config(&mut self, config: PerformanceAnalyticsConfig) {
        self.config = config;
    }
}

impl Default for PerformanceAnalyticsEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for PerformanceAnalyticsStatistics {
    fn clone(&self) -> Self {
        Self {
            current_stats: self.current_stats.clone(),
            total_sessions: self.total_sessions,
            total_alerts: self.total_alerts,
            total_optimizations: self.total_optimizations,
        }
    }
}
