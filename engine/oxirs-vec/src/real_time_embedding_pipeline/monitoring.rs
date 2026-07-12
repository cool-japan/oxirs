//! Performance monitoring and alerting for the real-time embedding pipeline

use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use uuid::Uuid;

use super::config::MonitoringConfig;
use super::traits::{
    Alert, AlertCategory, AlertConfig, AlertHandler, AlertSeverity, AlertThrottling, HealthStatus,
    MetricPoint, MetricsStorage,
};
use super::types::PerformanceMetrics;
use super::PipelineError;

// Display impls needed for formatting in throttle key
impl fmt::Display for AlertCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AlertCategory::Performance => write!(f, "Performance"),
            AlertCategory::Quality => write!(f, "Quality"),
            AlertCategory::Health => write!(f, "Health"),
            AlertCategory::Security => write!(f, "Security"),
            AlertCategory::Configuration => write!(f, "Configuration"),
        }
    }
}

impl fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AlertSeverity::Info => write!(f, "Info"),
            AlertSeverity::Warning => write!(f, "Warning"),
            AlertSeverity::Error => write!(f, "Error"),
            AlertSeverity::Critical => write!(f, "Critical"),
        }
    }
}

/// In-memory metrics storage implementation
pub struct InMemoryMetricsStorage {
    metrics: Vec<(String, MetricPoint)>,
    max_metrics: usize,
}

impl InMemoryMetricsStorage {
    pub fn new(max_metrics: usize) -> Self {
        Self {
            metrics: Vec::new(),
            max_metrics,
        }
    }
}

impl MetricsStorage for InMemoryMetricsStorage {
    fn store_metric(
        &mut self,
        name: &str,
        value: f64,
        timestamp: SystemTime,
        tags: HashMap<String, String>,
    ) -> anyhow::Result<()> {
        self.metrics.push((
            name.to_string(),
            MetricPoint {
                value,
                timestamp,
                tags,
            },
        ));
        while self.metrics.len() > self.max_metrics {
            self.metrics.remove(0);
        }
        Ok(())
    }

    fn get_metrics(
        &self,
        name: &str,
        start: SystemTime,
        end: SystemTime,
    ) -> anyhow::Result<Vec<MetricPoint>> {
        let filtered = self
            .metrics
            .iter()
            .filter(|(n, m)| n == name && m.timestamp >= start && m.timestamp <= end)
            .map(|(_, m)| m.clone())
            .collect();
        Ok(filtered)
    }

    fn get_metric_names(&self) -> anyhow::Result<Vec<String>> {
        let mut names: Vec<String> = self.metrics.iter().map(|(n, _)| n.clone()).collect();
        names.dedup();
        Ok(names)
    }

    fn cleanup_old_metrics(&mut self, cutoff: SystemTime) -> anyhow::Result<usize> {
        let before = self.metrics.len();
        self.metrics.retain(|(_, m)| m.timestamp >= cutoff);
        Ok(before - self.metrics.len())
    }
}

/// Console-based alert handler implementation
pub struct ConsoleAlertHandler;

impl AlertHandler for ConsoleAlertHandler {
    fn handle_alert(&self, alert: &Alert) -> anyhow::Result<()> {
        println!(
            "[ALERT] {} - {} - {} - {}",
            alert.severity, alert.category, alert.source, alert.message
        );
        Ok(())
    }

    fn get_config(&self) -> AlertConfig {
        AlertConfig {
            min_severity: AlertSeverity::Warning,
            throttling: AlertThrottling {
                enabled: false,
                window_duration: Duration::from_secs(60),
                max_alerts_per_window: 100,
            },
            enable_notifications: true,
        }
    }

    fn is_enabled(&self) -> bool {
        true
    }
}

/// Alert manager for handling pipeline alerts
pub struct AlertManager {
    alert_handler: Arc<dyn AlertHandler>,
    active_alerts: Arc<RwLock<HashMap<Uuid, Alert>>>,
    alert_history: Arc<tokio::sync::Mutex<VecDeque<Alert>>>,
    throttling_state: Arc<RwLock<HashMap<String, Instant>>>,
}

impl AlertManager {
    pub fn new(alert_handler: Arc<dyn AlertHandler>) -> Self {
        Self {
            alert_handler,
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            alert_history: Arc::new(tokio::sync::Mutex::new(VecDeque::new())),
            throttling_state: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn register_alert(&self, alert: Alert) -> Result<(), PipelineError> {
        if self.should_throttle_alert(&alert).await {
            return Ok(());
        }

        {
            let mut active_alerts = self.active_alerts.write().await;
            active_alerts.insert(alert.id, alert.clone());
        }

        {
            let mut history = self.alert_history.lock().await;
            history.push_back(alert.clone());
            while history.len() > 1000 {
                history.pop_front();
            }
        }

        self.alert_handler
            .handle_alert(&alert)
            .map_err(|e| PipelineError::MonitoringError {
                message: format!("Alert handling failed: {}", e),
            })
    }

    pub async fn process_alerts(&self) -> Result<(), PipelineError> {
        let alerts = {
            self.active_alerts
                .read()
                .await
                .values()
                .cloned()
                .collect::<Vec<_>>()
        };
        for alert in alerts {
            self.check_alert_status(alert).await?;
        }
        Ok(())
    }

    pub async fn get_active_alerts(&self) -> Vec<Alert> {
        self.active_alerts.read().await.values().cloned().collect()
    }

    async fn should_throttle_alert(&self, alert: &Alert) -> bool {
        let throttle_key = format!("{}:{}", alert.category, alert.source);
        let throttling_state = self.throttling_state.read().await;

        if let Some(last_sent) = throttling_state.get(&throttle_key) {
            let throttle_duration = match alert.severity {
                AlertSeverity::Critical => Duration::from_secs(60),
                AlertSeverity::Error => Duration::from_secs(300),
                AlertSeverity::Warning => Duration::from_secs(600),
                AlertSeverity::Info => Duration::from_secs(1200),
            };
            last_sent.elapsed() < throttle_duration
        } else {
            false
        }
    }

    async fn check_alert_status(&self, _alert: Alert) -> Result<(), PipelineError> {
        Ok(())
    }
}

/// Health checker for monitoring system health
pub struct HealthChecker {
    component_health: Arc<RwLock<HashMap<String, HealthStatus>>>,
}

impl HealthChecker {
    pub fn new() -> Self {
        Self {
            component_health: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn check_health(&self) -> Result<HealthStatus, PipelineError> {
        let component_health = self.component_health.read().await;

        if component_health.is_empty() {
            return Ok(HealthStatus::Healthy);
        }

        for status in component_health.values() {
            if matches!(status, HealthStatus::Unhealthy { .. }) {
                return Ok(HealthStatus::Unhealthy {
                    message: "Component unhealthy".to_string(),
                });
            }
        }

        for status in component_health.values() {
            if matches!(status, HealthStatus::Warning { .. }) {
                return Ok(HealthStatus::Warning {
                    message: "Component warning".to_string(),
                });
            }
        }

        Ok(HealthStatus::Healthy)
    }

    pub async fn get_overall_health(&self) -> Result<HealthStatus, PipelineError> {
        self.check_health().await
    }

    pub async fn update_component_health(&self, component: String, status: HealthStatus) {
        let mut component_health = self.component_health.write().await;
        component_health.insert(component, status);
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics collector for gathering performance data
pub struct MetricsCollector {
    current_metrics: Arc<RwLock<PerformanceMetrics>>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            current_metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
        }
    }

    /// Returns `(name, value, tags)` tuples — MetricPoint has no name field,
    /// so callers are responsible for associating the name when storing.
    pub async fn collect_metrics(
        &self,
    ) -> Result<Vec<(String, f64, HashMap<String, String>)>, PipelineError> {
        let metrics = vec![
            ("cpu_usage".to_string(), 50.0_f64, HashMap::new()),
            (
                "memory_usage".to_string(),
                (1024.0 * 1024.0 * 512.0_f64),
                HashMap::new(),
            ),
            (
                "embedding_throughput".to_string(),
                100.0_f64,
                HashMap::new(),
            ),
        ];
        Ok(metrics)
    }

    pub async fn get_current_metrics(&self) -> Result<PerformanceMetrics, PipelineError> {
        let metrics = self.current_metrics.read().await;
        Ok(metrics.clone())
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance monitoring manager for the pipeline
pub struct PipelinePerformanceMonitor {
    config: MonitoringConfig,
    metrics_storage: Arc<Mutex<InMemoryMetricsStorage>>,
    alert_manager: Arc<AlertManager>,
    health_checker: Arc<HealthChecker>,
    metrics_collector: Arc<MetricsCollector>,
    is_running: Arc<tokio::sync::Mutex<bool>>,
}

impl PipelinePerformanceMonitor {
    /// Create a new performance monitor
    pub fn new(config: MonitoringConfig, alert_handler: Arc<dyn AlertHandler>) -> Self {
        let alert_manager = Arc::new(AlertManager::new(alert_handler));
        let health_checker = Arc::new(HealthChecker::new());
        let metrics_collector = Arc::new(MetricsCollector::new());
        let metrics_storage = Arc::new(Mutex::new(InMemoryMetricsStorage::new(10_000)));

        Self {
            config,
            metrics_storage,
            alert_manager,
            health_checker,
            metrics_collector,
            is_running: Arc::new(tokio::sync::Mutex::new(false)),
        }
    }

    /// Start the monitoring system
    pub async fn start(&self) -> Result<(), PipelineError> {
        let mut running = self.is_running.lock().await;
        if *running {
            return Err(PipelineError::AlreadyRunning);
        }
        *running = true;
        drop(running);

        self.start_metrics_collection().await;
        self.start_health_monitoring().await;
        self.start_alert_processing().await;

        Ok(())
    }

    /// Stop the monitoring system
    pub async fn stop(&self) -> Result<(), PipelineError> {
        let mut running = self.is_running.lock().await;
        if !*running {
            return Err(PipelineError::NotRunning);
        }
        *running = false;
        Ok(())
    }

    /// Record a performance metric by name
    pub fn record_metric(
        &self,
        name: &str,
        value: f64,
        tags: HashMap<String, String>,
    ) -> Result<(), PipelineError> {
        self.metrics_storage
            .lock()
            .store_metric(name, value, SystemTime::now(), tags)
            .map_err(|e| PipelineError::MonitoringError {
                message: format!("Failed to store metric: {}", e),
            })
    }

    /// Get current performance metrics
    pub async fn get_current_metrics(&self) -> Result<PerformanceMetrics, PipelineError> {
        self.metrics_collector.get_current_metrics().await
    }

    /// Get health status
    pub async fn get_health_status(&self) -> Result<HealthStatus, PipelineError> {
        self.health_checker.get_overall_health().await
    }

    /// Register an alert
    pub async fn register_alert(&self, alert: Alert) -> Result<(), PipelineError> {
        self.alert_manager.register_alert(alert).await
    }

    async fn start_metrics_collection(&self) {
        let metrics_collector = Arc::clone(&self.metrics_collector);
        let metrics_storage = Arc::clone(&self.metrics_storage);
        let is_running = Arc::clone(&self.is_running);
        let collection_interval = Duration::from_millis(self.config.metrics_interval_ms);

        tokio::spawn(async move {
            loop {
                {
                    let running = is_running.lock().await;
                    if !*running {
                        break;
                    }
                }
                if let Ok(metrics) = metrics_collector.collect_metrics().await {
                    let timestamp = SystemTime::now();
                    let mut storage = metrics_storage.lock();
                    for (name, value, tags) in metrics {
                        let _ = storage.store_metric(&name, value, timestamp, tags);
                    }
                }
                tokio::time::sleep(collection_interval).await;
            }
        });
    }

    async fn start_health_monitoring(&self) {
        let health_checker = Arc::clone(&self.health_checker);
        let alert_manager = Arc::clone(&self.alert_manager);
        let is_running = Arc::clone(&self.is_running);

        tokio::spawn(async move {
            loop {
                {
                    let running = is_running.lock().await;
                    if !*running {
                        break;
                    }
                }
                if let Ok(health_status) = health_checker.check_health().await {
                    if !matches!(health_status, HealthStatus::Healthy) {
                        let alert = Alert {
                            id: Uuid::new_v4(),
                            category: AlertCategory::Health,
                            severity: AlertSeverity::Warning,
                            message: format!("Health check failed: {:?}", health_status),
                            timestamp: SystemTime::now(),
                            source: "health_monitor".to_string(),
                            details: HashMap::new(),
                        };
                        let _ = alert_manager.register_alert(alert).await;
                    }
                }
                tokio::time::sleep(Duration::from_secs(30)).await;
            }
        });
    }

    async fn start_alert_processing(&self) {
        let alert_manager = Arc::clone(&self.alert_manager);
        let is_running = Arc::clone(&self.is_running);

        tokio::spawn(async move {
            loop {
                {
                    let running = is_running.lock().await;
                    if !*running {
                        break;
                    }
                }
                let _ = alert_manager.process_alerts().await;
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::real_time_embedding_pipeline::config::MonitoringConfig;

    #[tokio::test]
    async fn test_metrics_collector_creation() {
        let collector = MetricsCollector::new();
        let metrics = collector.collect_metrics().await.expect("should collect");
        assert!(!metrics.is_empty());
    }

    #[tokio::test]
    async fn test_health_checker() {
        let checker = HealthChecker::new();
        let health = checker.get_overall_health().await.expect("should check");
        assert!(matches!(health, HealthStatus::Healthy));
    }

    #[tokio::test]
    async fn test_alert_manager() {
        let handler = Arc::new(ConsoleAlertHandler);
        let manager = AlertManager::new(handler);

        let alert = Alert {
            id: Uuid::new_v4(),
            category: AlertCategory::Performance,
            severity: AlertSeverity::Warning,
            message: "Test alert".to_string(),
            timestamp: SystemTime::now(),
            source: "test".to_string(),
            details: HashMap::new(),
        };

        let result = manager.register_alert(alert).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_in_memory_metrics_storage() {
        let mut storage = InMemoryMetricsStorage::new(100);
        let result = storage.store_metric("test_metric", 42.0, SystemTime::now(), HashMap::new());
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_monitor_creation() {
        let config = MonitoringConfig::default();
        let handler = Arc::new(ConsoleAlertHandler);
        let monitor = PipelinePerformanceMonitor::new(config, handler);
        let health = monitor
            .get_health_status()
            .await
            .expect("should check health");
        assert!(matches!(health, HealthStatus::Healthy));
    }

    #[test]
    fn test_console_alert_handler_config() {
        let handler = ConsoleAlertHandler;
        assert!(handler.is_enabled());
        let config = handler.get_config();
        assert!(config.enable_notifications);
    }

    #[test]
    fn test_alert_severity_display() {
        assert_eq!(format!("{}", AlertSeverity::Critical), "Critical");
        assert_eq!(format!("{}", AlertSeverity::Warning), "Warning");
    }

    #[test]
    fn test_alert_category_display() {
        assert_eq!(format!("{}", AlertCategory::Performance), "Performance");
        assert_eq!(format!("{}", AlertCategory::Health), "Health");
    }

    #[tokio::test]
    async fn test_health_checker_with_component() {
        let checker = HealthChecker::new();
        checker
            .update_component_health(
                "test_component".to_string(),
                HealthStatus::Warning {
                    message: "test warning".to_string(),
                },
            )
            .await;
        let health = checker.check_health().await.expect("should check");
        assert!(matches!(health, HealthStatus::Warning { .. }));
    }

    #[test]
    fn test_in_memory_metrics_storage_cleanup() {
        let mut storage = InMemoryMetricsStorage::new(100);
        let past = SystemTime::now()
            .checked_sub(Duration::from_secs(3600))
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let _ = storage.store_metric("old_metric", 1.0, past, HashMap::new());
        let _ = storage.store_metric("new_metric", 2.0, SystemTime::now(), HashMap::new());

        let cutoff = SystemTime::now()
            .checked_sub(Duration::from_secs(1800))
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let removed = storage.cleanup_old_metrics(cutoff).expect("cleanup ok");
        assert_eq!(removed, 1);
    }

    #[test]
    fn test_in_memory_metrics_storage_get() {
        let mut storage = InMemoryMetricsStorage::new(100);
        let now = SystemTime::now();
        let _ = storage.store_metric("cpu", 75.0, now, HashMap::new());

        let start = now.checked_sub(Duration::from_secs(1)).unwrap_or(now);
        let end = now
            .checked_add(Duration::from_secs(1))
            .unwrap_or_else(SystemTime::now);
        let results = storage.get_metrics("cpu", start, end).expect("should get");
        assert_eq!(results.len(), 1);
        assert!((results[0].value - 75.0).abs() < f64::EPSILON);
    }
}
